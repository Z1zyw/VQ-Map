import torch
from torch import nn
from typing import Any, Dict

from mmdet3d.models import VQDAE
from mmdet3d.models.builder import build_vqvae, build_tokenencoder

import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from collections import OrderedDict
import torch.distributed as dist

def load_model(model, checkpoint, strict=True):
    """Load checkpoint to model.

    Args:
        model (nn.Module): The model to be loaded.
        checkpoint (str): The checkpoint file.
        strict (bool, optional): Whether to allow different params for the model and checkpoint. Defaults to True.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an instance of nn.Module. Got {type(model)}")
    if not isinstance(checkpoint, str):
        raise TypeError(f"checkpoint must be a str. Got {type(checkpoint)}")
    checkpoint = torch.load(checkpoint, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=strict)

@VQDAE.register_module()
class BaseDAE(nn.Module):
    def __init__(
        self,
        vqvae: Dict[str, Any],
        noise_model: Dict[str, Any],
        token_encoder: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()
        self.vqvae = build_vqvae(vqvae)
        if "checkpoint" in vqvae and vqvae["checkpoint"] is not None:
            load_model(self.vqvae, vqvae["checkpoint"])
        self.freeze_vqvae()
        self.vqvae.eval()
        self.vqvae.codebook.embedding.update = False
        
        self.token_encoder = build_tokenencoder(token_encoder)
        self.token_encoder.pretraining = True
        if noise_model is not None:
            self.noise_model = build_tokenencoder(noise_model)
    
    def freeze_vqvae(self):
        for param in self.vqvae.parameters():
            param.requires_grad = False
    
    def cal_token_acc(self, token_gt, token_pred):
        return (token_gt == token_pred).sum().float() / token_gt.flatten().shape[0]
    
    def get_all_tokens(self, bev):
        return self.vqvae.get_all_tokens(bev)
    
    def forward_train(self, gt_masks_bev, **kwargs):
        raise NotImplementedError
    
    def sequence2grid(self, sequence: torch.Tensor):
        l, c = sequence.shape[1:]
        w, h = int(l ** 0.5), int(l ** 0.5)
        assert w * h == l
        return sequence.reshape(-1, h, w, c).permute(0, 3, 1, 2)
    
    def _generate_bev(self, token):
        features_seq = self.vqvae.codebook.embedding(token)
        features_grid = self.sequence2grid(features_seq)
        bev_logits = self.vqvae.decoder(features_grid)
        return bev_logits
    
    def token2bev(self, token):
        encoder_token = self.token_encoder(token)
        bev_pred = self._generate_bev(encoder_token.argmax(-1))
        return bev_pred
    
    def forward_test(self, gt_masks_bev):
        raise  NotImplementedError

    def weight_ce_loss(self, pred, target, weight):
        raise NotImplementedError
    
    def token_focal_loss(self, pred, target, gamma=2.0):
        c = pred.shape[-1]
        pred_ = pred.reshape(-1, c)
        target_ = target.reshape(-1)
        ce_loss = F.cross_entropy(pred_, target_, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
            
    def forward(self, gt_masks_bev, **kwargs):  
        if self.training:
            return self.forward_train(gt_masks_bev)
        else:
            return self.forward_test(gt_masks_bev)
        
        
    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data["metas"]))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data["metas"]))

        return outputs