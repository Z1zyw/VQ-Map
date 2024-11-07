import torch
from torch import nn
from typing import Any, Dict

from mmdet3d.models import VQVAE
from mmdet3d.models.builder import build_encoder, build_decoder, build_codebook
from mmcv.runner import auto_fp16, force_fp32

import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from collections import OrderedDict
import torch.distributed as dist

__all__ = ['CNNVQVAE']

@VQVAE.register_module()
class CNNVQVAE(nn.Module):
    def __init__(
        self,
        encoder: Dict[str, Any],
        codebook: Dict[str, Any],
        decoder: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.patch_size = kwargs.get("patch_size", 8)
        encoder.update({"patch_size": self.patch_size})
        decoder.update({"patch_size": self.patch_size})
        self.encoder = build_encoder(encoder)
        self.codebook = build_codebook(codebook)
        self.decoder = build_decoder(decoder)
        
        # self.fp16_enabled = False # if fp16_enabled variable is defined, the model can be converted to fp16
        
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                
    def patch_augmentation(self, x, p=0.9, p_rotate=0.5, p_translation=0.5, p_resize=0.5):
        # x.shape = (b, C, P, P)
        x_ = x.clone()
        ps = self.patch_size
        for i in range(x.shape[0]):
            if random.random() < p:
                if random.random() < p_rotate:
                    x_[i] = TF.rotate(x_[i], angle=torch.randint(-30, 30, (1,)).item())
                if random.random() < p_translation:
                    x_[i] = TF.affine(x_[i], angle=0, translate=(torch.randint(-2, 2, (1,)).item(), torch.randint(-2, 2, (1,)).item()), scale=1, shear=0)
                if torch.rand(1) < p_resize:
                    tmp = TF.resize(x_[i], (torch.randint(ps, ps+2, (1,)).item(), torch.randint(ps, ps+2, (1,)).item()), interpolation=TF.InterpolationMode.NEAREST)
                    x_[i] = TF.center_crop(tmp, (ps, ps))
        return x_
    
    
    @ staticmethod
    def get_mask_seq(mask_:torch.Tensor, p:int=16, threshold:float=0.4):
        with torch.no_grad():
            threshold = (threshold * torch.tensor([p*p])).to(mask_.device)
            B, C, H, W = mask_.shape
            assert H % p == 0 and W % p == 0
            mask = mask_.reshape(B, C, H//p, p, W//p, p)
            mask = torch.einsum("nchpwq->nhwcpq", mask)
            mask = mask.reshape(B, H//p * W//p, C*p*p)
            mask = mask > 0.5
            mask = mask.sum(dim=-1) > threshold
        return mask
    
    def _get_patches(self, x, logits_weight:torch.Tensor=None, mask=None):
        B = x.shape[0]
        x_patch, hw = self.patchify(x)
        if logits_weight is not None:
            C = x.shape[1]
            logits_weight = logits_weight.reshape(-1, C+1)
            weight = logits_weight[:,:C, None, None] * (x > 0).float()
            blank = (x.sum(dim=1, keepdim=True) < -C + 1).float() * logits_weight[:, -1, None, None]
            weight_patch, _ = self.patchify(weight)
            blank_patch, _ = self.patchify(blank)
            weight = torch.sum(weight_patch, dim=(1, 2, 3)) + torch.sum(blank_patch, dim=(1, 2, 3))
            weight = weight.reshape(B, 1, hw[0], hw[1])
        else:
            weight = None
            
        if mask is not None:
            mask_seq = self.get_mask_seq(mask, p=self.patch_size).reshape(B, 1, hw[0], hw[1])
        else:
            mask_seq = None
        return x_patch, hw, weight, mask_seq
        
    @ property
    def crop_sizes(self):
        crop_sizes = [
            (0, 0.5, 0, 1),
            (0.5, 0, 1, 0),
            (0.5, 0.5, 1, 1),
        ]
        return crop_sizes
    
    @ torch.no_grad()
    def get_latent(self, x, aug=False, logits_weight:torch.Tensor=None, mask=None, dense=False, offset=None):
        if offset is not None:
            dense = False
            shape = x.shape[-2:]
            offset = (
                int(offset[0]),
                int(offset[1]),
                (shape[0] - self.patch_size) if offset[0] > 0 else shape[0],
                (shape[1] - self.patch_size) if offset[1] > 0 else shape[1]
            )
            x = TF.crop(x, *offset)
            if mask is not None:
                mask = TF.crop(mask, *offset)
        
        x_patch, hw, weight, mask_seq = self._get_patches(x, logits_weight, mask)
        N, C, P, _ = x_patch.shape
        lat = self.encoder(x_patch, is_training=False) 
        lat = self.unpatchify(lat, hw) 
        
        if dense:
            h, w = hw
            B, C, H, W = lat.shape
            lat_dense = torch.zeros(B, C, H*2-1, W*2-1).to(lat.device)
            lat_dense[:, :, ::2, ::2] = lat
            if mask is not None:
                mask_dense = torch.zeros(B, 1, H*2-1, W*2-1).to(mask.device)
                mask_dense[:,:, ::2, ::2] = self.unpatchify(mask_seq, hw)
            
            if weight is not None:
                weight_dense = torch.zeros(B, 1, H*2-1, W*2-1).to(weight.device)
                weight_dense[:, :, ::2, ::2] = self.unpatchify(weight, hw)
            
            for crop_size in self.crop_sizes:
                begin_h = int(crop_size[0] * P)
                begin_w = int(crop_size[1] * P)
                crop_h = int((h - crop_size[2]) * P)
                crop_w = int((w - crop_size[3]) * P)
                x_ = TF.crop(x, begin_h, begin_w, crop_h, crop_w)
                mask_ = TF.crop(mask, begin_h, begin_w, crop_h, crop_w) if mask is not None else None
                x_patch_, hw_, weight_, mask_seq_ = self._get_patches(x_, logits_weight, mask_)
                lat_ = self.encoder(x_patch_, is_training=False)
                lat_ = self.unpatchify(lat_, hw_)
                lat_dense[:, :,int(crop_size[0]*2)::2, int(crop_size[1]*2)::2] = lat_
                
                if mask is not None:
                    mask_dense[:,:,int(crop_size[0]*2)::2, int(crop_size[1]*2)::2] = self.unpatchify(mask_seq_, hw_)
                
                if weight is not None:
                    weight_dense[:,:, int(crop_size[0]*2)::2, int(crop_size[1]*2)::2] = self.unpatchify(weight_, hw_)
            
            lat = lat_dense
            if mask is not None:
                mask_seq = mask_dense
            if weight is not None:
                weight = weight_dense        
        if aug:
            x_patch = self.patch_augmentation(x_patch)

        B, C, H, W = lat.shape
        lat = lat.reshape(B, C, -1).permute(0, 2, 1)
        
        _, lat_q, _, _, encoding_indices = self.codebook(lat)
        
        if weight is not None:
            weight = weight.reshape(B, -1)
        
        if mask_seq is not None:
            mask_seq = mask_seq.reshape(B, -1)
        
        return {
            "lat_q": lat_q,
            "lat": lat,
            "token": encoding_indices,
            "weight": weight,
            "mask_seq": mask_seq.bool() if mask_seq is not None else None
        }
    
    
    def get_all_tokens(self, x, logits_weight:torch.Tensor=None):

        x_patch, hw = self.patchify(x) 
        
        if logits_weight is not None:
            C = x.shape[1]
            logits_weight = logits_weight.reshape(-1, C+1)
            weight = logits_weight[:,:C, None, None] * (x > 0).float()
            blank = (x.sum(dim=1, keepdim=True) < -C + 1).float() * logits_weight[:, -1, None, None]
            weight_patch, _ = self.patchify(weight)
            blank_patch, _ = self.patchify(blank)
            weight = torch.sum(weight_patch, dim=(1, 2, 3)) + torch.sum(blank_patch, dim=(1, 2, 3))
        else:
            weight = None
            
        lat = self.encoder(x_patch, is_training=False)
        lat = self.unpatchify(lat, hw) 
        B, C, H, W = lat.shape
        lat = lat.reshape(B, C, -1).permute(0, 2, 1)
        
        _, lat_q, _, _, encoding_indices = self.codebook(lat)
        
        return {
            "token": encoding_indices,
            "lat_q": lat_q,
            "lat_c": lat,
            "weight": weight if logits_weight is not None else None
        }

    def _default_weight(self, y):
        H, W = y.shape[-2:]
        count = torch.sum(y, dim=(2, 3))
        weight = float(H * W) / (count + 1)
        return weight
    
    def forward_construction_loss(self, x, y, weight):
        # x.shape = (B, C, H, W)
        # loss = F.mse_loss(x, y)
        if weight is not None:
            assert x.shape[1] == weight.shape[0]

        loss = F.mse_loss(x, y, reduction='none')
        loss = torch.mean(loss, dim=(2, 3))
        if weight is None:
            weight = self._default_weight(y)
        loss = loss * weight
            
        loss = torch.mean(loss)
        return loss
    
    def patchify(self, x):
        p = self.patch_size
        B, C, H, W = x.shape
        h = H // p
        w = W // p
        x = x.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(-1, C, p, p)
        return x, (h, w)
    
    def unpatchify(self, x, hw):
        h, w = hw
        _, C, _, _ = x.shape
        x = x.reshape(-1, h, w, C)
        x = x.permute(0, 3, 1, 2)
        return x
    
    @auto_fp16()
    def forward_train(self, x, weight=None):
        x_patch, hw = self.patchify(x)
        lat = self.encoder(x_patch, self.training) 
        
        lat_list = None
        if isinstance(lat, tuple):
            lat, lat_list = lat


        lat = self.unpatchify(lat, hw)
        
        B, C_l, h, w = lat.shape 
        lat = lat.reshape(B, C_l, -1).permute(0, 2, 1)
        
        loss_codebook, lat_q, perplexity, _, encoding_indices = self.codebook(lat)
        
        if lat_list is not None:
            for lat_ in lat_list:
                loss_codebook = loss_codebook + self.codebook.beta * F.mse_loss(lat_.reshape(lat_q.shape), lat_q.detach())
        
        lat_q = lat_q.permute(0, 2, 1).reshape(B, C_l, h, w)
        y_q = self.decoder(lat_q)
        loss_reconstruction = self.forward_construction_loss(x, y_q, weight)
        
        outputs = {
            "loss/reconstruction": loss_reconstruction,
            "loss/codebook": loss_codebook,
            "perplexity": perplexity,
        }
        return outputs
        
    
    @auto_fp16(apply_to=("x"))
    def forward_test(self, x, weight=None):

        x_patch, hw = self.patchify(x)
        
        lat = self.encoder(x_patch, False) 

        lat = self.unpatchify(lat, hw)
        
        B, C_l, h, w = lat.shape 
        lat = lat.reshape(B, C_l, -1).permute(0, 2, 1)
        
        loss_codebook, lat_q, perplexity, _, encoding_indices = self.codebook(lat)

        lat_q = lat_q.permute(0, 2, 1).reshape(B, C_l, h, w)
        y_q = self.decoder(lat_q)
        
        b = x.shape[0]
        outputs = [{} for _ in range(b)]
        for i in range(b):
            outputs[i].update(
                {
                    "masks_bev": y_q[i].cpu(),
                    "gt_masks_bev": x[i].cpu(),
                }
            )
        return outputs
    
    def get_latent_code(self, x, weight=None):
        x = x.float()
        x_patch, hw = self.patchify(x)
        
        lat = self.encoder(x_patch, False) 

        lat = self.unpatchify(lat, hw)
        
        B, C_l, h, w = lat.shape 
        lat = lat.reshape(B, C_l, -1).permute(0, 2, 1)
        loss_codebook, lat_q, perplexity, _, encoding_indices = self.codebook(lat)

        return encoding_indices
    
    def forward(self, gt_masks_bev, weight=None, **args):
        gt_masks_bev = gt_masks_bev.float()
        if self.training:
            return self.forward_train(gt_masks_bev, weight)
        else:
            return self.forward_test(gt_masks_bev, weight)
    
    
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