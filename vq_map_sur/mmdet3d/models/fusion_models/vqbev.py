from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
    build_vqdae,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS

import pickle
from .base import Base3DFusionModel

__all__ = ["VQBev"]

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
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        print(f"Ignore: {e}")


@FUSIONMODELS.register_module()
class VQBev(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        encoders_freeze: bool = False,
        supervision: str = "index",
        use_mask: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.supervision = supervision
        self.use_mask = use_mask
        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                    "dense2sparse": build_vtransform(encoders["lidar"]["dense2sparse"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None
        if decoder is not None:
            self.decoder = build_vqdae(decoder)
            self.decoder.token_encoder.pretraining = False

            if "checkpoint" in decoder and decoder["checkpoint"] is not None:
                load_model(self.decoder, decoder["checkpoint"], strict=False)
            self.freeze_decoder()
        
        self.heads = nn.ModuleDict()
        if heads['token'] is not None:
            self.heads['token'] = build_head(heads['token'])


        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.init_weights()

    def get_mask_seq(self, mask_:torch.Tensor, p:int=16, threshold:float=0.4):
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
    
    def freeze_encoders(self):
        for sensor in self.encoders:
            for param in self.encoders[sensor].parameters():
                param.requires_grad = False
    
    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def extract_camera_features(
        self,
        x,
        points,
        camera2ego, # B, 6, 4, 4
        lidar2ego,  # B, 4, 4
        lidar2camera, # B, 6, 4, 4
        lidar2image,  # B, 6, 4, 4
        camera_intrinsics, # B, 6, 4, 4
        camera2lidar, # B, 6, 4, 4
        img_aug_matrix, # B, 6, 4, 4
        lidar_aug_matrix, # B, 4, 4
        img_metas, # 
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        x = list(x)
        for i in range(len(x)):
            BN, C, H, W = x[i].size()
            x[i] = x[i].view(B, int(BN / B), C, H, W) # B, 6, 256, 32, 88
            
        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)

        x = self.encoders["lidar"]["dense2sparse"](x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if self.use_mask and self.training:
            lidar_mask = kwargs['lidar_mask'].float()
            lidar_mask_seq = self.get_mask_seq(lidar_mask.unsqueeze(1), p=8, threshold=0.4)
            cam_valid_indices = lidar_mask_seq
        else:
            cam_valid_indices = None
        features = []
        
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
                if isinstance(feature, tuple):
                    feature, cam_valid_indices_ = feature
                    if cam_valid_indices is not None:
                        cam_valid_indices = cam_valid_indices & cam_valid_indices_
                    else:
                        cam_valid_indices = cam_valid_indices_
                else:
                    # cam_valid_indices = None
                    pass 
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)

            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = img.shape[0]
        
        # sequence features
        if len(x.shape) == 3:
            # pass 
            raise NotImplementedError   
        
        if self.training:
            if self.supervision == 'index':
                token_gt = self.decoder.get_all_tokens(gt_masks_bev.float().detach())['token'] + 1
            elif self.supervision == 'lat_q':
                token_gt = self.decoder.get_all_tokens(gt_masks_bev.float().detach())['lat_q']
            elif self.supervision == 'lat_c':
                token_gt = self.decoder.get_all_tokens(gt_masks_bev.float().detach())['lat_c']
                token_gt = F.normalize(token_gt, p=2, dim=-1)
            else:
                raise NotImplementedError
            outputs = {}
            for type, head in self.heads.items():
                if type == "token":
                    losses = head(x, token_gt, mask=cam_valid_indices)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "token":
                    token_pred = head(x, mask=cam_valid_indices)
                    logits = self.generate_bev(token_pred)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

    # must be fp32, because using sigmoid pretrained model in fp32
    @ torch.no_grad()
    @ force_fp32()
    def generate_bev(self, input):
        if self.supervision == 'index':
            return self.decoder.soft_token2bev(input)
        elif self.supervision == 'lat_q' or self.supervision == 'lat_c':
            return self.decoder.lat2bev_with_codebook(input)
        else:
            raise NotImplementedError
        