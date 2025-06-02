import os
import sys
sys.path.append(os.getcwd())
import ipdb
import math
import torch
import torch.nn as nn 
import torchvision.models as models
import torchvision.transforms.functional as TF
import timm
import random

from torchvision.models._utils import IntermediateLayerGetter
from src.utils.config import BasicConfig
from torchvision.models.resnet import BasicBlock
from src.model.img_backbone import build_backbone

from torch.nn.init import normal_

import copy
import numpy as np

from timm.models.vision_transformer import PatchEmbed, Block
from src.model.deformable_detr.misc import NestedTensor, nested_tensor_from_tensor_list
from src.model.ops.modules.ms_deform_attn import MSDeformAttn_v2 as MSDeformAttn
import torch.nn.functional as F 
        
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dropout = nn.Dropout(dropout)
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.dropout(x) if i > 0 else x
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class BEVDeformableLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None, 
                src_valid_ratios=None, attn_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=attn_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask,src_valid_ratios)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class TokenDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios=None,
                query_pos=None, src_padding_mask=None, attn_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            assert reference_points.shape[-1] == 2
            if src_valid_ratios is not None:
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            else:
                reference_points_input = reference_points[:, :, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, attn_mask)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
            
            
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points


class Image2TokenConfig(BasicConfig):
    def __init__(self, cfg=None):
        self.bev_size = [112, 112]
        self.z_range = [0, 50]
        self.x_range = [-25, 25]
        self.y_range = [0, 3]
        self.patch_size = 16
        
        self.img_shape = [900, 1600]
        
        self.predict_label_num = 4
        
        # build backbone
        self.backbone = "resnet50"
        # self.lr_backbone = 1e-5
        self.position_embedding = "sine"
        self.num_feature_levels = 4
        self.masks = False
        self.dilation = False
        
        self.y_sample_num = 5
        
        self.num_query_each_token = 4 # number of query to get a single token
        self.num_token_each_patch = 1 # corresponding to the number of codebook
        self.dim_model = 256
        self.dim_feedforward = 1024
        self.dropout = 0.1
        self.activation = "relu"

        self.num_decoder_layers = 4
        self.num_decoder_head = 8
        self.num_reference_points = 32
        
        self.sample_dense = 2
        self.y_sample_num = 4
        
        self.embedding = "all" # "all" or "z" or "one"
        self.codebook_size = 64
        self.codebook_dim = 256
        
        self.return_intermediate = False
        
        self.use_reference_refine = False
        
        self.use_semantic_depth_decoder = False
        # self.num_sd_decoder_layers = 4
        # self.num_sd_decoder_head = 8
        # self.num_sd_decoder_points = 4
        
        # self.semantic_label_num = 32
        
        self.pred_codebook_feature = False
        
        self.pred_confidence = False
        
        self.use_depth_level_weight = False
        
        self.use_global_attention = True
        
        self.query_whl = [1., 1., 1.]
        
        self.use_perspective = False
        self.token_fusion_persperctive = False
        
        self.level_independent = False
        
        self.use_pos_fusion = False
        
        self.self_attention_type = 'default'
        # self.self_attention_type = 'defromable' # or 'default'

        self.pixel_fusion = False
        
        self.use_cnn = False
        
        super().__init__(cfg)


from src.model.neck import build_neck
from src.model.pos_emb import PositionEmbeddingSine
class Image2Token(nn.Module):
    def __init__(self, cfg: Image2TokenConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_patches = (cfg.bev_size[0] // cfg.patch_size) * (cfg.bev_size[1] // cfg.patch_size)
        self.num_patches_z = cfg.bev_size[0] // cfg.patch_size
        self.num_anchors = cfg.y_sample_num * cfg.sample_dense ** 2
        assert cfg.num_reference_points % self.num_anchors == 0
    
        # query embedding 
        if cfg.embedding == "all":
            num_patches_x = self.num_patches // self.num_patches_z
            # self.query_embedding = nn.Parameter(torch.zeros(1, cfg.num_query_each_token, (self.num_patches_z * 2 - 1) * (num_patches_x * 2 - 1), cfg.dim_model))
            self.query_embedding = nn.Parameter(torch.zeros(1, cfg.num_query_each_token, self.num_patches, cfg.dim_model))
        elif cfg.embedding == "z":
            self.query_embedding = nn.Parameter(torch.zeros(1, cfg.num_query_each_token, self.num_patches_z, cfg.dim_model))
        elif cfg.embedding == "one":
            self.query_embedding = nn.Parameter(torch.zeros(1, cfg.num_query_each_token, 1, cfg.dim_model))
        elif cfg.embedding == "pers":
            assert cfg.use_perspective 
            assert cfg.num_token_each_patch == 1
            self.query_embedding = nn.Parameter(torch.zeros(1, cfg.predict_label_num + 1, 1, cfg.dim_model))
            self.pers_heads = nn.Linear(cfg.dim_model, cfg.predict_label_num + 1)
        else:
            raise NotImplementedError
        
        # build decoder to get bev patch feature
        decoder_layer = BEVDeformableLayer(
            d_model=cfg.dim_model, 
            d_ffn=cfg.dim_model * 4,
            dropout=cfg.dropout,
            activation=cfg.activation,
            n_levels=cfg.num_feature_levels,
            n_heads=cfg.num_decoder_head,
            n_points=cfg.num_reference_points
        )
        
        self.token_height_embed = None
        if cfg.use_reference_refine:
            token_height_embed = []
            for _ in range(cfg.num_decoder_layers - 1):
                token_height_embed.append(nn.Linear(cfg.dim_model, 6))
            self.token_height_embed = nn.ModuleList(token_height_embed)
            
        self.attn_mask = None
        self.attn_method = None
        self.img2bev_token_decoder = TokenDecoder(
            decoder_layer=decoder_layer, 
            num_layers=cfg.num_decoder_layers, 
            return_intermediate=cfg.return_intermediate,
        )

        self.token_head = MLP(
            input_dim=cfg.dim_model,
            hidden_dim=cfg.dim_model * 4,
            output_dim=cfg.codebook_size + 1,
            num_layers=3
        )

        if cfg.use_cnn:
            self.conv = nn.Sequential(
                nn.Conv2d(cfg.dim_model, cfg.dim_model, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(cfg.dim_model),
                nn.ReLU(),
            )
        else:
            self.conv = None
        
        self._reset_parameters()
        
        # build backbone and neck
        backbone = build_backbone(cfg.backbone)
        self.backbone_with_neck = build_neck(
            'fpn',
            backbone,
            out_channels=cfg.dim_model,
            # return_layers={"layer1": "1", "layer2": "2", "layer3": "3", "layer4": "4"},
            # return_layers={"layer2": "1", "layer3": "2", "layer4": "3", "conv5": "4"},
            return_layers={"layer2": "1", "layer3": "2", "layer4": "3"},
        )
        # build pos embedding 
        self.pos_emb = PositionEmbeddingSine(cfg.dim_model // 2, normalize=True)
        self.query_pos_emb = PositionEmbeddingSine(cfg.dim_model // 2, normalize=True)
    
    @ staticmethod
    def get_reference_points(
        offset=None, 
        num_token:int=196, 
        train_dense:int=2, 
        dense_sample:int=2, 
        h_sample:int=4,
        z_range=[1, 50],
        x_range=[-25, 25],
        y_range=[0, 3],
    ):
        # train_dense is query number
        # dense_sample dense_sample is the reference point number of each query: num_ref = dense_sample ** 2 * y_sample_num
        H, W = int(num_token ** 0.5), int(num_token ** 0.5)
        assert H * W == num_token
        H = (H - 1) * train_dense + 1
        W = (W - 1) * train_dense + 1
        if offset is not None:
            # assert NotImplementedError
            if offset[0] > 0:
                H = H - 1
                W = W - 1
                x_range[0] = x_range[0] + offset[1].item()
                z_range[0] = z_range[0] + offset[0].item()

        H = H * dense_sample
        W = W * dense_sample
        Z = h_sample
        
        # HWZ are in BEV space, Z is height
        w = torch.linspace(0.5, W - 0.5, W).view(1, 1, W).expand(Z, H, W) / W # cam_x 
        h = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(Z, H, W) / H # cam_z 
        z = torch.linspace(0.5, Z - 0.5, Z).view(Z, 1, 1).expand(Z, H, W) / Z # cam_y
        
        ref_3d = torch.stack((w, h, z), -1) # (Z, W, H, 3) 
        ref_3d = ref_3d.reshape(Z, W//dense_sample, dense_sample, H//dense_sample, dense_sample, 3) 
        ref_3d = ref_3d.permute(0, 2, 4, 1, 3, 5).reshape(Z * dense_sample * dense_sample, -1, 3) # (h_samples * dense * dense, num_token, 3))
        # return ref_3d[None] # (1, h_samples * dense * dense, num_token, 3) 
        ref_3d[..., 0] = ref_3d[..., 0] * (x_range[1] - x_range[0]) + x_range[0]
        ref_3d[..., 1] = ref_3d[..., 1] * (z_range[1] - z_range[0]) + z_range[0]
        ref_3d[..., 2] = ref_3d[..., 2] * (y_range[1] - y_range[0]) + y_range[0]
        
        return ref_3d[None] # (1, h_samples * dense * dense, num_token, 3) in camera
        
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def get_cam_patch_position(self, dense=1, offset=None):
        # get patch position(patch's center) in cam coordinate
        # offset is (z, x in cam coordinate)
        if offset is not None:
            dense = 1
        else:
            offset = (0, 0)
               
        z_range = torch.linspace(
            self.cfg.z_range[0] + self.patch_size_cam[0] / 2 + offset[0],
            self.cfg.z_range[1] - self.patch_size_cam[0] / 2 + offset[0],
            (int(self.cfg.bev_size[0] / self.cfg.patch_size) - 1) * dense + 1
        )
        z_range = z_range[
            (z_range <= self.cfg.z_range[1] - self.patch_size_cam[0] / 2) & 
            (z_range >= self.cfg.z_range[0] + self.patch_size_cam[0] / 2)
        ]
        
        x_range = torch.linspace(
            self.cfg.x_range[0] + self.patch_size_cam[1] / 2 + offset[1],
            self.cfg.x_range[1] - self.patch_size_cam[1] / 2 + offset[1],
            int(self.cfg.bev_size[1] / self.cfg.patch_size - 1) * dense + 1
        )
        x_range = x_range[
            (x_range <= self.cfg.x_range[1] - self.patch_size_cam[1] / 2) &
            (x_range >= self.cfg.x_range[0] + self.patch_size_cam[1] / 2)
        ]
        
        z, x = torch.meshgrid(z_range, x_range)
        return torch.stack([z, x], dim=-1).reshape(-1, 2)
        
    def get_bev2img_reference(self, calib, img_shape=None, y_sample=None, return_dim:int=6, whl=(1., 1., 1.), height=None, offset=None, scale=False):
        # return 2d img reference in normalized coordinate [0~1]
        # return shape [batch_size, y_sample_num, num_patch, 2]
        if img_shape is None:
            img_shape = self.cfg.img_shape
        assert len(img_shape) == 2
        
        if y_sample == None:
            y_sample = self.y_sample
        
        h, w = img_shape

        # calib = calib['camera_intrinsic'] # [b, 3, 3]
        instrinsic = calib['camera_intrinsic'] # [b, 3, 3]
        # cam_translation = calib['camera_translation'] # [b, 3]
        # cam_rotation = calib['camera_rotation'] # [b, 3, 3]
        if self.training:
            if offset is not None:
                patch_position_cam = self.get_cam_patch_position(offset=offset).to(device=instrinsic.device) # [num_patch, 2]
            else:
                patch_position_cam = self.patch_position_cam_dense.to(device=instrinsic.device) # [num_patch, 2]
        else:
            patch_position_cam = self.patch_position_cam.to(device=instrinsic.device) # [num_patch, 2]
        
        b = instrinsic.shape[0]
        f, cu, cv = instrinsic[:, 0, 0], instrinsic[:, 0, 2], instrinsic[:, 1, 2]
        f, cu, cv = f[:, None], cu[:, None], cv[:, None] # equal to unsqueeze(1), [b, 1, 1]
        z, x = patch_position_cam[:, 0], patch_position_cam[:, 1]
        
        
        # z, x = z.expand(b, -1), x.expand(b, -1) # [b, num_patch]
        if not self.cfg.use_semantic_depth_decoder:
            # import ipdb; ipdb.set_trace()
            y_sample = y_sample.mean(dim=-1,keepdim=True)
        y_sample_num = y_sample.shape[-1]
        z, x = z[None,:,None].expand(b, -1, y_sample_num), x[None,:,None].expand(b, -1, y_sample_num) # [b, num_patch, sample_num]
        y = y_sample.expand(b, z.shape[1], -1).to(device=instrinsic.device) # [b, num_patch, sample_num]
        
        if height is not None:
            y = y - y.mean(-1, keepdim=True) + torch.ones_like(y) * height[:, None, None]
        
        reference_point_xyz = torch.stack([x, y, z], dim=-1).transpose(1,2) # [b, sample_num, num_patch, 3]
        if return_dim == 6:
            reference_point_w = torch.ones_like(y) * self.patch_size_cam[0] * whl[0]
            reference_point_h = torch.ones_like(y) * self.patch_size_cam[0] * whl[1]
            reference_point_l = torch.ones_like(y) * self.patch_size_cam[0] * whl[2]
            
            if scale:
                s = torch.ones_like(z) + z / z.max()
                reference_point_h = reference_point_h * s
                reference_point_w = reference_point_w * s
            
            reference_point_whl = torch.stack([reference_point_w, reference_point_h, reference_point_l], dim=-1).to(device=instrinsic.device).transpose(1,2) # [b, sample_num, num_patch, 3]
            reference = torch.cat([reference_point_xyz, reference_point_whl], dim=-1)
        elif return_dim == 3:
            reference = reference_point_xyz
            
        else:
            raise NotImplementedError    

        z = z + self.patch_size_cam[0] # this z is translated just to get valid index
        v = (y * f[:, None]) / z + cv[:, None] # [b, num_patch, sample_num] height
        u = (x * f[:, None]) / z + cu[:, None] # [b, num_patch, sample_num] width
        reference_img = torch.stack([u, v], dim=-1) # [b, num_patch, sample_num, 2]
        reference_img_norm = reference_img / torch.tensor([w, h], dtype=torch.float32, device=instrinsic.device)
        reference_img_norm = reference_img_norm.transpose(1, 2) # [b, sample_num, num_patch, 2]
        
        valid = (reference_img_norm[..., 0] > 0) & \
                (reference_img_norm[..., 0] < 1) & \
                (reference_img_norm[..., 1] > 0) & \
                (reference_img_norm[..., 1] < 1).to(device=instrinsic.device)

        valid_union = valid.any(dim=0).any(dim=0)
        
        return reference[:, :, valid_union, :], reference_point_xyz[:, :, valid_union, :] ,valid_union, valid.any(dim=1)

    def ms_features_preprocess(self, features):
        # prepare data for deformable attention
        if self.cfg.num_feature_levels == 1:
            key = ['1']
            # key = ['2']
        elif self.cfg.num_feature_levels == 2:
            key = ['1', '2']
        elif self.cfg.num_feature_levels == 3:
            key = ['1', '2', '3']
        else:
            raise NotImplementedError
        src_flatten = []
        spatial_shapes = []
        pos_embed_flatten = []
        for k, src in features.items():
            if k not in key:
                continue
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # src = src + self.pos_emb(src)
            pos_embed_flatten.append(self.pos_emb(src).flatten(2).transpose(1, 2))
            src = src.flatten(2).transpose(1, 2)
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1)
        pos_embed_flatten = torch.cat(pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        return src_flatten, spatial_shapes, level_start_index, pos_embed_flatten

    def get_pos_embedding(self, valid_union):
        # valid_union: num_patch
        num_z = num_x = int(valid_union.shape[0] ** 0.5)
        query_pos_emb = self.query_pos_emb(torch.ones((1, self.cfg.dim_model, num_z, num_x)))
        query_pos_emb = query_pos_emb.permute(0, 2, 3, 1).reshape(1, num_z * num_x, self.cfg.dim_model).unsqueeze(1).repeat(1, self.cfg.num_query_each_token, 1, 1)
        query_pos_emb = query_pos_emb.to(device=valid_union.device)
        query_pos_emb = query_pos_emb[:, :, valid_union] 
        return query_pos_emb
    
    def cam2img(self, sampling_locations_cam, calib, img_shape):
        # sampling_location_cam.shape b, n_sample, n_patch, 3
        instrinsic = calib['camera_intrinsic']
        crop_height = calib['crop_height'][:, None, None]
        crop_width = calib['crop_width'][:, None, None]
        height_begin = calib['height_begin'][:, None, None]
        width_begin = calib['width_begin'][:, None, None]
        
        h, w = img_shape
        f, cu, cv = instrinsic[:, 0, 0], instrinsic[:, 0, 2], instrinsic[:, 1, 2]
        x, z, y = sampling_locations_cam[..., 0], sampling_locations_cam[..., 1], sampling_locations_cam[..., 2]
        f = f[:, None, None]
        cu = cu[:, None, None]
        cv = cv[:, None, None]
        z = torch.clamp(z, min=0.1)
        # v = (y * f) / z + cv + crop_height - h# [b, num_patch] height
        v = (y * f) / z + cv - height_begin # [b, num_patch] height
        # u = (x * f) / z + cu # [b, num_patch] width
        u = (x * f) / z + cu - width_begin # [b, num_patch] width
        sampling_locations = torch.stack([u, v], dim=-1) # [b, num_patch, 2]

        # sampling_locations = sampling_locations / torch.tensor([crop_width, crop_height], dtype=torch.float32, device=instrinsic.device)
        sampling_locations[..., 0] = sampling_locations[..., 0] / crop_width
        sampling_locations[..., 1] = sampling_locations[..., 1] / crop_height
        return sampling_locations
    
    def get_valid(self, ref):
        # ref.shape b, n_sample, n_patch, 2
        valid = (ref[..., 0] > 0) & \
                (ref[..., 0] < 1) & \
                (ref[..., 1] > 0) & \
                (ref[..., 1] < 1)
                
        valid_union = valid.any(dim=0).any(dim=0)
        valid = valid.any(dim=1)
        ref = ref[..., valid_union, :]
        return ref, valid_union, valid
     
    def get_query_embedding(self, valid_union=None, pers=None):
        query_embedding = self.query_embedding[0].unsqueeze(0)
        num_patches = valid_union.shape[0]
        num_z = num_x = int(num_patches ** 0.5)
        if self.cfg.embedding == "z":
            query_embedding = query_embedding.repeat(1, 1, num_x, 1)
            query_embedding = query_embedding.reshape(
                1,  self.cfg.num_query_each_token, 
                num_z, num_x, self.cfg.dim_model
            ).permute(0, 1, 3, 2, 4).reshape(1, self.cfg.num_query_each_token, num_patches, self.cfg.dim_model)
            query_embedding = query_embedding.to(device=valid_union.device)
            query_embedding = query_embedding[:, :, valid_union]    
        elif self.cfg.embedding == 'one':
            query_embedding = query_embedding.repeat(1, 1, num_patches, 1)
            query_embedding = query_embedding.to(device=valid_union.device)
            query_embedding = query_embedding[:, :, valid_union]    
        elif self.cfg.embedding == 'all':
            # if not self.training:
                # query_embedding = query_embedding.reshape(1, self.cfg.num_query_each_token, num_z * 2 - 1, num_x * 2 - 1, self.cfg.dim_model)
                # query_embedding = query_embedding[:, :, ::2, ::2].flatten(2, 3)
            query_embedding = query_embedding.to(device=valid_union.device)
            query_embedding = query_embedding[:, :, valid_union]    
        else:
            raise NotImplementedError
        
        
        return query_embedding
    
    def get_attn_mask_(self, mask_method:str="inner_local", kernel_size: int=5, valid_union: torch.Tensor=None, dense: bool=False):
        num_patches = valid_union.shape[0] # not equal to self.num_patches because when training, we use dense patch
        if self.attn_mask is None or self.attn_method != mask_method or num_patches != self.attn_mask.shape[0]:
            if mask_method == 'inner_global':
                attn_mask = torch.ones(num_patches, num_patches)

            elif mask_method == 'inner_local':
                num_z = num_x = int(num_patches ** 0.5)
                attn_mask = torch.zeros((num_patches, num_z, num_x))
                indices = torch.arange(num_patches, device=attn_mask.device)
                attn_mask[indices, indices // num_z, indices % num_z] = 1
                maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
                if self.training and dense: # dense patch 
                    attn_mask[..., ::2, ::2] = maxpool(attn_mask[..., ::2, ::2])
                    attn_mask[..., 1::2, 1::2] = maxpool(attn_mask[..., 1::2, 1::2])
                    attn_mask[..., 1::2, ::2] = maxpool(attn_mask[..., 1::2, ::2])
                    attn_mask[..., ::2, 1::2] = maxpool(attn_mask[..., ::2, 1::2])
                    attn_mask = attn_mask.reshape(num_patches, -1)
                else:
                    attn_mask = maxpool(attn_mask).reshape(num_patches, -1)
            
            elif mask_method == 'inner_col':
                num_z = num_x = int(num_patches ** 0.5)
                attn_mask = torch.zeros((self.cfg.num_query_each_token, num_patches,
                                        self.cfg.num_query_each_token, num_z, num_x))
                indices = torch.arange(num_patches, device=attn_mask.device)
                for i in range(self.cfg.num_query_each_token):
                    attn_mask[i, indices, i, :, indices % num_z] = 1
                attn_mask = attn_mask.reshape(self.cfg.num_query_each_token * num_patches, -1)
                
            else: 
                raise NotImplementedError
            
            attn_mask[attn_mask == 0] = float('-inf')
            attn_mask[attn_mask == 1] = 0
            self.attn_mask = attn_mask
            self.attn_method = mask_method
        
        else:
            attn_mask = self.attn_mask
    
        attn_mask = attn_mask.to(device=valid_union.device)
        attn_mask = attn_mask[valid_union, :][:, valid_union]
            
        return attn_mask    
    
    def get_attn_mask(self, mask_method:str="inner_local", kernel_size: int=5, valid_union: torch.Tensor=None, dense: bool=False):
        num_patches = valid_union.shape[0] # not equal to self.num_patches because when training, we use dense patch
        if self.attn_mask is None or self.attn_method != mask_method or num_patches != self.attn_mask.shape[0]:
            if mask_method == 'cross_local':
                num_z = num_x = int(num_patches ** 0.5)
                attn_mask = torch.zeros((self.cfg.num_query_each_token, num_patches,
                                        self.cfg.num_query_each_token, num_z, num_x))
                indices = torch.arange(num_patches, device=attn_mask.device)
                attn_mask[:, indices, :, indices // num_z, indices % num_z] = 1
                attn_mask = attn_mask.flatten(0, 1)
                maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
                attn_mask = maxpool(attn_mask).reshape(self.cfg.num_query_each_token * num_patches, -1)
                attn_mask = attn_mask.reshape(self.cfg.num_query_each_token * num_patches, -1)        
            else: 
                raise NotImplementedError
            
            attn_mask[attn_mask == 0] = float('-inf')
            attn_mask[attn_mask == 1] = 0
            self.attn_mask = attn_mask
            self.attn_method = mask_method
        
        else:
            attn_mask = self.attn_mask
    
        attn_mask = attn_mask.to(device=valid_union.device)
        attn_mask = attn_mask[valid_union.repeat(self.cfg.num_query_each_token), :][:, valid_union.repeat(self.cfg.num_query_each_token)]
            
        return attn_mask    
    
    def predict_token(self, token_feature, valid_union):
        if self.conv is not None:
            # not used 
            num_patches = valid_union.shape[0]
            num_z = num_x = int(num_patches ** 0.5)
            token_feature_grid = torch.zeros((token_feature.shape[0], self.cfg.dim_model, num_patches), dtype=torch.float32, device=token_feature.device)
            token_feature_grid[..., valid_union] = token_feature.transpose(1, 2)
            token_feature_grid = token_feature_grid.reshape(token_feature.shape[0], self.cfg.dim_model, num_z, num_x)
            token_feature_grid = self.conv(token_feature_grid)
            token_feature_grid = token_feature_grid.reshape(token_feature.shape[0], self.cfg.dim_model, num_patches)
            token_pred = self.token_head(token_feature_grid.transpose(1, 2)[:, valid_union])
        else:
            token_pred = self.token_head(token_feature)
        return token_pred
    
    def forward(self, img, calib, offset=None):
        if not isinstance(img, torch.Tensor):
            img = torch.stack(img, dim=0)
        # get multiscale img feature with same dimension
        ms_features = self.backbone_with_neck(img)
        # calculate multiscale img feature with same dimension
        features, spatial_shapes, level_start_index, features_pos = self.ms_features_preprocess(ms_features)
        
        # get reference point in img coordinate 
        ref_cam = self.get_reference_points(
            offset=offset, 
            num_token=self.num_patches, 
            # train_dense=2 if self.training else 1, 
            train_dense=1,
            dense_sample=self.cfg.sample_dense,
            h_sample=self.cfg.y_sample_num,
            z_range=[1, 50],
            x_range=[-25, 25],
            y_range=[0, 3],
            # y_range = [1., 2.5] # argoverse using
        )
        b = features.shape[0]
        ref_cam = ref_cam.repeat(b, 1, 1, 1).to(device=features.device)
        
        ref_img = self.cam2img(ref_cam, calib, img_shape=[900, 1600])
        # ref_img = self.cam2img(ref_cam, calib, img_shape=[900, 1600], crop_height=900)
        
        reference_select, valid_union, valid = self.get_valid(ref_img)
        
        # decoder to get token feature
        reference_select = reference_select.transpose(1, 2) # [b, num_patch, n_sample, 2]
        reference_select = reference_select.unsqueeze(1).repeat(1, self.cfg.num_query_each_token, 1, 1, 1).flatten(1,2)

        query_embedding_select = self.get_query_embedding(valid_union).repeat(b, 1, 1, 1).flatten(1,2)
        
        query_pos_emb = self.get_pos_embedding(valid_union).repeat(b, 1, 1, 1).flatten(1,2)
        attn_mask = self.get_attn_mask(mask_method='cross_local', kernel_size=5, valid_union=valid_union)

        token_feature, reference_points = self.img2bev_token_decoder(
            tgt=query_embedding_select, # [b, num_patch, d_model]
            query_pos=query_pos_emb,
            reference_points=reference_select, # [b, query_num, n_sample, 2]
            src=features, # [b, num_ms_feature, d_model]
            src_spatial_shapes=spatial_shapes, # [num_ms_feature, 2]
            src_level_start_index=level_start_index, # [num_ms_feature]
            attn_mask=attn_mask,
        )

        token_feature = token_feature.reshape(b, self.cfg.num_query_each_token, -1, self.cfg.dim_model)
        token_feature = token_feature.max(dim=1)[0]

        token_pred = self.predict_token(token_feature, valid_union)
        
        pred_valid = valid[:, valid_union]
        
        token_seq_all_patch = torch.zeros((b ,valid_union.shape[0], self.cfg.codebook_size + 1), dtype=torch.float32, device=features.device)
        token_seq_all_patch[:, :, 0] = 1 # set background token to token_0
        
        pred_valid_all_patch = valid.clone()
        pred_valid_all_patch[:, valid_union] = pred_valid
        token_seq_all_patch[:, valid_union] = token_pred # this use stable reference point
                
        output = {
            'token_pred': token_seq_all_patch,
            'valid': valid, # b, num_patch
            'valid_union': valid_union, # num_patch
            'pred_valid': pred_valid_all_patch, # b, num_patch
        }
        
        return output