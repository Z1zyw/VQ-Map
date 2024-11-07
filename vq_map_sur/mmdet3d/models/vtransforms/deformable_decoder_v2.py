from typing import Tuple, List
import torch
from mmcv.runner import force_fp32
from torch import nn
import ipdb
from mmdet3d.models.builder import VTRANSFORMS
from mmcv.utils import TORCH_VERSION, digit_version
from other.ops.modules import MSDeformAttn_v2
import torch.nn.functional as F
import math
import copy
import numpy as np

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

# detr position encoding rm mask
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, temperature=None):
        not_mask = torch.ones(x.shape[0], x.shape[-2], x.shape[-1], dtype=torch.bool, device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        if temperature is None:
            temperature = self.temperature
        dim_t = temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class BEVDeformableLayer(nn.Module):
    def __init__(
        self, d_model=256, d_ffn=1024,
        dropout=0.1, activation="relu",
        n_levels=4, n_heads=8, n_points=4, 
        use_global_attention=True, 
        global_attention_mode:str="default", # or deformable
    ):
        super().__init__()

        assert global_attention_mode == "default", \
            "Only support default global attention mode"
        
        self.use_global_attention = use_global_attention
        self.n_levels = n_levels
        self.n_heads = n_heads
        # cross attention
        self.cross_attn = MSDeformAttn_v2(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        if use_global_attention:
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

    def attn_prepare(self, attn_mask):
        attn_mask_ = attn_mask
        if attn_mask is not None:
            if len(attn_mask.shape) != 2:
                assert len(attn_mask.shape) == 3
                attn_mask_ = torch.repeat_interleave(attn_mask, self.n_heads, dim=0) 
        return attn_mask_
    
    def forward(
        self, 
        cam_indexes,
        tgt, 
        query_pos,        # B, num_query, D
        reference_points, # B, num_cam, num_query, num_z_anchor, 2
        src, 
        src_spatial_shapes, 
        level_start_index, 
        src_padding_mask=None, 
        src_valid_ratios=None, 
        attn_mask=None,
        is_last: bool=False,
    ):
        # self attention
        B, n_token, D = tgt.shape
        if self.use_global_attention:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(
                q.transpose(0, 1), 
                k.transpose(0, 1), 
                tgt.transpose(0, 1), 
                attn_mask=self.attn_prepare(attn_mask),
            )[0].transpose(0, 1)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        # rebatch query and reference_points_img to save memory
        BN = len(cam_indexes)
        max_len = max([len(each) for each in cam_indexes])
        n_sample, n_dim = reference_points.shape[-2:]

        tgt_rebatch = tgt.new_zeros([BN, max_len, D])
        query_pos_rebatch = query_pos.new_zeros([BN, max_len, D])
        reference_points_img_rebatch = reference_points.new_zeros([BN, max_len, n_sample, n_dim])

        N = BN // B

        for i, index_query_per_img in enumerate(cam_indexes):
            tgt_rebatch[i, :len(index_query_per_img)] = tgt[i // N, index_query_per_img]
            reference_points_img_rebatch[i, :len(index_query_per_img)] = reference_points[i, index_query_per_img]
            query_pos_rebatch[i, :len(index_query_per_img)] = query_pos[0, index_query_per_img]
        
        reference_points_img_rebatch = reference_points_img_rebatch.unsqueeze(2).repeat(1, 1, self.n_levels, 1, 1)
        
        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt_rebatch, query_pos_rebatch),
            reference_points_img_rebatch,
            src, 
            src_spatial_shapes, 
            level_start_index, 
            src_padding_mask,
            src_valid_ratios
        )
        tgt = tgt_rebatch + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        if is_last:
            tgt = self.forward_ffn(tgt)
            return tgt
        
        # tgt.shape = BN, n_token, D
        final_tgt = tgt.new_zeros([BN, n_token, D])
        cnt = torch.zeros([BN, n_token], dtype=torch.long, device=tgt.device)
        for i, index_query_per_img in enumerate(cam_indexes):
            final_tgt[i, index_query_per_img] += tgt[i, :len(index_query_per_img)]
            cnt[i, index_query_per_img] += 1

        # v3 add 
        final_tgt = final_tgt.view(B, -1, n_token, D).sum(1)
        cnt = cnt.view(B, -1, n_token).sum(1)
        final_tgt = final_tgt / (cnt.unsqueeze(-1) + 1e-3)
    
        # ffn
        tgt = self.forward_ffn(final_tgt)
        
        return tgt


class BEVDeformableDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, index, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios=None,
                query_pos=None, src_padding_mask=None, attn_mask=None):
        output = tgt
        is_last = False
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4 or reference_points.shape[-1] == 2:
                output = layer(index, output, query_pos, reference_points, 
                               src, src_spatial_shapes, src_level_start_index, src_padding_mask, 
                               src_valid_ratios, attn_mask, is_last)
            else:
                raise NotImplementedError

        return output, reference_points

        
        
@ VTRANSFORMS.register_module()
class BEVDeformableTokenizer_v2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        use_cnn: bool=True,
        dim_model: int=256,
        num_levels: int=4,
        num_points: int=8,
        num_heads: int=4,
        num_layers: int=4,
        num_token: int=625, # 25x25
        num_query_each_token: int=1,
        self_attn_mode: str='default', # or deformable
        self_attn_mask: str='inner_local', # or inner_global
        pc_range: Tuple[float, float, float, float, float, float]=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        h_samples: int=4,
        dense_sample: int=1, # xy sampling
        using_ref_dim: int=2,
        mask_kernel_size: int=5
    ) -> None:
        super().__init__()
        
        self.pc_range = pc_range
        self.num_token = num_token
        assert num_query_each_token == 1
        self.num_query_each_token = num_query_each_token
        self.dim_model = dim_model
        self.query_pos_emb = PositionEmbeddingSine(num_pos_feats=dim_model//2, normalize=True)
        self.query_emb = nn.Parameter(torch.randn(1, num_query_each_token, num_token, dim_model))
        
        assert num_points % (dense_sample * dense_sample * h_samples) == 0
        
        # build decoder 
        decoder_layer = BEVDeformableLayer(
            d_model=dim_model, 
            d_ffn=dim_model*4,
            n_levels=num_levels, 
            n_heads=num_heads, 
            n_points=num_points, 
            global_attention_mode=self_attn_mode
        )
        self.decoder = BEVDeformableDecoder(decoder_layer=decoder_layer, num_layers=num_layers)
        
        if use_cnn:
            self.conv = nn.Sequential(
                nn.Conv2d(dim_model, out_channels, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        
        self.ref_3d = self.get_reference_points(num_token, h_samples, dense_sample)
        self.attn_mask = self.get_attn_mask(method=self_attn_mask, num_token=num_token, kernel_size=mask_kernel_size)
        self.using_ref_dim = using_ref_dim
    
    # modified based on bevformer: get_reference_points
    @ staticmethod
    def get_reference_points(num_token, h_samples, dense_sample):
        H, W = int(num_token ** 0.5), int(num_token ** 0.5)
        assert H * W == num_token
        H = H * dense_sample
        W = W * dense_sample
        Z = h_samples
        xs = torch.linspace(0.5, W - 0.5, W).view(1, W, 1).expand(Z, W, H) / W
        ys = torch.linspace(0.5, H - 0.5, H).view(1, 1, H).expand(Z, W, H) / H
        zs = torch.linspace(0.5, Z - 0.5, Z).view(Z, 1, 1).expand(Z, W, H) / Z
        
        ref_3d = torch.stack((xs, ys, zs), -1) # (Z, W, H, 3)
        ref_3d = ref_3d.reshape(Z, W//dense_sample, dense_sample, H//dense_sample, dense_sample, 3) 
        ref_3d = ref_3d.permute(0, 2, 4, 1, 3, 5).reshape(Z * dense_sample * dense_sample, num_token, 3)
        return ref_3d[None] # (1, h_samples * dense * dense, num_token, 3)
    
    
    @ staticmethod
    def get_attn_mask(method:str="inner_local", num_token=625, kernel_size=5):
        if method == 'inner_local':
            H, W = int(num_token ** 0.5), int(num_token ** 0.5)
            assert H * W == num_token
            assert kernel_size % 2 == 1
            attn_mask = torch.zeros(num_token, num_token)
            indices = torch.arange(num_token)
            attn_mask[indices, indices] = 1
            attn_mask = attn_mask.view(-1, H, W)
            max_pool2d = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size//2)
            attn_mask = max_pool2d(attn_mask)
            attn_mask[attn_mask < 1] = float('-inf')
            attn_mask[attn_mask == 1] = 0
            return attn_mask.reshape(num_token, num_token)
        else:
            raise NotImplementedError
    
    
    @ force_fp32()
    @ torch.cuda.amp.autocast(enabled=False)
    def point_sampling(self, reference_points, pc_range, lidar2img, img_aug_matrix, lidar_aug_matrix, img_metas, return_dim=4):
        reference_points = reference_points.clone()
        
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]
        
        reference_points = torch.cat((
            reference_points, 
            torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]

        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
        
        # lidar aug
        lidar_aug_inverse = torch.inverse(lidar_aug_matrix)
        lidar_matrix = lidar_aug_inverse.view(
            1, B, 1, 1, 4, 4).repeat(D, 1, num_cam, num_query, 1, 1).to(reference_points.device)
        reference_points_ = torch.matmul(lidar_matrix.to(torch.float32), reference_points.to(torch.float32))
        
        # project to img
        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),reference_points_.to(torch.float32)).squeeze(-1)
        
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)

        reference_points_cam = reference_points_cam[..., 0:2] / reference_points_cam[..., 2:3]

        # img aug
        reference_points_cam = torch.cat(
            (reference_points_cam, torch.ones_like(reference_points_cam[..., :2])), -1)
        img_matrix = img_aug_matrix.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1).to(reference_points.device)
        reference_points_cam = torch.matmul(img_matrix.to(torch.float32),
                                            reference_points_cam.unsqueeze(-1).to(torch.float32)).squeeze(-1)
        reference_points_cam = reference_points_cam[..., :2]
        
        reference_points_cam[..., 0] /= 704
        reference_points_cam[..., 1] /= 256

        bev_mask = (bev_mask 
                    & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        
        if return_dim == 4:
            raise NotImplementedError
        
        lidar2img_aug = (lidar2img@lidar_matrix).flatten(-2).permute(2, 1, 3, 0, 4).mean(3)
        return reference_points_cam, bev_mask, lidar2img_aug

    def get_query_pos_embed(self, temperature=20):
        H, W = int(self.num_token ** 0.5), int(self.num_token ** 0.5)
        pos = self.query_pos_emb(torch.zeros(1, self.dim_model, H, W), temperature=temperature)
        return pos.flatten(2).permute(0, 2, 1)
    
    def img_feature_prepare(self, img):
        if isinstance(img, torch.Tensor):
            img = [img]
        
        value_flatten = []
        value_shape_flatten = []
        for i in range(len(img)):
        
            B, N_cam, C, H, W = img[i].shape
            value_shape_flatten.append((H, W))
            value = img[i].reshape(B * N_cam, C, H, W)
            value = value.flatten(2).transpose(1, 2)
            value_flatten.append(value)
        
        value_flatten = torch.cat(value_flatten, 1)
        value_shape_flatten = torch.as_tensor(value_shape_flatten, dtype=torch.long, device=value_flatten.device)
        value_start_index = torch.cat((value_shape_flatten.new_zeros((1, )), value_shape_flatten.prod(1).cumsum(0)[:-1]))
        return value_flatten, value_shape_flatten, value_start_index
    
    @ force_fp32() # because of deformable attention
    def forward(
        self,
        img, # B, 6, C, H, W
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
        **kwargs,
    ):  
        if isinstance(img, list):
            B, N_cam = img[0].shape[:2]
        elif isinstance(img, torch.Tensor):
            B, N_cam = img.shape[:2]
        query = self.query_emb.repeat(B, 1, 1, 1)
        
        if query.shape[2] == 1:
            query = query.repeat(1, 1, self.num_token, 1)
        
        query = query.flatten(1,2)
        ref_3d = self.ref_3d.repeat(B, 1, 1, 1).to(query.device)
        attn_masks = self.attn_mask.to(query.device)
        query_pos = self.get_query_pos_embed().to(query.device)
        
        reference_points_img, bev_masks, _ = self.point_sampling(
            ref_3d, 
            self.pc_range, 
            lidar2image, 
            img_aug_matrix, 
            lidar_aug_matrix, 
            metas,
            return_dim=self.using_ref_dim
        )
        
        reference_points_img = reference_points_img.to(query.device)
        bev_masks = bev_masks.to(query.device)

        reference_points_img = reference_points_img.permute(1, 0, 2, 3, 4).flatten(0, 1)# B*N, 4, 625, 2
        bev_masks = bev_masks.permute(1, 0, 2, 3).flatten(0, 1)# B*N, 4, 625

        indexes = []
        for i, mask_per_img in enumerate(bev_masks):
            index_query_per_img = mask_per_img.sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)

        # deformable decoder
        # prepare input
        value_flatten, value_shape_flatten, value_start_index = self.img_feature_prepare(img)
        
        features, _ = self.decoder(
            indexes,
            query, 
            reference_points_img, 
            value_flatten, 
            query_pos = query_pos,
            src_spatial_shapes = value_shape_flatten, 
            src_level_start_index = value_start_index, 
            attn_mask = attn_masks,
        )

        # new features shape: B, num_token, D
        if hasattr(self, 'conv'):
            n_x = n_y = int(self.num_token ** 0.5)
            features = features.view(B, n_x, n_y, -1)
            features = self.conv(features.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).flatten(1, 2)
            
        return features
    