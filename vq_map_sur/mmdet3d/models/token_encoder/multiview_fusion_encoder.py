import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import random

from typing import Union
# from timm.models.vision_transformer import Block
from mmdet3d.models.utils.transformer import TransformerDecoderLayer
from mmdet3d.models.codebook.norm_eam_quantizer import  NormEMAVectorQuantizer as Codebook, l2norm

from mmdet3d.models import TOKENENCODER

class TransformerDecoder(nn.Module):
    def __init__(self, emb_dim: int=256, num_heads: int=8, num_layers: int=4, norm: str="layer"):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(emb_dim, num_heads, norm=norm) for _ in range(num_layers)
        ])
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

@ TOKENENCODER.register_module()
class Cam2LidarTokenEncoder(nn.Module):
    def __init__(
        self,
        emb_dim: int=256,
        camera_channels: int=6, 
        mask_range: tuple=(0.0, 0.5),
        error_range: tuple=(0.0, 0.5),
        num_patches: int=625, # lidar token number
        pretrain_codebook_size: int=256,
        num_heads: int=8,
        num_layers: int=4,
        norm: str="layer",
    ):
        self.pretraining = True
        
        self.mask_range = mask_range
        self.error_range = error_range
        self.codebook_size = pretrain_codebook_size
        self.num_patches = num_patches
        embbeding = [nn.Embedding(pretrain_codebook_size + 1, emb_dim) for _ in range(camera_channels)]
        self.cam_token_embedding = nn.ModuleList(embbeding)
        # build transformer
        

        self.norm = norm
        self.head = nn.Linear(emb_dim*camera_channels, emb_dim)
        super().__init__()
        
    def forward(self, cam_token_pred):
        # shape (B, N_cam, MaxLen, D)
        return 