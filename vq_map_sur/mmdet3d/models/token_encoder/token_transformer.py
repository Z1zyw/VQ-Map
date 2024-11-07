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
from timm.models.vision_transformer import Block
from mmdet3d.models.codebook.norm_eam_quantizer import  NormEMAVectorQuantizer as Codebook, l2norm

from mmdet3d.models.utils.transformer import TransformerDecoderLayer

from mmdet3d.models import TOKENENCODER

@ TOKENENCODER.register_module()
class TokenIdentity(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        token = x - 1
        return F.one_hot(token, num_classes=256).float()



@ TOKENENCODER.register_module()
class TokenTransformer(nn.Module):
    def __init__(
        self,
        mask_range = (0.0, 0.5),
        error_range = (0.0, 0.2),
        pretrain_codebook_size=256,
        num_patches=625,
        embedding_dim=512,
        n_head=8,
        norm="layer",
        mlp_ratio=4,
        n_layers=2,
        n_layers_decoder=4,
        kernel_size=7,
        **kwargs
    ) -> None:
        super().__init__()
        # self.cfg = cfg
        self.pretraining = True
        
        self.mask_range = mask_range
        self.error_range = error_range
        self.codebook_size = pretrain_codebook_size
        self.num_patches = num_patches

        # build transformer        
        self.position_prob = self.get_postion_prob(num_patches).flatten()
        # build encoder
        self.embedding = nn.Embedding(pretrain_codebook_size + 1, embedding_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, embedding_dim))
        self.blocks = nn.ModuleList([
            Block(dim=embedding_dim, num_heads=n_head, qkv_bias=True, mlp_ratio=mlp_ratio) for _ in range(n_layers)
        ])
        
        # build decoder
        self.query_embedding = nn.Parameter(torch.randn(1, num_patches, embedding_dim))
        self.attn_mask = self.get_attn_mask(num_patches, kernel_size=kernel_size)
        self.decoder_block = nn.ModuleList([
            TransformerDecoderLayer(embedding_dim, n_head, dim_feedforward=embedding_dim*mlp_ratio) for _ in range(n_layers_decoder)
        ])
        
        # build head 
        self.norm = nn.LayerNorm(embedding_dim) if norm == "layer" else nn.Identity()
        self.pred = nn.Linear(embedding_dim, pretrain_codebook_size, bias=True)
    
    def get_postion_prob(self, num_patches: int, alpha: float=1.0):
        x = z = num_patches ** 0.5
        assert  x * z == num_patches
        c = x // 2
        xx = torch.arange(x).float()
        zz = torch.arange(z).float()
        xx, zz = torch.meshgrid(xx, zz)
        d = (xx - c) ** 2 + (zz - c) ** 2
        alpha = d.max() * alpha
        prob = torch.exp(d / alpha)
        prob = prob / prob.sum()
        return prob
    
    def get_attn_mask(self, num_patches, kernel_size=5):
        x = z = int(num_patches ** 0.5)
        assert  x * z == num_patches
        attn_mask = torch.zeros(num_patches, num_patches)
        indices = torch.arange(num_patches)
        attn_mask[indices, indices] = 1
        attn_mask = attn_mask.view(-1, x, z)
        max_pool2d = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size//2)
        attn_mask = max_pool2d(attn_mask)
        attn_mask[attn_mask < 1] = float('-inf')
        attn_mask[attn_mask == 1] = 0
        return attn_mask.reshape(num_patches, num_patches)
        
    
    def shuffle(self, token: torch.Tensor):
        # token: [token_num]
        token_shuffle = token.clone()
        index = torch.randperm(token.shape[0])
        return token_shuffle[index]
    
    def mask_patch_seq(self, mask_: torch.Tensor, threshold:float=0.2, p:int=16):
        threshold = torch.tensor([p * p * threshold]).to(mask_.device)
        C, H, W = mask_.shape
        assert H % p == 0 and W % p == 0
        mask = mask_.reshape(C, H//p, p, W//p, p)
        mask = torch.einsum("chpwq->hwcpq", mask)
        mask = mask.reshape(H//p * W//p, C*p*p)
        mask = mask > 0.5
        mask = mask.sum(dim=-1) > threshold
        return mask
    
    def add_mask(self, mask_range: tuple, token: torch.Tensor):
        mask_ratio = random.uniform(*mask_range)
        for i in range(token.shape[0]):
            mask_index = random.sample(range(self.num_patches), int(self.num_patches * mask_ratio))
            token[i, mask_index] = 0
        return token
    
    def add_noise(self, token: torch.Tensor, view_mask: torch.Tensor = None):
        mask_ratio = random.uniform(*self.mask_range)
        error_ratio = random.uniform(*self.error_range)
        noise_ratio = mask_ratio + error_ratio

        for i in range(token.shape[0]):
            if random.random() < 0.0 and view_mask is not None:
                mask_index = self.mask_patch_seq(view_mask[i], threshold=0.1, p=16).cpu()
                token[i, mask_index] = 0
                # import ipdb; ipdb.set_trace()
                index_remain = torch.tensor(range(self.num_patches))[mask_index].numpy().tolist()
                error_index = random.sample(index_remain, int(len(index_remain) * error_ratio))
                token[i, error_index] = self.shuffle(token[i, error_index])
            else:
                noise_index = torch.multinomial(self.position_prob, int(self.num_patches * noise_ratio), replacement=False).tolist()
                # noise_index = random.sample(range(self.num_patches), int(self.num_patches * noise_ratio))
                mask_index = random.sample(noise_index, int(self.num_patches * mask_ratio))
                error_index = list(set(noise_index) - set(mask_index))
                token[i, mask_index] = 0
                # token[i, error_index] = torch.randint(1, self.codebook_size, (len(error_index),), device=token.device)
                token[i, error_index] = self.shuffle(token[i, error_index])
        return token

    def soft_forward(self, token_index: torch.Tensor, token_weight: torch.Tensor):
        features = self.embedding(token_index) * token_weight.unsqueeze(-1)
        features = features.sum(-2) + self.position_embedding
        
        for blk in self.blocks:
            features = blk(features)
            
        features = self.norm(features)
        token = self.pred(features)
        
        return token
    
    def forward(self, token: torch.Tensor, view_mask: torch.Tensor = None):
        if len(token.shape) == 3:
            token = token.reshape(-1, token.shape[2])
        
        B = token.shape[0]
        if self.pretraining:
            token = token + 1
            token = self.add_noise(token, view_mask)
            index = (token != 0)
            features = self.embedding(token[index]).reshape(B, -1, self.embedding.embedding_dim)
            features = features + self.position_embedding.repeat(B, 1, 1)[index].reshape(B, -1, self.embedding.embedding_dim)
        
        else:
            if B == 1:
                index = (token != 0) # just one sample for test
            else:
                index = token >= -1 # all 
            features = self.embedding(token[index]).reshape(B, -1, self.embedding.embedding_dim)
            features = features + self.position_embedding.repeat(B, 1, 1)[index].reshape(B, -1, self.embedding.embedding_dim)
        
        # features = self.embedding(token[index]).reshape(B, -1, self.embedding.embedding_dim)
        # features = features + self.position_embedding.repeat(B, 1, 1)[index].reshape(B, -1, self.embedding.embedding_dim)
        
        # encoder
        for blk in self.blocks:
            features = blk(features)
        
        # decoder
        # pad mask
        features2 = self.embedding(token)
        features2[index] = features.reshape(-1, self.embedding.embedding_dim)
        features2 = features2.reshape(B, -1, self.embedding.embedding_dim)
        
        query = self.query_embedding.repeat(B, 1, 1) + self.position_embedding
        query = query.permute(0, 2, 1)
        features2 = features2.permute(0, 2, 1)
        attn_mask = self.attn_mask.to(features2.device)
        for blk in self.decoder_block:
            # import ipdb; ipdb.set_trace()
            features2 = blk(query, features2, query_pos=None, key_pos=None, attn_mask=attn_mask)
        
        features2 = features2.permute(0, 2, 1)
        features2 = self.norm(features2)
        token = self.pred(features2)
        
        return token
    
    
    
