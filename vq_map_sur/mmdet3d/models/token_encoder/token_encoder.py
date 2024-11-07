
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

from mmdet3d.models import TOKENENCODER
@ TOKENENCODER.register_module()
class TokenIdenity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return x


@ TOKENENCODER.register_module()
class BevAugNosie(nn.Module):
    def __init__(
        self,
        patch_size,
        patch_size_resize_max,
        p_rotate,
        p_translate,
        p_resize,
        p_mask,
        translate_max,
        p_pers,
        pers_ratio,
        mask_size,
        bev_length,
        bev_width,
    ) -> None:
        super().__init__()
        self.p = patch_size
        self.patch_max = patch_size_resize_max
        self.p_rotate = p_rotate
        self.p_translate = p_translate
        self.p_resize = p_resize
        self.p_mask = p_mask
        self.translate_max = translate_max
        
        self.p_pers = p_pers
        self.pers_ratio = pers_ratio
        
        self.mask_generator= nn.MaxPool2d(kernel_size=mask_size, stride=1, padding=mask_size // 2)
        
        padding_size = (bev_length // 2, bev_width // 2, bev_length // 2, bev_width // 2)
        self.bev_aug = transforms.Compose([
            transforms.Pad(padding=padding_size, padding_mode='edge'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((-20, 20), fill=0), 
            transforms.CenterCrop((bev_length, bev_width)),
        ])
    
    @ staticmethod
    def random_perspective(img, p=0.5, fill=-1, ratio=0.1):
    # ratio is the ratio of the image to be distorted
        if torch.rand(1) > p:
            return img
        
        shape = img.shape[-2:]
        x_mid = shape[1] // 2
        z_max, x_max = shape
        _range = int(min(shape) * ratio)
        sp1 = [_range, z_max - 1 - _range]
        sp2 = [x_max - 1 - _range, z_max - 1 - _range]
        ep1 = [torch.randint(0, 2 * _range, size=(1,)), z_max - 1 - torch.randint(0, 2 * _range, size=(1,))]
        ep2 = [x_max - 1 - torch.randint(0,  2 * _range, size=(1,)), z_max - 1 - torch.randint(0, 2 * _range, size=(1,))]

        img_ = TF.perspective(
            img=img,
            startpoints=[[x_mid, 1], [x_mid+1, 1], sp1, sp2], # (left-right, up-down)
            endpoints=[[x_mid, 1], [x_mid+1, 1], ep1, ep2],
            interpolation=TF.InterpolationMode.NEAREST,
            fill=fill,
        )
        return img_
        
    
    def patchify(self, x):
        p = self.p
        B, C, H, W = x.shape
        h = H // p
        w = W // p
        x = x.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(-1, C, p, p)
        return x, (h, w)
    
    def unpatchify(self, x, hw):
        h, w = hw
        p = self.p
        _, C, _, _ = x.shape
        x = x.reshape(-1, h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(-1, C, h * p, w * p)
        return x
    
    def patch_add_noise_each_sample(self, x: torch.Tensor):
        x = x.clone()
        p = self.p
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p_mask:
                pass # TODO
            if torch.rand(1) < self.p_rotate:
                pad_size = p // 4
                tmp = F.pad(x[i], mode="reflect", pad=(pad_size, pad_size, pad_size, pad_size))
                tmp = TF.rotate(tmp, angle=torch.randint(-20, 20, (1,)).item())
                x[i] = TF.center_crop(tmp, (p, p))
            # random translation
            if torch.rand(1) < self.p_translate:
                t_max = self.translate_max
                tmp = F.pad(x[i], mode="reflect", pad=(t_max, t_max, t_max, t_max))
                tmp = TF.affine(tmp, angle=0, translate=(
                    torch.randint(-t_max, t_max, (1,)).item(), 
                    torch.randint(-t_max, t_max, (1,)).item()), scale=1, shear=0)
                x[i] = TF.center_crop(tmp, (p, p))
            # random resize
            if torch.rand(1) < self.p_resize:
                p_max = self.patch_max
                tmp = TF.resize(x[i], (torch.randint(p, p_max, (1,)).item(), torch.randint(p, p_max, (1,)).item()), interpolation=TF.InterpolationMode.NEAREST)
                x[i] = TF.center_crop(tmp, (p, p))
        return x
    
    def patch_add_noise(self, x: torch.Tensor):
        x = x.clone()
        p = self.p
        c = x.shape[1]
        b = x.shape[0]
        
        # Add mask 
        mask = torch.zeros((b * c * p * p), device=x.device)
        mask_num = int(b * c * p * p * self.p_mask)
        mask_index = torch.randperm(b * c * p * p, device=x.device)[:mask_num]
        mask[mask_index] = 1
        mask = mask.reshape(b, c, p, p)
        mask = self.mask_generator(mask.float())
        x[mask == 1] = -1   
    
        if torch.rand(1) < self.p_rotate:
            pad_size = p // 4
            tmp = F.pad(x, mode="reflect", pad=(pad_size, pad_size, pad_size, pad_size))
            tmp = TF.rotate(tmp, angle=torch.randint(-20, 20, (1,)).item())
            x = TF.center_crop(tmp, (p, p))
        # random translation
        if torch.rand(1) < self.p_translate:
            t_max = self.translate_max
            tmp = F.pad(x, mode="reflect", pad=(t_max, t_max, t_max, t_max))
            tmp = TF.affine(tmp, angle=0, translate=(
                torch.randint(-t_max, t_max, (1,)).item(), 
                torch.randint(-t_max, t_max, (1,)).item()), scale=1, shear=0)
            x = TF.center_crop(tmp, (p, p))
        # random resize
        if torch.rand(1) < self.p_resize:
            p_max = self.patch_max
            tmp = TF.resize(x, (torch.randint(p, p_max, (1,)).item(), torch.randint(p, p_max, (1,)).item()), interpolation=TF.InterpolationMode.NEAREST)
            x = TF.center_crop(tmp, (p, p))
        return x

    def smoothing(self, bev: torch.Tensor):
        # smoothing using 3x3 kernel maxpool as dilate and erode
        bev = bev.clone()
        for i in range(bev.shape[0]):
            tmp = F.pad(bev[i].unsqueeze(0), mode="reflect", pad=(1, 1, 1, 1))
            tmp = F.max_pool2d(tmp, kernel_size=3, stride=1)
            tmp = F.pad(tmp, mode="reflect", pad=(1, 1, 1, 1))
            tmp = F.max_pool2d(1 - tmp, kernel_size=3, stride=1)
            bev[i] = 1 - tmp.squeeze(0)
        return bev

    def forward(self, bev:torch.Tensor):
        bev = self.bev_aug(bev.float())
        bev_gt = bev.clone()
        # bev = BevAugNosie.random_perspective(bev, p=self.p_pers, fill=-1, ratio=self.pers_ratio)
        bev_patch, hw = self.patchify(bev)
        bev_patch_noised = self.patch_add_noise(bev_patch)
        bev = self.unpatchify(bev_patch_noised, hw)
        return bev, bev_gt


@ TOKENENCODER.register_module()
class TokenEncoder(nn.Module):
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
        n_layers=4
    ) -> None:
        super().__init__()
        # self.cfg = cfg
        self.pretraining = True
        
        self.mask_range = mask_range
        self.error_range = error_range
        self.codebook_size = pretrain_codebook_size
        self.num_patches = num_patches
        
        self.embedding = nn.Embedding(pretrain_codebook_size + 1, embedding_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, embedding_dim))
        self.blocks = nn.ModuleList([
            Block(dim=embedding_dim, num_heads=n_head, qkv_bias=True, mlp_ratio=mlp_ratio) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim) if norm == "layer" else nn.Identity()
        self.pred = nn.Linear(embedding_dim, pretrain_codebook_size, bias=True)
        
    
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
                noise_index = random.sample(range(self.num_patches), int(self.num_patches * noise_ratio))
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
        
        if self.pretraining:
            token = token + 1
            token = self.add_noise(token, view_mask)
        
        features = self.embedding(token)
        features = features + self.position_embedding
        
        for blk in self.blocks:
            features = blk(features)
            
        features = self.norm(features)
        token = self.pred(features)
        
        return token

