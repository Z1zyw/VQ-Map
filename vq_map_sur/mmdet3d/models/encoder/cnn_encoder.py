import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF
from typing import List
from ..utils.utils_cnn import GroupNorm, Swish, ResidualBlock, UpSampleBlock, DownSampleBlock, NonLocalBlock

import torchvision.transforms.functional as TF 

from mmdet3d.models import ENCODER

@ENCODER.register_module()
class SampleEncoderAutoAugment(nn.Module):
    def __init__(
        self,
        latent_dim,
        image_channels: int=3,
        channels: List[int]=[64, 128],
        augment: int=3,
        patch_size: int=8,
    ) -> None:
        super().__init__()
        if len(channels) != 2:
            raise Warning("channels: only first 3 layers work and patch size is 8x8 or 4x4 or 20x20")
        self.augment = augment
        self.patch_size = patch_size
        if patch_size == 8:
            self.model = nn.Sequential(
                self.make_block(image_channels, channels[0], padding=0), # 8 -> 6
                nn.MaxPool2d(2, 2), # 6 -> 3
                self.make_block(channels[0], channels[1], padding=0), # 3 -> 1
                nn.Conv2d(channels[1], latent_dim, 1, 1, padding=0),
            )
        elif patch_size == 20: 
            self.model = nn.Sequential(
                self.make_block(image_channels, channels[0], padding=0, kernel_size=5, stride=2), # 20 -> 8
                nn.MaxPool2d(2, 2), # 8 -> 4
                self.make_block(channels[0], channels[1], padding=0, kernel_size=4), # 4 -> 1
                nn.Conv2d(channels[1], latent_dim, 1, 1, padding=0),
            )
        elif patch_size == 4:
            self.model = nn.Sequential(
                self.make_block(image_channels, channels[0], padding=0, kernel_size=3, stride=2), # 4 -> 2
                self.make_block(channels[0], channels[1], padding=0, kernel_size=2), # 2 -> 1
                nn.Conv2d(channels[1], latent_dim, 1, 1, padding=0),
            )
        else:
            raise Warning("patch size: only 8x8 or 4x4 or 20x20")
        
    def make_block(self, in_channel, out_channel, padding=0, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    
    def patch_random_augument(self, x, p_rotate=0.5, p_translate=0.5, p_resize=0.5):
        # random rotate
        x_ = x.clone()
        p = self.patch_size
        p_aug = int(p * 1.25)
        if torch.rand(1) < p_rotate:
            x_ = TF.rotate(x_, angle=torch.randint(-10, 10, (1,)).item())
        # random translation
        if torch.rand(1) < p_translate:
            x_ = TF.affine(x_, angle=0, translate=(torch.randint(-(p_aug - p), (p_aug - p), (1,)).item(), torch.randint(-(p_aug - p), (p_aug - p), (1,)).item()), scale=1, shear=0)
        # random resize
        if torch.rand(1) < p_resize:
            x_ = TF.resize(x_, (torch.randint(p, p_aug, (1,)).item(), torch.randint(p, p_aug, (1,)).item()), interpolation=TF.InterpolationMode.NEAREST)
            x_ = TF.center_crop(x_, (p, p))
        return x_
    
    def forward(self, x, is_training=True):
        # assert x.shape[-1] == x.shape[-2] == 8, "only support 8x8 patch"
        if not is_training:
            # return self.model(x), None 
            return self.model(x)
        else:
            x_aug_list = [x]
            for _ in range(self.augment):
                x_aug_list.append(self.patch_random_augument(x))
            # x_aug = torch.stack(x_aug_list, dim=-1)
            # lat = self.model(x_aug.mean(dim=-1))
            lat = self.model(x)
            
            lat_aug = []
            for x_ in x_aug_list:
                lat_aug.append(self.model(x_))
            
            return lat, lat_aug
