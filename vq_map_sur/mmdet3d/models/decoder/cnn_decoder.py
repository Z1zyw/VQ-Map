import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF
from typing import List
from ..utils.utils_cnn import GroupNorm, Swish, ResidualBlock, UpSampleBlock, DownSampleBlock, NonLocalBlock

from mmdet3d.models import DECODER
@ DECODER.register_module()
class Decoder(nn.Module):
    def __init__(
        self, 
        latent_dim,
        image_channels: int=3,
        channels: List[int]=[512, 256, 256, 128],
        attn_resolutions: List[int]=[16],
        num_res_blocks: int=3,
        resolution: int=16,
        patch_size: int=8,
        output_shape: int=200,
    ):
        super(Decoder, self).__init__()
        in_channels = channels[0]
        layers = [nn.Conv2d(latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2
                

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        feat = self.model(x)
        logits = torch.sigmoid(feat)
        return logits
