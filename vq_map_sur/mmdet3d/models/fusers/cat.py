import random
from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["CatFuser"]

# ! this only support for token fusion instead of feature fusion
@FUSERS.register_module()
class CatFuser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(inputs, dim=-1)
