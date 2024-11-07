from .lss import *
from .depth_lss import *
from .deformable_decoder_v2 import *

import torch
import torch.nn as nn

from mmdet3d.models.builder import VTRANSFORMS
@ VTRANSFORMS.register_module()
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *args):
        return args