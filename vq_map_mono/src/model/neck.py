import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection.backbone_utils import BackboneWithFPN

from torchvision.models.detection import FasterRCNN
def build_neck(
    neck: str="fpn",
    backbone: nn.Module=None,
    # return_layers={"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"},
    return_layers={"layer2": "1", "layer3": "2", "layer4": "3", "conv5": "4"},
    out_channels: int=256,
):  
    if neck == "fpn":
        in_channel_stage = backbone.inplanes // 8 # 2048 // 8 = 256
        # in_channels_list = [in_channel_stage * 2 * (2**i) for i in range(len(return_layers))]
        in_channels_list = [
            # in_channel_stage,
            in_channel_stage * 2,
            in_channel_stage * 4,
            in_channel_stage * 8,
            in_channel_stage * 8,
        ]
        in_channels_list = in_channels_list[:len(return_layers)]
        return BackboneWithFPN(
            backbone, 
            return_layers=return_layers,
            in_channels_list=in_channels_list,  
            out_channels=out_channels,
        )
    else:
        raise NotImplementedError
