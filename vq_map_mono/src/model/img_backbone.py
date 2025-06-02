import torch
import torch.nn as nn
import timm
from torchvision.models import resnet50, resnet18, resnet34, resnet101, resnet152, ResNet50_Weights, ResNet18_Weights, ResNet34_Weights, ResNet101_Weights, ResNet152_Weights, ResNet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
# add mobile net
class ResNetwithFPN(nn.Module):
    def __init__(
        self, 
        backbone: str="resnet50", 
        pretrained: bool=True,
        out_channels: int=256,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        self.out_channels = out_channels
        self._build_backbone()
        self._build_fpn()
        
    def _build_backbone(self):
        return build_backbone(self.backbone, self.pretrained)
    
    def _build_fpn(self):
        self.body = IntermediateLayerGetter(
            self.backbone, 
            return_layers={
                # "layer1": "0", 
                "layer2": "1", 
                "layer3": "2", 
                "layer4": "3"
            }
        ) 
        out_channels = self.out_channels
        in_channel_stage = self.backbone.inplanes // 8
        in_channels_list = [
            in_channel_stage,
            in_channel_stage * 2,
            in_channel_stage * 4,
            in_channel_stage * 8,
        ]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        
    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x['0']


class _backbone_resnet(nn.Module):
    def __init__(self, backbone:ResNet):
        super().__init__()
        # Resnet
        # self.inplanes = backbone.inplanes
        self.inplanes = 2048
        
        self.conv1 = backbone.conv1
        # import ipdb; ipdb.set_trace()
        self.bn1 = backbone.bn1
        try:
            self.relu = backbone.relu
        except:
            self.relu = backbone.act1
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        # Add extra conv layer
        self.conv5 = nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        # Resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Add extra conv layer
        x = self.conv5(x)
        return x
    
def build_backbone(backbone:str, pretrained:bool=True, return_layers:str=None):
    '''
    Build the Image Backbone.
    '''
    if backbone == 'resnet50':
        net = resnet50(weights=ResNet50_Weights.DEFAULT) if pretrained else resnet50()
    elif backbone == 'resnet18':
        net = resnet18(weights=ResNet18_Weights.DEFAULT) if pretrained else resnet18()
    elif backbone == 'resnet34':
        net = resnet34(weights=ResNet34_Weights.DEFAULT) if pretrained else resnet34()
    elif backbone == 'resnet101':
        net = resnet101(weights=ResNet101_Weights.DEFAULT) if pretrained else resnet101()
    elif backbone == 'resnet152':
        net = resnet152(weights=ResNet152_Weights.DEFAULT) if pretrained else resnet152()
    elif backbone == 'ssl_resnet50':
        net = timm.create_model('ssl_resnet50', pretrained=pretrained)
    else:
        raise NotImplementedError
    
    if 'resnet' in backbone:
        net = _backbone_resnet(net)
    
    if return_layers is not None:
        # assert backbone in ['resnet18', 'resnet34'], "return_layers only support resnet18 and resnet34"
        net = IntermediateLayerGetter(net, return_layers=return_layers)
    
    return net