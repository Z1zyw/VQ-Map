from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import HEADS
from mmcv.runner import auto_fp16, force_fp32
import numpy as np

def mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    scale: float = 100.0,
) -> torch.Tensor:
    loss =  F.mse_loss(pred, target, reduction='none')
    if mask is not None:
        mask = mask.reshape(-1)
        if mask.sum() == 0:
            return torch.tensor(0.0).to(pred.device)
        loss = loss[mask]
        
    if reduction == "mean":
        return loss.mean() * scale
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    class_num = pred.shape[-1]
    pred = pred.reshape(-1, class_num)
    target = target.reshape(-1)
    
    if mask is not None:
        mask = mask.reshape(-1)
        if mask.sum() == 0:
            return torch.tensor(0.0).to(pred.device)
        pred = pred[mask]
        target = target[mask]

    ce_loss = F.cross_entropy(pred, target.long(), reduction="none")
    pt = torch.exp(-ce_loss)
    f_loss = (1 - pt) ** gamma * ce_loss
    
    #? why
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    
    if reduction == "mean":
        return f_loss.mean()
    elif reduction == "sum":
        return f_loss.sum()
    else:
        return f_loss

def cross_entropy_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    class_num = pred.shape[-1]
    pred = pred.reshape(-1, class_num)
    target = target.reshape(-1)
    
    if mask is not None:
        mask = mask.reshape(-1)
        if mask.sum() == 0:
            return torch.tensor(0.0).to(pred.device)
        pred = pred[mask]
        target = target[mask]
    
    loss = F.cross_entropy(pred, target.long(), reduction=reduction)

    return loss
    
@HEADS.register_module()
class TokenMLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
        dropout: float = 0.0, 
        loss: str= "focal_loss",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dropout = nn.Dropout(dropout)
        self.sigmoid_output = sigmoid_output
        self.loss = loss 

    @ force_fp32()
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        
        for i, layer in enumerate(self.layers):
            x = self.dropout(x) if i > 0 else x
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
            
        if target is not None:
            if self.loss == "focal_loss":
                loss = focal_loss(x, target, mask)
            else:
                raise NotImplementedError
            return {
                'loss/focal_loss': loss,
            }
        
        return x
    