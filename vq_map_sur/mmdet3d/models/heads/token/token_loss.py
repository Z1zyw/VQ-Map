from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import HEADS
from mmcv.runner import auto_fp16, force_fp32
import numpy as np
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
class TokenLossHead(nn.Module):
    def __init__(
        self,
        loss: str= "focal_loss",
    ) -> None:
        super().__init__()
        self.loss = loss 

    @ force_fp32()
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        if target is not None:
            if self.loss == "focal_loss":
                n = x.shape[0]
                loss = []
                for i in range(n):
                    loss.append(focal_loss(x[i], target, mask))
            else:
                raise NotImplementedError
            return {
                f'loss/focal_loss_{i}': loss[i] for i in range(n)
            }
        
        return x[-1]