import torch 
from typing import Union
import torch.nn.functional as F

def focal_loss(pred:torch.Tensor, target, gamma=2, alpha=0.5, mask=None, beta:float=1.0):
    # if mask is not None:
    #     raise NotImplementedError
    if isinstance(pred, list):
        loss = 0
        if isinstance(target, list):
            assert len(pred) == len(target)
            for i, p in enumerate(pred):
                loss += focal_loss(p, target[i], gamma, alpha)
        else:
            for p in pred:
                loss += focal_loss(p, target, gamma, alpha)
        return loss

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
    f_loss = alpha * (1 - pt) ** gamma * ce_loss * beta

    return f_loss.mean()
