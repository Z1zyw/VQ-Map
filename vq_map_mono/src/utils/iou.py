import torch
import torch.nn as nn
import torch.nn.functional as F

@ torch.no_grad()
def cal_batch_iou(pred_:torch.Tensor, target_:torch.Tensor, threshold:float=0.5, mask_=None):
    # pred: [B, C, H, W]
    # target: [B, C, H, W]
    # return: [C]
    pred_size = pred_.shape[-2:]
    target = target_.detach().clone()
    pred = pred_.detach().clone()
    target = F.interpolate(target, size=pred_size, mode='bilinear')
    
    if mask_ is not None:
        mask = mask_.detach().clone()
        mask = F.interpolate(mask, size=pred_size, mode='bilinear')
        mask = (mask > 0.5).float()
        target = target * mask
        pred = pred * mask
    
    target = target > 0.5
    pred = pred > threshold
    intersection = torch.sum(pred & target, dim=(2, 3)).float()
    union = torch.sum(pred | target, dim=(2, 3)).float()
    iou = (intersection) / (union + 1e-6)
    iou = torch.sum(iou, dim=0)
    
    count = torch.sum(target.sum(-1).sum(-1) > 0, dim=0)
    
    return iou, count



@ torch.no_grad()
def cal_batch_loss_each_class(pred_:torch.Tensor, target_:torch.Tensor, threshold:float=0.5, mask_=None):
    pred_size = pred_.shape[-2:]
    target = target_.detach().clone()
    pred = pred_.detach().clone()
    target[target > 0.2] = 1
    target[target <= 0.2] = 0
    target = F.interpolate(target, size=pred_size, mode='bilinear')
    
    if mask_ is not None:
        mask = mask_.detach().clone()
        mask = F.interpolate(mask, size=pred_size, mode='bilinear')
        mask = (mask > 0).float()
        target = target * mask
        pred = pred * mask
    
    pred = torch.sigmoid(pred) > threshold
    intersection = (pred * target).sum(dim=(2, 3)).float()
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    dice = torch.mean(dice, dim=0)
    return 1 - dice