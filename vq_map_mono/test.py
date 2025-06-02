import os
import sys
sys.path.append(os.getcwd())
import warnings 
warnings.filterwarnings("ignore")
import wandb
import tqdm
import torch
import time
import random
import numpy as np
from typing import Union
from argparse import ArgumentParser
from src.model.vq_bev_seg import build_vq_bev_seg_model as build_model
from src.utils.config import TrainConfig, load_config
from src.utils.checkpoints import save_checkpoint_accelerate as save_checkpoint, load_checkpoint_accelerate as load_checkpoint
from src.utils.optimizer import build_optimizer
from src.model.vq_bev_seg import VQBEVSegConfig as ModelConfig, VQBEVSeg as Model   
from src.model.cnn_vq import CNNVQ as PretrainModel, CNNVQConfig as PretrainModelConfig

import torch.nn.functional as F
from PIL import Image
import numpy as np
import accelerate
from src.data.nuscenes.data_factory import nuScenesDatasetConfig as DatasetConfig
from src.data.nuscenes.data_factory import build_dataloader_seg as build_dataloader

train_cfg = TrainConfig()
train_cfg.data_cfg = 'config/data/nuscenes_eval.yaml'
train_cfg.resume = True

train_cfg.model_cfg = 'config/model/vq_map.yaml'
train_cfg.resume_dir = "YOUR_CHECKPOINT_PATH" 
train_cfg.resume_epoch = 40

data_cfg = DatasetConfig(load_config(train_cfg.data_cfg))
model_cfg = ModelConfig(load_config(train_cfg.model_cfg))

train_cfg.num_workers = 4
train_cfg.batch_size = 1
train_cfg.seed = 1231

random.seed(train_cfg.seed)
torch.manual_seed(train_cfg.seed)
torch.cuda.manual_seed_all(train_cfg.seed)

from accelerate import DistributedDataParallelKwargs
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs], step_scheduler_with_optimizer=False)

model, pretrain_model = build_model(model_cfg)
try:
    model = model.cuda()
    pretrain_model = pretrain_model.cuda()
except: 
    pass
train_loader, val_loader = build_dataloader(data_cfg, train_cfg)
optimizer, scheduler = build_optimizer(train_cfg, model=model, group_params=True)

if train_cfg.resume:
    begin_epoch = load_checkpoint(train_cfg, model, optimizer, scheduler, accelerator) + 1
else: 
    begin_epoch = 1
    
device = accelerator.device
model = model.to(device)
model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
pretrain_model = accelerator.prepare(pretrain_model)

from src.utils.confusion import BinaryConfusionMatrix
model = model.eval()
confusion = {
    "0.5": BinaryConfusionMatrix(4),
}
data_pbar = tqdm.tqdm(val_loader, ncols=100)
threshold = 0.0
pred_valid = []
mask_seq = []
with torch.no_grad():
    max_pooling = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    pred_resize1 = torch.nn.Upsample(size=(100, 100), mode='nearest')
    pred_resize = torch.nn.Upsample(size=(200, 200), mode='nearest')
    for i, (img, bev, view_mask, mask, calib) in enumerate(data_pbar):
        out_put = model(img, calib, using_confidence=False, threshold=threshold, is_soft=True)

        token_pred = out_put['token_pred']
        pred = out_put['bev_logits'].detach()

        pred = max_pooling(pred)

        pred = pred_resize(pred)
        gt = bev.detach()

        mask_gt = mask.detach()

        for key in confusion.keys():
            confusion[key].update(pred > float(key), gt > 0, mask_gt > 0.5)

    for key in confusion:
        print(key)
        print(confusion[key].iou)
        print(confusion[key].mean_iou)
