# accelerate launch --main_process_port 25541 src/train.py  -bs 1 --model-cfg config/model/vq_map.yaml  -d 
import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# sys.path.append("~/code/VQ-BEV/argoverse_api")

sys.path.append(os.getcwd())
import warnings 
warnings.filterwarnings("ignore")
import wandb
import tqdm
import torch 
import time
import random
import accelerate
import numpy as np
from typing import Union, List
from argparse import ArgumentParser
from src.model.vq_bev_seg import build_vq_bev_seg_model as build_model
from src.model.loss_factory import focal_loss
from src.data.nuscenes.data_factory import build_dataloader_seg as build_dataloader
from src.data.nuscenes.dataset import nuScenesDatasetConfig as DatasetConfig
from src.data.argoverse.dataset import argoverseDatasetConfig as DatasetConfig_argoverse
from src.data.argoverse.data_factory import build_dataloader as build_dataloader_argoverse
from src.utils.config import TrainConfig, load_config, parser2config, save_config
from src.utils.checkpoints import save_checkpoint_accelerate as save_checkpoint, load_checkpoint_accelerate as load_checkpoint
from src.utils.optimizer import build_optimizer
from src.utils.confusion import BinaryConfusionMatrix
from src.model.vq_bev_seg import VQBEVSeg as Model, VQBEVSegConfig as ModelConfig
from src.model.cnn_vq import CNNVQ as PretrainModel, CNNVQConfig as PretrainModelConfig

def parser() -> TrainConfig:
    parser = ArgumentParser("Train Vector Quantizer Bev Semantic Segmentation")
    
    parser.add_argument("--model-cfg", type=str, default="config/vq_bev_seg.yaml", help="path to model config file")
    parser.add_argument("--data-cfg", type=str, default="config/data/nuscenes4_v2.yaml", help="path to dataset config file")
    parser.add_argument("-n", "--exp-name", type=str, default=None, help="experiment name")
    parser.add_argument("--num-epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--save-interval", type=int, default=10, help="save interval")
    parser.add_argument("--val-interval", type=int, default=1, help="validation interval")
    parser.add_argument("--log-interval", type=int, default=50, help="log interval")
    parser.add_argument("--ckpt-dir", type=str, default="bin", help="checkpoint directory")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("-s", "--seed", type=int, default=0, help="random seed")
    
    parser.add_argument("-bs","--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("-nw","--num-workers", type=int, default=4, help="number of workers")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="prefetch factor")
    parser.add_argument("--shuffle", type=bool, default=True, help="shuffle dataset")
    parser.add_argument("--pin-memory", type=bool, default=True, help="pin memory")
    parser.add_argument("--drop-last", type=bool, default=True, help="drop last batch")
    
    parser.add_argument("-r","--resume", action="store_true", help="resume")
    parser.add_argument("--resume-dir", type=str, default="", help="resume directory")
    parser.add_argument("--resume-epoch", type=int, default=0, help="resume epoch")
    parser.add_argument("--resume-path-other", type=str, default="", help="resume path other")
    
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimizer")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--lr-backbone", type=float, default=1e-3, help="learning rate backbone")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    
    parser.add_argument("--lr-scheduler", type=str, default="step", help="lr scheduler")
    parser.add_argument("--lr-step-size", type=int, default=100, help="lr step size")
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="lr gamma")
    
    parser.add_argument("-d","--debug", action="store_true", help="debug mode")
    parser.add_argument("-t","--test", action="store_true", help="test mode, only running val set")
    parser.add_argument('-u',"--use-new-optimizer", action="store_true", help="using new optimizer when resume")
    
    return parser2config(parser, TrainConfig(), verbose=False)

def build_exp(train_cfg: TrainConfig, data_cfg: DatasetConfig, model_cfg: ModelConfig, accelerator):
    if train_cfg.debug or not accelerator.is_local_main_process:
        return
    wandb.init(project="6cam-VQ-Map-OpenDebug", name=train_cfg.exp_name)
    wandb.config.update(train_cfg)
    wandb.config.update(data_cfg)
    wandb.config.update(model_cfg)
    cfg_save_dir = wandb.run.dir
    ckpt_save_dir = os.path.join(train_cfg.ckpt_dir, train_cfg.exp_name)
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)
    save_config(train_cfg, os.path.join(cfg_save_dir, "train.yaml"))
    save_config(data_cfg, os.path.join(cfg_save_dir, "data.yaml"))
    save_config(model_cfg, os.path.join(cfg_save_dir, "model.yaml"))
    
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def get_token_label(token:torch.Tensor, valid_index:torch.Tensor):
    B, N = token.shape
    token_label = torch.zeros(B, N, device=token.device, dtype=torch.long)
    token_label[valid_index] = token[valid_index] + 1
    return token_label
    
def train_one_epoch(
    model: Model,
    pretrain_model: PretrainModel, 
    dataloader, 
    optimizer, 
    scheduler, 
    epoch, 
    train_cfg: TrainConfig, 
    accelerator
):
    if not train_cfg.debug and accelerator.is_local_main_process:
        wandb.log({
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/lr_backbone": optimizer.param_groups[1]["lr"],
            "train/epoch": epoch
        })
    
    model.train()
    freeze_model(pretrain_model)
    if isinstance(pretrain_model, torch.nn.parallel.DistributedDataParallel):
        pretrain_model = pretrain_model.module
    pretrain_model.eval()

    accelerator.print("\n==> Epoch: {} Train".format(epoch))
    accelerator.print("==> LR: {}".format(optimizer.param_groups[0]["lr"]))
    accelerator.print("==> Num batches: {}".format(len(dataloader)))
    
    train_pbar = tqdm.tqdm(dataloader, total=len(dataloader), ncols=80, desc="Train", disable=not accelerator.is_local_main_process)
    
    sum_loss = 0
    sum_loss_token = 0
    
    accumulate_loss = 0
    current_time = time.time()
    
    
    for i, (img, bev, view_mask, mask, calib) in enumerate(train_pbar):
        bev = (bev > 0).float()
        
        offset = torch.zeros(2)
        out_put = model(img, calib, is_tokenizer_train=True,offset=offset)
        token_pred = out_put["token_pred"]
        valid = out_put["valid"] # shape: [N]

        pred_valid = out_put['pred_valid']

        bev_ = bev

        logits_weight = torch.Tensor([1.7, 5.9, 3.3, 4.6, 2.1]).to(bev.device)
        
        cnn_offset = offset / 3.5 * 16

        lat_dict = pretrain_model.get_latent(bev_, False, logits_weight=logits_weight, dense=True, mask=mask, offset=cnn_offset)
        token_gt = lat_dict["token"]
        mask_seq = lat_dict['mask_seq']

        
        
        token_label = get_token_label(token_gt, valid)

        loss_token = focal_loss(pred=token_pred, target=token_label, mask=valid * mask_seq * pred_valid)

        loss = loss_token

        if loss.item() > 1e6 or torch.isnan(loss):
            import ipdb; ipdb.set_trace()
        try:
            accelerator.backward(loss)
        except RuntimeError:
            import ipdb; ipdb.set_trace()
        
        accumulate_loss += loss.item()

        sum_loss += loss.item()
        sum_loss_token += loss_token.item()        
        
        train_pbar.set_postfix_str("loss:{:.4}, time:{:.2}".format(loss.item(), time.time() - current_time))
        
        current_time = time.time()
        
        if (i + 1) % train_cfg.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if not train_cfg.debug and accelerator.is_local_main_process and i % train_cfg.log_interval == 0:
                wandb.log({
                    "train/accumulate_mean_loss": accumulate_loss / train_cfg.accumulation_steps,
                    "train/mini_batch": (epoch-1) * len(dataloader) + i + 1
                })
            accumulate_loss = 0
        
        if not train_cfg.debug and accelerator.is_local_main_process and i % train_cfg.log_interval == 0:
            wandb.log({
                "train/batch/loss": loss.item(),
                "train/batch/loss_token": loss_token.item(),
                "train/batch": (epoch-1) * len(dataloader) + i + 1,
            })

    if not train_cfg.debug and accelerator.is_local_main_process:        
        wandb.log({
            "train/epoch/loss": sum_loss / len(dataloader),
            "train/epoch/loss_token": sum_loss_token / len(dataloader),
            "train/epoch": epoch
        })

        
    scheduler.step()
    

# ! This evaluation (IoU) is inaccurate as it only assessed the data on a single GPU. A complete evaluation needs to be conducted again.
@ torch.no_grad() 
def evaluate(model, pretrain_model, dataloader, epoch, train_cfg, accelerator):
    accelerator.print("\n==> Epoch: {} Val".format(epoch))
    accelerator.print("==> Num batches: {}".format(len(dataloader)))
    
    model.eval()
    if isinstance(pretrain_model, torch.nn.parallel.DistributedDataParallel):
        pretrain_model = pretrain_model.module
    pretrain_model.eval()
    
    eval_pbar = tqdm.tqdm(dataloader, total=len(dataloader), ncols=80, desc="Val", disable=not accelerator.is_local_main_process)
    
    sum_loss = 0
    sum_loss_token = 0
    
    current_time = time.time()
    
    confusion = BinaryConfusionMatrix(len(dataloader.dataset.cfg.bev_classes))

    for i, (img, bev, view_mask, mask, calib) in enumerate(eval_pbar):
        
        bev = (bev > 0).float()
        
        out_put = model(img, calib)
        token_pred = out_put["token_pred"]
        bev_logits = out_put["bev_logits"]
        valid = out_put["valid"] # shape: [N]
        pred_valid = out_put['pred_valid']
        bev_ = bev
            
        logits_weight = torch.Tensor([1.7, 5.9, 3.3, 4.6, 2.1]).to(bev.device)
        
        lat_dict = pretrain_model.get_latent(bev_, False, logits_weight=logits_weight, mask=mask)
        token_gt = lat_dict["token"]
        weight = lat_dict["weight"] 
        mask_seq = lat_dict['mask_seq']
        
        weight[~(mask_seq * pred_valid)] = 1
        
        token_label = get_token_label(token_gt, valid)

        loss_token = focal_loss(pred=token_pred, target=token_label, mask=valid * mask_seq * pred_valid)
        
        with torch.no_grad():
            confusion.update(bev_logits > 0.5, bev > 0.5, mask > 0.5)
    
        loss = loss_token

        if loss.item() > 1e6 or torch.isnan(loss):
            import ipdb; ipdb.set_trace()
        

        sum_loss += loss.item()
        sum_loss_token += loss_token.item()        
        
        
        eval_pbar.set_postfix_str("loss:{:.4}, time:{:.2}".format(loss.item(), time.time() - current_time))
        
        current_time = time.time()
        
        if not train_cfg.debug and accelerator.is_local_main_process and i % train_cfg.log_interval == 0:
            wandb.log({
                "val/batch/loss": loss.item(),
                "val/batch/loss_token": loss_token.item(),
                "val/batch": (epoch-1) * len(dataloader) + i + 1,
            })
            
    if not train_cfg.debug and accelerator.is_local_main_process:        
        wandb.log({
            "val/epoch/loss": sum_loss / len(dataloader),
            "val/epoch/loss_token": sum_loss_token / len(dataloader),
            "val/epoch/iou_mean": confusion.mean_iou,
            "val/epoch": epoch
        })
            
        class_name = dataloader.dataset.cfg.bev_classes
        
        for i, name in enumerate(class_name):
            wandb.log({
                "val/epoch/iou/{}".format(name): confusion.iou[i],
                "val/epoch": epoch
            })

    if train_cfg.test:
        class_name = dataloader.dataset.cfg.bev_classes
        print("==> Epoch: {} val".format(epoch))
        print("loss: {:.4}".format(sum_loss / len(dataloader)))
        print("loss_token: {:.4}".format(sum_loss_token / len(dataloader)))
        print("iou_mean: {:.4}".format(confusion.mean_iou))
        
        for  i, name in enumerate(class_name):
            print("iou/{}: {:.4}".format(name, confusion.iou[i]))
            

def main():
    train_cfg = parser()

    if "argoverse" in train_cfg.data_cfg:
        data_cfg = DatasetConfig_argoverse(load_config(train_cfg.data_cfg))
    else:
        data_cfg = DatasetConfig(load_config(train_cfg.data_cfg))
    model_cfg = ModelConfig(load_config(train_cfg.model_cfg))
    
    if train_cfg.debug:
        train_cfg.num_workers = 4
        # train_cfg.batch_size = 2
    
    if train_cfg.test:
        # train_cfg.batch_size = 1
        pass
    
    random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    torch.cuda.manual_seed_all(train_cfg.seed)
    
    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # ddp_kwargs = DistributedDataParallelKwargs()
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs], step_scheduler_with_optimizer=False)
    # accelerator = accelerate.Accelerator(step_scheduler_with_optimizer=False)
    build_exp(train_cfg, data_cfg, model_cfg, accelerator)
    model, pretrain_model = build_model(model_cfg)
    model = model.cuda()
    pretrain_model = pretrain_model.cuda()
    if "argoverse" in train_cfg.data_cfg:
        train_loader, val_loader = build_dataloader_argoverse(data_cfg, train_cfg)
    else:
        train_loader, val_loader = build_dataloader(data_cfg, train_cfg)
    optimizer, scheduler = build_optimizer(train_cfg, model=model, group_params=True)
    
    if train_cfg.resume:
        begin_epoch = load_checkpoint(train_cfg, model, optimizer, scheduler, accelerator) + 1
        if train_cfg.test:
            if train_cfg.resume_path_other != "":
                model.load_token_encoder(train_cfg.resume_path_other)
    else: 
        begin_epoch = 1
        
    device = accelerator.device
    model = model.to(device)
        
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
    pretrain_model = accelerator.prepare(pretrain_model)

    for epoch in range(begin_epoch, train_cfg.num_epochs + 1):
        if not train_cfg.test:
            train_one_epoch(model, pretrain_model, train_loader, optimizer, scheduler, epoch, train_cfg, accelerator)
        if epoch % train_cfg.val_interval == 0:
            evaluate(model, pretrain_model, val_loader, epoch, train_cfg, accelerator)
            if train_cfg.test or train_cfg.debug:
                break
        if epoch % train_cfg.save_interval == 0:
            save_checkpoint(train_cfg, epoch, model, optimizer, scheduler, accelerator)

if __name__ == "__main__":
    main()