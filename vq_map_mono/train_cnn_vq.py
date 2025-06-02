import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append(os.getcwd())
import wandb
import tqdm
import torch
import time
import random
import accelerate
from argparse import ArgumentParser
from src.model.model_factory import build_cnn_vq as build_model
from src.data.nuscenes.data_factory import build_dataloader_pretrain as build_dataloader
from src.data.nuscenes.dataset import nuScenesDatasetConfig as DatasetConfig
from src.utils.config import TrainConfig, load_config, parser2config, save_config
from src.utils.checkpoints import save_checkpoint_accelerate as save_checkpoint, load_checkpoint_accelerate as load_checkpoint
from src.utils.optimizer import build_optimizer
from src.utils.iou import cal_batch_iou
from src.model.cnn_vq import CNNVQConfig as ModelConfig, CNNVQ as Model

import numpy as np

def parser() -> TrainConfig:
    parser = ArgumentParser("Train CNNVQ.")
    
    parser.add_argument("--model-cfg", type=str, default="config/model/cnn_vq.yaml", help="path to model config file")
    parser.add_argument("--data-cfg", type=str, default="config/data/nuscenes.yaml", help="path to dataset config file")
    parser.add_argument("-n", "--exp-name", type=str, default=None, help="experiment name")
    parser.add_argument("--num-epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--save-interval", type=int, default=5, help="save interval")
    parser.add_argument("--val-interval", type=int, default=5, help="validation interval")
    parser.add_argument("--ckpt-dir", type=str, default="bin", help="checkpoint directory")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=1231, help="random seed")
    
    parser.add_argument("-bs","--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("-nw","--num-workers", type=int, default=16, help="number of workers")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="prefetch factor")
    parser.add_argument("--shuffle", type=bool, default=True, help="shuffle dataset")
    parser.add_argument("--pin-memory", type=bool, default=True, help="pin memory")
    parser.add_argument("--drop-last", type=bool, default=True, help="drop last batch")
    
    parser.add_argument("-r", "--resume", action="store_true", help="resume training")
    parser.add_argument("--resume-dir", type=str, default="", help="resume directory")
    parser.add_argument("--resume-epoch", type=int, default=0, help="resume epoch")
    
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    
    parser.add_argument("--lr-scheduler", type=str, default="step", help="lr scheduler")
    parser.add_argument("--lr-step-size", type=int, default=20, help="lr step size")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="lr gamma")
    
    parser.add_argument("-d", "--debug", action="store_true", help="debug mode")
    parser.add_argument("-t", "--test", action="store_true", help="test mode")
    
    return parser2config(parser, TrainConfig(), verbose=False)

def build_exp(train_cfg: TrainConfig, data_cfg: DatasetConfig, model_cfg: ModelConfig, accelerator):
    if train_cfg.debug or not accelerator.is_local_main_process:
        return
    wandb.init(project="CNN-VQ-OpenDebug", name=train_cfg.exp_name)
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


def train_one_epoch(model, dataloader, optimizer, scheduler, epoch, train_cfg:TrainConfig, accelerator):
    if not train_cfg.debug and accelerator.is_local_main_process:
        wandb.log({
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/epoch": epoch
        })
    
    # new_ratio = update_mask_ratio(model, epoch, train_cfg, accelerator)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.train()
        model = model.module
    else:
        model.train()
    
    accelerator.print("\n==> Epoch: {} Train".format(epoch))
    accelerator.print("==> LR: {}".format(optimizer.param_groups[0]["lr"]))
    accelerator.print("==> Num batches: {}".format(len(dataloader)))
    
    train_pbar = tqdm.tqdm(dataloader, total=len(dataloader), ncols=80, desc="Train", disable=not accelerator.is_local_main_process)
    
    sum_loss = 0
    sum_loss_reconstruction = 0
    sum_loss_codebook = 0
    sum_perplexity = 0
    
    sum_iou = 0
    sum_count = 0
    
    accumulate_loss = 0
    
    # weight = torch.tensor(dataloader.dataset.cfg.bev_class_pretrain_weights).to(accelerator.device)
    current_time = time.time()
    

    temp_decay = 0.0001
    
    for i, (bev, _, _, _) in enumerate(train_pbar):
        # import ipdb; ipdb.set_trace()
        bev = (bev > 0).float()
        loss_reconstruction, loss_codebook, perplexity, token, pred, lat_q = model(bev, weight=None)
        
        iou, count = cal_batch_iou(pred, bev)
        
        loss = loss_codebook + loss_reconstruction
        accelerator.backward(loss)
        
        accumulate_loss += loss.item()
        
        sum_loss += loss.item()
        sum_loss_reconstruction += loss_reconstruction.item()
        sum_loss_codebook += loss_codebook.item()
        sum_perplexity += perplexity.item()
        sum_iou += iou
        sum_count += count
        
        train_pbar.set_postfix_str("loss:{:.4}, time:{:.2}".format(loss.item(), time.time() - current_time))
        
        current_time = time.time()
        
        if (i + 1) % train_cfg.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            step = i + (epoch - 1) * len(dataloader)
            temp = np.max([np.exp(- temp_decay * step), 1e-5])
            try:
                model.codebook.set_temperature(temp)
            except:
                pass
            
            if not train_cfg.debug and accelerator.is_local_main_process:
                wandb.log({
                    "train/accumulate_mean_loss": accumulate_loss / train_cfg.accumulation_steps,
                    "train/mini_batch": (epoch-1) * len(dataloader) + i + 1
                })
                try:
                    wandb.log({
                        "train/temperature": model.codebook.temperature,
                        "train/mini_batch": (epoch-1) * len(dataloader) + i + 1
                    })
                except:
                    pass
            accumulate_loss = 0
            
        if not train_cfg.debug and accelerator.is_local_main_process:
            wandb.log({
                "train/batch/loss": loss.item(),
                "train/batch/loss_reconstruction": loss_reconstruction.item(),
                "train/batch/loss_codebook": loss_codebook.item(),
                "train/batch/perplexity": perplexity.item(),
                "train/batch": (epoch-1) * len(dataloader) + i + 1
            })
    
    if not train_cfg.debug and accelerator.is_local_main_process:
        wandb.log({
            "train/epoch/loss": sum_loss / len(dataloader),
            "train/epoch/loss_reconstruction": sum_loss_reconstruction / len(dataloader),
            "train/epoch/loss_codebook": sum_loss_codebook / len(dataloader),
            "train/epoch/perplexity": sum_perplexity / len(dataloader),
            "train/epoch": epoch
        })
        class_name = dataloader.dataset.cfg.bev_classes
        for i ,name in enumerate(class_name):
            wandb.log({
                "train/epoch/iou/{}".format(name): sum_iou[i] / sum_count[i],
                "train/epoch": epoch
            })
    
    scheduler.step()
    
    
@ torch.no_grad()
def evaluate(model, dataloader, epoch, train_cfg:TrainConfig, accelerator):
    accelerator.print("\n==> Epoch: {} Val".format(epoch))
    accelerator.print("==> Num batches: {}".format(len(dataloader)))
    
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.eval()
        model = model.module
    else:
        model.eval()
    
    eval_pbar = tqdm.tqdm(dataloader, total=len(dataloader), ncols=80, desc="Val", disable=not accelerator.is_local_main_process)
    sum_loss = 0
    sum_loss_reconstruction = 0
    sum_loss_codebook = 0
    sum_perplexity = 0
    sum_iou = 0
    sum_count = 0
    
    current_time = time.time()
    for i, (bev, _, _, _) in enumerate(eval_pbar):
        bev = (bev > 0).float()
        loss_reconstruction, loss_codebook, perplexity, token, pred, lat_q = model(bev, weight=None)
        iou, count = cal_batch_iou(pred, bev)
        loss = loss_codebook + loss_reconstruction
        
        sum_loss += loss.item()
        sum_loss_reconstruction += loss_reconstruction.item()
        sum_loss_codebook += loss_codebook.item()
        sum_perplexity += perplexity.item()
        sum_iou += iou
        sum_count += count
        
        if not train_cfg.debug and accelerator.is_local_main_process:
            wandb.log({
                "val/batch/loss": loss.item(),
                "val/batch/loss_reconstruction": loss_reconstruction.item(),
                "val/batch/loss_codebook": loss_codebook.item(),
                "val/batch/perplexity": perplexity.item(),
                "val/batch": (epoch-1) * len(dataloader) + i + 1
            })
        
        eval_pbar.set_postfix_str("loss:{:.4}, time:{:.2}".format(loss.item(), time.time() - current_time))
        
        current_time = time.time()
        
    if not train_cfg.debug and accelerator.is_local_main_process:
        wandb.log({
            "val/epoch/loss": sum_loss / len(dataloader),
            "val/epoch/loss_reconstruction": sum_loss_reconstruction / len(dataloader),
            "val/epoch/loss_codebook": sum_loss_codebook / len(dataloader),
            "val/epoch/perplexity": sum_perplexity / len(dataloader),
            "val/epoch": epoch
        })
        class_name = dataloader.dataset.cfg.bev_classes
        for i ,name in enumerate(class_name):
            wandb.log({
                "val/epoch/iou/{}".format(name): sum_iou[i] / sum_count[i],
                "val/epoch": epoch
            })
    if train_cfg.debug:
        print("loss:{:.4}".format(sum_loss / len(dataloader)))
        print("loss_reconstruction:{:.4}".format(sum_loss_reconstruction / len(dataloader)))
        print("loss_codebook:{:.4}".format(sum_loss_codebook / len(dataloader)))
        print("perplexity:{:.4}".format(sum_perplexity / len(dataloader)))
        print("iou:{}".format(sum_iou / sum_count))
        class_name = dataloader.dataset.cfg.bev_classes
        for i ,name in enumerate(class_name):
            print("iou/{}:{}".format(name, sum_iou[i] / sum_count[i]))


def main():
    train_cfg = parser()
    data_cfg = DatasetConfig(load_config(train_cfg.data_cfg))
    model_cfg = ModelConfig(load_config(train_cfg.model_cfg))
    
    random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    torch.cuda.manual_seed_all(train_cfg.seed)
    
    accelerator = accelerate.Accelerator(step_scheduler_with_optimizer=False)
    
    if train_cfg.debug:
        train_cfg.num_workers = 4
    
    build_exp(train_cfg, data_cfg, model_cfg, accelerator)
    model = build_model(model_cfg)
    train_loader, val_loader = build_dataloader(data_cfg, train_cfg)
    optimizer, scheduler = build_optimizer(train_cfg, model=model)
    
    device = accelerator.device
    model = model.to(device)
    if train_cfg.resume:
        begin_epoch = load_checkpoint(train_cfg, model, optimizer, scheduler, accelerator) + 1
    else: 
        begin_epoch = 1
        
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

    for epoch in range(begin_epoch, train_cfg.num_epochs + 1):
        if not train_cfg.test:
            train_one_epoch(model, train_loader, optimizer, scheduler, epoch, train_cfg, accelerator)
        if epoch % train_cfg.val_interval == 0 or train_cfg.test:
            evaluate(model, val_loader, epoch, train_cfg, accelerator)
            if train_cfg.debug or train_cfg.test:
                sys.exit()
        if epoch % train_cfg.save_interval == 0:
            save_checkpoint(train_cfg, epoch, model, optimizer, scheduler, accelerator)

if __name__ == "__main__":
    main()