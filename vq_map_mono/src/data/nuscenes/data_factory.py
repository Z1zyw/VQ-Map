import os
import sys
sys.path.append(os.getcwd())

import torch    
from torch.utils.data import DataLoader
from src.data.nuscenes.dataset import nuScenesDataset, nuScenesDatasetTransform, nuScenesDatasetConfig
from src.utils.config import load_config, TrainConfig


def build_dataset(cfg: nuScenesDatasetConfig):
    if cfg.mode.lower() == "train":
        print("\n==> Builiding train dataset...")
    elif cfg.mode.lower() == "val":
        print("\n==> Builiding val dataset...")
    elif cfg.mode.lower() == "test":
        print("\n==> Builiding test dataset...")
    transform = nuScenesDatasetTransform(cfg)
    dataset = nuScenesDataset(transform, cfg)
    return dataset

def build_dataloader_pretrain(data_cfg: nuScenesDatasetConfig, train_cfg: TrainConfig, mode="trainval"):
    # trainval means return two dataloaders, one for train and one for val
    if mode == "trainval":
        data_cfg.mode = "train"
        train_dataset = build_dataset(data_cfg)
        data_cfg.mode = "val"
        val_dataset = build_dataset(data_cfg)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=train_cfg.shuffle,
            num_workers=train_cfg.num_workers,
            pin_memory=train_cfg.pin_memory,
            drop_last=train_cfg.drop_last,
            prefetch_factor=train_cfg.prefetch_factor,
            collate_fn=nuScenesDataset.collate_bev_pretrain
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.num_workers,
            pin_memory=train_cfg.pin_memory,
            drop_last=train_cfg.drop_last,
            prefetch_factor=train_cfg.prefetch_factor,
            collate_fn=nuScenesDataset.collate_bev_pretrain
        )
        return train_dataloader, val_dataloader
    # elif mode == "train" or mode == "val":
    dataset = build_dataset(data_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=train_cfg.shuffle,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        drop_last=train_cfg.drop_last,
        collate_fn=nuScenesDataset.collate_bev_pretrain
    )
    return dataloader


def build_dataloader_seg(data_cfg, train_cfg: TrainConfig, mode="trainval"):
    if mode == "trainval":
        data_cfg.mode = "train"
        train_dataset = build_dataset(data_cfg)
        data_cfg.mode = "val"
        val_dataset = build_dataset(data_cfg)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=train_cfg.shuffle,
            num_workers=train_cfg.num_workers,
            pin_memory=train_cfg.pin_memory,
            drop_last=train_cfg.drop_last,
            prefetch_factor=train_cfg.prefetch_factor,
            collate_fn=nuScenesDataset.collate_train
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.num_workers,
            pin_memory=train_cfg.pin_memory,
            drop_last=False,
            prefetch_factor=train_cfg.prefetch_factor,
            collate_fn=nuScenesDataset.collate_train
        )
        return train_dataloader, val_dataloader
    # elif mode == "train" or mode == "val":
    dataset = build_dataset(data_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=train_cfg.shuffle,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        drop_last=train_cfg.drop_last,
        collate_fn=nuScenesDataset.collate_train
    )
    return dataloader