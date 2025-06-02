import os
import sys
sys.path.append(os.getcwd())
sys.path.append("/home/ywzhang/code/VQ-BEV/argoverse_api")
import torch    
from torch.utils.data import DataLoader
from src.data.argoverse.dataset import ArgoverseMapDataset, argoverseDatasetConfig, argoverseDatasetTransform
from src.data.argoverse.constant import TRAIN_LOGS, VAL_LOGS
from src.utils.config import load_config, TrainConfig

from argoverse_api.argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader


def build_argoverse_datasets(config:argoverseDatasetConfig):
    print('==> Loading Argoverse dataset...')
    dataroot = os.path.expandvars(config.img_root_dir)
    
    # Load native argoverse splits
    loaders = {
        'train' : ArgoverseTrackingLoader(os.path.join(dataroot, 'train')),
        'val' : ArgoverseTrackingLoader(os.path.join(dataroot, 'val'))
    }
    transform = argoverseDatasetTransform(config)
    # Create datasets using new argoverse splits
    train_data = ArgoverseMapDataset(loaders, config, config.bev_root_dir, TRAIN_LOGS, transforms=transform)
    
    config.mode = 'val'
    transform = argoverseDatasetTransform(config)
    val_data = ArgoverseMapDataset(loaders, config, config.bev_root_dir, VAL_LOGS, transforms=transform)
    return train_data, val_data

def build_dataloader(config:argoverseDatasetConfig, train_cfg:TrainConfig):
    train_data, val_data = build_argoverse_datasets(config)
    train_loader = DataLoader(train_data, batch_size=train_cfg.batch_size, 
                              shuffle=train_cfg.shuffle, num_workers=train_cfg.num_workers,
                              pin_memory=train_cfg.pin_memory, drop_last=train_cfg.drop_last, 
                              collate_fn=ArgoverseMapDataset.collate_train)

    val_loader = DataLoader(val_data, 
                            # batch_size=train_cfg.batch_size, 
                            batch_size=1,
                            shuffle=train_cfg.shuffle, 
                            # num_workers=train_cfg.num_workers,
                            num_workers=1,
                            pin_memory=train_cfg.pin_memory, drop_last=train_cfg.drop_last, 
                            collate_fn=ArgoverseMapDataset.collate_train)
    return train_loader, val_loader


if __name__ == '__main__':
    import ipdb
    config = argoverseDatasetConfig(load_config('config/data/argoverse.yaml'))
    train_cfg = TrainConfig()
    train_loader, val_loader = build_dataloader(config, train_cfg)
    from tqdm import tqdm
    for i, batch in enumerate(tqdm(train_loader)):
        ipdb.set_trace()
        if i > 10:
            break
    print('==> Done.')