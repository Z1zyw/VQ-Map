# version 2: Index is camera token
# view_mask == all_mask because view_mask is not define 
import os
import sys
sys.path.append(os.getcwd())
import torch
import cv2 as cv
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import pickle
import cv2 as cv
import copy
from torch.utils.data import Dataset

from PIL import Image
from src.utils.config import BasicConfig

from pyquaternion import Quaternion
from src.data.nuscenes.constant import GT_CLASS_WEIGHT_DICT

DECODER_CHANNEL = 14

DECODERED_CLASS_INDEX = {
    "drivable_area" : 0,
    "ped_crossing": 1,
    "walkway": 2,
    "carpark_area": 3,
    "view_mask": -1,
    "all_mask": -1
}

def decode_binary_labels(labels, nclass):
    bits = torch.pow(2, torch.arange(nclass))
    return (labels & bits.view(-1, 1, 1)) > 0

class nuScenesDatasetConfig(BasicConfig):
    def __init__(self, cfg=None):
        self.img_height: int = 900
        self.img_width: int = 1600
        self.original_img_height: int = 900
        self.crop_height: int = 582 # 256 / 704 * 900; 900-582 = 318
        self.crop_width: int = 1600
        self.original_img_width: int = 1600
        self.img_mean: list = [0.485, 0.456, 0.406]
        self.img_std: list = [0.229, 0.224, 0.225]
        self.img_root_dir: str = 'data/nuscenes'
        
        self.camera_channels = [
            'CAM_FRONT', 
            'CAM_FRONT_LEFT', 
            'CAM_FRONT_RIGHT', 
            'CAM_BACK_LEFT', 
            'CAM_BACK_RIGHT', 
            'CAM_BACK'
        ]
        
        self.bev_pretrain: bool = True
        self.bev_classes: list = [
            "drivable_area",
            "ped_crossing",
            "walkway",
            "carpark_area"
        ]
        
        self.bev_length: int = 50
        self.bev_width: int = 50
        self.bev_mean: list = [0.5]
        self.bev_std: list = [0.5]
        self.bev_root_dir: str = 'data/nuscenes_ground_truth'
        # self.lidar_gt_root_dir: str = 'data/nuscenes_lidar'
        # self.divider_kernel_size: list = [5, 5]
        # self.lidar_ray_kernel_size: list = [5, 5]
                
        self.info_path: str = 'data/nuscenes_info.pkl'
        self.train_split_path: str = 'data/nuscenes_splits/train_roddick.txt'
        self.val_split_path: str = 'data/nuscenes_splits/val_roddick.txt'
        self.mode: str = 'train'
        
        self.augment: bool = True
        self.pretrain_augment: bool = True
        
        super().__init__(cfg)
        self.bev_class_pretrain_weights = [GT_CLASS_WEIGHT_DICT[cls] for cls in self.bev_classes]
        
class nuScenesDatasetTransform:
    def __init__(self, cfg: nuScenesDatasetConfig):
        self.cfg = copy.deepcopy(cfg)
        self.img_transform_base = transforms.Compose([
            transforms.Resize((cfg.img_height, cfg.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.img_mean, cfg.img_std),
        ])
        self.mask_transform = transforms.Compose([
            # transforms.Pad((0, 4, 0, 0), fill=0),
            transforms.Resize((cfg.bev_length, cfg.bev_width), interpolation=Image.NEAREST),
        ])
        self.bev_transform = transforms.Compose([
            # transforms.Pad((0, 4, 0, 0), fill=0),
            transforms.Resize((cfg.bev_length, cfg.bev_width), interpolation=Image.NEAREST),
            transforms.Normalize(cfg.bev_mean, cfg.bev_std),
        ])
        # only use in pretrain
        padding_size = (cfg.bev_length // 2, cfg.bev_width // 2, cfg.bev_length // 2, cfg.bev_width // 2)
        self.bev_augment = transforms.Compose([
            transforms.Pad(padding=padding_size, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90, fill=-1), # -1 beacuse of the normalize,
            # transforms.RandomRotation(180, fill=-1), # -1 beacuse of the normalize
            transforms.CenterCrop((cfg.bev_length, cfg.bev_width)),
        ])

    def img2bev_augment(self, img, bev, mask, view_mask, info):
        img, bev, mask, view_mask, info = self._hflip(0.5, img, bev, mask, view_mask, info)
        # img = self._resize((0.5, 1.), 0.9, img)
        return img, bev, mask, view_mask, info
    
    def _hflip(self, p, img, bev, mask, view_mask, info):
        # horizontal flip for img, bev, mask, view_mask and calib
        if np.random.rand() < p:
            img = TF.hflip(img)
            u = info['camera_intrinsic'][0][2] - info['width_begin']
            tu = (2 * u - info['crop_width']) / info['crop_width'] * self.cfg.img_width
            img = TF.affine(img, angle=0, translate=(tu, 0), scale=1, shear=0)
            bev = TF.hflip(bev)
            mask = TF.hflip(mask)
            view_mask = TF.hflip(view_mask)
            
            try:
                info['points_img'][:, 0] = 1 - info['points_img'][:, 0] + tu / self.cfg.img_width
                info['points_cam'][:, 0] = - info['points_cam'][:, 0]
            except:
                pass 
            
            try:
                info['reference_ground_truth_14x14'] = np.flip(info['reference_ground_truth_14x14'], -1).copy()
                info['reference_ground_truth_7x7'] = np.flip(info['reference_ground_truth_7x7'], -1).copy()
            except:
                pass
            
        return img, bev, mask, view_mask, info
    
    def _resize(self, scale_range, p, img, keep_ratio=False):
        # resize img
        if np.random.rand() < p:
            if keep_ratio:
                scale = np.random.uniform(*scale_range)
                img = TF.resize(img, (int(self.cfg.img_height * scale), int(self.cfg.img_width * scale)))
            else:
                scale_h = np.random.uniform(*scale_range)
                scale_w = np.random.uniform(*scale_range)
                img = TF.resize(img, (int(self.cfg.img_height * scale_h), int(self.cfg.img_width * scale_w)))
        return img
    
    def _htrans(self, p, img, bev, mask, view_mask, info):
        pass 
    
    def _color_jitter(self, p, img):
        # color jitter for img 
        # this need to be done before normalization, so we need to use PIL image
        if np.random.rand() < p:
            img = TF.adjust_brightness(img, np.random.uniform(0.5, 1.5))
            img = TF.adjust_contrast(img, np.random.uniform(0.5, 1.5))
            img = TF.adjust_saturation(img, np.random.uniform(0.5, 1.5))
            img = TF.adjust_hue(img, np.random.uniform(-0.1, 0.1))
        return img
    
    def __call__(self, input):
        if self.cfg.bev_pretrain:
            bev, view_mask, mask = input
            
            bev = self.bev_transform(bev)
            
            if self.cfg.pretrain_augment and self.cfg.mode == 'train':
                bev = self.bev_augment(bev)
            
            view_mask = self.mask_transform(view_mask)
            mask = self.mask_transform(mask)
            
            return bev, view_mask, mask
        else:
            img, bev, view_mask, mask, info = input
            
            # print(img.size)
            
            # if self.cfg.mode == 'train' and self.cfg.augment:
            #     crop_height = np.random.randint(500, 650)
            #     crop_width = np.random.randint(1300, self.cfg.crop_width)
            #     height_begin = np.random.randint(200, self.cfg.original_img_height - crop_height)
            #     width_begin = np.random.randint(0, self.cfg.original_img_width - crop_width)
            # else :
            crop_height = self.cfg.crop_height
            crop_width = self.cfg.crop_width
            height_begin = self.cfg.original_img_height - crop_height
            width_begin = (self.cfg.original_img_width - crop_width) // 2
            
            img = img.crop((width_begin, height_begin, width_begin + crop_width, height_begin + crop_height))
            info.update({
                "crop_height": crop_height,
                "crop_width": crop_width,
                "height_begin": height_begin,
                "width_begin": width_begin,
            })
            
            # if self.cfg.augment and self.cfg.mode == 'train':
            #     img = self._color_jitter(0.8, img)
            
            img = self.img_transform_base(img)
            
            # bev_ = []
            # for i in range(bev.shape[0]):
            #     bev_.append(self.bev_transform(bev[i]))
            # bev = bev_
            bev = self.bev_transform(bev)
            
            
            view_mask = self.mask_transform(view_mask)
            mask = self.mask_transform(mask)
            
            if self.cfg.augment and self.cfg.mode == 'train':
                img, bev, mask, view_mask, info = self.img2bev_augment(img, bev, mask, view_mask, info)
            
            return img, bev, view_mask, mask, info




class nuScenesDataset(Dataset):
    def __init__(
        self,
        transform: nuScenesDatasetTransform,
        cfg : nuScenesDatasetConfig
    ) -> None:
        super().__init__()
        self.transform = transform
        self.cfg = cfg
        
        if cfg.mode == 'train':
            self.sample_tokens = self._read_split(cfg.train_split_path)
        else:
            self.sample_tokens = self._read_split(cfg.val_split_path)
            
        self.sample_infos = self._load_info(cfg.info_path)
        
        self.dataset_preprocess()

    def dataset_preprocess(self):
        # only run once
        self.tokens = []
        self.infos = {}
        for token in self.sample_tokens:
            for channel in self.cfg.camera_channels:
                self.tokens.append(token + '_' + channel)
                self.infos[token + '_' + channel] = self.sample_infos[token][channel]
    
    def _read_split(self, filename):
        """
        Read a list of NuScenes sample tokens
        """
        with open(filename, "r") as f:
            lines = f.read().split("\n")
            return [val for val in lines if val != ""]
    
    def _load_info(self, info_path):
        return pickle.load(open(info_path, 'rb'))
    
    def _decode_bev(self, encode_bev_mask):
        encode_bev_mask = TF.to_tensor(encode_bev_mask)
        tmp = decode_binary_labels(encode_bev_mask, DECODER_CHANNEL + 1)
        bev = []
        for class_name in self.cfg.bev_classes:
            bev.append(tmp[DECODERED_CLASS_INDEX[class_name]])
        bev = torch.stack(bev).float()
        mask = (~tmp[DECODERED_CLASS_INDEX['all_mask']]).float().unsqueeze(0)
        return bev, mask, mask
    
    def _load_bev(self, path):
        return Image.open(path)
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, index):
        token = self.tokens[index]
        info = self.infos[token]
        bev_path = os.path.join(self.cfg.bev_root_dir , info['camera_token'] + ".png")
        bev = self._load_bev(bev_path)
        bev, view_mask, mask = self._decode_bev(bev)
        
        if self.cfg.bev_pretrain:
            bev, view_mask, mask = self.transform((bev, view_mask, mask))
            return bev, view_mask, mask, info
        
        img_path = info["img_path"]
        img = Image.open(img_path)
        
        cam_token = info['camera_token']     
        calib = info
            
        img, bev, view_mask, mask, calib = self.transform((img, bev, view_mask, mask, calib))
        return img, bev, view_mask, mask, calib
    
    
    @ staticmethod
    def collate_train(batch):
        imgs, bevs, view_masks, masks, infos = zip(*batch)
        # imgs = torch.stack(imgs) # if deformable encoder, this will be a list
        bevs = torch.stack(bevs)
        view_masks = torch.stack(view_masks)
        masks = torch.stack(masks)
        
        merge_infos = {}
        keys = [
            "camera_token", # for debug
            
            "crop_height", # for img augment
            "crop_width",  
            "height_begin",
            "width_begin",
            
            "camera_intrinsic", 
            "camera_translation", 
            "camera_rotation", 
            "ego_pose_translation", 
            "ego_pose_rotation", 
            
            # "reference_ground_truth_14x14",
            # "reference_ground_truth_7x7",
            
            # lidar info
            # "points_img",
            # "semantic_label_img",
            # "depth_img",
            # "points_cam",
            # "index_cam2img", # index from raw lidar points cloud
            
            # "token",
            # "weight",
            # "mask_seq",
        ]
        
        keys = list(set(keys).intersection(set(infos[0].keys())))
        
        for key in keys:
            tmp = []
            if key == 'camera_rotation' or key == 'ego_pose_rotation':
                for i in range(len(infos)):
                    tmp_ = Quaternion(infos[i][key]).rotation_matrix
                    tmp.append(torch.tensor(tmp_, dtype=torch.float32))
                merge_infos[key] = torch.stack(tmp)
            elif key == 'camera_token':
                merge_infos[key] = [infos[i][key] for i in range(len(infos))]
            elif key == 'points_img' or key == 'semantic_label_img' or key == 'depth_img' or key == 'points_cam':
                for i in range(len(infos)):
                    tmp.append(torch.tensor(infos[i][key], dtype=torch.float32))
                merge_infos[key] = tmp
            elif key == 'mask_seq' or key == 'weight' or key == 'token':
                for i in range(len(infos)):
                    tmp.append(torch.tensor(infos[i][key], dtype=torch.float32))
                merge_infos[key] = torch.cat(tmp)
            else:
                for i in range(len(infos)):
                    tmp.append(torch.tensor(infos[i][key], dtype=torch.float32))        
                merge_infos[key] = torch.stack(tmp)
        # return imgs list instead of tensor to allow different size and allign to the deformable detr
        
        return imgs, bevs, view_masks, masks, merge_infos
    
    @ staticmethod
    def collate_bev_pretrain(batch):
        bevs, view_masks, masks, calibs = zip(*batch)
        bevs = torch.stack(bevs)
        view_masks = torch.stack(view_masks)
        masks = torch.stack(masks)
        merge_calibs = {}
        keys = {"camera_intrinsic", "camera_translation", "camera_rotation"}
        
        for key in keys:
            tmp = []
            if key == 'camera_rotation':
                for i in range(len(calibs)):
                    tmp_ = Quaternion(calibs[i][key]).rotation_matrix
                    tmp.append(torch.tensor(tmp_, dtype=torch.float32))
            else:
                for i in range(len(calibs)):
                    tmp.append(torch.tensor(calibs[i][key], dtype=torch.float32))
            merge_calibs[key] = torch.stack(tmp)
        return bevs, view_masks, masks, merge_calibs