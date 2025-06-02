import os
import sys
sys.path.append(os.getcwd())
import copy
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_tensor
from argoverse_api.argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse_api.argoverse.utils.camera_stats import RING_CAMERA_LIST
from src.data.argoverse.constant import IMAGE_WIDTH, IMAGE_HEIGHT, ARGOVERSE_CLASS_NAMES

def decode_binary_labels(labels, nclass):
    bits = torch.pow(2, torch.arange(nclass))
    return (labels & bits.view(-1, 1, 1)) > 0

from src.utils.config import BasicConfig

class argoverseDatasetConfig(BasicConfig):
    def __init__(self, cfg=None):
        self.img_size: list = [960, 600]
        self.original_img_height: int = IMAGE_HEIGHT
        self.original_img_width: int = IMAGE_WIDTH
        self.crop_height: int = IMAGE_HEIGHT
        self.crop_width: int = IMAGE_WIDTH
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
                
        self.mode: str = 'train'
        
        self.augment: bool = True
        self.pretrain_augment: bool = True
        super().__init__(cfg)

class argoverseDatasetTransform:
    def __init__(self, cfg: argoverseDatasetConfig):
        self.cfg = copy.deepcopy(cfg)
        self.img_transform_base = transforms.Compose([
            transforms.Resize((cfg.img_size[1], cfg.img_size[0])),
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
            u = info[0][2]
            tu = (2 * u - self.cfg.original_img_width) / self.cfg.original_img_width * self.cfg.img_size[0]
            img = TF.affine(img, angle=0, translate=(tu, 0), scale=1, shear=0)
            bev = TF.hflip(bev)
            mask = TF.hflip(mask)
            view_mask = TF.hflip(view_mask)
            
        return img, bev, mask, view_mask, info
    
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
            
            # if self.cfg.augment and self.cfg.mode == 'train':
            #     img = self._color_jitter(0.8, img)
            crop_height = self.cfg.crop_height
            crop_width = self.cfg.crop_width
            height_begin = self.cfg.original_img_height - crop_height
            width_begin = (self.cfg.original_img_width - crop_width) // 2
            
            img = img.crop((width_begin, height_begin, width_begin + crop_width, height_begin + crop_height))
            img = self.img_transform_base(img)
            bev = self.bev_transform(bev)
            
            view_mask = self.mask_transform(view_mask)
            mask = self.mask_transform(mask)
            
            if self.cfg.augment and self.cfg.mode == 'train':
                img, bev, mask, view_mask, info = self.img2bev_augment(img, bev, mask, view_mask, info)
            
            new_info = {
                "camera_intrinsic": info,
                "crop_height": self.cfg.crop_height,
                "crop_width": self.cfg.crop_width,
                "height_begin": 0,
                "width_begin": 0,
            }
            
            return img, bev, view_mask, mask, new_info

class ArgoverseMapDataset(Dataset):

    def __init__(self, 
        argo_loaders:ArgoverseTrackingLoader,
        cfg:argoverseDatasetConfig,
        label_root, 
        log_names=None,
        transforms=None
    ):

        self.label_root = label_root
        self.cfg = cfg
        self.examples = dict()
        self.calibs = dict()
        
        self.transform = transforms
        self.image_size = cfg.img_size
        # Preload training examples from Argoverse train and test sets
        self.loaders = argo_loaders
        for split, loader in argo_loaders.items():
            self.preload(split, loader, log_names)

        self.timestamps = list(sorted(self.examples.keys()))


    def preload(self, split, loader, log_names=None):

        # Iterate over sequences
        for log in loader:

            # Check if the log is within the current dataset split
            logid = log.current_log
            if log_names is not None and logid not in log_names:
                continue

            self.calibs[logid] = dict()
            for camera, timestamps in log.image_timestamp_list_sync.items():

                if camera not in RING_CAMERA_LIST:
                    continue

                # Load image paths
                for timestamp in timestamps:
                    self.examples[timestamp] = (split, logid, camera)

    def __len__(self):
        return len(self.examples)
    

    # def __getitem__(self, timestamp):
    def __getitem__(self, index):
        timestamp = self.timestamps[index]
        # Get the split, log and camera ids corresponding to the given timestamp
        split, log, camera = self.examples[timestamp]

        image = self.load_image(split, log, camera, timestamp)
        calib = self.load_calib(split, log, camera)
        labels, view_mask, mask = self.load_labels(split, log, camera, timestamp)
        image, labels, view_mask, mask, calib = self.transform((image, labels, view_mask, mask, calib))
        # return image, calib, labels, mask
        return image, labels, view_mask, mask, calib


    def load_image(self, split, log, camera, timestamp):
        
        # Load image
        loader = self.loaders[split]
        image = loader.get_image_at_timestamp(timestamp, camera, log)
        
        # Resize to the desired dimensions
        # image = image.resize(self.image_size)

        return Image.fromarray(image)
    

    def load_calib(self, split, log, camera):

        # Get the loader for the current split
        loader = self.loaders[split]

        # Get intrinsics matrix and rescale to account for downsampling
        calib = loader.get_calibration(camera, log).K[:,:3]
        # calib[0] *= self.image_size[0] / IMAGE_WIDTH
        # calib[1] *= self.image_size[1] / IMAGE_HEIGHT
        
        # Convert to a torch tensor
        return torch.from_numpy(calib)
    

    def load_labels(self, split, log, camera, timestamp):

        # Construct label path from example data
        label_path = os.path.join(self.label_root, split, log, camera, 
                                     f'{camera}_{str(timestamp)}.png')
        
        # Load encoded label image as a torch tensor
        encoded_labels = to_tensor(Image.open(label_path)).long()

        # Decode to binary labels
        num_class = len(ARGOVERSE_CLASS_NAMES)
        labels = decode_binary_labels(encoded_labels, num_class+ 1)
        labels, mask = labels[:-1], ~labels[-1]
        
        return labels.float(), mask.float().unsqueeze(0), mask.float().unsqueeze(0)
        
    @ staticmethod
    def collate_train(batch):
        imgs, labels, masks, view_masks, infos = zip(*batch)
        imgs = torch.stack(imgs)
        labels = torch.stack(labels)
        labels2 = labels[:, :4]
        labels2[:, 1:] = -1.
        
        
        masks = torch.stack(masks)
        view_masks = torch.stack(view_masks)
        # calibs = torch.stack(calibs)
        
        keys = {   
            "crop_height",
            "crop_width",  
            "height_begin",
            "width_begin",   
            "camera_intrinsic", 
        }
        
        keys = list(set(keys).intersection(set(infos[0].keys())))
        merge_infos = {}
        for key in keys:
            tmp = []
            for i in range(len(infos)):
                tmp.append(torch.tensor(infos[i][key], dtype=torch.float32))        
            merge_infos[key] = torch.stack(tmp)
        
        return imgs, labels2, masks, view_masks, merge_infos   
    