import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF
from typing import List
from src.utils.config import BasicConfig
from src.model.norm_ema_quantizer import NormEMAVectorQuantizer as Codebook, NormEMAVectorQuantizerConfig as CodebookConfig

class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A
   
 
class Decoder(nn.Module):
    def __init__(
        self, 
        latent_dim,
        image_channels: int=3,
        channels: List[int]=[512, 256, 256, 128, 128],
        attn_resolutions: List[int]=[16],
        num_res_blocks: int=3,
        resolution: int=16,
    ):
        super(Decoder, self).__init__()
        # channels = [512, 256, 256, 128, 128]
        # attn_resolutions = [16]
        # num_res_blocks = 3
        # resolution = 16

        in_channels = channels[0]
        layers = [nn.Conv2d(latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        feats = self.model(x)
        return torch.sigmoid(feats)


import torchvision.transforms.functional as TF  
class SampleEncoderAutoAugment(nn.Module):
    def __init__(
        self,
        latent_dim,
        image_channels: int=3,
        channels: List[int]=[64, 128, 128],
        augment: int=3,
        patch_size: int=16
    ) -> None:
        super().__init__()
        self.augment = augment
        if patch_size == 16:
            if len(channels) != 3:
                raise Warning("channels: only first 3 layers work for 16x16 patch")
            self.model = nn.Sequential(
                self.make_block(image_channels, channels[0], padding=0), # 14x14
                nn.MaxPool2d(2, 2), # 7x7
                self.make_block(channels[0], channels[1], padding=1), # 7x7
                nn.MaxPool2d(2, 2), # 3x3
                self.make_block(channels[1], channels[2], padding=1), # 3x3
                nn.Conv2d(channels[2], latent_dim, 3, 1, padding=0)
            )
        elif patch_size == 8:
            if len(channels) != 2:
                raise Warning("channels: only first 2 layers work for 8x8 patch")
            self.model = nn.Sequential(
                self.make_block(image_channels, channels[0], padding=0), # 8 -> 6
                nn.MaxPool2d(2, 2), # 6 -> 3
                self.make_block(channels[0], channels[1], padding=0), # 3 -> 1
                nn.Conv2d(channels[1], latent_dim, 1, 1, padding=0),
            )
        
    def make_block(self, in_channel, out_channel, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    
    def patch_random_augument(self, x, p_rotate=0.5, p_translate=0.5, p_resize=0.5):
        # random rotate
        patch_size = x.shape[-1]
        patch_max = int(patch_size * 1.2)
        patch_translate = int(patch_size * 0.2)
        x_ = x.clone()
        if torch.rand(1) < p_rotate:
            x_ = TF.rotate(x_, angle=torch.randint(-10, 10, (1,)).item())
        # random translation
        if torch.rand(1) < p_translate:
            x_ = TF.affine(x_, angle=0, translate=(torch.randint(-patch_translate, patch_translate, (1,)).item(), torch.randint(-patch_translate, patch_translate, (1,)).item()), scale=1, shear=0)
        # random resize
        if torch.rand(1) < p_resize:
            x_ = TF.resize(x_, (torch.randint(patch_size, patch_max, (1,)).item(), torch.randint(patch_size, patch_max, (1,)).item()), interpolation=TF.InterpolationMode.NEAREST)
            x_ = TF.center_crop(x_, (patch_size, patch_size))
        return x_
    
    def forward(self, x, is_training=True):
        assert x.shape[-1] == x.shape[-2] == 16 or x.shape[-1] == x.shape[-2] == 8, "only support 16x16 or 8x8 patch size"
        if not is_training:
            # return self.model(x), None 
            return self.model(x)
        else:
            x_aug_list = [x]
            for _ in range(self.augment):
                x_aug_list.append(self.patch_random_augument(x))
            # x_aug = torch.stack(x_aug_list, dim=-1)
            # lat = self.model(x_aug.mean(dim=-1))
            lat = self.model(x)
            lat_aug = []
            for x_ in x_aug_list:
                lat_aug.append(self.model(x_))
            
            return lat, lat_aug

# pure cnn architecture
# transformer architecture used when predict bev token 
class CNNVQConfig(BasicConfig):
    def __init__(self, cfg=None):
        self.codebook_dim = 32
        self.codebook_size = 64
        self.codebook_beta = 1
        
        self.image_channels = 8
        self.resolution = 256
        self.atte_resolutions = [16]
        
        self.patch_size = 16
        self.encoder_channels = [128, 128, 128, 256, 256, 512]
        self.encoder_num_res_block = 2
        self.encoder_augment = 3
        
        self.decoder_channels = [512, 256, 256, 128, 128]
        self.decoder_num_res_block = 3
        
        self.lambda_reconstruction = 0.6
        
        self.is_training = True
        self.channel_independent = False
        
        super().__init__(cfg)
        self.cfg_check()
        
    def cfg_check(self):
        pass 

class CNNVQ(nn.Module):
    def __init__(
        self,
        cfg: CNNVQConfig
    ) -> None:
        super().__init__()
        self.patch_size = cfg.patch_size
        self.lambda_reconstruction = cfg.lambda_reconstruction
        self.is_training = cfg.is_training
        self.channel_independent = cfg.channel_independent
        
        # build encoder
        self.encoder = SampleEncoderAutoAugment(
            latent_dim=cfg.codebook_dim,
            image_channels=cfg.image_channels,
            channels= cfg.encoder_channels[:3],
            augment=cfg.encoder_augment,
            patch_size=cfg.patch_size
        )

        # self.encoder = nn.Conv2d(
        #     in_channels=cfg.image_channels,
        #     out_channels=cfg.codebook_dim,
        #     kernel_size=16,
        #     padding=0
        # )
        
        # build codebook
        codeobok_cfg = CodebookConfig()
        codeobok_cfg.codebook_beta = cfg.codebook_beta
        codeobok_cfg.codebook_dim = cfg.codebook_dim
        codeobok_cfg.codebook_size = cfg.codebook_size
        codeobok_cfg.channel_independent = True
        codeobok_cfg.channels = cfg.image_channels
    
        self.codebook = Codebook(cfg=codeobok_cfg)
        # build decoder
        self.decoder = Decoder(
            latent_dim=cfg.codebook_dim,
            image_channels=cfg.image_channels,
            channels=cfg.decoder_channels,
            attn_resolutions=cfg.atte_resolutions,
            num_res_blocks=cfg.decoder_num_res_block,
            resolution=cfg.resolution
        )
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    
    def patch_augmentation(self, x, p=0.9, p_rotate=0.5, p_translation=0.5, p_resize=0.5):
        # x.shape = (b, C, P, P)
        x_ = x.clone()
        for i in range(x.shape[0]):
            if random.random() < p:
                if random.random() < p_rotate:
                    x_[i] = TF.rotate(x_[i], angle=torch.randint(-30, 30, (1,)).item())
                if random.random() < p_translation:
                    x_[i] = TF.affine(x_[i], angle=0, translate=(torch.randint(-2, 2, (1,)).item(), torch.randint(-2, 2, (1,)).item()), scale=1, shear=0)
                if torch.rand(1) < p_resize:
                    tmp = TF.resize(x_[i], (torch.randint(16, 20, (1,)).item(), torch.randint(16, 20, (1,)).item()), interpolation=TF.InterpolationMode.NEAREST)
                    x_[i] = TF.center_crop(tmp, (16, 16))
        return x_
    
    @ staticmethod
    def get_mask_seq(mask_:torch.Tensor, p:int=16, threshold:float=0.1):
    # def get_mask_seq(mask_:torch.Tensor, p:int=16, threshold:float=0.4):
        with torch.no_grad():
            threshold = (threshold * torch.tensor([p*p])).to(mask_.device)
            B, C, H, W = mask_.shape
            assert H % p == 0 and W % p == 0
            mask = mask_.reshape(B, C, H//p, p, W//p, p)
            mask = torch.einsum("nchpwq->nhwcpq", mask)
            mask = mask.reshape(B, H//p * W//p, C*p*p)
            mask = mask > 0.5
            mask = mask.sum(dim=-1) > threshold
        return mask
    
    def _get_patches(self, x, logits_weight:torch.Tensor=None, mask=None):
        B = x.shape[0]
        x_patch, hw = self.patchify(x)
        if logits_weight is not None:
            C = x.shape[1]
            logits_weight = logits_weight.reshape(-1, C+1)
            weight = logits_weight[:,:C, None, None] * (x > 0).float()
            blank = (x.sum(dim=1, keepdim=True) < -C + 1).float() * logits_weight[:, -1, None, None]
            weight_patch, _ = self.patchify(weight)
            blank_patch, _ = self.patchify(blank)
            weight = torch.sum(weight_patch, dim=(1, 2, 3)) + torch.sum(blank_patch, dim=(1, 2, 3))
            weight = weight.reshape(B, 1, hw[0], hw[1])
        else:
            weight = None
            
        if mask is not None:
            mask_seq = self.get_mask_seq(mask, p=self.patch_size).reshape(B, 1, hw[0], hw[1])
        else:
            mask_seq = None
        return x_patch, hw, weight, mask_seq
        
    @ property
    def crop_sizes(self):
        crop_sizes = [
            (0, 0.5, 0, 1),
            (0.5, 0, 1, 0),
            (0.5, 0.5, 1, 1),
        ]
        return crop_sizes
    
    @ torch.no_grad()
    def get_latent(self, x, aug=False, logits_weight:torch.Tensor=None, mask=None, dense=False, offset=None):
        if offset is not None:
            dense = False
            shape = x.shape[-2:]
            offset = (
                int(offset[0]),
                int(offset[1]),
                (shape[0] - self.patch_size) if offset[0] > 0 else shape[0],
                (shape[1] - self.patch_size) if offset[1] > 0 else shape[1]
            )
            x = TF.crop(x, *offset)
            if mask is not None:
                mask = TF.crop(mask, *offset)
        
        x_patch, hw, weight, mask_seq = self._get_patches(x, logits_weight, mask)
        N, C, P, _ = x_patch.shape
        lat = self.encoder(x_patch, is_training=False) 
        lat = self.unpatchify(lat, hw) 

        if dense:
            # not use dense
            h, w = hw
            B, C, H, W = lat.shape
            lat_dense = torch.zeros(B, C, H*2-1, W*2-1).to(lat.device)
            lat_dense[:, :, ::2, ::2] = lat
            if mask is not None:
                mask_dense = torch.zeros(B, 1, H*2-1, W*2-1).to(mask.device)
                mask_dense[:,:, ::2, ::2] = self.unpatchify(mask_seq, hw)
            
            if weight is not None:
                weight_dense = torch.zeros(B, 1, H*2-1, W*2-1).to(weight.device)
                weight_dense[:, :, ::2, ::2] = self.unpatchify(weight, hw)
            
            for crop_size in self.crop_sizes:
                begin_h = int(crop_size[0] * P)
                begin_w = int(crop_size[1] * P)
                crop_h = int((h - crop_size[2]) * P)
                crop_w = int((w - crop_size[3]) * P)
                x_ = TF.crop(x, begin_h, begin_w, crop_h, crop_w)
                mask_ = TF.crop(mask, begin_h, begin_w, crop_h, crop_w) if mask is not None else None
                x_patch_, hw_, weight_, mask_seq_ = self._get_patches(x_, logits_weight, mask_)
                lat_ = self.encoder(x_patch_, is_training=False)
                lat_ = self.unpatchify(lat_, hw_)
                lat_dense[:, :,int(crop_size[0]*2)::2, int(crop_size[1]*2)::2] = lat_
                
                if mask is not None:
                    mask_dense[:,:,int(crop_size[0]*2)::2, int(crop_size[1]*2)::2] = self.unpatchify(mask_seq_, hw_)
                
                if weight is not None:
                    weight_dense[:,:, int(crop_size[0]*2)::2, int(crop_size[1]*2)::2] = self.unpatchify(weight_, hw_)
            
            lat = lat_dense
            if mask is not None:
                mask_seq = mask_dense
            if weight is not None:
                weight = weight_dense        
        if aug:
            x_patch = self.patch_augmentation(x_patch)

        B, C, H, W = lat.shape
        lat = lat.reshape(B, C, -1).permute(0, 2, 1)
        
        _, lat_q, _, _, encoding_indices = self.codebook(lat)
        
        if weight is not None:
            weight = weight.reshape(B, -1)
        
        if mask_seq is not None:
            mask_seq = mask_seq.reshape(B, -1)
        
        return {
            "lat_q": lat_q,
            "lat": lat,
            "token": encoding_indices,
            "weight": weight,
            "mask_seq": mask_seq.bool() if mask_seq is not None else None
        }
    
    def get_token_from_latent(self, lat):
        _, _, _, _, encoding_indices = self.codebook(lat)
        return encoding_indices
    
    def get_all_tokens(self, x, logits_weight:torch.Tensor=None):
        if self.channel_independent:
            # transpose to (B*C_i, 1, H, W)
            B, C_i, H, W = x.shape
            x_ = x.reshape(B * C_i, 1, H, W)
            x_patch, hw = self.patchify(x_)
        else:
            x_patch, hw = self.patchify(x) 
        
        if logits_weight is not None:
            C = x.shape[1]
            logits_weight = logits_weight.reshape(-1, C+1)
            weight = logits_weight[:,:C, None, None] * (x > 0).float()
            blank = (x.sum(dim=1, keepdim=True) < -C + 1).float() * logits_weight[:, -1, None, None]
            weight_patch, _ = self.patchify(weight)
            blank_patch, _ = self.patchify(blank)
            weight = torch.sum(weight_patch, dim=(1, 2, 3)) + torch.sum(blank_patch, dim=(1, 2, 3))
        else:
            weight = None
            
        
        lat = self.encoder(x_patch, is_training=False)
        lat = self.unpatchify(lat, hw) 
        B, C, H, W = lat.shape
        lat = lat.reshape(B, C, -1).permute(0, 2, 1)
        
        _, _, _, _, encoding_indices = self.codebook(lat)
        
        if self.channel_independent:
            encoding_indices = encoding_indices.reshape(B // C_i, C_i, -1)
        
        return {
            "token": encoding_indices,
            "weight": weight if logits_weight is not None else None
        }
    
    def forward_construction_loss(self, x, y, weight, is_log=False):
        # x.shape = (B, C, H, W)
        # loss = F.mse_loss(x, y)
        if weight is not None:
            assert x.shape[1] == weight.shape[0]
        

        if is_log:
            raise NotImplementedError
        else :
            loss = F.mse_loss(x, y, reduction='none')
            loss = torch.mean(loss, dim=(2, 3))
            if weight is None:
                weight = self._default_weight(y)
            loss = loss * weight
            loss = torch.mean(loss)
        
        return loss
    
    def patchify(self, x):
        p = self.patch_size
        B, C, H, W = x.shape
        h = H // p
        w = W // p
        x = x.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(-1, C, p, p)
        return x, (h, w)
    
    def unpatchify(self, x, hw):
        h, w = hw
        _, C, _, _ = x.shape
        x = x.reshape(-1, h, w, C)
        x = x.permute(0, 3, 1, 2)
        return x
    
    def _default_weight(self, y):
        H, W = y.shape[-2:]
        count = torch.sum(y, dim=(2, 3))
        weight = float(H * W) / (count + 1)
        return weight
    
    def forward(self, x, weight=None):
        x_patch, hw = self.patchify(x)
        
        lat = self.encoder(x_patch, self.is_training) 
        
        lat_list = None
        if isinstance(lat, tuple):
            lat, lat_list = lat

        lat = self.unpatchify(lat, hw)
        
        B, C_l, h, w = lat.shape 
        lat = lat.reshape(B, C_l, -1).permute(0, 2, 1)

        loss_codebook, lat_q, perplexity, _, encoding_indices = self.codebook(lat)
        
        if lat_list is not None:
            for lat_ in lat_list:
                loss_codebook = loss_codebook + self.codebook.beta * F.mse_loss(lat_.reshape(lat_q.shape), lat_q.detach())
        
        lat_q = lat_q.permute(0, 2, 1).reshape(B, C_l, h, w)

        y_q = self.decoder(lat_q)
        loss_reconstruction = self.forward_construction_loss(x, y_q, weight, False)
        
        return loss_reconstruction, loss_codebook, perplexity, encoding_indices, y_q, lat_q
        