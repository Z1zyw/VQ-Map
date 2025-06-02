import torch
import torch.nn as nn
from src.utils.config import BasicConfig
from timm.models.vision_transformer import Block

class DecoderConfig(BasicConfig):
    def __init__(self, cfg=None) -> None:
        self.bev_size = [50, 50]
        self.patch_size = 5
        self.out_channels = 8
        self.decoder_in_dim = 32
        self.decoder_embed_dim = 1024
        self.decoder_depth = 4
        self.decoder_num_heads = 8
        self.decoder_norm = 'layer'
        self.decoder_mlp_ratio = 4
        self.is_mask = True
        super().__init__(cfg)


class Decoder(nn.Module):
    '''
    Decoder recover the masked BEV feature to the original BEV image, 
    which is a semantic segmentation head.
    '''
    def __init__(self, cfg:DecoderConfig) -> None:
        super().__init__()
        num_patches = (cfg.bev_size[0] // cfg.patch_size) * (cfg.bev_size[1] // cfg.patch_size)
        self.cfg = cfg
        self.decoder_embed = nn.Linear(cfg.decoder_in_dim, cfg.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_in_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.decoder_embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(dim=cfg.decoder_embed_dim, num_heads=cfg.decoder_num_heads, qkv_bias=True, mlp_ratio=cfg.decoder_mlp_ratio)
        ])
        
        # self.pred = nn.Linear(cfg.decoder_embed_dim, cfg.patch_size**2 * cfg.out_channels, bias=True)
        self.pred = nn.Linear(cfg.decoder_embed_dim, cfg.decoder_embed_dim, bias=True)
        
        if cfg.decoder_norm == 'layer':
            self.norm = nn.LayerNorm(cfg.decoder_embed_dim)
        else:
            raise NotImplementedError
    
    def forward(self, x, ids_restore=None):
        # x: [B, N, C], N = num_patches + 1 (cls token)

        if self.cfg.is_mask and ids_restore is not None:
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            # append cls token
            x = torch.cat([x[:, :1, :], x_], dim=1)  
        
        # embed tokens
        x = self.decoder_embed(x)
        
        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # predictor projection
        x = self.pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x