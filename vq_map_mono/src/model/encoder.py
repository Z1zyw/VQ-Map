import torch
import torch.nn as nn
from src.utils.config import BasicConfig
from timm.models.vision_transformer import PatchEmbed, Block

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


class EncoderConfig(BasicConfig):
    def __init__(self, cfg=None) -> None:
        self.bev_size = [50, 50]
        self.patch_size = 5
        self.in_channels = 8
        self.encoder_embed_dim = 256
        self.encoder_depth = 2
        self.encoder_num_heads = 4
        self.encoder_norm: str='layer'
        self.encoder_out_dim = 32
        self.is_mask: bool = True
        self.encoder_mask_ratio: float = 0.40
        super().__init__(cfg)


class Encoder(nn.Module):
    '''
    Encoder extract the feature from the BEV image,
    only the unmasked position will be extracted.
    '''
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(
            img_size=cfg.bev_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_channels,
            embed_dim=cfg.encoder_embed_dim
        )
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.encoder_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, cfg.encoder_embed_dim))
        self.blocks = nn.ModuleList([
            Block(dim=cfg.encoder_embed_dim, num_heads=cfg.encoder_num_heads,qkv_bias=True)
        ])
        self.head = nn.Linear(cfg.encoder_embed_dim, cfg.encoder_out_dim)
        if cfg.encoder_norm == 'layer':
            self.norm = nn.LayerNorm(cfg.encoder_embed_dim)
        else :
            raise NotImplementedError
    
    def update_mask_ratio(self, mask_ratio: float):
        self.cfg.encoder_mask_ratio = mask_ratio
    
    def forward(self, x: torch.Tensor):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if not self.cfg.is_mask:
            self.cfg.encoder_mask_ratio = 0.0
        x, mask, ids_restore = random_masking(x, mask_ratio=self.cfg.encoder_mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        
        x = self.head(x)

        return x, mask, ids_restore
