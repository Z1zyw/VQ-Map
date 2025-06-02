import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import ipdb

from typing import Union, List
from timm.models.vision_transformer import Block

from src.utils.config import BasicConfig, load_config
from src.model.norm_ema_quantizer import NormEMAVectorQuantizerConfig as CodebookConfig, NormEMAVectorQuantizer as Codebook, l2norm
from src.model.cnn_vq import CNNVQConfig, CNNVQ, Decoder
from src.model.token_encoder import TokenIdentity


class VQBEVSegConfig(BasicConfig):
    def __init__(self, cfg=None):
        self.tokenizer_cfg_path = "config/tokenizer.yaml"
        
        self.bev_pretrain_cfg_path = "config/bev_pretrain.yaml"
        self.bev_pretrain_ckpt_path = "bin/to/path"
        
        self.use_pretrain_encoder = True
        self.finetune_encoder = False
        self.bev_encoder_cfg_path = "config/bev_pretrain_transformer.yaml"
        self.bev_pretrain_encoder_ckpt_path = "bin/to/path"
        
        self.finetune_decoder = False
        
        self.embedding_dim = 256
        self.n_head = 8
        self.norm = "layer"
        self.mlp_ratio = 4
        self.n_layers = 4
        self.out_shape = [100, 100]
        super().__init__(cfg)


class VQBEVSeg(nn.Module):
    def __init__(
        self, 
        cfg: VQBEVSegConfig,
        pretrain_cfg: CNNVQConfig,
        tokenizer: Image2Token,
        encoder: TokenIdentity,
        codebook: Union[Codebook, List[Codebook]],
        decoder: Decoder
    ):
        super().__init__()
        self.cfg = cfg
        self.pretrain_cfg = pretrain_cfg
        
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.codebook = codebook
        
        if cfg.finetune_decoder:
            raise NotImplementedError
        else:
            self.decoder = decoder
        
        if isinstance(codebook, Codebook):
            assert tokenizer.cfg.codebook_size == codebook.num_tokens
        else:
            raise NotImplementedError
        
        # for evaluation
        self.max_pooling = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pred_resize = torch.nn.Upsample(size=(200, 200), mode='nearest')
    
    def load_token_encoder(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.load_state_dict(ckpt["model"])
        self.encoder.eval()
        print("==> Load Token Encoder from %s"%ckpt_path)
        freeze_module(self.encoder)
    
    def sequence_to_grid(self, sequence: torch.Tensor):
        # sequence: [B, L, C]
        # grid: [B, C, H, W]
        l, c = sequence.shape[1:]
        w, h = int(l ** 0.5), int(l ** 0.5)
        assert w * h == l
        return sequence.reshape(-1, h, w, c).permute(0, 3, 1, 2)

    def _get_logits_training(self, token_pred):
        if not self.cfg.finetune_decoder and not self.cfg.finetune_encoder:
            return None, None
        token = token_pred.argmax(dim=-1).detach()
        try:
            token = token.reshape(token.shape[0], 27, 27)
            token = token[:, ::2, ::2]
            token = token.reshape(token.shape[0], -1)
        except:
            pass
        encoder_token_pred = self.encoder(token)
        if not self.cfg.finetune_decoder:
            return encoder_token_pred, None
        
        encoder_token = encoder_token_pred.argmax(dim=-1)
        
        with torch.no_grad():
            bev_feature_sequnence = self.codebook.embedding(encoder_token)
        bev_feature_grid = self.sequence_to_grid(bev_feature_sequnence)
        bev_logits = self.decoder(bev_feature_grid)
        return encoder_token_pred, bev_logits
        
    def forward(
        self, 
        img:torch.Tensor, 
        calib:dict, 
        is_tokenizer_train:bool=False, 
        is_soft:bool=False,
        using_confidence:bool=False, # not used
        threshold:float=0.5, # not used
        offset=None # not used 
    ):
        tokenizer_output = self.tokenizer(img, calib, offset)
        if is_tokenizer_train:
            encoder_token_pred, bev_logits = self._get_logits_training(tokenizer_output["token_pred"])
            return {
                **tokenizer_output,
                'encoder_token_pred': encoder_token_pred,
                'bev_logits': bev_logits,
            }
            
        token_pred = tokenizer_output["token_pred"]
        if isinstance(token_pred, torch.Tensor):
            if len(token_pred.shape) == 4:
                token_pred = token_pred.max(dim=1)[0]
            
            token = token_pred.argmax(dim=-1).detach()
            
            if using_confidence:
                raise NotImplementedError
            
        elif isinstance(token_pred, list):
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        with torch.no_grad():
            if not is_soft:
                if isinstance(self.encoder, TokenIdentity):
                    index = token != 0
                    token[index] = token[index] - 1
                    bev_feature_sequnence = torch.zeros(token.shape[0], token.shape[1], self.codebook.codebook_dim, device=token.device)
                    bev_feature_sequnence[index] = self.codebook.embedding(token[index])
                    encoder_token_pred = None
                else:
                    encoder_token_pred = self.encoder(token)
                    encoder_token = encoder_token_pred.argmax(dim=-1)
                    bev_feature_sequnence = self.codebook.embedding(encoder_token)
            else:
                if isinstance(self.encoder, TokenIdentity):
                    index = token != 0
                    token_pred = token_pred[:, :, 1:]
                    soft_token = token_pred.softmax(dim=-1)
                    bev_feature_sequnence = torch.zeros(token.shape[0], token.shape[1], self.codebook.codebook_dim, device=token.device)
                    codebook_weight = self.codebook.embedding.weight
                    bev_feature_sequnence[index] = torch.matmul(soft_token[index], codebook_weight)
                    encoder_token_pred = None
            
        bev_feature_grid = self.sequence_to_grid(bev_feature_sequnence)
        bev_logits = self.decoder(bev_feature_grid)

        if self.tokenizer.cfg.num_token_each_patch > 1:
            _, _, h, w = bev_logits.shape
            bev_logits = bev_logits.reshape(-1, self.tokenizer.cfg.num_token_each_patch, *self.cfg.out_shape)
            
        # resize bev_logits to 200x200 for evaluation
        bev_logits = self.max_pooling(bev_logits)
        bev_logits = self.pred_resize(bev_logits)
        
        output = {
            **tokenizer_output,
            'encoder_token_pred': encoder_token_pred,
            'bev_logits': bev_logits
        }
        
        return output
    
def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False

from src.model.img2token import Image2Token, Image2TokenConfig

def build_vq_bev_seg_model(cfg: VQBEVSegConfig) -> VQBEVSeg:
    if "tokenizer" in cfg.tokenizer_cfg_path:
        tokenizer_cfg = Image2TokenConfig(load_config(cfg.tokenizer_cfg_path))
        tokenizer = Image2Token(tokenizer_cfg)
    else:
        raise NotImplementedError
    
    pretrain_cfg = CNNVQConfig(load_config(cfg.bev_pretrain_cfg_path))
    pretrain_model = CNNVQ(pretrain_cfg)
    
    ckpt = torch.load(cfg.bev_pretrain_ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        pretrain_model.load_state_dict(ckpt["model"])
    except RuntimeError as e:
        print("==> Model state dict not Matched")
        print("RuntimeError: %s"%e)
        pretrain_model.load_state_dict(ckpt["model"], strict=False)
    pretrain_model.eval()
    
    codebook = pretrain_model.codebook
    decoder = pretrain_model.decoder
    freeze_module(codebook)
    freeze_module(decoder)
    
    if 'identity' in cfg.bev_encoder_cfg_path:
        encoder = TokenIdentity()
    else:
        raise NotImplementedError
    
    if cfg.use_pretrain_encoder:
        raise NotImplementedError

    model = VQBEVSeg(cfg, pretrain_cfg, tokenizer, encoder, codebook, decoder)
    return model, pretrain_model