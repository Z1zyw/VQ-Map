import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional

from src.model.img_backbone import ResNetwithFPN
from src.model.encoder import Encoder, EncoderConfig
from src.model.decoder import Decoder, DecoderConfig

from src.model.norm_ema_quantizer import NormEMAVectorQuantizerConfig as CodebookConfig, NormEMAVectorQuantizer as Codebook
from src.model.cnn_vq import CNNVQConfig, CNNVQ
from src.model.token_encoder import TokenEncoder, TokenEncoderConfig

# from src.model.tokenizer import TokenizerDeformableAttentionConfig, TokenizerDeformableAttention
from src.utils.config import load_config

def build_encoder(cfg: Union[EncoderConfig, str]) -> Encoder:
    '''
    Build the Encoder.
    '''
    if isinstance(cfg, str):
        cfg = load_config(cfg, EncoderConfig())
    encoder = Encoder(cfg)
    return encoder


def build_decoder(cfg: Union[DecoderConfig, str]) -> Decoder:
    '''
    Build the Decoder.
    '''
    if isinstance(cfg, str):
        cfg = load_config(cfg, DecoderConfig())
    decoder = Decoder(cfg)
    return decoder


def build_codebook(cfg: Union[CodebookConfig, str]) -> Codebook:
    '''
    Build the Codebook
    '''
    if isinstance(cfg, str):
        cfg = load_config(cfg, CodebookConfig())
    codebook = Codebook(cfg)
    return codebook



def build_cnn_vq(cfg: Union[CNNVQConfig, str]):
    '''
    Build the CNN Vector Quantized VAE.
    '''
    if isinstance(cfg, str):
        cfg = CNNVQConfig(load_config(cfg))
    model = CNNVQ(cfg)
    return model

def build_token_encoder(cfg: Union[TokenEncoderConfig, str]):
    '''
    Build the CNN Vector Quantized VAE.
    '''
    if isinstance(cfg, str):
        cfg = TokenEncoderConfig(load_config(cfg))
    model = TokenEncoder(cfg)
    return model

def build_cnn_vq_and_token_encoder(cnnvq_cfg: Union[CNNVQConfig, str], encoder_cfg):
    cnnvq = build_cnn_vq(cnnvq_cfg)
    token_encoder = build_token_encoder(encoder_cfg)
    return cnnvq, token_encoder

def build_backbone(backbone:str):
    '''
    Build the Image Backbone.
    '''
    if backbone.lower() == "resnet50_fpn":
        return ResNetwithFPN()
    else:
        raise NotImplementedError(f"Backbone {backbone} is not implemented.")


def freeze_model(model: nn.Module):
    '''
    Freeze the model.
    '''
    for param in model.parameters():
        param.requires_grad = False




