#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 10:17:10 2026

@author: jacob
"""

import ultradino_finetune.models.dinov2.load as vit_load
import logging
from models.Transform import Transform, CNNTransform
from models.Predictor import FCPredictor
from models.BirthModel import BirthModel
import torch.nn as nn
import re
import torch

logger = logging.getLogger("model_loader")

def vit_from_conf(cfg):    
    if cfg.data.img_size[0] != cfg.data.img_size[1]:
        raise Exception(f"Models only support square images. You have {cfg.data.img_size}")
    img_size = cfg.data.img_size[0]

    #Stochastic depth while the ViT trains. Configs without the key keep ultradino-finetune's 0.3
    drop_path_rate = cfg.model.vit.get("drop_path_rate", 0.3)

    if cfg.model.vit.weights_path.endswith('.pth'):
        match = re.search(r'vit\w{3}', cfg.model.vit.weights_path)
        model_type = match.group() if match else None

        if model_type is not None:
            model = vit_load.load_pretrained_ultradino(model_type, cfg.model.vit.weights_path, img_size=img_size,
                                                       drop_path_rate=drop_path_rate)
            logger.info('Loading pretrained encoder from %s', cfg.model.vit.weights_path)
        else:
            raise Exception("Weight path must contain the model type [Vits16/Vitb16/Vitl16]\n"
                            "E.g. 'oversampled_vitb16.pth'")
        
    else:
        model = vit_load.load_from_scratch(cfg.model.vit.weights_path, img_size=img_size,
                                           drop_path_rate=drop_path_rate)
        logger.info(f"Loading {cfg.model.vit.weights_path} encoder from scratch")
    return model


# def seg_vit_from_conf(cfg, **kwargs):
#     if 'vitb16' in cfg.weights_path:
#         logger.info('Loading pretrained encoder from %s', cfg.weights_path)
#         model = UltraDINOSegmentationModel.from_pretrained(model_type='vitb16',
#                                                            weights_path=cfg.weights_path,
#                                                            num_classes=1)
#     else:
#         raise Exception(f"Segmentation model does not support model type from weights path {cfg.weights_path}")
#     return model


def ehr_from_conf(cfg, **kwargs):
    #Placeholder for EHR model
    #Remember to update freezeing in BirthModel to accomedate EHR model if implemented!
    class PassThrough(nn.Module):
        def __init__(self):
            super(PassThrough, self).__init__()
            self.embed_dim = len(cfg.data.ehr_data)
            
        def forward(self, x):
            return x
    model = PassThrough()
    
    return model


def set_dropout(model, p=0.1):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p


def model_from_conf(cfg, weights_path=None):
    """Create GA model from configuration"""
    
    device = cfg.device.type
    
    model_parts = {}
    
    #Foundation models
    vit_model = vit_from_conf(cfg)
    ehr_model = ehr_from_conf(cfg)
    
    model_parts['vit_model'] = vit_model
    model_parts['ehr_model'] = ehr_model
    
    #Transforms
    #Configs without the key keep raw inputs, so their checkpoints still load
    normalize_inputs = cfg.model.transforms.get("normalize_inputs", False)

    img_data_transform = Transform(len(cfg.data.img_data),
                                   vit_model.embed_dim,
                                   layer_dims=cfg.model.transforms.vit_layer_dims,
                                   normalize_inputs=normalize_inputs)

    ehr_transform = Transform(ehr_model.embed_dim,
                              vit_model.embed_dim,
                              layer_dims=cfg.model.transforms.ehr_layer_dims,
                              normalize_inputs=normalize_inputs)
        
    segmentation_transform = CNNTransform(cfg.data.img_size, 
                                          vit_model.embed_dim,
                                          channel_dims=cfg.model.transforms.seg_channel_dims)
    
    model_parts['img_data_transform'] = img_data_transform
    model_parts['ehr_transform'] = ehr_transform
    model_parts['segmentation_transform'] = segmentation_transform

    #Predictor heads
    preterm_heads = nn.ModuleDict({})
    aux_task_heads = nn.ModuleDict({})
    
    for task in cfg.tasks.keys():
        if task == 'preterm':
            for cutoff in cfg.tasks[task].cutoffs:
                preterm_heads[str(cutoff)] = FCPredictor(vit_model.embed_dim,
                                                         cfg.model.head.dropout,
                                                         cfg.model.head.layer_dims)
                
        elif task == 'aux_tasks':
            for aux_cfg in cfg.tasks[task]:
                name = aux_cfg['var'] if isinstance(aux_cfg['var'], str) else '_'.join(aux_cfg['var'])
                aux_task_heads[name] = FCPredictor(vit_model.embed_dim,
                                                   cfg.model.head.dropout,
                                                   cfg.model.head.layer_dims)
              
    model_parts['preterm_heads'] = preterm_heads
    model_parts['aux_task_heads'] = aux_task_heads

    model = BirthModel(model_parts, aux_method=cfg.auxiliary.method)

    if weights_path is not None:
        weights=torch.load(weights_path, weights_only=True)
        model.load_state_dict(weights)
    
    return model.to(device)


def update_freezing(model, epoch, cfg):
    if epoch == 0:
        model.freeze_model(model.vit_model)
        model.freeze_model(model.ehr_model)
    if epoch >= cfg.training.vit_frozen_until:
        n = epoch - cfg.training.vit_frozen_until
        model.unfreeze_vit(model.vit_model, n, cfg)


def set_train_mode(model):
    """Train mode, except frozen ViT blocks, so stochastic depth is off while they are frozen"""
    model.train()
    for block in model.vit_model.blocks:
        if not any(p.requires_grad for p in block.parameters()):
            block.eval()

            
