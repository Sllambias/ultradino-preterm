#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:30:43 2026

@author: jacob
"""

from utils.model_utils import model_from_conf
from omegaconf import OmegaConf
import torch
import albumentations as A
   
FUS13M_MEAN = 0.1842924807
FUS13M_STD = 0.2187705424

def load_model_and_transform(cfg, weight_path):
    model = model_from_conf(cfg)
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    model.eval()
    
    transforms = A.Compose([A.Resize(height=cfg.data.img_size[0], width=cfg.data.img_size[1]),
                                 A.ToGray(p=1.0, num_output_channels=1),
                                 A.Normalize(mean=FUS13M_MEAN, std=FUS13M_STD),
                                 A.ToTensorV2()])       
    
    return model, transforms
    

def inference(model, data, cutoff, device_type='cuda'):
    
    outputs, _ = model(data['imgs'].to(device_type),
                       data['img_data'].to(device_type),
                       data['ehr_data'].to(device_type))

    pred = outputs['preterm'][str(cutoff)]['preds'].flatten().cpu().numpy()
    
    return pred

   
    
    