#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:55:55 2026

@author: jacob
"""
import torch
import math
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer(model, cfg):
    if cfg.optimizer.type == "AdamW":
        optim = torch.optim.AdamW(decay_lr(model,
                                           base_lr=cfg.optimizer.lr2,
                                           lr_decay=cfg.optimizer.lr_decay,
                                           weight_decay=cfg.optimizer.weight_decay), 
                                  lr=cfg.optimizer.lr2,
                                  betas=cfg.optimizer.adamw_params[0:2],
                                  eps=cfg.optimizer.adamw_params[2])

    elif cfg.optimizer.type == "Muon":
        optim = torch.optim.Muon(model.parameters(),
                                 lr=cfg.optimizer.lr,
                                 weight_decay=cfg.optimizer.weight_decay)

    else:
        raise Exception(f"Optimizer {cfg.optimzier.type} not implemented")        
    
    return optim

def get_cosine_schedule_with_warmup(optimizer, cfg):
    n_warmup_steps = cfg.scheduler.num_warmup_steps
    vit_frozen = cfg.training.vit_frozen_until
    num_cycles = cfg.scheduler.num_cycles 
    epochs = cfg.training.epochs
    lr1 = cfg.optimizer.lr1
    lr2 = cfg.optimizer.lr2
    

    def lr_lambda(current_step):
        if current_step < vit_frozen:
            t = current_step / max(1, vit_frozen)
            alpha = 0.5 * (1 - math.cos(math.pi * t))
            lr = lr1 + (lr2 - lr1) * alpha
            return lr / lr2
        elif current_step < vit_frozen + n_warmup_steps:
            t = (current_step - vit_frozen) / max(1, n_warmup_steps)
            return t
        else:
            t = (current_step - vit_frozen - n_warmup_steps) / (epochs - vit_frozen - n_warmup_steps)
            t = min(max(t, 0.0), 1.0)
            return 0.5 * (1 + math.cos(math.pi * num_cycles * t))
    return LambdaLR(optimizer, lr_lambda)

def get_layer_id(name, n_layers):
    name = '.'.join(name.split('.')[1:])
    if name.startswith("patch_embed"):
        return 0

    #Embeddings get the lowest learning rate, as in DINOv2
    elif name in ["cls_token", "pos_embed", "register_tokens", "mask_token"]:
        return 0

    elif name.startswith("blocks"):
        block_id = int(name.split(".")[1])
        return block_id + 1

    else:
        return n_layers
    
def decay_lr(model, base_lr, lr_decay, weight_decay):
    n_layers = len(model.vit_model.blocks) + 1  # blocks + patch embed´

    param_groups = []

    for name, param in model.named_parameters():
        layer_id = get_layer_id(name, n_layers)
        scale = lr_decay ** (n_layers - layer_id)

        #No weight decay on biases, norms, LayerScale and tokens
        if param.ndim != 1 and name.endswith('.weight'):
            wd = weight_decay
        else:
            wd = 0.0

        param_groups.append({"params": [param],
                             "lr": base_lr * scale,
                             'weight_decay': wd})

    return param_groups