#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:30:20 2026

@author: jacob
"""

from datetime import datetime
from omegaconf import OmegaConf
import os
import shutil


def timestamp(pattern):
    """Current time in a strftime pattern, used as ${now:<pattern>} in configs"""
    return datetime.now().strftime(pattern)


def load_config(path):
    """Load a config and resolve it once, so ${now:...} is the same timestamp for the whole run"""
    OmegaConf.register_new_resolver("now", timestamp, use_cache=True, replace=True)
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return cfg


def setup(cfg):
    if cfg.info.name is None:
        raise Exception("Model experiment must be named")

    if len(cfg.tasks.preterm.cutoffs) == 1:
        name = f"{cfg.info.name}_{cfg.tasks.preterm.cutoffs[0]}"
    else:
        name = f"{cfg.info.name}_multihead"

    #The start time goes after the cutoff, e.g. baseline_34_2026-11-25_21-42-00
    if cfg.info.get("timestamp") is not None:
        name = f"{name}_{cfg.info.timestamp}"

    #Configs without info.output_dir keep the old training_runs folder
    output_dir = cfg.info.get("output_dir", "/projects/users/data/UCPH/DeepFetal/projects/preterm/training_runs/Running/")
    #Expand ~, which os.path.join would otherwise keep as a folder named "~"
    path = os.path.join(os.path.expanduser(output_dir), name) + "/"

    if cfg.info.name == "test":
        if os.path.exists(path):
            shutil.rmtree(path)
    
    else:
        if os.path.exists(path):
            raise Exception("Model experiment exists in Running folder.\n"
                            "You can disable this check by naming the model 'test'")
        if os.path.exists(path.replace('Running', 'Evaluated')):
            raise Exception("Model experiment exists in Evaluated folder.\n"
                            "You can disable this check by naming the model 'test'")
            
    os.makedirs(path + 'misc', exist_ok=True)
    for i in range(cfg.data.folds):
        os.makedirs(path + 'weights/fold_' + str(i), exist_ok=True)
        
    OmegaConf.save(cfg, path + 'conf.yaml')        
    return path