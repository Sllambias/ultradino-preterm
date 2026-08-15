#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:30:20 2026

@author: jacob
"""

import os
from omegaconf import OmegaConf


def setup(cfg):
    if cfg.output_dir is None:
        raise Exception("Model experiment must be named")

    os.makedirs(os.path.join(cfg.output_dir, "weights"), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.output_dir, "conf.yaml"))
    return cfg.output_dir
