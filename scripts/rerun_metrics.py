#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:06:52 2026

@author: jacob
"""

from utils.metrics import Metrics
from omegaconf import OmegaConf

path = '/users/data/UCPH/DeepFetal/projects/preterm/training_runs/Running/baseline_32/'

cfg = OmegaConf.load(path + 'conf.yaml')

metric = Metrics(cfg, path)

metric.log_model()