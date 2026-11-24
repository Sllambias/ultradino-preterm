#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:35:15 2026

@author: jacob
"""

from bias_analysis import generate_insight
import os
from utils.metrics import Metrics
from omegaconf import OmegaConf

path = '/users/data/UCPH/DeepFetal/projects/preterm/training_runs/Evaluated/'

folders = os.listdir(path)

for folder in folders:
    cfg = OmegaConf.load(path + folder + '/conf.yaml')

    metric = Metrics(cfg, path)

    metric.log_model()
    
    generate_insight(path + folder, 
                     '/users/data/UCPH/DeepFetal/projects/preterm/ultradino-preterm/confs/bias_analysis.yaml')
    
    