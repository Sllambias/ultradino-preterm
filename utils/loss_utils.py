#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:08:41 2026

@author: jacob
"""

import torch
from omegaconf import ListConfig


def get_loss(cfg):
    loss_map = {
        "bce": torch.nn.BCEWithLogitsLoss(reduction="mean"),
        "l2": torch.nn.MSELoss(reduction="mean"),
        "l1": torch.nn.L1Loss(reduction="mean"),
    }

    losses = {}

    for config in cfg.tasks.values():
        tasks = config if isinstance(config, (list, ListConfig)) else [config]

        for task in tasks:
            loss_name = task["loss"]

            if loss_name not in loss_map:
                raise ValueError(f"Loss type '{loss_name}' not implemented")

            else:
                losses[loss_name] = loss_map[loss_name]

    return losses


def fix_labels(label, cutoff, label_smoothing_param):
    positive = label < cutoff

    if label_smoothing_param > 0:
        labels = torch.sigmoid((cutoff - label) / label_smoothing_param)
    else:
        labels = positive.float()

    return labels
