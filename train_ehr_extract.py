#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py on EHR_extract population parquets (e.g. preterm_P1_img_v7) instead of the preprocess.py dataset.

The train and test parquets are pooled and split into data.folds folds by mother.
Model `fold` tests on fold `fold`, validates on fold `fold + 1` and trains on the other folds.
The test predictions of all models are concatenated into predictions/predictions_<cutoff>.parquet,
so every child is test data exactly once, and GA_<cutoff>.txt reports the metrics of that concatenation.

Usage: python train_ehr_extract.py confs/training_confs/ehr_extract_P1_img_v7.yaml
"""
#%%Imports
import argparse

import torch
from tqdm import tqdm

from dataloader.ehr_extract_dataloader import EHRExtractDataSplits
from utils.model_utils import model_from_conf, update_freezing, set_train_mode, mixed_precision
from utils.optim_loader import get_optimizer, get_cosine_schedule_with_warmup
from utils.loss_utils import ModelLoss
from utils.metrics import Metrics
from utils.utils import setup, load_config

import warnings
warnings.filterwarnings("ignore", message="The image is already gray.")
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")

#%%Load config and setup logger(s)
parser = argparse.ArgumentParser()
parser.add_argument("config", help="Training config with the EHR_extract parquets in data.train_path/test_path, which are pooled")
args = parser.parse_args()

cfg = load_config(args.config)

save_path = setup(cfg)

#%% Load and train n folds of models

Data = EHRExtractDataSplits(cfg, save_path + 'misc/')

Loss = ModelLoss(cfg)
metrics = Metrics(cfg, save_path)

for fold in range(cfg.data.folds):

    TrainLoader, ValLoader, TestLoader = Data.get_split(fold, cfg)

    model = model_from_conf(cfg)

    optimizer = get_optimizer(model, cfg)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg)
    Loss.reset_weight()

    for epoch in range(cfg.training.epochs):
        update_freezing(model, epoch, cfg)
        set_train_mode(model)

        for data in tqdm(TrainLoader, desc='Training'):
            optimizer.zero_grad()
            with mixed_precision(cfg):
                outputs, _ = model(data['imgs'].to(cfg.device.type),
                                   data['img_data'].to(cfg.device.type),
                                   data['ehr_data'].to(cfg.device.type),
                                   data['segmentation'].to(cfg.device.type))

            loss = Loss(outputs, data)
            loss.backward()
            optimizer.step()

        train_loss = Loss.get_loss()

        scheduler.step()
        model.eval()

        with torch.no_grad():
            for data in tqdm(ValLoader, desc='Validating'):
                with mixed_precision(cfg):
                    outputs, _ = model(data['imgs'].to(cfg.device.type),
                                       data['img_data'].to(cfg.device.type),
                                       data['ehr_data'].to(cfg.device.type),
                                       data['segmentation'].to(cfg.device.type))

                metrics.update(outputs, data)

                loss = Loss(outputs, data)

            val_loss = Loss.get_loss()

            metrics.log_epoch(train_loss, val_loss)

            torch.save(model.state_dict(), f"{save_path}/weights/fold_{fold}/{str(epoch+1).zfill(3)}.pth")
        Loss.update_weight()
    best_epoch = metrics.get_best_epoch()

    for cutoff in cfg.tasks.preterm.cutoffs:
        model.load_state_dict(torch.load(f"{save_path}/weights/fold_{fold}/{best_epoch[str(cutoff)]}.pth", weights_only=True))
        model.eval()
        with torch.no_grad():
            for data in tqdm(TestLoader, desc=f"Testing GA {cutoff} model"):
                with mixed_precision(cfg):
                    outputs, _ = model(data['imgs'].to(cfg.device.type),
                                       data['img_data'].to(cfg.device.type),
                                       data['ehr_data'].to(cfg.device.type),
                                       data['segmentation'].to(cfg.device.type))

                metrics.update(outputs, data, cutoff)

            metrics.log_fold(cutoff)

    metrics.reset()

metrics.log_model()
