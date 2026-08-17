#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:41:00 2026

@author: jacob
"""

import argparse
import hydra
import torch
import warnings
from dataloader.dataloader import PreTermDataset, collate_fn, make_data_split
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.loss_utils import fix_labels, get_loss
from utils.metrics import Metrics
from utils.model_utils import model_from_conf, update_freezing
from utils.optim_loader import get_cosine_schedule_with_warmup, get_optimizer
from utils.test_utils import test_model
from utils.utils import setup

warnings.filterwarnings("ignore", message="The image is already gray.")
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")


@hydra.main(
    config_path="./confs/training_confs",
    config_name="default",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    save_path = setup(cfg)
    print(cfg)
    train_df, val_df = make_data_split(cfg, cfg.data.path, unique_column="CPR_MOR")
    TrainData = PreTermDataset(train_df, cfg, train=True)
    ValData = PreTermDataset(val_df, cfg, train=False)

    TrainLoader = DataLoader(
        TrainData,
        cfg.data.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=cfg.data.workers,
        collate_fn=collate_fn,
    )

    ValLoader = DataLoader(
        ValData,
        cfg.data.batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=cfg.data.workers,
        collate_fn=collate_fn,
    )

    model = model_from_conf(cfg, TrainData.ehr_df.shape[1] + TrainData.img_metadata_df.shape[1])

    optimizer = get_optimizer(model, cfg)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg)
    loss_fns = get_loss(cfg)
    metrics = Metrics(cfg, save_path)

    for epoch in range(cfg.training.epochs):
        update_freezing(model, epoch, cfg)

        model.train()
        train_loss = 0.0
        pbar = tqdm(TrainLoader, desc=f"Train epoch: {epoch} / {cfg.training.epochs}. Prev SensAtSpec: {metrics.sens_at_spec}")
        for data in pbar:
            optimizer.zero_grad()
            outputs, _ = model(
                data["imgs"].to(cfg.device.type),
                data["tabular_data"].to(cfg.device.type),
                data["aux_vars"].to(cfg.device.type),
            )
            loss = 0
            for task in cfg.tasks.keys():
                if task == "preterm":
                    cutoff, loss_fn, weight = cfg.tasks[task].values()
                    labels = fix_labels(data["labels"], cutoff, cfg.data.label_smoothing_param)
                    labels = labels.to(cfg.device.type)
                    loss += loss_fns[loss_fn](outputs[task][str(cutoff)]["logits"], labels) * weight
                else:
                    for idx, aux_task in enumerate(cfg.tasks[task]):
                        var, loss_fn, weight = aux_task.values()
                        labels = data["aux_vars"][:, idx].to(cfg.device.type).float()
                        loss += loss_fns[loss_fn](outputs[task][var]["logits"], labels.unsqueeze(1)) * weight
            loss.backward()

            train_loss += loss.item() / len(TrainLoader)
            optimizer.step()
            pbar.set_postfix({"train_loss": train_loss})

        scheduler.step()
        model.eval()
        val_loss = 0

        with torch.no_grad():
            pbar = tqdm(ValLoader, desc=f"Val epoch: {epoch} / {cfg.training.epochs}")
            for data in pbar:
                outputs, _ = model(
                    data["imgs"].to(cfg.device.type),
                    data["tabular_data"].to(cfg.device.type),
                    data["aux_vars"].to(cfg.device.type),
                )
                metrics.update(outputs, data["labels"], data["ID"])

                loss = 0

                for task in cfg.tasks.keys():
                    if task == "preterm":
                        cutoff, loss_fn, weight = cfg.tasks[task].values()
                        labels = fix_labels(data["labels"], cutoff, cfg.data.label_smoothing_param)
                        labels = labels.to(cfg.device.type)
                        loss += loss_fns[loss_fn](outputs[task][str(cutoff)]["logits"], labels) * weight
                    else:
                        for idx, aux_task in enumerate(cfg.tasks[task]):
                            var, loss_fn, weight = aux_task.values()
                            labels = data["aux_vars"][:, idx].to(cfg.device.type).float()
                            loss += loss_fns[loss_fn](outputs[task][var]["logits"], labels.unsqueeze(1)) * weight
                val_loss += loss.item() / len(ValLoader)
                pbar.set_postfix({"val_loss": val_loss})
        metrics.log_metrics(train_loss, val_loss)

        if (epoch + 1 % 5 == 0) or (epoch == cfg.training.epochs) or (epoch == 0):
            torch.save(model.state_dict(), save_path + "/weights/" + str(epoch).zfill(3) + ".pth")

    test_model(save_path)


if __name__ == "__main__":
    main()
