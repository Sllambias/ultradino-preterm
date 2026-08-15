#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:03:57 2026

@author: jacob
"""

import torchmetrics.classification as tm
import polars as pl
import torch
from pathlib import Path
import matplotlib.pyplot as plt


class Metrics:
    def __init__(self, cfg, save_path):
        cutoff = cfg.tasks.preterm.cutoff
        metric = tm.SensitivityAtSpecificity(min_specificity=0.85, task="binary").to(cfg.device.type)

        df = {str(cutoff): []}
        metrics = {
            "train_loss": [],
            "val_loss": [],
            **{agg: {str(cutoff): {"SensAtSpec": [], "SensAtSpec_cutoff": []}} for agg in ("avg", "max")},
        }

        self.cutoff = cutoff
        self.metric = metric
        self.metrics = metrics
        self.df = df
        self.save_path = save_path

    def update(self, outputs, labels, ID):
        self.df[str(self.cutoff)].append(
            pl.DataFrame(
                {
                    "ID": ID,
                    "preds": outputs["preterm"][str(self.cutoff)]["preds"].flatten().cpu().numpy(),
                    "label": (labels < float(self.cutoff)).flatten().cpu().numpy(),
                }
            )
        )

    def plot_metrics(self):
        for agg in ["avg", "max"]:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(self.metrics["train_loss"], label="Train Loss")
            ax.plot(self.metrics["val_loss"], label="Val Loss")
            ax.plot(self.metrics[agg][str(self.cutoff)]["SensAtSpec"], label=f"{self.cutoff} weeks")

            ax.set_title(agg.capitalize())
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Sensitivity @ 85% Specificity")
            ax.set_ylim(0, 1.05)
            ax.legend(loc="upper left")

            plt.tight_layout()
            fig.savefig(Path(self.save_path) / f"{agg}_metrics.png", dpi=300)
            plt.close(fig)

    def log_metrics(self, train_loss, val_loss):

        self.metrics["train_loss"].append(round(train_loss, 5))
        self.metrics["val_loss"].append(round(val_loss, 5))

        row = {"train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5)}

        df = pl.concat(self.df[str(self.cutoff)])
        df = df.group_by("ID").agg(
            [
                pl.col("preds").mean().alias("avg"),
                pl.col("preds").max().alias("max"),
                pl.col("label").first().alias("label"),
            ]
        )

        labels = torch.tensor(df["label"].to_numpy(), dtype=torch.int32)

        for agg in ["avg", "max"]:
            preds = torch.tensor(df[agg].to_numpy(), dtype=torch.float32)

            self.metric.reset()

            sens_spec, sens_spec_cutoff = self.metric(preds, labels)

            self.metrics[agg][str(self.cutoff)]["SensAtSpec"].append(sens_spec.item())
            self.metrics[agg][str(self.cutoff)]["SensAtSpec_cutoff"].append(sens_spec_cutoff.item())

            for name, values in self.metrics[agg][str(self.cutoff)].items():
                row[f"{name}_{self.cutoff}_{agg}"] = values[-1]

        metrics_df = pl.DataFrame([row])

        path = Path(self.save_path) / "metrics.csv"

        if path.exists():
            existing = pl.read_csv(path)
            pl.concat([existing, metrics_df], how="vertical").write_csv(path)
        else:
            metrics_df.write_csv(path)

        self.plot_metrics()
        self.df = {str(self.cutoff): []}


def get_metrics(cfg, t=0.5):
    metrics = {
        "Recall": tm.Recall(task="binary", threshold=t).to(cfg.device.type),
        "Specificity": tm.Specificity(task="binary", threshold=t).to(cfg.device.type),
        "SensAtSpec": tm.SensitivityAtSpecificity(min_specificity=0.85, task="binary").to(cfg.device.type),
    }

    return metrics
