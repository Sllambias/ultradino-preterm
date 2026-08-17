#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:41:00 2026

@author: jacob
"""

import os
import polars as pl
import torch
import warnings
from dataloader.dataloader import PreTermDataset, collate_fn, make_data_split
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import get_metrics
from utils.model_utils import model_from_conf

warnings.filterwarnings("ignore", message="The image is already gray.")
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")


def test_model(folder_path, move=True, batch_size=2):
    cfg = OmegaConf.load(os.path.join(folder_path, "conf.yaml"))

    df = make_data_split(cfg, cfg.data.test_path, unique_column="CPR_MOR", training=False)
    TestData = PreTermDataset(df, cfg, train=False)
    TestLoader = DataLoader(
        TestData,
        batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=cfg.data.workers,
        collate_fn=collate_fn,
    )

    model = model_from_conf(cfg, TestData.ehr_df.shape[1] + TestData.img_metadata_df.shape[1])

    dirs = os.listdir(os.path.join(folder_path, "weights"))
    dirs.sort()

    cutoff = cfg.tasks.preterm.cutoff

    val_metrics_df = pl.read_csv(os.path.join(folder_path, "metrics.csv"))
    best_epoch = {}
    val_metrics = {}

    population_all = {}

    population_all[str(cutoff)] = {
        "Total Population": df["CPR_BARN"].n_unique(),
        "Non-preterm_births": df.filter(pl.col("GA") >= cutoff)["CPR_BARN"].n_unique(),
        "Preterm births": df.filter(pl.col("GA") < cutoff)["CPR_BARN"].n_unique(),
    }

    best_epoch[str(cutoff)] = {
        "all": {
            "Total Population": population_all[str(cutoff)]["Total Population"],
            "Preterm births": population_all[str(cutoff)]["Preterm births"],
            "Non-preterm_births": population_all[str(cutoff)]["Non-preterm_births"],
            "SensAtSpec": 0.0,
        },
    }

    val_metrics[str(cutoff)] = {
        "avg": val_metrics_df[f"SensAtSpec_threshold_{cutoff}_avg"],
        "max": val_metrics_df[f"SensAtSpec_threshold_{cutoff}_max"],
    }

    for weights in dirs:
        weight_path = os.path.join(folder_path, "weights", weights)
        print("evaluating: ", weights)
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        model.eval()

        with torch.no_grad():
            df = {str(cutoff): []}
            pbar = tqdm(TestLoader, desc=f"Checkpoint: {weights} / {len(dirs)}")
            for data in pbar:
                outputs, _ = model(
                    data["imgs"].to(cfg.device.type),
                    data["tabular_data"].to(cfg.device.type),
                    data["aux_vars"].to(cfg.device.type),
                )

                df[str(cutoff)].append(
                    pl.DataFrame(
                        {
                            "ID": data["ID"],
                            "preds": outputs["preterm"][str(cutoff)]["preds"].flatten().cpu().numpy(),
                            "label": list(data["labels"] < float(cutoff)),
                        },
                    )
                )
            df = pl.concat(df[str(cutoff)])
            df = df.group_by("ID").agg(
                [
                    pl.col("preds").mean().alias("pred_avg"),
                    pl.col("preds").max().alias("pred_max"),
                    pl.col("label").first().alias("label"),
                ]
            )
            df = df.with_columns(pl.col("label").cast(pl.Int32))
            preds = {
                "avg": torch.tensor(df["pred_avg"].to_numpy(), dtype=torch.float32),
                "max": torch.tensor(df["pred_max"].to_numpy(), dtype=torch.float32),
            }

            labels = torch.tensor(df["label"].to_numpy(), dtype=torch.int32)

            for eval_type in ["avg", "max"]:
                val_metric = val_metrics[str(cutoff)][eval_type].item(i)
                metrics = get_metrics(cfg, 0.5)

                for metric in metrics.values():
                    metric(preds[eval_type], labels)

                sens_spec, sens_spec_threshold = metrics["SensAtSpec"].compute()
                if sens_spec.item() > best_epoch[str(cutoff)]["all"]["SensAtSpec"]:
                    best_epoch[str(cutoff)]["all"]["Checkoipoint"] = weights
                    best_epoch[str(cutoff)]["all"]["SensAtSpec"] = sens_spec.item()
                    best_epoch[str(cutoff)]["all"]["SensAtSpec_threshold"] = sens_spec_threshold.item()
                    best_epoch[str(cutoff)]["all"]["AUC"] = roc_auc_score(df["label"] * 1.0, df[f"pred_{eval_type}"])
                    best_epoch[str(cutoff)]["all"]["Type"] = eval_type
                    best_epoch[str(cutoff)]["all"]["Sensitivity"] = metrics["Recall"].compute().item()
                    best_epoch[str(cutoff)]["all"]["Specificity"] = metrics["Specificity"].compute().item()
                    best_epoch[str(cutoff)]["all"]["val_sens_at_spec"] = val_metric
                    best_epoch[str(cutoff)]["all"]["weights"] = weight_path.replace("Running", "Evaluated")

    os.makedirs(os.path.join(folder_path, "preds"), exist_ok=True)
    pl.DataFrame(best_epoch[str(cutoff)]["all"]).write_csv(os.path.join(folder_path, f"preds/GA_{cutoff}_all.csv"))
    with open(os.path.join(folder_path, "test_results.txt"), "w") as f:
        f.write(f"\n----------GA {str(cutoff)}----------\n")
        f.write("--All patients--\n")
        for key, value in best_epoch[str(cutoff)]["all"].items():
            if isinstance(value, float):
                value = round(value, 3)
            f.write(f"\t {key} : {value}\n")
        f.write("\n")
        f.write("raw predictions: \n")
        for row in df.rows(named=True):
            f.write(f"Pred: {row['pred_avg']} Label: {row['label']} ID: {row['ID']} \n")
