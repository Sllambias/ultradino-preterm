#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the fold models of a train_ehr_extract.py run on another EHR_extract population, e.g. prospective_img_v1.

Usage: python test_ehr_extract.py <run_dir> confs/test_confs/prospective_img_v1.yaml
"""
#%%Imports
import argparse
import os

import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.ehr_extract_dataloader import (EHRExtractDataset, ehr_extract_collate_fn, load_population,
                                               remove_excluded)
from utils.metrics import ensemble_predictions, evaluate_predictions, patient_predictions
from utils.model_utils import model_from_conf, mixed_precision

import warnings
warnings.filterwarnings("ignore", message="The image is already gray.")
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")


def best_checkpoint(run_dir, cutoff, fold):
    """Epoch, aggregation, TTA and threshold that training picked for a fold, redone from its validation metrics."""
    tables = {agg: pl.read_csv(f"{run_dir}/metrics/metrics_{cutoff}_{agg}_fold_{fold}.csv") for agg in ["avg", "quantile", "max"]}

    # Same rule as Metrics.log_epoch: epochs in order, then avg, quantile, max; a later choice must score higher
    best = {"score": -float("inf")}
    for i in range(len(tables["avg"])):
        for agg in ["avg", "quantile", "max"]:
            row = tables[agg].row(i, named=True)
            tta = max(["tta", "no-tta"], key=lambda t: row[f"SensAtSpec_{agg}_{t}"])
            score = row[f"SensAtSpec_{agg}_{tta}"]
            if score > best["score"]:
                best = {"score": score,
                        "epoch": row["epoch"],
                        "agg": agg,
                        "tta": tta,
                        "threshold": row[f"SensAtSpec_cutoff_{agg}_{tta}"]}

    # Training saved the agg, TTA and threshold of the best epoch with the fold's validation predictions
    saved = pl.read_parquet(f"{run_dir}/predictions/validation/predictions_{cutoff}_fold_{fold}.parquet").row(0, named=True)
    if (saved["agg"], saved["tta"], saved["threshold"]) != (best["agg"], best["tta"], best["threshold"]):
        raise ValueError(f"Fold {fold}, GA {cutoff}: the metrics CSVs give {best}, but training saved "
                         f"agg {saved['agg']}, tta {saved['tta']}, threshold {saved['threshold']}")
    return best


def predict_images(model, loader, cfg, cutoff):
    """Image predictions of one head, with every test time augmentation as its own row (as in Metrics.update)."""
    frames = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(loader, desc=f"Predicting GA {cutoff}"):
            with mixed_precision(cfg):
                outputs, _ = model(data['imgs'].to(cfg.device.type),
                                   data['img_data'].to(cfg.device.type),
                                   data['ehr_data'].to(cfg.device.type),
                                   data['segmentation'].to(cfg.device.type))

            frames.append(pl.DataFrame({'CPR_CHILD': data['IDs'],
                                        'img_path': data['img_paths'],
                                        'preds': outputs['preterm'][str(cutoff)]['preds'].flatten().cpu().numpy(),
                                        'label': (data['labels']['GA_weeks'] < float(cutoff)).flatten().cpu().numpy(),
                                        'treatment': data['treatment']}))
    return pl.concat(frames)


#%%Load the training run and the population
parser = argparse.ArgumentParser()
parser.add_argument("run_dir", help="Output folder of a train_ehr_extract.py run (holds conf.yaml, weights/, metrics/)")
parser.add_argument("test_config", help="Population to test on, e.g. confs/test_confs/prospective_img_v1.yaml")
args = parser.parse_args()

run_dir = os.path.expanduser(args.run_dir)
test_cfg = OmegaConf.load(args.test_config)
cfg = OmegaConf.load(os.path.join(run_dir, "conf.yaml"))
cfg.data.image_root = test_cfg.image_root
cfg.data.workers = test_cfg.workers

save_path = os.path.join(run_dir, test_cfg.name)
os.makedirs(save_path, exist_ok=True)

df = load_population(test_cfg.population_path, cfg.data.image_root)
# The same exclusions as in training: the no-MP rule and data.remove
df = remove_excluded(df, cfg, test_cfg.name)

missing = [column for column in test_cfg.subgroups if column not in df.columns]
if missing:
    raise ValueError(f"{test_cfg.population_path} has no subgroup columns {missing}")
child_subgroups = (df.group_by("CPR_BARN").agg([pl.col(column).first() for column in test_cfg.subgroups])
                   .rename({"CPR_BARN": "CPR_CHILD"}))

# FILE_PATH of each loaded image, so the image predictions can be joined to other image tables (e.g. us-ocr)
image_files = df.select(["image_path", "FILE_PATH"]).unique()
if image_files["image_path"].n_unique() != len(image_files):
    raise ValueError("Some image files belong to several FILE_PATHs")

loader = DataLoader(EHRExtractDataset(df, cfg, train=False),
                    cfg.data.batch_size // 4,
                    shuffle=False,
                    pin_memory=False,
                    drop_last=False,
                    num_workers=cfg.data.workers,
                    collate_fn=ehr_extract_collate_fn)

#%%Predict with every fold model and evaluate
rng = np.random.default_rng(test_cfg.seed)

for cutoff in cfg.tasks.preterm.cutoffs:
    fold_dfs = []
    image_dfs = []
    for fold in range(cfg.data.folds):
        best = best_checkpoint(run_dir, cutoff, fold)
        print(f"\nGA {cutoff}, fold {fold}: epoch {best['epoch']}, {best['agg']} {best['tta']}, threshold {best['threshold']}")

        model = model_from_conf(cfg, weights_path=f"{run_dir}/weights/fold_{fold}/{str(best['epoch']).zfill(3)}.pth")
        images = predict_images(model, loader, cfg, cutoff)
        del model

        image_dfs.append(images.with_columns([pl.lit(f"fold_{fold}").alias("model"),
                                              pl.lit(best['agg']).alias("agg"),
                                              pl.lit(best['tta']).alias("tta"),
                                              pl.lit(best['threshold']).alias("threshold")]))

        fold_dfs.append(patient_predictions(images, best['agg'], best['tta'], best['threshold'])
                        .with_columns([pl.lit(f"fold_{fold}").alias("model"),
                                       pl.lit(best['epoch']).alias("epoch"),
                                       pl.lit(best['agg']).alias("agg"),
                                       pl.lit(best['tta']).alias("tta"),
                                       pl.lit(best['threshold']).alias("threshold")]))

    folds_df = pl.concat(fold_dfs)
    ensemble_df = ensemble_predictions(folds_df)

    predictions = pl.concat([folds_df, ensemble_df], how="diagonal").join(child_subgroups, on="CPR_CHILD", how="left")
    predictions.write_parquet(os.path.join(save_path, f"predictions_{cutoff}.parquet"))

    # Every image and TTA flip of every fold, with each fold's settings, e.g. for analyse_ocr_protocols.py
    image_predictions = (pl.concat(image_dfs)
                         .with_columns(pl.col("img_path").str.replace(r"_\d+$", "").alias("image_path"))
                         .join(image_files, on="image_path", how="left", validate="m:1")
                         .drop("image_path")
                         .join(child_subgroups, on="CPR_CHILD", how="left"))
    image_predictions.write_parquet(os.path.join(save_path, f"image_predictions_{cutoff}.parquet"))

    # Metrics on every child and on each subgroup; bootstrap CIs for the ensemble only
    rows = []
    groups = {"all": pl.lit(True)}
    for column in test_cfg.subgroups:
        groups[column] = pl.col(column)
    for group, condition in groups.items():
        for model_name in [f"fold_{fold}" for fold in range(cfg.data.folds)] + ["ensemble"]:
            model_df = predictions.filter((pl.col("model") == model_name) & condition)
            if model_name == "ensemble":
                n_bootstrap = test_cfg.n_bootstrap
            else:
                n_bootstrap = 0
            rows.append({"group": group, "model": model_name, **evaluate_predictions(model_df, n_bootstrap, rng)})

    metrics_df = pl.DataFrame(rows)
    metrics_df.write_csv(os.path.join(save_path, f"metrics_{cutoff}.csv"))
    print(f"\nGA {cutoff} on {test_cfg.name}:")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(metrics_df.select(["group", "model", "children", "preterm", "AUC", "SensAtSpec", "Sensitivity", "Specificity"]))

print(f"\nPredictions and metrics written to {save_path}")
