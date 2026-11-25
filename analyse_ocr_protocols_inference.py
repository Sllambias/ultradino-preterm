#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyse_ocr_protocols.py on the image predictions of scripts/inference.py instead of a test_ehr_extract.py run.

The prediction CSV has img_path (relative) and fold_<k>_pred, fold_<k>_cutoff and fold_<k>_binary_pred for every
fold. Each image's child and GA at birth come from the EHR_extract population, matched on img_path =
NO_OCR_FILE_PATH, after the training run's exclusions (as in test_ehr_extract.py). Each fold's child prediction is
the mean (--agg) of its image predictions, called at fold_<k>_cutoff, and the folds are ensembled as in
test_ehr_extract.py. The protocol combinations and Shapley values are the same as in analyse_ocr_protocols.py.

Usage: python analyse_ocr_protocols_inference.py <results.csv> <population.parquet> <run_dir>/conf.yaml
       <us-ocr results.csv> [--ga-cutoff 37] [--image-root <folder of img_path>] [--ocr-path-column file_path]
       [--ocr-path-root <folder joined onto the OCR paths>]
"""

# %%Imports
import argparse
import itertools
import math
import os
import re

import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf

from dataloader.ehr_extract_dataloader import keep_subgroup, load_population, remove_excluded
from utils.metrics import (
    binary_metrics,
    ensemble_predictions,
    evaluate_predictions,
    metric_value,
    patient_predictions,
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")


def cervix_protocols():
    """The 4 cervix protocols, as in us-ocr's get_OCR_stats.py."""
    return ["Cervix", "Cervix dybde 1", "Cervix dybde 2", "Cervix dybde 3"]


def load_fold_predictions(predictions_path, image_root):
    """One row per image and fold of a scripts/inference.py results CSV, with the fold's threshold."""
    predictions = pl.read_csv(predictions_path)

    folds = []
    for column in predictions.columns:
        match = re.fullmatch(r"fold_(\d+)_pred", column)
        if match:
            folds.append(int(match.group(1)))
    if not folds:
        raise ValueError(f"{predictions_path} has no fold_<k>_pred columns. Its columns are {predictions.columns}")

    fold_dfs = []
    for fold in sorted(folds):
        threshold_column = f"fold_{fold}_cutoff"
        if threshold_column not in predictions.columns:
            raise ValueError(f"{predictions_path} has fold_{fold}_pred but no {threshold_column}")
        thresholds = predictions[threshold_column].unique().to_list()
        if len(thresholds) != 1:
            raise ValueError(f"{predictions_path} has several thresholds for fold {fold}: {thresholds}")
        fold_dfs.append(
            predictions.select(
                [
                    pl.col("img_path"),
                    pl.col(f"fold_{fold}_pred").alias("preds"),
                    pl.col(threshold_column).alias("threshold"),
                    pl.lit(f"fold_{fold}").alias("model"),
                ]
            )
        )
    images = pl.concat(fold_dfs)

    # normpath so that e.g. "./a.png" matches "a.png"
    image_paths = [os.path.normpath(os.path.join(image_root, img_path)) for img_path in images["img_path"]]
    return images.with_columns(pl.Series("image_path", image_paths))


def load_protocols(ocr_path, path_column, path_root):
    """Cervix protocol of each image in a us-ocr results CSV, keyed by image_path (null if OCR found none).

    image_path is the path column with path_root joined onto it if given (absolute paths are kept as they are).
    """
    ocr = pl.read_csv(ocr_path, infer_schema=False)
    if path_column not in ocr.columns:
        raise ValueError(
            f"{ocr_path} has no column {path_column}. Its columns are {ocr.columns}"
        )

    # Rows without a path cannot be matched to an image
    ocr = ocr.drop_nulls(path_column)

    # TEMP fix: the OCR paths use an old folder name. Remove when us-ocr is rerun on the current folders
    old_folder = "RegionH_pseudonymized"
    new_folder = "RegionH/regionH/2026"
    n_renamed = ocr.filter(pl.col(path_column).str.contains(old_folder, literal=True)).height
    print(f"TEMP fix: replaced {old_folder} with {new_folder} in {n_renamed:,} of {ocr.height:,} OCR paths")
    ocr = ocr.with_columns(pl.col(path_column).str.replace_all(old_folder, new_folder, literal=True))

    paths = ocr[path_column].to_list()
    if path_root is not None:
        paths = [os.path.join(path_root, path) for path in paths]
    # normpath so that e.g. a trailing slash on the root does not stop a match
    ocr = ocr.with_columns(
        pl.Series("image_path", [os.path.normpath(path) for path in paths])
    )

    ocr = ocr.select(
        [
            pl.col("image_path"),
            pl.when(pl.col("protocol").is_in(cervix_protocols()))
            .then(pl.col("protocol"))
            .alias("protocol"),
        ]
    )

    conflicting = (
        ocr.drop_nulls("protocol")
        .group_by("image_path")
        .agg(pl.col("protocol").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
        .height
    )
    if conflicting > 0:
        print(
            f"Warning: {conflicting} images have rows with different protocols in {ocr_path}; the first is used"
        )

    # Resumed us-ocr runs can append a path twice, so keep one row per image, preferring one with a protocol
    return ocr.sort(pl.col("protocol").is_null(), maintain_order=True).unique(
        subset="image_path", keep="first", maintain_order=True
    )


def print_coverage(images):
    """Images and children per protocol, and children by how many of the 4 protocols they have."""
    image_rows = images.select(["CPR_CHILD", "image_path", "protocol"]).unique()
    children = image_rows["CPR_CHILD"].n_unique()

    print(
        f"\n{image_rows['image_path'].n_unique():,} images of {children:,} children, "
        f"{image_rows['protocol'].is_not_null().sum():,} with a cervix protocol"
    )
    per_protocol = (
        image_rows.group_by("protocol")
        .agg(
            [pl.len().alias("images"), pl.col("CPR_CHILD").n_unique().alias("children")]
        )
        .sort("protocol", nulls_last=True)
    )
    print(per_protocol)

    n_protocols = (
        image_rows.group_by("CPR_CHILD")
        .agg(pl.col("protocol").drop_nulls().n_unique().alias("n_protocols"))
        .group_by("n_protocols")
        .agg(pl.len().alias("children"))
        .sort("n_protocols")
    )
    print("Children by how many of the 4 protocols they have")
    print(n_protocols)


def child_predictions(images, agg):
    """Each fold's child predictions, combined by agg and called at the fold's threshold, and their ensemble."""
    fold_dfs = []
    for model_name in sorted(images["model"].unique()):
        fold_images = images.filter(pl.col("model") == model_name)
        threshold = fold_images["threshold"][0]
        # scripts/inference.py makes one prediction per image without flips, so "tta" keeps every row
        fold_dfs.append(
            patient_predictions(fold_images, agg, "tta", threshold).with_columns(
                pl.lit(model_name).alias("model")
            )
        )
    folds_df = pl.concat(fold_dfs)
    return folds_df, ensemble_predictions(folds_df)


def evaluate_images(images, agg, n_bootstrap, rng):
    """Ensemble metrics with bootstrap CIs, plus the mean and SD of the fold models' AUCs."""
    folds_df, ensemble_df = child_predictions(images, agg)
    result = {
        "images": images["image_path"].n_unique(),
        **evaluate_predictions(ensemble_df, n_bootstrap, rng),
    }

    auc = binary_metrics("cpu")["AUC"]
    fold_aucs = []
    for model_name in sorted(folds_df["model"].unique()):
        fold_df = folds_df.filter(pl.col("model") == model_name)
        fold_aucs.append(
            metric_value(
                auc,
                torch.tensor(
                    fold_df["preds_normalized"].to_numpy(), dtype=torch.float32
                ),
                torch.tensor(fold_df["label"].to_numpy(), dtype=torch.int32),
            )
        )
    result["fold_AUC_mean"] = round(float(np.mean(fold_aucs)), 4)
    result["fold_AUC_sd"] = round(float(np.std(fold_aucs)), 4)
    return result


def shapley_values(aucs, protocols):
    """Each protocol's Shapley share of the AUC above chance, from the AUC of every combination (none = 0.5)."""
    n = len(protocols)
    values = {}
    for protocol in protocols:
        others = [other for other in protocols if other != protocol]
        value = 0.0
        for size in range(n):
            weight = (
                math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            )
            for combination in itertools.combinations(others, size):
                if size == 0:
                    without = 0.5
                else:
                    without = aucs[frozenset(combination)]
                value += weight * (aucs[frozenset(combination) | {protocol}] - without)
        values[protocol] = round(value, 4)
    return values


# %%Load the predictions, the population and the protocols
parser = argparse.ArgumentParser()
parser.add_argument(
    "predictions",
    help="results.csv of scripts/inference.py: img_path and fold_<k>_pred/_cutoff/_binary_pred for every fold",
)
parser.add_argument(
    "population",
    help="EHR_extract population parquet of the images; img_path must equal its NO_OCR_FILE_PATH",
)
parser.add_argument(
    "run_config",
    help="conf.yaml of the training run the predictions came from, for its exclusions (no-MP rule and data.remove)",
)
parser.add_argument(
    "ocr_results", help="us-ocr results CSV (main.py output) for the same images"
)
parser.add_argument(
    "--ga-cutoff",
    type=int,
    default=37,
    help="Preterm is born before this many weeks; the cutoff the predictions were made for (37 in scripts/inference.py)",
)
parser.add_argument(
    "--agg",
    choices=["avg", "quantile", "max"],
    default="avg",
    help="How a child's image predictions are combined. fold_<k>_cutoff was picked for the fold's own aggregation",
)
parser.add_argument(
    "--image-root",
    default="",
    help="Folder joined onto img_path and NO_OCR_FILE_PATH before matching them to the OCR paths "
    "(input_path of scripts/inference.py). By default the relative paths are matched",
)
parser.add_argument(
    "--ocr-path-column",
    default="file_path",
    help="us-ocr column with the image paths. With --ocr-path-root joined onto it, it must give img_path with "
    "--image-root joined onto it",
)
parser.add_argument(
    "--ocr-path-root",
    help="Folder joined onto every path in --ocr-path-column before matching",
)
parser.add_argument(
    "--subgroup", help="Only use children where this population column is True, e.g. parity_0"
)
parser.add_argument(
    "--save-path",
    help="Output folder. Default: ocr_protocols/ next to the prediction CSV",
)
parser.add_argument(
    "--n-bootstrap",
    type=int,
    default=1000,
    help="Bootstrap resamples for the ensemble's 95%% CIs",
)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

cfg = OmegaConf.load(args.run_config)
if args.ga_cutoff not in cfg.tasks.preterm.cutoffs:
    raise ValueError(
        f"--ga-cutoff {args.ga_cutoff} is not one of the run's cutoffs {list(cfg.tasks.preterm.cutoffs)}"
    )

protocols = cervix_protocols()
predictions = load_fold_predictions(args.predictions, args.image_root)
ocr = load_protocols(args.ocr_results, args.ocr_path_column, args.ocr_path_root)

save_path = args.save_path
if save_path is None:
    save_path = os.path.join(os.path.dirname(os.path.abspath(args.predictions)), "ocr_protocols")
os.makedirs(save_path, exist_ok=True)

population = load_population(args.population, args.image_root)
population = population.with_columns(
    pl.Series("image_path", [os.path.normpath(path) for path in population["image_path"]])
)
n_images = predictions["image_path"].n_unique()
n_in_population = predictions.filter(
    pl.col("image_path").is_in(population["image_path"].implode())
)["image_path"].n_unique()
print(
    f"\n{n_in_population:,} of {n_images:,} predicted images are in the population (on img_path = NO_OCR_FILE_PATH)"
)

# The same exclusions as in training and test_ehr_extract.py
population = remove_excluded(population, cfg, "population")
if args.subgroup is not None:
    population = keep_subgroup(population, args.subgroup, cfg, "population")

# One row per image and child; a twin pregnancy's image belongs to both children
children = population.select(
    [
        pl.col("image_path"),
        pl.col("CPR_BARN").alias("CPR_CHILD"),
        (pl.col("GA") // 7 < args.ga_cutoff).alias("label"),
        (pl.col("progesterone") | pl.col("cerclage")).alias("treatment"),
    ]
).unique()
images = predictions.join(children, on="image_path", how="inner")

n_kept = images["image_path"].n_unique()
images = images.join(ocr, on="image_path", how="left")
n_matched = images.filter(pl.col("image_path").is_in(ocr["image_path"].implode()))[
    "image_path"
].n_unique()
print(
    f"\n========== GA {args.ga_cutoff}{' (' + args.subgroup + ')' if args.subgroup else ''} =========="
)
print(f"{n_kept:,} predicted images are left after the exclusions")
print(
    f"{n_matched:,} of {n_kept:,} images are in the us-ocr results (on {args.ocr_path_column} = img_path)"
)
print_coverage(images)

# Children with images of all 4 protocols, for the view where only the images change
complete = (
    images.drop_nulls("protocol")
    .group_by("CPR_CHILD")
    .agg(pl.col("protocol").n_unique().alias("n"))
    .filter(pl.col("n") == len(protocols))["CPR_CHILD"]
)
views = {
    "with 1-4 protocols": images,
    "with all 4 protocols": images.filter(
        pl.col("CPR_CHILD").is_in(complete.implode())
    ),
}

# %%Evaluate every combination
rng = np.random.default_rng(args.seed)
rows = []
aucs = {}
for view, view_images in views.items():
    if view_images.is_empty():
        print(f"No children in the view '{view}'")
        continue

    # Reference: all of the children's images, also those without a cervix protocol
    rows.append(
        {
            "view": view,
            "combination": "all images",
            "n_protocols": None,
            **evaluate_images(view_images, args.agg, args.n_bootstrap, rng),
        }
    )

    for size in range(1, len(protocols) + 1):
        for combination in itertools.combinations(protocols, size):
            combination_images = view_images.filter(
                pl.col("protocol").is_in(list(combination))
            )
            if combination_images.is_empty():
                continue
            result = evaluate_images(combination_images, args.agg, args.n_bootstrap, rng)
            rows.append(
                {
                    "view": view,
                    "combination": " + ".join(combination),
                    "n_protocols": size,
                    **result,
                }
            )
            if view == "with all 4 protocols":
                aucs[frozenset(combination)] = result["AUC"]

combinations_df = pl.DataFrame(rows)
suffix = f"{args.ga_cutoff}_{args.subgroup}" if args.subgroup else str(args.ga_cutoff)
combinations_df.write_csv(os.path.join(save_path, f"combinations_{suffix}.csv"))

with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=60):
    for view in views:
        print(f"\n{view}: ensemble metrics (95% CI) and the fold models' AUC")
        print(
            combinations_df.filter(pl.col("view") == view).select(
                [
                    "combination",
                    "children",
                    "preterm",
                    "images",
                    "AUC",
                    "AUC_lower",
                    "AUC_upper",
                    "SensAtSpec",
                    "Sensitivity",
                    "Specificity",
                    "fold_AUC_mean",
                    "fold_AUC_sd",
                ]
            )
        )

    if len(aucs) == 2 ** len(protocols) - 1:
        shapley = shapley_values(aucs, protocols)
        shapley_df = pl.DataFrame(
            {"protocol": list(shapley), "AUC_contribution": list(shapley.values())}
        )
        shapley_df.write_csv(os.path.join(save_path, f"shapley_{suffix}.csv"))
        print(
            f"\nShapley contribution of each protocol to the AUC above 0.5, children with all 4 protocols "
            f"(sums to {aucs[frozenset(protocols)]:.4f} - 0.5)"
        )
        print(shapley_df)

print(f"\nWritten to {save_path}")
