#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance of a test_ehr_extract.py run on each cervix protocol read by the us-ocr pipeline, and on every
combination of the protocols.

For a combination, each child's prediction is rebuilt from only its images of those protocols, with each fold's
aggregation, TTA and threshold, and the folds are ensembled as in test_ehr_extract.py. Only the folds in the image
predictions are used, i.e. those that had finished training when test_ehr_extract.py ran. Two views:
  - "with protocols": every child with at least one image of the combination. Shows how each combination does
    where it is available, but different combinations are tested on different children.
  - "all 4 protocols": only children with images of all 4 protocols, so the combinations differ only in which
    images are used. The Shapley values split this view's AUC above chance (0.5) between the 4 protocols.

The OCR rows are matched to the absolute image paths that the models were tested on (img_path in the image
predictions, without its TTA suffix).

Usage: python analyse_ocr_protocols.py <run_dir>/<test name> <us-ocr results.csv> [--ocr-path-column file_path]
       [--ocr-path-root <folder joined onto the OCR paths>]
"""

# %%Imports
import argparse
import glob
import itertools
import math
import os
import re

import numpy as np
import polars as pl
import torch

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


def child_predictions(images):
    """Each fold's child predictions with its own aggregation, TTA and threshold, and their ensemble."""
    fold_dfs = []
    for model_name in sorted(images["model"].unique()):
        fold_images = images.filter(pl.col("model") == model_name)
        settings = fold_images.row(0, named=True)
        fold_dfs.append(
            patient_predictions(
                fold_images, settings["agg"], settings["tta"], settings["threshold"]
            ).with_columns(pl.lit(model_name).alias("model"))
        )
    folds_df = pl.concat(fold_dfs)
    return folds_df, ensemble_predictions(folds_df)


def evaluate_images(images, n_bootstrap, rng):
    """Ensemble metrics with bootstrap CIs, plus the mean and SD of the fold models' AUCs."""
    folds_df, ensemble_df = child_predictions(images)
    result = {
        "images": images["image_path"].n_unique(),
        "folds": folds_df["model"].n_unique(),
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


# %%Load the test predictions and the protocols
parser = argparse.ArgumentParser()
parser.add_argument(
    "test_dir", help="Output folder of test_ehr_extract.py, i.e. <run_dir>/<test name>"
)
parser.add_argument(
    "ocr_results", help="us-ocr results CSV (main.py output) for the same images"
)
parser.add_argument(
    "--ocr-path-column",
    default="file_path",
    help="us-ocr column with the image paths. With --ocr-path-root joined onto it, it must give the absolute "
    "img_path of the test predictions",
)
parser.add_argument(
    "--ocr-path-root",
    help="Folder joined onto every path in --ocr-path-column before matching, when the OCR paths are relative",
)
parser.add_argument(
    "--subgroup", help="Only use children where this column is True, e.g. parity_0"
)
parser.add_argument(
    "--n-bootstrap",
    type=int,
    default=1000,
    help="Bootstrap resamples for the ensemble's 95%% CIs",
)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

protocols = cervix_protocols()
ocr = load_protocols(args.ocr_results, args.ocr_path_column, args.ocr_path_root)
save_path = os.path.join(args.test_dir, "ocr_protocols")
os.makedirs(save_path, exist_ok=True)

prediction_files = sorted(
    glob.glob(os.path.join(args.test_dir, "image_predictions_*.parquet"))
)
if not prediction_files:
    raise FileNotFoundError(
        f"No image_predictions_<cutoff>.parquet in {args.test_dir}; rerun test_ehr_extract.py"
    )

for path in prediction_files:
    cutoff = re.search(r"image_predictions_(\d+)\.parquet", path).group(1)
    images = pl.read_parquet(path)
    if args.subgroup is not None:
        images = images.filter(pl.col(args.subgroup))

    # img_path is the absolute path of the loaded image with the TTA flip appended, e.g. <image>.png_1
    image_paths = images["img_path"].str.replace(r"_\d+$", "").to_list()
    images = images.with_columns(
        pl.Series(
            "image_path", [os.path.normpath(image_path) for image_path in image_paths]
        )
    )

    n_images = images["image_path"].n_unique()
    images = images.join(ocr, on="image_path", how="left")
    n_matched = images.filter(pl.col("image_path").is_in(ocr["image_path"].implode()))[
        "image_path"
    ].n_unique()
    print(
        f"\n========== GA {cutoff}{' (' + args.subgroup + ')' if args.subgroup else ''} =========="
    )
    # test_ehr_extract.py only tests the folds that had finished training
    fold_names = sorted(images["model"].unique().to_list())
    print(f"Ensemble of the {len(fold_names)} folds in the test predictions: {', '.join(fold_names)}")
    print(
        f"{n_matched:,} of {n_images:,} test images are in the us-ocr results (on {args.ocr_path_column} = img_path without the TTA suffix)"
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
                **evaluate_images(view_images, args.n_bootstrap, rng),
            }
        )

        for size in range(1, len(protocols) + 1):
            for combination in itertools.combinations(protocols, size):
                combination_images = view_images.filter(
                    pl.col("protocol").is_in(list(combination))
                )
                if combination_images.is_empty():
                    continue
                result = evaluate_images(combination_images, args.n_bootstrap, rng)
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
    suffix = f"{cutoff}_{args.subgroup}" if args.subgroup else cutoff
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
