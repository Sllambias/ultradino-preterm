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

Next to the protocols and all images, both views also have the images that the PN_CompleteV1 classifier predicts
as cervix (max_pred = --pn-cervix-class), whatever their protocol. It is a class of its own: it is not combined
with the protocols and not part of the Shapley values. Images missing from the PN_CompleteV1 CSV count as not cervix.

With --auc-by-n-images, also the ensemble AUC by number of images: each child keeps at most n of its images (with
or without a protocol), drawn at random --n-draws times, for n = 1, 2, ... in both views. It shows how much of a
combination's AUC could come from simply having more images.

The OCR and PN_CompleteV1 rows are matched to the absolute image paths that the models were tested on (img_path in
the image predictions, without its TTA suffix).

Usage: python analyse_ocr_protocols.py <run_dir>/<test name> <us-ocr results.csv>
       [--ocr-path-column file_path] [--ocr-path-root <folder joined onto the OCR paths>]
       [--pn-predictions <PN_CompleteV1_cls_outputs.csv>] [--pn-path-column file_path]
       [--pn-path-root <folder joined onto the PN_CompleteV1 paths>]
       [--auc-by-n-images] [--n-draws 20] [--max-images <n>]
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


def image_paths(paths, path_root):
    """paths as an image_path column, with path_root joined onto them if given (absolute paths are kept as they are)."""
    if path_root is not None:
        paths = [os.path.join(path_root, path) for path in paths]
    # normpath so that e.g. a trailing slash on the root does not stop a match
    return pl.Series("image_path", [os.path.normpath(path) for path in paths])


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

    ocr = ocr.with_columns(image_paths(ocr[path_column].to_list(), path_root))

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


def load_pn_cervix(pn_path, path_column, path_root, cervix_class):
    """Whether PN_CompleteV1 predicts each image in its classifier outputs CSV as cervix, keyed by image_path.

    image_path is the path column with path_root joined onto it if given (absolute paths are kept as they are). An
    image is cervix if any of its rows has max_pred == cervix_class.
    """
    pn = pl.scan_csv(pn_path, schema_overrides={path_column: pl.String})
    columns = pn.collect_schema().names()
    for column in [path_column, "max_pred"]:
        if column not in columns:
            raise ValueError(f"{pn_path} has no column {column}. Its columns are {columns}")

    # Rows without a path cannot be matched to an image
    pn = pn.select([path_column, "max_pred"]).drop_nulls(path_column).collect()
    pn = pn.with_columns(image_paths(pn[path_column].to_list(), path_root))

    # An image can have several rows, e.g. when it was predicted twice
    return pn.group_by("image_path").agg(
        (pl.col("max_pred") == cervix_class).any().alias("pn_cervix")
    )


def print_coverage(images):
    """Images and children per protocol, how many of them PN_CompleteV1 predicts as cervix, and children by how
    many of the 4 protocols they have."""
    image_rows = images.select(["CPR_CHILD", "image_path", "protocol", "pn_cervix"]).unique()
    children = image_rows["CPR_CHILD"].n_unique()
    pn_children = image_rows.filter(pl.col("pn_cervix"))["CPR_CHILD"].n_unique()

    print(
        f"\n{image_rows['image_path'].n_unique():,} images of {children:,} children, "
        f"{image_rows['protocol'].is_not_null().sum():,} with a cervix protocol and "
        f"{image_rows['pn_cervix'].sum():,} predicted as cervix by PN_CompleteV1 (of {pn_children:,} children)"
    )
    per_protocol = (
        image_rows.group_by("protocol")
        .agg(
            [
                pl.len().alias("images"),
                pl.col("CPR_CHILD").n_unique().alias("children"),
                pl.col("pn_cervix").sum().alias("PN_cervix_images"),
            ]
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


def sorted_mean(preds, starts, counts):
    """Mean of each child's predictions."""
    return np.add.reduceat(preds, starts) / counts


def sorted_max(preds, starts, counts):
    """Largest of each child's predictions, which is its last as they are sorted."""
    return preds[starts + counts - 1]


def sorted_quantile(preds, starts, counts):
    """0.75 quantile of each child's predictions, taking the nearest one as polars does."""
    nearest = np.floor(0.75 * (counts - 1) + 0.5).astype(np.int64)
    return preds[starts + nearest]


def sorted_aggregation_functions():
    """aggregation_functions() of utils/metrics.py in numpy, for predictions sorted within each child."""
    return {"avg": sorted_mean, "max": sorted_max, "quantile": sorted_quantile}


def auc(scores, labels):
    """ROC AUC as the chance that a preterm child scores above a term child, ties counting half as in torchmetrics."""
    term = np.sort(scores[~labels])
    preterm = scores[labels]
    term_below = np.searchsorted(term, preterm, side="left")
    term_not_above = np.searchsorted(term, preterm, side="right")
    return (term_below + term_not_above).sum() / (2 * len(preterm) * len(term))


class FoldImages:
    """One fold's image predictions as arrays, to rebuild its child predictions from a subset of the images."""

    def __init__(self, fold_images, image_table):
        settings = fold_images.row(0, named=True)
        self.aggregate = sorted_aggregation_functions()[settings["agg"]]
        self.threshold = settings["threshold"]

        # The rows that patient_predictions aggregates, sorted by child and prediction
        if settings["tta"] == "no-tta":
            fold_images = fold_images.filter(pl.col("img_path").str.ends_with("_1"))
        rows = fold_images.join(image_table, on=["CPR_CHILD", "image_path"]).sort(["child", "preds"])
        if rows["image"].n_unique() != image_table.height:
            raise ValueError(f"{settings['model']} has no {settings['tta']} predictions for some of the images")
        self.child = rows["child"].cast(pl.Int64).to_numpy()
        self.image = rows["image"].cast(pl.Int64).to_numpy()
        self.preds = rows["preds"].to_numpy()

        n_children = image_table["child"].n_unique()
        self.all_images = self.first_children(np.ones(image_table.height, dtype=bool), n_children)

    def first_children(self, keep_image, n_children):
        """preds_normalized of children 0 to n_children - 1 from their kept images."""
        end = np.searchsorted(self.child, n_children)
        keep = keep_image[self.image[:end]]
        child = self.child[:end][keep]
        preds = self.preds[:end][keep]

        counts = np.bincount(child, minlength=n_children)
        starts = np.cumsum(counts) - counts
        aggregated = self.aggregate(preds, starts, counts)
        return np.where(
            aggregated < self.threshold,
            0.5 * aggregated / self.threshold,
            0.5 + 0.5 * (aggregated - self.threshold) / (1 - self.threshold),
        )

    def child_predictions(self, keep_image, n_changed):
        """preds_normalized of every child, rebuilding only the first n_changed children from their kept images."""
        predictions = self.all_images.copy()
        if n_changed > 0:
            predictions[:n_changed] = self.first_children(keep_image, n_changed)
        return predictions


def evaluate_n_images(images, max_images, n_draws, rng):
    """Ensemble AUC when each child keeps at most n random images, for n = 1 to max_images, over n_draws draws.

    Each draw shuffles every child's images once and keeps the first n for every n.
    """
    # Children with the most images first, so the children with more than n images are the first ones
    children = (
        images.group_by("CPR_CHILD")
        .agg([pl.col("image_path").n_unique().alias("n"), pl.col("label").first()])
        .sort(["n", "CPR_CHILD"], descending=[True, False])
        .with_row_index("child")
    )
    image_table = (
        images.select(["CPR_CHILD", "image_path"])
        .unique()
        .join(children.select(["CPR_CHILD", "child"]), on="CPR_CHILD")
        .sort(["child", "image_path"])
        .with_row_index("image")
    )
    folds = []
    for model_name in sorted(images["model"].unique()):
        folds.append(FoldImages(images.filter(pl.col("model") == model_name), image_table))

    images_per_child = children["n"].to_numpy()
    labels = children["label"].cast(pl.Boolean).to_numpy()
    if max_images is None:
        max_images = int(images_per_child.max())
    else:
        max_images = min(max_images, int(images_per_child.max()))

    ensemble_aucs = np.zeros((max_images, n_draws))
    fold_aucs = np.zeros((max_images, n_draws))
    for draw in range(n_draws):
        order = (
            image_table.with_columns(pl.Series("key", rng.random(image_table.height)))
            .select(pl.col("key").rank(method="ordinal").over("child"))["key"]
            .to_numpy()
        )
        for n in range(1, max_images + 1):
            keep_image = order <= n
            # Children with at most n images keep all of them, as with every image
            n_changed = int((images_per_child > n).sum())
            fold_predictions = []
            for fold in folds:
                fold_predictions.append(fold.child_predictions(keep_image, n_changed))
            ensemble = np.mean(fold_predictions, axis=0)

            ensemble_aucs[n - 1, draw] = auc(ensemble, labels)
            fold_aucs[n - 1, draw] = np.mean([auc(predictions, labels) for predictions in fold_predictions])

    rows = []
    for n in range(1, max_images + 1):
        aucs = ensemble_aucs[n - 1]
        rows.append(
            {
                "n_images": n,
                "children": children.height,
                "children_with_n": int((images_per_child >= n).sum()),
                "preterm": int(labels.sum()),
                "images": int(np.minimum(images_per_child, n).sum()),
                "folds": len(folds),
                "AUC_mean": round(float(np.mean(aucs)), 4),
                "AUC_sd": round(float(np.std(aucs)), 4),
                "AUC_min": round(float(np.min(aucs)), 4),
                "AUC_max": round(float(np.max(aucs)), 4),
                "fold_AUC_mean": round(float(np.mean(fold_aucs[n - 1])), 4),
            }
        )
    return rows


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
    "--pn-predictions",
    default="/storage/archive/UCPH/DeepFetal/ultrasound/derived_data/PN_CompleteV1/PN_CompleteV1_cls_outputs.csv",
    help="PN_CompleteV1 classifier outputs CSV with the predicted class (max_pred) of each image",
)
parser.add_argument(
    "--pn-path-column",
    default="file_path",
    help="PN_CompleteV1 column with the image paths. With --pn-path-root joined onto it, it must give the absolute "
    "img_path of the test predictions",
)
parser.add_argument(
    "--pn-path-root",
    help="Folder joined onto every path in --pn-path-column before matching, when the PN_CompleteV1 paths are relative",
)
parser.add_argument(
    "--pn-cervix-class",
    type=int,
    default=12,
    help="max_pred of cervix in PN_CompleteV1 (Cervix in its configs/label_mapping/class_label_map.yaml)",
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
parser.add_argument(
    "--auc-by-n-images",
    action="store_true",
    help="Also compute the ensemble AUC when each child keeps at most n random images",
)
parser.add_argument(
    "--n-draws",
    type=int,
    default=20,
    help="Random draws of each child's images for --auc-by-n-images",
)
parser.add_argument(
    "--max-images",
    type=int,
    help="Largest number of images per child for --auc-by-n-images. Default: the most any child has",
)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

protocols = cervix_protocols()
ocr = load_protocols(args.ocr_results, args.ocr_path_column, args.ocr_path_root)
pn = load_pn_cervix(
    args.pn_predictions, args.pn_path_column, args.pn_path_root, args.pn_cervix_class
)
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
    loaded_paths = images["img_path"].str.replace(r"_\d+$", "").to_list()
    images = images.with_columns(image_paths(loaded_paths, None))

    n_images = images["image_path"].n_unique()
    images = images.join(ocr, on="image_path", how="left")
    n_matched = images.filter(pl.col("image_path").is_in(ocr["image_path"].implode()))[
        "image_path"
    ].n_unique()
    images = images.join(pn, on="image_path", how="left")
    n_in_pn = images.filter(pl.col("pn_cervix").is_not_null())["image_path"].n_unique()
    print(
        f"\n========== GA {cutoff}{' (' + args.subgroup + ')' if args.subgroup else ''} =========="
    )
    # test_ehr_extract.py only tests the folds that had finished training
    fold_names = sorted(images["model"].unique().to_list())
    print(f"Ensemble of the {len(fold_names)} folds in the test predictions: {', '.join(fold_names)}")
    print(
        f"{n_matched:,} of {n_images:,} test images are in the us-ocr results (on {args.ocr_path_column} = img_path without the TTA suffix)"
    )
    print(
        f"{n_in_pn:,} of {n_images:,} test images are in the PN_CompleteV1 predictions (on {args.pn_path_column} = img_path without the TTA suffix)"
    )
    if n_in_pn == 0:
        raise ValueError(
            f"No test image is in {args.pn_predictions}, e.g. test image {images['image_path'][0]} and PN_CompleteV1 "
            f"image {pn['image_path'][0]}. Check --pn-path-column and --pn-path-root"
        )
    # Images that PN_CompleteV1 has no prediction for count as not cervix
    images = images.with_columns(pl.col("pn_cervix").fill_null(False))
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

        # The images that PN_CompleteV1 predicts as cervix, whatever their protocol
        pn_images = view_images.filter(pl.col("pn_cervix"))
        if pn_images.is_empty():
            print(f"No images predicted as cervix by PN_CompleteV1 in the view '{view}'")
        else:
            rows.append(
                {
                    "view": view,
                    "combination": "PN_CompleteV1 cervix",
                    "n_protocols": None,
                    **evaluate_images(pn_images, args.n_bootstrap, rng),
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

    # %%AUC by the number of images per child, drawn from all of a child's images
    if not args.auc_by_n_images:
        continue

    draw_rng = np.random.default_rng(args.seed)
    n_image_rows = []
    for view, view_images in views.items():
        if view_images.is_empty():
            continue
        for row in evaluate_n_images(view_images, args.max_images, args.n_draws, draw_rng):
            n_image_rows.append({"view": view, **row})

    n_images_df = pl.DataFrame(n_image_rows)
    n_images_df.write_csv(os.path.join(save_path, f"n_images_{suffix}.csv"))

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
        for view in views:
            print(
                f"\n{view}: ensemble AUC with at most n random images per child, over {args.n_draws} draws "
                f"(children_with_n have at least n images)"
            )
            print(n_images_df.filter(pl.col("view") == view).drop("view"))

print(f"\nWritten to {save_path}")
