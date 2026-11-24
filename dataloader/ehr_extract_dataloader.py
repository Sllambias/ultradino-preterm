#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training data read directly from EHR_extract population parquets, e.g. preterm_P1_img_v7.
"""

import os

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataloader.dataloader import PreTermDataset, collate_fn


def load_population(path, image_root):
    """Load an EHR_extract population parquet with absolute image paths and numeric physical deltas."""
    df = pl.read_parquet(path)

    required = ["CPR_BARN", "CPR_MOR", "GA", "FILE_PATH", "NO_OCR_FILE_PATH", "PDX", "PDY",
                "c-section_acute", "induced_acute", "progesterone", "cerclage"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing the columns {missing}")

    n_rows = len(df)
    df = df.filter(pl.col("NO_OCR_FILE_PATH").is_not_null())
    if len(df) < n_rows:
        print(f"{path}: dropped {n_rows - len(df)} of {n_rows} rows without a NO_OCR_FILE_PATH")

    # os.path.join keeps paths that are already absolute
    image_paths = [os.path.join(image_root, file) for file in df["NO_OCR_FILE_PATH"]]

    # One-region images have one delta, written as "0.01" or "[0.01]"
    df = df.with_columns(pl.Series("image_path", image_paths),
                         pl.col("GA").cast(pl.Int64),
                         pl.col("PDX").cast(pl.String).str.strip_chars("[] ").cast(pl.Float64),
                         pl.col("PDY").cast(pl.String).str.strip_chars("[] ").cast(pl.Float64))
    return df


def assign_folds(df, folds, seed):
    """Give each mother one fold, spreading every GA group evenly over the folds."""
    groups = (df.group_by("CPR_MOR").agg(pl.col("GA").min().alias("GA"))
              .sort("CPR_MOR")
              .with_columns(pl.when(pl.col("GA") // 7 < 28).then(0)
                            .when(pl.col("GA") // 7 < 32).then(1)
                            .when(pl.col("GA") // 7 < 34).then(2)
                            .when(pl.col("GA") // 7 < 37).then(3)
                            .otherwise(4)
                            .alias("GA_group")))

    groups = (groups.with_columns(pl.int_range(pl.len()).shuffle(seed=seed).over("GA_group").alias("fold"))
              .with_columns((pl.col("fold") % folds).alias("fold"))
              .select(["CPR_MOR", "fold"]))

    return df.join(groups, on="CPR_MOR", how="left")


class EHRExtractDataset(PreTermDataset):
    """PreTermDataset on an EHR_extract population, with every value resolved in __init__."""

    def __init__(self, df, cfg, train):
        super().__init__(df, cfg, train, ID="CPR_BARN")

        if self.get_segs:
            raise ValueError("EHR_extract populations have no segmentation_path, so segmentation can't be used")

        missing = [var for var in list(self.ehr_vars) + self.aux_vars + list(self.remove) if var not in df.columns]
        if missing:
            raise ValueError(f"The EHR_extract population has no columns {missing}")

        unknown = [var for var in self.img_data_vars if var not in ["PDX", "PDY"]]
        if unknown:
            raise ValueError(f"img_data can only hold PDX and PDY, not {unknown}")

        self.image_paths = df["image_path"].to_list()
        self.ids = df["CPR_BARN"].to_list()
        self.ga = df["GA"].to_numpy()
        self.ehr_data = np.zeros((len(df), len(self.ehr_vars)))
        for i, var in enumerate(self.ehr_vars):
            self.ehr_data[:, i] = df[var].cast(pl.Float64).fill_null(0.0).to_numpy()
        self.aux_labels = {var: df[var].cast(pl.Float64).fill_null(0.0).to_numpy() for var in self.aux_vars}
        self.physical_deltas = {"PDX": df["PDX"].fill_null(0.0).to_numpy(),
                                "PDY": df["PDY"].fill_null(0.0).to_numpy()}
        self.treatment = (df["progesterone"] | df["cerclage"]).to_list()

        # No-MP loss mask: preterm births after a planned C-section or induction are left out
        planned = (df["c-section_acute"] | df["induced_acute"]).to_numpy()
        removed = np.zeros(len(df), dtype=bool)
        for var in self.remove:
            removed |= df[var].cast(pl.Boolean).fill_null(False).to_numpy()

        self.masks = {}
        for cutoff in self.cutoffs:
            preterm = self.ga // 7 < cutoff
            self.masks[str(cutoff)] = ~(preterm & planned) & ~removed

    def __len__(self):
        return len(self.image_paths)

    def getitem(self, idx):
        img_path = self.image_paths[idx]
        img = np.asarray(Image.open(img_path))
        h = img.shape[0]
        w = img.shape[1]
        img = self.transforms(image=img)['image']

        # Test time augmentations
        if not self.train:
            imgs = torch.stack([img,
                                torch.flip(img, dims=[-1]),
                                torch.flip(img, dims=[-2]),
                                torch.flip(img, dims=[-2, -1])], dim=0)
        else:
            imgs = img.unsqueeze(0)

        # Physical deltas of the resized image
        scale = {"PDX": w / self.img_size[0], "PDY": h / self.img_size[1]}
        img_data = [scale[var] * self.physical_deltas[var][idx] for var in self.img_data_vars]
        img_data = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0)

        ehr_data = torch.tensor(self.ehr_data[idx], dtype=torch.float32).unsqueeze(0)

        labels = {'GA_weeks': torch.tensor([float(self.ga[idx] // 7)])}
        for var in self.aux_vars:
            labels[var] = torch.tensor([float(self.aux_labels[var][idx])])

        masks = {cutoff: torch.tensor(bool(mask[idx])) for cutoff, mask in self.masks.items()}

        return {'imgs': imgs,
                'img_path': img_path,
                'img_data': img_data,
                'ehr_data': ehr_data,
                'masks': masks,
                'ID': self.ids[idx],
                'treatment': self.treatment[idx],
                'labels': labels,
                'segmentation': torch.tensor([], dtype=torch.bool)}


class EHRExtractDataSplits:
    """DataSplits on EHR_extract's train and test parquets: CV folds over test, train always in training."""

    def __init__(self, cfg, save_path):
        self.oversample_ratio = cfg.data.oversample_ratio
        self.highest_GA = max(cfg.tasks.preterm.cutoffs)
        self.folds = cfg.data.folds
        self.seed = cfg.data.seed

        if self.folds < 2:
            raise Exception(f"{self.folds} folds set, must be atleast 2")

        if self.folds == 2:
            print('Warning: Only 2 folds selected, one fold will be used for validation and one for testing')

        test_df = load_population(cfg.data.test_path, cfg.data.image_root)
        train_df = load_population(cfg.data.train_path, cfg.data.image_root)

        self.train_df = train_df.with_columns(pl.lit(-1, dtype=pl.Int64).alias("fold"))
        self.test_df = assign_folds(test_df, self.folds, self.seed)

        # Check that every mother is in exactly one fold
        assert (self.test_df.group_by("CPR_MOR").agg(pl.col("fold").n_unique().alias("n_folds"))
                .filter(pl.col("n_folds") != 1).height == 0)

        self.save_distributions(save_path)

    def save_distributions(self, save_path):
        df = self.test_df.unique("CPR_BARN")

        distributions = []

        for cutoff in [32, 34, 37]:
            inclusion = (pl.col("GA") // 7 >= cutoff) | ~(pl.col("c-section_acute") | pl.col("induced_acute"))

            counts = (df.filter(inclusion).group_by("fold").agg([pl.len().alias(f"n_{cutoff}"),
                                                                 (pl.col("GA") // 7 < cutoff).sum().alias(f"GA < {cutoff}")]))

            distributions.append(counts)

        fold_counts = distributions[0]

        for counts in distributions[1:]:
            fold_counts = fold_counts.join(counts, on="fold")

        fold_counts = fold_counts.sort("fold")

        fold_counts.write_csv(save_path + 'GA_distribution.csv')

        for fold in range(self.folds):
            test_df_fold = self.test_df.filter(pl.col("fold") == fold)

            test_df_fold.write_parquet(save_path + f"test_df_fold_{fold}.parquet")

    def get_split(self, fold, cfg):
        if not 0 <= fold < self.folds:
            raise Exception(f"Data only contains {self.folds} folds")

        test_df = self.test_df.filter(pl.col('fold') == fold)
        val_fold = (fold + 1) % self.folds
        val_df = self.test_df.filter(pl.col('fold') == val_fold)

        train_df = self.test_df.filter((pl.col('fold') != fold) & (pl.col('fold') != val_fold))
        train_df = pl.concat([self.train_df, train_df])

        for col in ["CPR_MOR", "CPR_BARN", "FILE_PATH"]:
            overlap_test = (set(train_df[col].drop_nulls().unique()) & set(test_df[col].drop_nulls().unique()))
            overlap_val = (set(train_df[col].drop_nulls().unique()) & set(val_df[col].drop_nulls().unique()))
            if overlap_test:
                raise ValueError(f"{col}: contains {len(overlap_test)} overlaps in test/train. \n Examples: {list(overlap_test)[:10]}")
            if overlap_val:
                raise ValueError(f"{col}: contains {len(overlap_val)} overlaps in val/train. \n Examples: {list(overlap_val)[:10]}")

        if self.oversample_ratio != 0:
            positives = train_df.filter(pl.col("GA") // 7 < self.highest_GA)
            negatives = train_df.filter(pl.col("GA") // 7 >= self.highest_GA)

            if len(negatives) > len(positives):
                if self.oversample_ratio == 'sqrt':
                    n_target = int(np.sqrt(len(negatives)*len(positives)))
                else:
                    n_target = int(len(negatives) * self.oversample_ratio)
                positives = positives.sample(n=n_target, with_replacement=True, seed=self.seed + fold)

            elif len(positives) > len(negatives):
                if self.oversample_ratio == 'sqrt':
                    n_target = int(np.sqrt(len(negatives)*len(positives)))
                else:
                    n_target = int(len(positives) * self.oversample_ratio)
                negatives = negatives.sample(n=n_target, with_replacement=True, seed=self.seed + fold)

            train_df = pl.concat([negatives, positives])

        TrainData = EHRExtractDataset(train_df, cfg, train=True)
        ValData = EHRExtractDataset(val_df, cfg, train=False)
        TestData = EHRExtractDataset(test_df, cfg, train=False)

        batch_size = cfg.data.batch_size - (cfg.data.batch_size % 4)
        if batch_size != cfg.data.batch_size:
            print(f"Batch size not divisible by 4. Changing it from {cfg.data.batch_size} to {batch_size}")
        val_test_batch_size = batch_size//4

        TrainLoader = DataLoader(TrainData,
                                 batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 drop_last=True,
                                 num_workers=cfg.data.workers,
                                 collate_fn=collate_fn)

        ValLoader = DataLoader(ValData,
                               val_test_batch_size,
                               shuffle=False,
                               pin_memory=False,
                               drop_last=False,
                               num_workers=cfg.data.workers,
                               collate_fn=collate_fn)

        TestLoader = DataLoader(TestData,
                                val_test_batch_size,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False,
                                num_workers=cfg.data.workers,
                                collate_fn=collate_fn)

        return TrainLoader, ValLoader, TestLoader
