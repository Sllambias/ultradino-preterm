#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:33:56 2026

@author: jacob
"""

from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import albumentations as A
import polars as pl
import polars.selectors

FUS13M_MEAN = 0.1842924807
FUS13M_STD = 0.2187705424


class PreTermDataset(Dataset):
    def __init__(self, df, cfg, train, ID_column="CPR_BARN"):

        super().__init__()
        self.img_size = cfg.data.img_size
        self.norm_mean = 0.1842924807
        self.norm_std = 0.2187705424
        self.train = train
        self.df = df
        self.aux_vars = []
        for task in cfg.tasks.aux_tasks:
            self.aux_vars.append(task.var)

        self.id_df = df.select(pl.col(ID_column))
        self.label_df = df.select(pl.col(cfg.data.label_column))
        self.img_df = df.select(pl.col(cfg.data.img_path_column))
        self.img_metadata_df = df.select(pl.col(cfg.data.img_metadata_columns))
        if len(cfg.data.ehr_categorical_columns) > 0:
            self.ehr_df = pl.concat(
                [
                    df.select(pl.col(cfg.data.ehr_noncategorical_columns)),
                    df.select(polars.selectors.starts_with(cfg.data.ehr_categorical_columns)),
                ],
                how="horizontal",
            )
        else:
            self.ehr_df = df.select(pl.col(cfg.data.ehr_noncategorical_columns))

        self.aux_df = df.select(pl.col([task.var for task in cfg.tasks.aux_tasks]))
        self.setup_transforms()

    def setup_transforms(self):
        if self.train:
            self.transforms = A.Compose(
                [
                    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                    A.GaussNoise(std_range=(0.05, 0.2), p=0.5),
                    A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.Resize(height=self.img_size[0], width=self.img_size[1]),
                    A.ToGray(p=1.0, num_output_channels=1),
                    A.Normalize(mean=self.norm_mean, std=self.norm_std),
                    A.ToTensorV2(),
                ]
            )

        else:
            self.transforms = A.Compose(
                [
                    A.Resize(height=self.img_size[0], width=self.img_size[1]),
                    A.ToGray(p=1.0, num_output_channels=1),
                    A.Normalize(mean=self.norm_mean, std=self.norm_std),
                    A.ToTensorV2(),
                ]
            )

    def __getitem__(self, idx):
        return self.getitem(idx)

    def __len__(self):
        return len(self.df)

    def getitem(self, idx):
        # Get data as named dict
        # data = self.df.row(idx, named=True)

        # Prepare auxilary task vars
        aux_vars = torch.tensor([i or 28 for i in self.aux_df.row(idx)])

        # Prepare Image
        img = Image.open(self.img_df.row(idx)[0])
        img = np.asarray(img)
        h = img.shape[0]
        w = img.shape[1]
        img = self.transforms(image=img)["image"]

        # Prepare image metadata
        img_metadata = self.img_metadata_df.row(idx, named=True)

        if img_metadata.get("PDX", None) is not None:
            img_metadata["PDX"] = (w / self.img_size[0]) * img_metadata["PDX"]
        if img_metadata.get("PDY", None) is not None:
            img_metadata["PDY"] = (h / self.img_size[1]) * img_metadata["PDX"]

        img_metadata = torch.tensor(list(img_metadata.values()), dtype=torch.float32)

        ehr_data = torch.tensor(self.ehr_df.row(idx))

        tabular_data = torch.cat([img_metadata, ehr_data])
        tabular_data = tabular_data.unsqueeze(0)

        labels = torch.tensor(self.label_df.row(idx))
        return {"img": img, "tabular_data": tabular_data, "aux_vars": aux_vars, "labels": labels, "ID": self.id_df.row(idx)[0]}


def collate_fn(batch):
    imgs = torch.stack([sample["img"] for sample in batch])
    tabular_data = torch.stack([sample["tabular_data"] for sample in batch])
    aux_vars = torch.stack([sample["aux_vars"] for sample in batch])
    labels = torch.stack([sample["labels"] for sample in batch])
    ID = [sample["ID"] for sample in batch]

    sample = {"imgs": imgs, "tabular_data": tabular_data, "aux_vars": aux_vars, "labels": labels, "ID": ID}

    return sample


def read_dataframe(path, columns=None):
    if path.endswith(".csv"):
        return pl.read_csv(path, infer_schema=False, columns=columns)
    if path.endswith(".parquet"):
        return pl.read_parquet(path, columns=columns)
    raise ValueError(f"Unsupported data file format: {path}")


def make_data_split(cfg, data_path, unique_column="CPR_MOTHER", training=True):
    df = pl.read_csv(data_path, infer_schema_length=10000000)
    df = df.to_dummies(columns=cfg.data.ehr_categorical_columns)
    df = df.with_columns(pl.all().fill_null(-1))
    df = df.with_columns(pl.col(cfg.data.ehr_noncategorical_columns).cast(pl.Float32))
    if training:
        unique_keys = df.select(unique_column).unique()

        rng = np.random.default_rng(cfg.random_seed)
        keys = unique_keys.to_series().to_list()
        rng.shuffle(keys)

        split_idx = int(len(keys) * (1 - cfg.data.val_frac))
        train_keys = keys[:split_idx]
        val_keys = keys[split_idx:]

        train_df = df.filter(pl.col(unique_column).is_in(train_keys))
        val_df = df.filter(pl.col(unique_column).is_in(val_keys))

        if cfg.data.oversample_ratio != 0:
            df_1 = train_df.filter(pl.col("GA") < cfg.tasks.preterm.cutoff)
            df_0 = train_df.filter(pl.col("GA") >= cfg.tasks.preterm.cutoff)
            n1 = df_1.height
            n0 = df_0.height
            if n1 > n0:
                df_0 = df_0.sample(n=n1 * cfg.data.oversample_ratio, with_replacement=True)
            else:
                df_1 = df_1.sample(n=n0 * cfg.data.oversample_ratio, with_replacement=True)
            train_df = pl.concat([df_1, df_0])
            train_df = train_df.sample(fraction=1.0, shuffle=True)
        if (train_df[unique_column].is_in(val_df[unique_column].implode())).any():
            raise Exception(f"Traindata and Validation data overlap on column {unique_column}")

        return train_df, val_df

    else:
        return df
