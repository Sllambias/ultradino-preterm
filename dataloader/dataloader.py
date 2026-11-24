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
from torch.utils.data import DataLoader


FUS13M_MEAN = 0.1842924807
FUS13M_STD = 0.2187705424
        
class PreTermDataset(Dataset):
    def __init__(self, df, cfg, train, ID='CPR_CHILD'):

        super().__init__()
        self.img_size = cfg.data.img_size
        self.ehr_vars = cfg.data.ehr_data
        self.img_data_vars = cfg.data.img_data
        self.norm_mean = 0.1842924807
        self.norm_std = 0.2187705424       
        self.train = train
        self.setup_transforms()
        self.ID_var = ID
        self.df = df
        self.get_segs = 'segmentation' in cfg.tasks.keys() or cfg.data.segmentation
        self.cutoffs = cfg.tasks.preterm.cutoffs
        self.remove = cfg.data.remove

        self.aux_vars = []      
        if "aux_tasks" in cfg.tasks:
            for task in cfg.tasks.aux_tasks:
                self.aux_vars.append(task.var)

    def setup_transforms(self):
        if self.train:
            self.transforms = A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.1),
                                         A.RandomGamma(gamma_limit=(80, 120), p=0.1),
                                         A.GaussNoise(std_range=(0.05, 0.2), p=0.1),
                                         #A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), p=0.5),
                                         A.HorizontalFlip(p=0.5),
                                         A.VerticalFlip(p=0.5),
                                         A.Resize(height=self.img_size[0], width=self.img_size[1]), #move this up
                                         A.ToGray(p=1.0, num_output_channels=1),
                                         A.Normalize(mean=self.norm_mean, std=self.norm_std),
                                         A.ToTensorV2()])
        
        else:
            self.transforms = A.Compose([A.Resize(height=self.img_size[0], width=self.img_size[1]),
                                         A.ToGray(p=1.0, num_output_channels=1),
                                         A.Normalize(mean=self.norm_mean, std=self.norm_std),
                                         A.ToTensorV2()])        
        
    def __getitem__(self, idx):
        return self.getitem(idx)
        
    
    def __len__(self):
        return len(self.df)


    def getitem(self, idx):
        #Get data as named dict
        data = self.df.row(idx, named=True)

        #Prepare EHR data
        ehr_data = []
        for key in self.ehr_vars:
            if isinstance(key, str):                
                if key == 'Treatment':
                    ehr_data.append(float(data.get('progesterone') or data.get('cerclage')))
                else:
                    ehr_data.append(float(data.get(key) or 0.0))
            else:
                values = [float(data[k]) for k in key if data.get(k) is not None]
                ehr_data.append(sum(values) / len(values) if values else 0.0)

        ehr_data = torch.tensor(ehr_data).unsqueeze(0)
                
        #Prepare labels
        labels = {}

        GA_weeks = data.get('GA')//7
        GA_weeks = torch.tensor([float(GA_weeks)])        

        labels['GA_weeks'] = GA_weeks
        
        for var in self.aux_vars:
            if var == 'GA_weeks':
                continue
            else:
                if isinstance(var, str):
                    labels[var] = torch.tensor([float(data.get(var) or 0.0)])
                else:
                    values = [float(data[v]) for v in var if data.get(v) is not None]
                    labels['_'.join(var)] = torch.tensor([sum(values) / len(values) if values else 0.0])
                    
        #Get segmentations
        if self.get_segs:
            segmentation = np.load(data.get('segmentation_path'))['seg_logits']
            segmentation = np.isin(segmentation, self.seg_labels)
            segmentation = torch.from_numpy(segmentation)
        else:
            segmentation = torch.tensor([], dtype=torch.bool)
 
        #Prepare Image    
        img_path = data.get('no_ocr_preprocessed_file_path')
        img = Image.open(img_path)
        img = np.asarray(img)
        h = img.shape[0]
        w = img.shape[1]
        img = self.transforms(image=img)['image']
        
        #Test time agumentations
        if not self.train:
            imgs = torch.stack([img,
                                torch.flip(img, dims=[-1]),
                                torch.flip(img, dims=[-2]),
                                torch.flip(img, dims=[-2, -1])], dim=0)
        else:
            imgs = img.unsqueeze(0)

        #Prepare image metadata
        img_data = []
        for key in self.img_data_vars:
            data_temp = data.get(key) or 0.0
            #Correct physical deltas    
            if key == 'physical_delta_x':
                data_temp = (w/self.img_size[0]) * data_temp[0]
            elif key == 'physical_delta_y':
                data_temp = (h/self.img_size[1]) * data_temp[0]
            
            img_data.append(data_temp)
        
        img_data = torch.tensor(img_data).unsqueeze(0)
        
        #Get patient identifier
        ID = data.get(self.ID_var)
        
        #Get loss masks
        masks = {}

        c_sec = data.get('c-section')
        c_sec_preg = data.get('c-section_during_birth')
        contrac = data.get('contractions_with_preterm_birth')
        induced = data.get('induced')
        pprom = data.get('pprom')
        
        for cutoff in self.cutoffs:
            mask = 0
            preterm = data.get('GA')//7 < cutoff
            
            if not preterm:
                mask = 1
            elif pprom:
                mask = 1
            elif not induced and not c_sec:
                mask = 1
            elif c_sec and (c_sec_preg or contrac) and not induced:
                mask = 1
            
            for var in self.remove:
                if data.get(var):
                    mask = 0
            
            mask = torch.tensor(mask, dtype=torch.bool)
            masks[str(cutoff)] = mask
                            
        #Get treatment status
        progesterone = data.get('progesterone')
        cerclage = data.get('cerclage')
        
        treatment = progesterone or cerclage

        return {'imgs': imgs, 
                'img_path': img_path,
                'img_data': img_data, 
                'ehr_data': ehr_data, 
                'masks': masks,
                'ID': ID, 
                'treatment': treatment,
                'labels': labels,
                'segmentation': segmentation}


def collate_fn(batch):
    #Reform imgs into (B,C,H,W)
    imgs = torch.stack([sample['imgs'] for sample in batch])
    B,R = imgs.shape[:2]
    imgs = imgs.reshape(B*R, *imgs.shape[2:])
    
    #Repeat all data for each img
    img_paths = [f"{sample['img_path']}_{i+1}" for sample in batch for i in range(R)]
    img_data = torch.stack([sample['img_data'] for sample in batch]).repeat_interleave(R, dim=0)
    ehr_data = torch.stack([sample['ehr_data'] for sample in batch]).repeat_interleave(R, dim=0)
    masks = {cutoff: torch.stack([sample['masks'][cutoff] for sample in batch]).repeat_interleave(R, dim=0) for cutoff in batch[0]['masks']}
    IDs = [sample['ID'] for sample in batch for _ in range(R)]
    treatment = [sample['treatment'] for sample in batch for _ in range(R)]    
    labels = {key: torch.stack([sample['labels'][key] for sample in batch]).repeat_interleave(R, dim=0) for key in batch[0]['labels']}
    segmentation = torch.stack([sample['segmentation'] for sample in batch]).repeat_interleave(R, dim=0)
    

    sample =  {"imgs": imgs,
               "img_paths": img_paths,
               "img_data": img_data,
               "ehr_data": ehr_data,
               "masks": masks,
               "IDs": IDs,
               'treatment': treatment,
               'labels': labels,
               'segmentation': segmentation}

    return sample
   
    
class DataSplits:
    def __init__(self, cfg, save_path, unique_column='CPR_MOTHER'):
        self.unique_column=unique_column
        self.oversample_ratio = cfg.data.oversample_ratio
        self.highest_GA = max(cfg.tasks.preterm.cutoffs)
        self.folds=cfg.data.folds
        
        if self.folds < 2:
            raise Exception(f"{self.folds} folds set, must be atleast 2")        

        if self.folds == 2:
            print('Warining: Only 2 folds selected, one fold will be used for validation and one for testing')
        
        test_df = pl.read_parquet(cfg.data.test_path)

        if cfg.data.train_path is not None:
            train_df = pl.read_parquet(cfg.data.train_path)
        else:
            train_df = pl.DataFrame(schema=test_df.schema)

        self.train_df = train_df.with_columns(pl.lit(-1, dtype=pl.Int64).alias("fold"))
        
        # Group GA cutoffs
        groups = (test_df.group_by(self.unique_column).agg(pl.col("GA").min().alias("GA"))
                  .with_columns(pl.when(pl.col("GA")//7 < 28).then(0)
                                .when(pl.col("GA")//7 < 32).then(1)
                                .when(pl.col("GA")//7 < 34).then(2)
                                .when(pl.col("GA")//7 < 37)
                                .otherwise(3)
                                .alias("GA_group")))
        
        
        # Distribute groups evenly within each GA group
        groups = (groups.with_columns(pl.int_range(pl.len())
                                      .shuffle().over("GA_group").alias("fold"))
                  .with_columns((pl.col("fold") % self.folds).alias("fold"))
                  .select([self.unique_column, "fold"]))

        # Assign the group's fold to every row
        self.test_df = test_df.join(groups, on=self.unique_column, how="left")
        
        #Check that self.unique_column is in exactly one fold
        assert (self.test_df.group_by(self.unique_column).agg(pl.col("fold").n_unique().alias("n_folds"))
                .filter(pl.col("n_folds") != 1).height == 0)
        
        self.save_distributions(save_path)
        
    def save_distributions(self, save_path):
        df = self.test_df.unique("CPR_CHILD")
        
        distributions = []
        
        for cutoff in [32,34,37]:
            inclusion = ((pl.col("GA") // 7 >= cutoff)
                         | (~pl.col("induced") & ~pl.col("c-section"))
                         | pl.col("pprom")
                         | (pl.col("c-section")
                            & (pl.col("c-section_during_birth") | pl.col("contractions_with_preterm_birth"))
                            & ~pl.col("induced")))
            
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
                
        for col in ["CPR_MOTHER", "CPR_CHILD", "file_path"]:
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
                positives = positives.sample(n=n_target, with_replacement=True)
                
            elif len(positives) > len(negatives):
                if self.oversample_ratio == 'sqrt':
                    n_target = int(np.sqrt(len(negatives)*len(positives)))
                else:
                    n_target = int(len(positives) * self.oversample_ratio)
                negatives = negatives.sample(n=n_target, with_replacement=True)
                
            train_df = pl.concat([negatives, positives]) 
        
        TrainData = PreTermDataset(train_df, cfg, train=True)
        ValData = PreTermDataset(val_df, cfg, train=False)
        TestData = PreTermDataset(test_df, cfg, train=False)
        
        
        if (cfg.data.batch_size % 4) != 0:
            print(f"Batch size not divisible by 4. Changing it from {cfg.data.batch_size} to {cfg.data.batch_size - (cfg.data.batch_size % 3)}")
        batch_size = cfg.data.batch_size - (cfg.data.batch_size % 4)
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
        