#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:59:37 2026

@author: jacob
"""

from utils.model_utils import model_from_conf
from omegaconf import OmegaConf
from pathlib import Path
import polars as pl
import re
import torch
from PIL import Image
import albumentations as A
import numpy as np


def inference(model_path, input_path, cutoff=37, metric='SensAtSpec'):
    model_path = Path(model_path)
    input_path = Path(input_path)
    cfg = OmegaConf.load(model_path / 'conf.yaml')
    
    best_models = get_best_model(model_path / 'metrics', cutoff)
    
    df = pl.read_csv(f"{input_path}/input.csv")
    
    results = {img_path: {} for img_path in df["no_ocr_file_path"].to_list()}
    
    for fold in best_models.keys():
        threshold = best_models[fold]["threshold"]
        weights_path = str(best_models[fold]['epoch']).zfill(3)
        model = model_from_conf(cfg, weights_path=model_path / f"weights/{fold}/{weights_path}.pth")
        model = model.to(cfg.device.type)
        model.eval()
 
        with torch.no_grad():
            for row in df.iter_rows(named=True):
                data = prepare_data(row, cfg, input_path)
                output, _ = model(data['img'].to(cfg.device.type), 
                                  data['img_data'].to(cfg.device.type), 
                                  data['ehr_data'].to(cfg.device.type),
                                  data['segmentation'].to(cfg.device.type))
                
                pred = output['preterm'][str(cutoff)]['preds'].flatten().cpu().numpy()
                
                results[row['no_ocr_file_path']][f"{fold}_pred"] = pred[0]
                results[row['no_ocr_file_path']][f"{fold}_cutoff"] = threshold
                results[row['no_ocr_file_path']][f"{fold}_binary_pred"] = int(pred[0] >= threshold)
                results[row['no_ocr_file_path']][f"{fold}_weights"] = str(model_path / f"weights/{fold}/{weights_path}.pth")
    results_df = pl.DataFrame([{"img_path": img_path, **fold_results} for img_path, fold_results in results.items()])
    results_df.write_csv(input_path / "results.csv")
    
def prepare_data(row, cfg, input_path):
    norm_mean = 0.1842924807
    norm_std = 0.2187705424  
    transforms = A.Compose([A.Resize(height=cfg.data.img_size[0], width=cfg.data.img_size[1]),
                            A.ToGray(p=1.0, num_output_channels=1),
                            A.Normalize(mean=norm_mean, std=norm_std),
                            A.ToTensorV2()])  
    
    img_path = Path(row['no_ocr_file_path'])
    if not img_path.is_absolute():
        img_path = input_path / img_path

    img = Image.open(img_path)
    img = np.asarray(img)
    h = img.shape[0]
    w = img.shape[1]

    img = transforms(image=img)['image']

    ehr_data = []
    for var in cfg.data.ehr_data:
        ehr_data.append(float(row[var]))
    ehr_data = torch.tensor(ehr_data).unsqueeze(0)
    
    img_data = []
    for var in cfg.data.img_data:
        data_temp = row[var]
        if var =='physical_delta_x':
            data_temp = (w/cfg.data.img_size[0])*data_temp
        elif var == 'physical_delta_y':
            data_temp = (h/cfg.data.img_size[1])*data_temp
        img_data.append(float(data_temp))
        
    img_data = torch.tensor(img_data).unsqueeze(0)
    
    segmentation = torch.tensor([], dtype=torch.bool)
    
    data = {'img': img.unsqueeze(0),
            'img_data': img_data.unsqueeze(0),
            'ehr_data': ehr_data.unsqueeze(0),
            'segmentation': segmentation.unsqueeze(0)}
    
    return data
    
    
    


def get_best_model(metrics_folder, cutoff, metric='SensAtSpec'):
    best_models = {}
    files = sorted((Path(metrics_folder)).glob(f"metrics_{cutoff}_*.csv"))
    
    folds = sorted({int(match.group(1)) for file in files if (match := re.search(r"_fold_(\d+)\.csv$", file.name))})
    
    for fold in folds:
        best_metric = 0
        fold_files = [file for file in files if file.stem.endswith(f"_fold_{fold}")]
        #Find best epoch
        for file in fold_files:
            #Read metrics file and get best epoch, cutoff and tta
            agg = file.stem.split(f"metrics_{cutoff}_")[1].split("_fold_")[0]
            df = pl.read_csv(file)
            
            cutoff_cols = [col for col in df.columns if "cutoff" in col]
            df_cutoff = df.select(['epoch'] + cutoff_cols)
            df = df.drop(cutoff_cols)

            columns = [col for col in df.columns if metric in col]
            df = df.select(['epoch'] + columns)
            max_row = df.with_columns(pl.max_horizontal(columns).alias("max_value")).sort("max_value", descending=True).row(0, named=True)

            best_epoch = max_row["epoch"]
            value = max_row["max_value"] 
            best_col = max(columns, key=lambda col: max_row[col])
            tta = False if "no-tta" in best_col else True
            
            cutoff_col = f"SensAtSpec_cutoff_{agg}_{'tta' if tta else 'no-tta'}"


            threshold = (df_cutoff.filter(pl.col("epoch") == best_epoch)
                         .select(cutoff_col).item())

            if value > best_metric:
                best_metric = value
                best_results = {"agg": agg,
                                "epoch": best_epoch,
                                "tta": tta,
                                "threshold": threshold}
        
        best_models[f"fold_{fold}"] = best_results
    
    return best_models

if __name__ == '__main__':
    model_path = '/users/data/UCPH/DeepFetal/projects/preterm/training_runs/Running/baseline_37'
    input_path = '/users/data/UCPH/DeepFetal/projects/preterm/png_test_data/'
    inference(model_path, input_path)