#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:41:00 2026

@author: jacob
"""
#%%Imports
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch
import os
import polars as pl
import shutil
from tqdm import tqdm
from filelock import FileLock
from sklearn.metrics import roc_auc_score

from dataloader.dataloader import PreTermDataset, collate_fn
from utils.model_utils import model_from_conf
from utils.metrics import get_metrics
from bias_analysis.bias_analysis import run_analysis

import warnings
warnings.filterwarnings("ignore", message="The image is already gray.")
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")

#%%Main

def make_data_split(cfg, data_path, unique_column='CPR_MOTHER', training=True):
    df = pl.read_parquet(data_path)
    
    for col, cond in cfg.dataset.items():
        if cond == 'remove':
            df = df.filter(~pl.col(col))
    return df

def test_model(folder_path, move=True, batch_size=128):
    cfg = OmegaConf.load(folder_path + 'conf.yaml')
    
    cfg.dataset.progesterone = 'ignore'
    df = make_data_split(cfg, "/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/OnlyFirstPreg_v5/test.parquet", training=False)
    TestData = PreTermDataset(df, cfg, train=False)
    TestLoader = DataLoader(TestData,
                            batch_size,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False,
                            num_workers=cfg.data.workers,
                            collate_fn=collate_fn)
    
    model = model_from_conf(cfg)
    
    dirs = os.listdir(folder_path + 'weights/')
    dirs.sort()

    cutoffs = cfg.tasks.preterm.cutoffs
    
    metrics_df =  pl.read_csv(folder_path + 'metrics.csv')
    best_epoch = {}
    thresholds = {}
    best_preds = {str(c): {'all': None, 'no_prog': None} for c in cutoffs}
    population_all, population_no_prog = TestData.population_count(cutoffs)

    for cutoff in cutoffs:        
        best_epoch[str(cutoff)] = {'all': {'Total Population': population_all[str(cutoff)]['Total Population'],
                                           'Preterm births': population_all[str(cutoff)]['Preterm births'],
                                           'Non-preterm_births': population_all[str(cutoff)]['Non-preterm_births'],
                                           'SensAtSpec': 0.},
                                   'no_prog': {'Total Population': population_no_prog[str(cutoff)]['Total Population'],
                                               'Preterm births': population_no_prog[str(cutoff)]['Preterm births'],
                                               'Non-preterm_births': population_no_prog[str(cutoff)]['Non-preterm_births'],
                                               'SensAtSpec': 0.}}
        
        thresholds[str(cutoff)] = {'avg': metrics_df[f"SensAtSpec_cutoff_{cutoff}_avg"],
                                   'max': metrics_df[f"SensAtSpec_cutoff_{cutoff}_max"]}
        
    for i, weights in enumerate(tqdm(dirs)):
        weight_path = folder_path + 'weights/' + weights
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        model.eval()
        
        with torch.no_grad():
            dfs = {str(c): [] for c in cutoffs}
            for data in TestLoader:
                outputs, _ = model(data['imgs'].to(cfg.device.type),
                                   data['img_data'].to(cfg.device.type),
                                   data['ehr_data'].to(cfg.device.type))
                
                for cutoff in cutoffs:
                    dfs[str(cutoff)].append(pl.DataFrame({'CPR_CHILD': data['IDs'],
                                                          'preds': outputs['preterm'][str(cutoff)]['preds'].flatten().cpu().numpy(),
                                                          'label': (data['GA_weeks'] < float(cutoff)).flatten().cpu().numpy(),
                                                          'prog': data['progesterone'],
                                                          'remove_on_GA': data['remove_on_GA'].flatten().cpu().numpy()}))
                                                
            for cutoff in cutoffs:
                pred_df = pl.concat(dfs[str(cutoff)])    
                patient_df = (pred_df.group_by("CPR_CHILD").agg([pl.col('preds').mean().alias('pred_avg'),
                                                                 pl.col('preds').max().alias('pred_max'),
                                                                 pl.col('label').first().alias('label'),
                                                                 pl.col('prog').first().alias('prog'),
                                                                 pl.col('remove_on_GA').first().alias('remove')]))
                
                populations = {'all': patient_df.filter(~pl.col('remove')),
                               'no_prog': patient_df.filter(~pl.col('prog') & ~pl.col('remove'))}
                
                for population, df in populations.items():
                    if df.height == 0:
                        #Handle no prog patients
                        continue

                    preds = {'avg': torch.tensor(df['pred_avg'].to_numpy(), dtype=torch.float32),
                             'max': torch.tensor(df['pred_max'].to_numpy(), dtype=torch.float32)}
        
                    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.int32)
                
                    for eval_type in ['avg', 'max']: 
                        t = thresholds[str(cutoff)][eval_type].item(i)
                        metrics = get_metrics(cfg, t)
                
                        for metric in metrics.values():
                            metric(preds[eval_type], labels)
                
                        sens_spec, sens_spec_cutoff = metrics['SensAtSpec'].compute()
                                                
                        best = best_epoch[str(cutoff)][population]
                        
                        if sens_spec.item() > best['SensAtSpec']:
                            best['Epoch'] = i
                            best['SensAtSpec'] = sens_spec.item()
                            best['AUC'] = roc_auc_score(df['label']*1., df[f"pred_{eval_type}"])
                            best['Type'] = eval_type
                            best['Sensitivity'] = metrics['Recall'].compute().item()
                            best['Specificity'] = metrics['Specificity'].compute().item()
                            best['SensAtSpec_cutoff'] = sens_spec_cutoff.item()
                            best['Val_Cutoff'] = t
                            best['weights'] = weight_path.replace('Running', 'Evaluated')
                            best_preds[str(cutoff)][population] = df[['CPR_CHILD', f"pred_{eval_type}", 'label']]
                            
    
    os.makedirs(folder_path + 'preds/', exist_ok=True)
    for cutoff in cutoffs:
        for population in ['all', 'no_prog']:
            best_preds[str(cutoff)][population].write_csv(folder_path + f"preds/GA_{cutoff}_{population}.csv")
        
    with open(folder_path + 'test_results.txt', 'w') as f:
        for cutoff in cutoffs:
            f.write(f"\n----------GA {str(cutoff)}----------\n")
            f.write('--All patients--\n')
            for key, value in best_epoch[str(cutoff)]['all'].items():
                if isinstance(value, float):
                    value = round(value, 3)
                f.write(f"\t {key} : {value}\n")
            f.write('\n')
            f.write('--No Progesterone patients--\n')
            for key, value in best_epoch[str(cutoff)]['no_prog'].items():
                if isinstance(value, float):
                    value = round(value, 3)
                f.write(f"\t {key} : {value}\n")
    
    
    sota_path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/SOTA.csv'
    
    lock_file = sota_path + ".lock"
    with FileLock(lock_file):
        if os.path.exists(sota_path):
            df = pl.read_csv(sota_path)
        else:
            df = pl.DataFrame({"SensAtSpec": pl.Series([], dtype=pl.Float64),
                               "population": pl.Series([], dtype=pl.String),
                               "GA": pl.Series([], dtype=pl.Int64),
                               "Weight_path": pl.Series([], dtype=pl.String)})
            
    
        for cutoff in cutoffs:
            for population in ['all', 'no_prog']:
                best = best_epoch[str(cutoff)][population]
                
                cond = (pl.col("GA") == cutoff) & (pl.col("population") == population)
                existing = df.filter(cond)
                current_best = (existing["SensAtSpec"].item() if existing.height == 1 else 0.0)
                
                if best['SensAtSpec'] > current_best:
                    result = pl.DataFrame({"SensAtSpec": [best['SensAtSpec']],
                                           "population": [population],
                                           "GA": [cutoff],
                                           "Weight_path": [best['weights']]})
                
                    df = (pl.concat([df.filter(~cond), result]).sort(["GA", "population"]))

        df.write_csv(sota_path)

    bias_cfg = OmegaConf.load("/projects/users/data/UCPH/DeepFetal/projects/preterm/ultradino-preterm/confs/Bias_analysis.yaml")          
    for cutoff in cutoffs:
        bias_cfg.config.save_path = folder_path + f"bias_analysis_{cutoff}/"
        run_analysis(bias_cfg, folder_path + f"preds/GA_{cutoff}_all.csv", cfg.data.test_path)

    if move:
        shutil.move(folder_path, folder_path.replace('Running', 'Evaluated'))
    
  

    
    
    