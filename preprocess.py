#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:29:55 2026

@author: jj@di.ku.dk
"""
#%%Imports
import logging
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
import polars as pl
import json

from utils.calc_stats import calc_stats
from utils.preprocessing_utils import (merge_population_tables, 
                                       merge_population_and_image_df, 
                                       apply_inclusion_exclusion, 
                                       link_tables,
                                       make_train_test_split,
                                       sqlite_extractor,
                                       get_CL,
                                       merge_EHR)

#%%Load variable YAML and setup logger and dirs
cfg = OmegaConf.load('/projects/users/data/UCPH/DeepFetal/projects/preterm/ultradino-preterm/confs/Population.yaml')
incl_excl_cfgs = {'train': OmegaConf.load(cfg.paths.train_cfg),
                  'test': OmegaConf.load(cfg.paths.test_cfg)}

cfg.paths.data_dir += cfg.version + '/'

#Setup dirs
Path(cfg.paths.data_dir).mkdir(exist_ok=False)
Path(cfg.paths.data_dir + 'data_dump/').mkdir(exist_ok=True)
Path(cfg.paths.data_dir + 'logs/').mkdir(exist_ok=True)
Path(cfg.paths.data_dir + 'tables/').mkdir(exist_ok=True)

OmegaConf.save(cfg, cfg.paths.data_dir + 'logs/preprocessing.yaml')
OmegaConf.save(incl_excl_cfgs['train'], cfg.paths.data_dir + 'logs/train_incl_excl.yaml')
OmegaConf.save(incl_excl_cfgs['test'], cfg.paths.data_dir + 'logs/test_incl_excl.yaml')

#Setup logger
logging.basicConfig(filename=cfg.paths.data_dir + 'logs/preprocess.log', filemode='w', level=logging.INFO)
logger = logging.getLogger(__name__)

#%%Build population data
link_tables(cfg)

df_pop = merge_population_tables(cfg)

df_pop.write_csv(cfg.paths.data_dir + 'data_dump/population.csv')

logger.info(f"Found {df_pop['CPR_MOTHER'].n_unique()} mothers - " + str(datetime.now().strftime('%H:%M:%S')))
    

#%%Extract info from database

df_img = sqlite_extractor(cfg, list(df_pop['CPR_MOTHER'].unique()))

#Link cervix preds and image df
df_cervix_preds = pl.read_csv(cfg.paths.cervix_preds, infer_schema=False)
df_img = df_img.join(df_cervix_preds, on='file_path', how='left')
del df_cervix_preds

df_img.write_parquet(cfg.paths.data_dir + 'data_dump/img_data.parquet')

logger.info(f"Found {len(df_img)} images - " + str(datetime.now().strftime('%H:%M:%S')))
logger.info(f"Found images for {df_img['CPR_MOTHER'].n_unique()} mothers - " + str(datetime.now().strftime('%H:%M:%S')))

#%%Merge image and population dfs
df = merge_population_and_image_df(df_img, df_pop, cfg)
df.write_parquet(cfg.paths.data_dir + "data_dump/total_data.parquet")

#%%Apply inclusion/exclusion criteria for train and test set
for incl_excl in['test', 'train']:
    df_temp = df.clone()
    cfg_incl_excl = incl_excl_cfgs[incl_excl]
    cfg_incl_excl.paths = cfg.paths

    df_temp, discards, conditioned = apply_inclusion_exclusion(df_temp, cfg_incl_excl)
    with open(cfg.paths.data_dir + f"logs/{incl_excl}_discards.json", "w") as file:
        json.dump(discards, file)
    
    with open(cfg.paths.data_dir + f"logs/{incl_excl}_conditioned.json", "w") as file:
        json.dump(conditioned, file)
    
    #Get cervix length for remaining images
    df_temp = get_CL(df_temp, cfg)

    df_temp = merge_EHR(df_temp, cfg)

    #Make train/test split and save the data
    df_temp = make_train_test_split(df_temp, cfg, split=incl_excl)
    df_temp.write_parquet(cfg.paths.data_dir + f"{incl_excl}.parquet")

    logger.info("\n")
    logger.info(f"{incl_excl} data contains {len(df_temp)} images - " + str(datetime.now().strftime('%H:%M:%S')))
    logger.info(f"{incl_excl} data contains {df_temp['CPR_MOTHER'].n_unique()} mothers - " + str(datetime.now().strftime('%H:%M:%S')))
    logger.info(f"{incl_excl} data contains {df_temp['CPR_CHILD'].n_unique()} children - " + str(datetime.now().strftime('%H:%M:%S')))

    
df_train = pl.read_parquet(cfg.paths.data_dir + 'train.parquet')
df_test = pl.read_parquet(cfg.paths.data_dir + 'test.parquet')

cols_to_check=['CPR_MOTHER', 'CPR_CHILD', 'no_ocr_preprocessed_file_path']

for col in cols_to_check:
    overlap = (df_train.select(col).unique().join(df_test.select(col).unique(),
                                                  on=col, how="inner")
               .get_column(col).to_list())  
    
    if len(overlap) > 0:
        logger.warning("WARNING - Overlap found in test and train split for %s.\n"
                       "Removing duplicates from test split", col)
        logger.info("Overlap: %s", overlap)

        df_test = df_test.filter(~pl.col(col).is_in(overlap))
df_test.write_parquet(cfg.paths.data_dir + 'test.parquet')

    
#%% Calculate stats

"""
logger.info("Calculating stats - " + str(datetime.now().strftime('%H:%M:%S')))
calc_stats('/'.join(path.split('/')[:-2]) + '/')

logger.info("Preprocessing done - " + str(datetime.now().strftime('%H:%M:%S')))
"""