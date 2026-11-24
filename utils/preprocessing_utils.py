#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:28:32 2026

@author: jacob
"""
#%%Imports
import polars as pl
import operator
import sqlite3
from tqdm import tqdm
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import json
from omegaconf import ListConfig
import logging
from pathlib import Path

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
logger = logging.getLogger()

#%%Operator functions

def unique(column, value):
    key = pl.struct(column)
    if value is True:
        return pl.len().over(key) == 1
    elif value is False:
        return pl.len().over(key) != 1


def in_list(column, value):
    return column.is_in(value)


def starts_with(column, value):
    if isinstance(value, (list, ListConfig)):
        return pl.any_horizontal([column.str.starts_with(v) for v in value])
    else:
        return column.str.starts_with(value)


def is_null(column, value):
    return column.is_null()


OPS = {">": operator.gt,
       "<": operator.lt,
       ">=": operator.ge,
       "<=": operator.le,
       "==": operator.eq,
       "!=": operator.ne,
       '-': operator.sub,
       '+': operator.add,
       "unique": unique,
       "in": in_list,
       "starts_with": starts_with,
       "is_null": is_null}

type_map = {"str": pl.Utf8,
            "float": pl.Float64,
            "int": pl.Int64,
            "date": pl.Utf8,
            "bool": pl.Boolean,
            "list": pl.List(pl.Float64)}

#%%Utility functions for inclusion / exclusion

def load_table(path, ignore_errors=False, has_header=True):
    if path.endswith(".csv"):
        return pl.read_csv(path, ignore_errors=ignore_errors, has_header=has_header, infer_schema=False)

    else:
        raise NotImplementedError(f"Unknown file type for path: {path}")


def filter_conditions(df, condition, filter_on, table, action, external=True):
    if condition.operator in [">", "<", ">=", "<=", "-", "+"]:
        df_temp = df.with_columns(OPS[condition.operator](pl.col(condition.column).cast(pl.Float64, strict=False), 
                                                               condition.value).alias("_matching"))
    else:
        df_temp = df.with_columns(OPS[condition.operator](pl.col(condition.column), condition.value).alias("_matching"))
    if external:
        match_on = [condition.match_on] if isinstance(condition.match_on, str) else condition.match_on
        df_temp = df_temp.with_columns(pl.col(src).alias(dst) for src, dst in zip(match_on, filter_on))
        if action in ['exclude_birth', 'include_birth']:
            filter_on = filter_on + ["date_of_occurence"]
            df_temp = df_temp.with_columns(pl.col(condition.date_column)
                                           .str.slice(0,10).str.strptime(pl.Date, strict=False).alias('date_of_occurence'))
            
    if condition.condition is None:
        table = df_temp.select(filter_on + ['_matching'])
    
    elif condition.condition == "or":
        table = pl.concat([table, df_temp.select(filter_on + ['_matching'])])
        table = (table.group_by(filter_on).agg(pl.col("_matching").any().alias("_matching")))
        
    elif condition.condition == "and":
        table = (table.join(df_temp.select(filter_on + ['_matching']), on=filter_on, how="inner", suffix="_new")
                 .with_columns((pl.col("_matching") & pl.col("_matching_new")).alias("_matching"))
                 .drop("_matching_new"))
    return table

    
def filter_df(df, criteria):
    logger.info(criteria.name)
    children = df['CPR_CHILD'].n_unique()
    df = df.with_columns(pl.lit(None, dtype=pl.Boolean).alias("remove"))
    for action in criteria.actions: 
        table = None
        filter_on = [action.filter_on] if isinstance(action.filter_on, str) else action.filter_on
        for condition in action.conditions:
            if 'table' in condition:
                df_temp = load_table(condition.table)
                table = filter_conditions(df_temp, condition, filter_on, table, action.action)
            else:
                table = filter_conditions(df, condition, filter_on, table, action.action, external=False)
    
        if action.action == 'include':
            matches = (table.group_by(filter_on).agg(pl.col("_matching").any()
                                                            .alias("_matching")).filter(pl.col("_matching")))
            
            matches = matches.filter(pl.col("_matching")).select(filter_on + ['_matching'])
            
            df = df.join(matches, on=filter_on, how='left')
            df = df.with_columns(pl.when(pl.col('remove') == True).then(True)
                                 .when(pl.col("_matching") == True).then(False)
                                 .otherwise(pl.col("remove")).alias("remove")).drop("_matching")
                   
        elif action.action == 'exclude':
            matches = (table.group_by(filter_on).agg(pl.col("_matching").any()
                                                            .alias("_matching")).filter(pl.col("_matching")))
            
            matches = matches.filter(pl.col("_matching")).select(filter_on + ['_matching'])
            
            df = df.join(matches, on=filter_on, how='left')
            df = df.with_columns(pl.when(pl.col("_matching") == True).then(True)
                                 .otherwise(pl.col("remove")).alias("remove")).drop("_matching")
        
        elif action.action == 'include_birth':
            matches = df.join(table, on=filter_on, how="left")
            matches = matches.filter(pl.col("date_of_occurence").is_not_null())
            matches = matches.filter((pl.col("date_of_occurence") <= pl.col("BIRTHDAY") + pl.duration(days=7)) &
                                         (pl.col("date_of_occurence") >= pl.col("BIRTHDAY") - pl.duration(days=280)))

            if getattr(action, "strict", False):
                matches = matches.group_by(filter_on + ['BIRTHDAY']).agg([pl.col("_matching").drop_nulls().all().alias("_matching"),
                                                                          pl.col("_matching").drop_nulls().len().alias("_n_valid")
                                                                          ]).filter((pl.col("_n_valid") > 0) & pl.col("_matching"))
            else:
                matches = matches.group_by(filter_on + ['BIRTHDAY']).agg(pl.col("_matching").any())

            matches = matches.filter(pl.col("_matching")).select(filter_on + ['BIRTHDAY', '_matching'])
          
            df = df.join(matches, on=filter_on + ["BIRTHDAY"], how='left')
            df = df.with_columns(pl.when(pl.col('remove') == True).then(True)
                                 .when(pl.col("_matching") == True).then(False)
                                 .otherwise(pl.col("remove")).alias("remove")).drop("_matching")
    
        elif action.action == 'exclude_birth':
            matches = df.join(table, on=filter_on, how="left")
            matches = matches.filter((pl.col("date_of_occurence") <= pl.col("BIRTHDAY") + pl.duration(days=7)) &
                                         (pl.col("date_of_occurence") >= pl.col("BIRTHDAY") - pl.duration(days=280)))
            
            if getattr(action, "strict", False):
                matches = matches.group_by(filter_on + ['BIRTHDAY']).agg(pl.col("_matching").all())
            else:
                matches = matches.group_by(filter_on + ['BIRTHDAY']).agg(pl.col("_matching").any())
            
            matches = matches.filter(pl.col("_matching")).select(filter_on + ['BIRTHDAY', '_matching'])
            
            df = df.join(matches, on=filter_on + ["BIRTHDAY"], how='left')
            df = df.with_columns(pl.when(pl.col("_matching") == True).then(True)
                                 .otherwise(pl.col("remove")).alias("remove")).drop("_matching")
            

    if criteria.default == 'keep':
        df = df.with_columns(pl.col("remove").fill_null(False).alias("remove"))
    elif criteria.default == 'remove':
        df = df.with_columns(pl.col("remove").fill_null(True).alias("remove"))
    else:
        raise Exception(f"Default behaviour {criteria.default} not implemented")
        
    final_df = df.filter(~pl.col("remove")).drop("remove")
    logger.info(f"Removed: {children - final_df['CPR_CHILD'].n_unique()}")
    logger.info(f"Children left: {final_df['CPR_CHILD'].n_unique()}\n")
    return final_df


def mark_df(df, criteria):
    df = df.with_columns(pl.lit(None, dtype=pl.Boolean).alias(criteria.mark_name))
    for action in criteria.actions:
        table = None
        filter_on = [action.filter_on] if isinstance(action.filter_on, str) else action.filter_on
        for condition in action.conditions:
            if 'table' in condition:
                df_temp = load_table(condition.table)
                table = filter_conditions(df_temp, condition, filter_on, table, action.action)
            else:
                table = filter_conditions(df, condition, filter_on, table, action.action, external=False)

        if action.action == 'include':
            matches = (table.group_by(filter_on).agg(pl.col("_matching").any()
                                                            .alias("_matching")).filter(pl.col("_matching")))
            
            matches = matches.filter(pl.col("_matching")).select(filter_on + ['_matching'])
            
            df = df.join(matches, on=filter_on, how='left')
            df = df.with_columns(pl.when(pl.col("_matching") == True).then(True)
                                 .otherwise(pl.col(criteria.mark_name)).alias(criteria.mark_name)).drop("_matching")

        elif action.action == 'exclude':
            matches = (table.group_by(filter_on).agg(pl.col("_matching").any()
                                                            .alias("_matching")).filter(pl.col("_matching")))
            
            matches = matches.filter(pl.col("_matching")).select(filter_on + ['_matching'])
            
            df = df.join(matches, on=filter_on, how='left')
            df = df.with_columns(pl.when(pl.col(criteria.mark_name) == True).then(True)
                                 .when(pl.col("_matching") == True).then(False)
                                 .otherwise(pl.col(criteria.mark_name)).alias(criteria.mark_name)).drop("_matching")

        elif action.action == 'include_birth':
            matches = df.join(table, on=filter_on, how="left")
            matches = matches.filter(pl.col("date_of_occurence").is_not_null())
            matches = matches.filter((pl.col("date_of_occurence") <= pl.col("BIRTHDAY") + pl.duration(days=7)) &
                                         (pl.col("date_of_occurence") >= pl.col("BIRTHDAY") - pl.duration(days=280)))
            if getattr(action, "strict", False):
                matches = matches.group_by(filter_on + ['BIRTHDAY']).agg(pl.col("_matching").all())
            else:
                matches = matches.group_by(filter_on + ['BIRTHDAY']).agg(pl.col("_matching").any())

            matches = matches.filter(pl.col("_matching")).select(filter_on + ['BIRTHDAY', '_matching'])
          
            df = df.join(matches, on=filter_on + ["BIRTHDAY"], how='left')
            df = df.with_columns(pl.when(pl.col("_matching") == True).then(True)
                                 .otherwise(pl.col(criteria.mark_name)).alias(criteria.mark_name)).drop("_matching")

        elif action.action == 'exclude_birth':
            matches = df.join(table, on=filter_on, how="left")
            matches = matches.filter((pl.col("date_of_occurence") <= pl.col("BIRTHDAY") + pl.duration(days=7)) &
                                         (pl.col("date_of_occurence") >= pl.col("BIRTHDAY") - pl.duration(days=280)))
            
            if getattr(action, "strict", False):
                matches = matches.group_by(filter_on + ['BIRTHDAY']).agg(pl.col("_matching").all())
            else:
                matches = matches.group_by(filter_on + ['BIRTHDAY']).agg(pl.col("_matching").any())
            
            matches = matches.filter(pl.col("_matching")).select(filter_on + ['BIRTHDAY', '_matching'])
            
            df = df.join(matches, on=filter_on + ["BIRTHDAY"], how='left')
            df = df.with_columns(pl.when(pl.col(criteria.mark_name) == True).then(True)
                                 .when(pl.col("_matching") == True).then(False)
                                 .otherwise(pl.col(criteria.mark_name)).alias(criteria.mark_name)).drop("_matching")
        
        if criteria.default == 'no_mark':
            df = df.with_columns(pl.col(criteria.mark_name).fill_null(False).alias(criteria.mark_name))
        elif criteria.default == 'mark':
            df = df.with_columns(pl.col(criteria.mark_name).fill_null(True).alias(criteria.mark_name))
        else:
            raise Exception(f"Default behaviour {criteria.default} not implemented")
        
        logger.info(f"{criteria.name} marked {df.unique('CPR_CHILD')[criteria.mark_name].sum()} children")
        
        return df    


def find_close_births(df, criteria):
    logger.info(criteria.name)
    children = df['CPR_CHILD'].n_unique()
    for action in criteria.actions:
        births = df.select(["CPR_MOTHER", "CPR_CHILD", action.column]).unique()
        
        pairs = (births.join(births, on="CPR_MOTHER", how="inner", suffix="_2")
                .filter(pl.col("CPR_CHILD") != pl.col("CPR_CHILD_2")))
                 
        pairs = pairs.with_columns((pl.col(action.column) - pl.col(f"{action.column}_2"))
                                   .dt.total_days().abs().alias("birth_gap"))
             
        close_births = (pairs.filter((pl.col('birth_gap') >= action.min_threshold)
                                     & (pl.col('birth_gap') <= action.max_threshold))
                        .select(['CPR_MOTHER', 'CPR_CHILD']).unique())
    
        if action.action == 'include':
            df = df.join(close_births, on=action.filter_on, how="semi")
        if action.action == 'exclude':
            df = df.join(close_births, on=action.filter_on, how="anti")

    logger.info(f"Removed: {children - df['CPR_CHILD'].n_unique()}")
    logger.info(f"Children left: {df['CPR_CHILD'].n_unique()}\n")

    return df

def remove_missed_para(df, criteria):
    logger.info(criteria.name)
    children = df['CPR_CHILD'].n_unique()
    for action in criteria.actions:
        earliest_birth = (df.select(action.filter_on).unique().sort("BIRTHDAY")
                          .group_by("CPR_MOTHER", maintain_order=True).first())

        if action.action == 'include':
            df = df.join(earliest_birth, on=action.filter_on, how="semi")
        if action.action == 'exclude':
            df = df.join(earliest_birth, on=action.filter_on, how="anti")
    
    logger.info(f"Removed: {children - df['CPR_CHILD'].n_unique()}")
    logger.info(f"Children left: {df['CPR_CHILD'].n_unique()}\n")
    
    return df

def discard(discards, df, criteria, mothers, children):
    if criteria.name in discards.keys():
        mothers_temp = df['CPR_MOTHER'].unique()
        children_temp = df['CPR_CHILD'].unique()
        mothers_discarded = discards[criteria.name]['mothers_discarded']
        children_discarded = discards[criteria.name]['children_discarded']
        mothers_cpr = discards[criteria.name]['mothers_cpr']
        children_cpr = discards[criteria.name]['children_cpr']
        
        discards[criteria.name] = {'mothers_discarded': mothers_discarded + len(mothers)-len(mothers_temp),
                                   'children_discarded': children_discarded + len(children)-len(children_temp),
                                   'mothers_cpr': [mothers.filter(~mothers.is_in(mothers_temp)).to_list(), mothers_cpr],
                                   'children_cpr': [children.filter(~children.is_in(children_temp)).to_list(), children_cpr]}
        
        
    else:
        mothers_temp = df['CPR_MOTHER'].unique()
        children_temp = df['CPR_CHILD'].unique()
        discards[criteria.name] = {'mothers_discarded': len(mothers)-len(mothers_temp),
                                   'children_discarded': len(children)-len(children_temp),
                                   'mothers_cpr': mothers.filter(~mothers.is_in(mothers_temp)).to_list(),
                                   'children_cpr': children.filter(~children.is_in(children_temp)).to_list()}
    
    return discards, mothers_temp, children_temp


def condition(conditioned, df, criteria):    
    n_mothers = df.filter(pl.col(criteria.mark_name)).get_column("CPR_MOTHER").n_unique()
    n_children = df.filter(pl.col(criteria.mark_name)).get_column("CPR_CHILD").n_unique()
    cpr_mothers = df.filter(pl.col(criteria.mark_name))["CPR_MOTHER"].unique().to_list()
    cpr_children = df.filter(pl.col(criteria.mark_name))["CPR_CHILD"].unique().to_list()
        
        
    conditioned[criteria.name] = {'mothers_conditioned': n_mothers,
                                  'children_conditioned': n_children,
                                  'mothers_cpr': cpr_mothers,
                                  'childrens_cpr': cpr_children}
    
    return conditioned

#%%High level inclusion / exclusion functions

custom_funcs = {'filter_df': filter_df,
                'mark_df': mark_df,
                'find_close_births': find_close_births,
                'remove_missed_para': remove_missed_para}

def link_tables(cfg):
    for merge in cfg.merge_tables.merges:
        table = pl.read_csv(merge.table, infer_schema=False)
        merge_table = pl.read_csv(merge.merge_table, infer_schema=False)
        table = table.join(merge_table.select(merge.include + [merge.merge_table_link]), 
                           left_on=merge.table_link, right_on=merge.merge_table_link)
        
        table.write_csv(cfg.paths.data_dir + 'tables/' + merge.table.split('/')[-1])
    

def merge_population_tables(cfg, ignore_errors=False):
    df = pl.DataFrame()
    for cfg_table in cfg.population.tables:
        table = load_table(cfg_table.table, ignore_errors=ignore_errors)
        table = table.select(list(cfg_table.columns.values()))
        table = table.rename({v: k for k, v in cfg_table.columns.items()})
        table = table.select(sorted(table.columns))
        
        if df.height > 0:
            table = table.join(df.select(["CPR_MOTHER", "CPR_CHILD"]),
                               on=["CPR_MOTHER", "CPR_CHILD"],
                               how="anti")
            
        df = pl.concat([df, table])
        
    for name, t in cfg.population.types.items():
        if t == 'date':
            df = df.with_columns(pl.col(name).str.slice(0,10).str.strptime(pl.Date, strict=False))
        else:
            df = df.with_columns(pl.col(name).cast(type_map[t], strict=False))

    return df


def merge_population_and_image_df(df_img, df_pop, cfg):
    df = df_img.join(df_pop, on=cfg.merge.population_key, how='left')
    for config in cfg.merge.create_variables:
        if config.var_type == "days":
            df = df.with_columns(OPS[config.operator](pl.col(config.column_1),
                                                      pl.col(config.column_2)).dt.total_days().alias(config.var_name))
        else:
            df = df.with_columns(OPS[config.operator](pl.col(config.column_1),
                                                      pl.col(config.column_2)).cast(type_map[config.var_type]).alias(config.var_name))

    return df


def make_train_test_split(df, cfg, split):
    if split == 'test':
        df_holdout = pl.read_csv(cfg.paths.holdout_csv)
        df = df.join(df_holdout, left_on="CPR_MOTHER", right_on="CPR_MOR", how="semi")
    elif split == 'train':
        df_holdout = pl.read_parquet(cfg.paths.data_dir + 'test.parquet')
        df = df.join(df_holdout, on="CPR_MOTHER", how="anti")
    else:
        raise Exception(f"Split {split} not understood.")
    return df


def apply_inclusion_exclusion(df, cfg):
    discards = {}
    conditioned = {}
    mothers = df['CPR_MOTHER'].unique()
    children = df['CPR_CHILD'].unique()
    if cfg.image_criteria is not None:
        for criteria in cfg.image_criteria:
            fn = custom_funcs[criteria.function]
            df = fn(df, criteria)
            discards, mothers, children = discard(discards, df, criteria, mothers, children)
    
    if cfg.population_criteria is not None:
        for criteria in cfg.population_criteria:       
            fn = custom_funcs[criteria.function]
            df = fn(df, criteria)
            discards, mothers, children = discard(discards, df, criteria, mothers, children)

    if cfg.conditional_criteria is not None:
        for criteria in cfg.conditional_criteria:
            fn = custom_funcs[criteria.function]
            df = fn(df, criteria)
            conditioned = condition(conditioned, df, criteria)
    
    return df, discards, conditioned

def merge_EHR(df, cfg):
    bias_df = pl.read_csv(cfg.paths.bias_variables, ignore_errors=True)
    df = df.join(bias_df, how='left', left_on='CPR_CHILD', right_on='b_cpr')
    return df
   
#%%Cervical length functions

def get_CL(df, cfg):
    #Get CL from segmentation
    required_cols = ['physical_delta_x', 'physical_delta_y', 
                     'region_location_min_x0', 'region_location_max_x1',
                     'region_location_min_y0', 'region_location_max_y1']

    missing = set(required_cols) - set(df.columns)

    assert not missing, f"Missing columns: {missing}"
        
    with ProcessPoolExecutor(max_workers=64) as executor:
        results = list(tqdm(executor.map(calculate_CL,
                                         df.iter_rows(named=True),
                                         chunksize=1000),
                            total=df.height,
                            desc="Calculating CL",
                            unit="img"))
    
    df = df.with_columns(pl.Series("CL", results))
    
    # Get CL from manual readings
    cl_df = pl.read_parquet(cfg.paths.manual_cl)
    cl_df = (cl_df.group_by(["CPR_MOTHER", "study_date"]).agg(pl.col("cervix_length").mean().alias("CL_manual"))
             .select(["CPR_MOTHER", "study_date", "CL_manual"]))

    df = df.join(cl_df, how="left", on=["CPR_MOTHER", "study_date"])
    
    return df

def calculate_CL(row, cervix_label=3):

    img_path = row['no_ocr_preprocessed_file_path']    
    seg_path = row['segmentation_path']    
    #TEMP UNTIL SQL DB IS UPDATED!
    seg_path = seg_path.replace('segmentations', 'derived_data/PN_COMPLETE_CERVIX')
    x0 = row['region_location_min_x0'][0]
    x1 = row['region_location_max_x1'][0]
    y0 = row['region_location_min_y0'][0] 
    y1 = row['region_location_max_y1'][0]
    delta_x = row['physical_delta_x'][0]
    delta_y = row['physical_delta_y'][0]
    
    if any(v is None for v in [x0, x1, y0, y1, delta_x, delta_y, img_path, seg_path]):
        return 0.0
        
    img = Image.open(img_path)
    seg = np.load(seg_path)['seg_logits']
    
    img_x, img_y = img.size
    seg_x, seg_y = seg.shape
    
    x_crop = x0 + (img_x - x1)
        
    ratio_x = (img_x - x_crop) / seg_x
    new_phys_delta_x = delta_x*ratio_x*10
    
    y_crop = y0 + (img_y - y1) 
    ratio_y = (img_y - y_crop) / seg_y
    new_phys_delta_y = delta_y*ratio_y*10
    
    xs, ys = np.where(seg == cervix_label)

    if len(xs) < 2:
        return 0.0
    
    if xs.max() - xs.min() <= ys.max() - ys.min():
        return 0.0

    coords = np.column_stack([xs * new_phys_delta_x,
                              ys * new_phys_delta_y])


    diff = coords[:, None, :] - coords[None, :, :]
    CL = np.sqrt((diff ** 2).sum(axis=2)).max()
            
    return CL

#%%SQL functions
  
def sqlite_extractor(cfg, cpr_mothers, chunk_size=100000):
    
    db_path = Path(cfg.paths.SQL_DB)

    if not db_path.is_file():
        raise FileNotFoundError(f"SQL database not found: {db_path}")
    
    conn = sqlite3.connect(cfg.paths.SQL_DB)
    cur = conn.cursor()
    
    cur.execute("CREATE TEMP TABLE tmp_hashes (phair_hash TEXT PRIMARY KEY)")
    cur.executemany("INSERT INTO tmp_hashes VALUES (?)", [(h,) for h in cpr_mothers])
    conn.commit()
    
    metadata_dicom_variables = cfg.imaging.metadata_dicom_variables

    dicom_select = ",\n".join(f"d.{column}" for column, _ in metadata_dicom_variables)

    schema = [("CPR_MOTHER", pl.Utf8),
              ("file_path", pl.Utf8),
              ("no_ocr_preprocessed_file_path", pl.Utf8),
              ("segmentation_path", pl.Utf8),
              *[(column, type_map[dtype]) for column, dtype in metadata_dicom_variables],
              ("is_flow", pl.Boolean)]
    
    list_columns = {column for column, dtype in metadata_dicom_variables if dtype == "list"}
    date_cols = [col for col, dtype in metadata_dicom_variables if dtype == "date"]
    
    cur.execute("""
                SELECT COUNT(*)
                FROM tmp_hashes t
                LEFT JOIN cpr_hashes c
                    ON c.phair_hash = t.phair_hash
                LEFT JOIN path_table pt
                    ON pt.file_hash = c.xxhash
                LEFT JOIN dicom_metadata_table d
                    ON d.sop_instance_uid = pt.sop_instance_uid
                """)
    
    total_rows = cur.fetchone()[0]

    
    cur.execute(f"""
                SELECT
                    t.phair_hash,
                    pt.file_path,
                    pt.no_ocr_preprocessed_file_path,
                    pt.segmentation_path,
                    {dicom_select}
                FROM tmp_hashes t 
                LEFT JOIN cpr_hashes c
                    ON c.phair_hash = t.phair_hash
                LEFT JOIN path_table pt
                    ON pt.file_hash = c.xxhash
                LEFT JOIN dicom_metadata_table d
                    ON d.sop_instance_uid = pt.sop_instance_uid
                """)
    
    chunks = []
    
    with tqdm(total=total_rows, desc="Extracting Images", unit="rows") as pbar:
        while True:
            rows = cur.fetchmany(chunk_size)
            
            if not rows:
                break
            
            processed = []
            
            for row in rows:
                row = list(row)
                
                is_flow = any(isinstance(s, str) and "[" in s for s in row[4:])
                
                for i, (column, dtypes) in enumerate(metadata_dicom_variables, start=4):
                    if column in list_columns:
                        row[i] = to_list(row[i])
                        
                processed.append((*row, is_flow))
                
            chunk = pl.DataFrame(processed, schema=schema,
                                 orient="row", strict=False)
            
            chunks.append(chunk)
            pbar.update(len(rows))
            
        conn.close()
        
    df = pl.concat(chunks, rechunk=False)
    df = df.drop_nulls(subset="file_path")

    df = df.with_columns([pl.col(col).str.strptime(pl.Date, format="%Y%m%d", strict=False) for col in date_cols])
                
    return df

def to_list(x):
    if x is None:
        return None

    if isinstance(x, str):
        if x.startswith("["):
            return [float(v) for v in json.loads(x)]
        return [float(x)]

    return [float(x)]