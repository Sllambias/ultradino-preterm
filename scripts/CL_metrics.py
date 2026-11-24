#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:59:31 2026

@author: jacob
"""

import polars as pl
from sklearn.metrics import confusion_matrix
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import roc_curve
import numpy as np
from sklearn.metrics import roc_auc_score

path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/Data/'

df1 = pl.read_parquet(path + 'OnlyFirstPreg_v5/test.parquet')
df2 = pl.read_parquet(path + 'OnlyFirstPreg_v5/train.parquet')

df_full = pl.concat([df1, df2])

cl = pl.read_parquet(path + 'misc/CL_2009-2025.parquet')

cl_avg = (df_full.filter(pl.col("CL") != 0).group_by("CPR_MOTHER").agg(pl.col("CL").mean().alias("CL_avg")))

comparison = (cl_avg.join(cl.select(["CPR_MOTHER", "cervix_length"]), on="CPR_MOTHER",how="inner")
              .filter(pl.col("cervix_length").is_not_null() & (pl.col("cervix_length") != 0)))


comparison = comparison.with_columns((pl.col("CL_avg") - pl.col("cervix_length")).alias("difference"),
                                     (pl.col("CL_avg") - pl.col("cervix_length")).abs().alias("abs_difference"))

comparison = comparison.with_columns(((pl.col("CL_avg") - pl.col("cervix_length"))/ pl.col("cervix_length")* 100)
                                     .alias("percent_difference"))

print("Average % difference:", comparison["percent_difference"].mean())

print("Mean difference:", comparison["difference"].mean())
print("Median difference:", comparison["difference"].median())
print("Mean absolute difference:", comparison["abs_difference"].mean())
print("SD difference:", comparison["difference"].std())

comparison = comparison.with_columns(((pl.col("CL_avg") + pl.col("cervix_length")) / 2).alias("mean_measurement"),)

mean_diff = comparison["difference"].mean()
sd_diff = comparison["difference"].std()

loa_upper = mean_diff + 1.96 * sd_diff
loa_lower = mean_diff - 1.96 * sd_diff

print("Bias:", mean_diff)
print("95% limits of agreement:", loa_lower, "to", loa_upper)


for prog in [True, False]:
    print(f"-----Progesterone included: {prog}-----")
    if not prog:
        df_full = df_full.filter(~pl.col('progesterone'))
    for cutoff in [32,34,37]:
        print(f"\t-----GA {cutoff}-----")
        #Filter correct subsample
        df = df_full.filter(~((pl.col("GA") // 7 < cutoff) & (pl.col("induced") | pl.col("c-section"))))
        
        n_before = df.select(pl.col("CPR_CHILD").n_unique()).item()
        df = df.filter(pl.col("CL").max().over("CPR_CHILD") != 0)
        n_after = df.select(pl.col("CPR_CHILD").n_unique()).item()
        
        print(f"\tFraction removed due to CL = 0: {((n_before - n_after) / n_before):.4f}")
    
    
        #Do sens and spec at CL < 25
        df_child = (df.group_by("CPR_CHILD").agg([pl.col("GA").first().alias("GA"),
                                                  pl.col("CL").mean().alias("CL")]))
        
        y_true = (df_child["GA"] // 7 < cutoff).to_numpy()
        y_pred = (df_child["CL"] < 25).to_numpy()
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        sens_ci = proportion_confint(tp, tp + fn, alpha=0.05, method="wilson")
        spec_ci = proportion_confint(tn, tn + fp, alpha=0.05, method="wilson")
        spec_ci = proportion_confint(tn, tn + fp, alpha=0.05, method="wilson")
            
        print(f"\tSensitivity CL < 25: {sensitivity:.4f} (95% CI {sens_ci[0]:.4f}-{sens_ci[1]:.4f})")
        print(f"\tSpecificity CL < 25: {specificity:.4f} (95% CI {spec_ci[0]:.4f}-{spec_ci[1]:.4f})")
    
        #Do sens@85spec        
        # CL values
        cl = df_child["CL"].to_numpy()
        
        # Original cutoff: 85% specificity
        cl_cutoff = np.quantile(cl[~y_true], 0.15)
        
        # Sensitivity at this cutoff
        sens = np.mean(cl[y_true] < cl_cutoff)
                
        # Bootstrap
        rng = np.random.default_rng()
        n_bootstrap = 2000
        
        boot_sens = []
        boot_cutoff = []
        
        for _ in range(n_bootstrap):
        
            sample_idx = rng.integers(0, len(y_true), len(y_true))
        
            y_boot = y_true[sample_idx]
            cl_boot = cl[sample_idx]
        
            # Need both classes
            if len(np.unique(y_boot)) < 2:
                continue
        
            # 85% specificity = 15th percentile among negatives
            cl_cutoff_b = np.quantile(cl_boot[~y_boot], 0.15)
        
            # Sensitivity at that cutoff
            sens_b = np.mean(cl_boot[y_boot] < cl_cutoff_b)
        
            boot_cutoff.append(cl_cutoff_b)
            boot_sens.append(sens_b)
        
        # 95% CIs
        sens_ci = np.percentile(boot_sens, [2.5, 97.5])
        cutoff_ci = np.percentile(boot_cutoff, [2.5, 97.5])
        
        print(f"\tSensitivity @ 85% spec: {sens:.4f} (95% CI {sens_ci[0]:.4f}-{sens_ci[1]:.4f})")
        
        print(f"\tCL cutoff: {cl_cutoff:.4f} (95% CI {cutoff_ci[0]:.4f}-{cutoff_ci[1]:.4f})")
        
        #Get AUC and CI
        auc = roc_auc_score(y_true, -cl)
        
        rng = np.random.default_rng()
        aucs = []
        
        for _ in range(2000):
            idx = rng.integers(0, len(y_true), len(y_true))
            aucs.append(roc_auc_score(y_true[idx], -cl[idx]))
        
        ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
        
        print(f"\tAUC: {auc:.4f} (95% CI {ci_low:.4f}-{ci_high:.4f})")