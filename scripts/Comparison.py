#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 10:07:01 2026

@author: jacob
"""
import numpy as np
import polars as pl

from sklearn.metrics import confusion_matrix
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/'

CL = pl.read_parquet(path + 'Data/OnlyFirstPreg_v5/test.parquet')

#Preds is CPR_CHILD | pred_{avg/max} | label
preds = pl.read_csv(path + 'some_path')

MODEL_PRED = "pred_avg"

PROG = True

cutoff = 34

N_BOOTSTRAP = 2000
N_PERMUTATIONS = 2000

RANDOM_SEED = 42

# Keep only the CL information we need
cl = (
    CL
    .filter(pl.col("CL") != 0)
    .group_by("CPR_CHILD")
    .agg([
        pl.col("CL").mean().alias("CL"),
        pl.col("GA").first().alias("GA"),
        pl.col("progesterone").first().alias("progesterone"),
        pl.col("induced").first().alias("induced"),
        pl.col("c-section").first().alias("c-section"),
    ])
)

# Combine model predictions with CL
df = (
    preds
    .join(cl, on="CPR_CHILD", how="inner")
)


if not PROG:
    df = df.filter(pl.col("progesterone") == False)

if df["CPR_CHILD"].n_unique() != df.height:
    raise ValueError("There are multiple prediction rows per CPR_CHILD")


print(f"Total children: {df.height}")

def sensitivity_specificity(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True]
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    sens_ci = proportion_confint(
        tp,
        tp + fn,
        alpha=0.05,
        method="wilson"
    )

    spec_ci = proportion_confint(
        tn,
        tn + fp,
        alpha=0.05,
        method="wilson"
    )

    return (
        sensitivity,
        specificity,
        sens_ci,
        spec_ci
    )

def paired_auc_permutation_test(
    y_true,
    score_baseline,
    score_comparison,
    n_permutations=10000,
    seed=42
):
    """
    Paired permutation test for difference in AUC.

    Predictions/scores are paired within each child.
    """

    y_true = np.asarray(y_true, dtype=bool)
    score_baseline = np.asarray(score_baseline, dtype=float)
    score_comparison = np.asarray(score_comparison, dtype=float)

    auc_baseline = roc_auc_score(
        y_true,
        score_baseline
    )

    auc_comparison = roc_auc_score(
        y_true,
        score_comparison
    )

    observed_difference = auc_comparison - auc_baseline

    rng = np.random.default_rng(seed)

    permutation_differences = np.empty(n_permutations)

    for i in range(n_permutations):

        swap = rng.random(len(y_true)) < 0.5

        perm_baseline = np.where(
            swap,
            score_comparison,
            score_baseline
        )

        perm_comparison = np.where(
            swap,
            score_baseline,
            score_comparison
        )

        permutation_differences[i] = (
            roc_auc_score(y_true, perm_comparison)
            -
            roc_auc_score(y_true, perm_baseline)
        )

    p_value = (
        np.sum(
            np.abs(permutation_differences)
            >= abs(observed_difference)
        ) + 1
    ) / (n_permutations + 1)

    return (
        auc_baseline,
        auc_comparison,
        observed_difference,
        p_value
    )

def optimal_cl_cutoff(cl, y_true, specificity_target=0.85):
    """
    Find the largest CL cutoff giving at least the requested specificity.
    Smaller CL = positive.
    """
    cl = np.asarray(cl)
    y_true = np.asarray(y_true, dtype=bool)

    thresholds = np.sort(np.unique(cl))

    valid_cutoffs = []

    for cutoff in thresholds:
        y_pred = cl < cutoff

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[False, True]
        ).ravel()

        if tn + fp == 0:
            continue

        specificity = tn / (tn + fp)

        if specificity >= specificity_target:
            valid_cutoffs.append(cutoff)

    return max(valid_cutoffs) if valid_cutoffs else None


def paired_permutation_test(
    y_true,
    pred_baseline,
    pred_comparison,
    n_permutations=10000,
    seed=42
):
    """
    Paired permutation test comparing sensitivity.

    For every child, randomly swap the baseline and comparison
    predictions. This preserves the pairing between predictions.
    """

    y_true = np.asarray(y_true, dtype=bool)
    pred_baseline = np.asarray(pred_baseline, dtype=bool)
    pred_comparison = np.asarray(pred_comparison, dtype=bool)

    # Only cases with y_true = 1 contribute to sensitivity
    positive = y_true

    baseline_sens = pred_baseline[positive].mean()
    comparison_sens = pred_comparison[positive].mean()

    observed_difference = comparison_sens - baseline_sens

    rng = np.random.default_rng(seed)

    permutation_differences = np.empty(n_permutations)

    for i in tqdm(range(n_permutations), desc="AUC permutation test"):

        swap = rng.random(len(y_true)) < 0.5

        perm_baseline = np.where(
            swap,
            pred_comparison,
            pred_baseline
        )

        perm_comparison = np.where(
            swap,
            pred_baseline,
            pred_comparison
        )

        permutation_differences[i] = (
            perm_comparison[positive].mean()
            - perm_baseline[positive].mean()
        )

    # Two-sided p-value
    p_value = (
        np.sum(
            np.abs(permutation_differences)
            >= abs(observed_difference)
        ) + 1
    ) / (n_permutations + 1)

    return (
        baseline_sens,
        comparison_sens,
        observed_difference,
        p_value
    )


def bootstrap_cl_sensitivity(
    cl,
    y_true,
    specificity_target=0.85,
    n_bootstrap=2000,
    seed=42
):
    """
    Bootstrap CI for sensitivity at the subgroup-specific
    optimal CL cutoff.

    The optimal cutoff is recalculated in every bootstrap sample.
    """

    cl = np.asarray(cl)
    y_true = np.asarray(y_true, dtype=bool)

    rng = np.random.default_rng(seed)

    boot_sens = []
    boot_cutoffs = []

    n = len(y_true)

    for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):

        idx = rng.integers(0, n, n)

        y_boot = y_true[idx]
        cl_boot = cl[idx]

        # Need both classes
        if len(np.unique(y_boot)) < 2:
            continue

        cutoff = optimal_cl_cutoff(
            cl_boot,
            y_boot,
            specificity_target
        )

        if cutoff is None:
            continue

        sens = np.mean(
            cl_boot[y_boot] < cutoff
        )

        boot_sens.append(sens)
        boot_cutoffs.append(cutoff)

    sens_ci = np.percentile(
        boot_sens,
        [2.5, 97.5]
    )

    cutoff_ci = np.percentile(
        boot_cutoffs,
        [2.5, 97.5]
    )

    return sens_ci, cutoff_ci

def optimal_model_threshold(score, y_true, specificity_target=0.85):
    score = np.asarray(score, dtype=float)
    y_true = np.asarray(y_true, dtype=bool)

    thresholds = np.sort(np.unique(score))

    valid_thresholds = []

    for threshold in thresholds:
        y_pred = score >= threshold

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[False, True]
        ).ravel()

        if tn + fp == 0:
            continue

        specificity = tn / (tn + fp)

        if specificity >= specificity_target:
            valid_thresholds.append(threshold)

    return min(valid_thresholds) if valid_thresholds else None

# ============================================================
# Analysis
# ============================================================

results = []




print()
print("=" * 70)
print(f"GA < {cutoff}")
print("=" * 70)

# --------------------------------------------------------
# Subgroup
# --------------------------------------------------------

# Exclude induced / C-section only among preterm births
df_sub = df.filter(
    ~(
        (pl.col("GA") // 7 < cutoff)
        &
        (pl.col("induced") | pl.col("c-section"))
    )
)

# Remove children without a valid CL
df_sub = df_sub.filter(
    pl.col("CL").is_not_null()
)

if df_sub.height == 0:
    print("No data")

# --------------------------------------------------------
# Arrays
# --------------------------------------------------------

y_true = (
    (df_sub["GA"] // 7 < cutoff)
    .to_numpy()
    .astype(bool)
)

cl = df_sub["CL"].to_numpy()

# Continuous model prediction [0, 1]
model_score = df_sub[MODEL_PRED].to_numpy()

# Binary model prediction
# Change threshold if your model uses a different threshold
model_threshold = optimal_model_threshold(
    model_score,
    y_true,
    specificity_target=0.85
)

model_pred = model_score >= model_threshold
# CL score
# Lower CL = higher preterm risk
cl_score = -cl

# CL < 25
cl_25_pred = cl < 25

# --------------------------------------------------------
# Model performance
# --------------------------------------------------------

(
    model_sens,
    model_spec,
    model_sens_ci,
    model_spec_ci
) = sensitivity_specificity(
    y_true,
    model_pred
)

print(
    f"Model sensitivity: "
    f"{model_sens:.4f} "
    f"(95% CI {model_sens_ci[0]:.4f}-{model_sens_ci[1]:.4f})"
)

print(
    f"Model specificity: "
    f"{model_spec:.4f} "
    f"(95% CI {model_spec_ci[0]:.4f}-{model_spec_ci[1]:.4f})"
)

# --------------------------------------------------------
# AUC: Model vs CL
# --------------------------------------------------------

model_auc = roc_auc_score(
    y_true,
    model_score
)

cl_auc = roc_auc_score(
    y_true,
    cl_score
)

# Bootstrap AUC CIs
rng = np.random.default_rng(RANDOM_SEED)

model_boot_auc = []
cl_boot_auc = []

for _ in range(N_BOOTSTRAP):

    idx = rng.integers(
        0,
        len(y_true),
        len(y_true)
    )

    y_boot = y_true[idx]

    # Need both classes
    if len(np.unique(y_boot)) < 2:
        continue

    model_boot_auc.append(
        roc_auc_score(
            y_boot,
            model_score[idx]
        )
    )

    cl_boot_auc.append(
        roc_auc_score(
            y_boot,
            cl_score[idx]
        )
    )

model_auc_ci = np.percentile(
    model_boot_auc,
    [2.5, 97.5]
)

cl_auc_ci = np.percentile(
    cl_boot_auc,
    [2.5, 97.5]
)

print()
print(
    f"Model AUC: "
    f"{model_auc:.4f} "
    f"(95% CI {model_auc_ci[0]:.4f}-{model_auc_ci[1]:.4f})"
)

print(
    f"CL AUC: "
    f"{cl_auc:.4f} "
    f"(95% CI {cl_auc_ci[0]:.4f}-{cl_auc_ci[1]:.4f})"
)

# --------------------------------------------------------
# Model vs CL AUC
# --------------------------------------------------------

(
    _,
    _,
    auc_difference,
    auc_p
) = paired_auc_permutation_test(
    y_true,
    model_score,
    cl_score,
    n_permutations=N_PERMUTATIONS,
    seed=RANDOM_SEED
)

print(
    f"Model vs CL AUC: "
    f"Δ AUC = {auc_difference:+.4f}, "
    f"p = {auc_p:.4f}"
)

# --------------------------------------------------------
# CL < 25
# --------------------------------------------------------

(
    cl25_sens,
    cl25_spec,
    cl25_sens_ci,
    cl25_spec_ci
) = sensitivity_specificity(
    y_true,
    cl_25_pred
)

print()
print(
    f"CL < 25 sensitivity: "
    f"{cl25_sens:.4f} "
    f"(95% CI {cl25_sens_ci[0]:.4f}-{cl25_sens_ci[1]:.4f})"
)

print(
    f"CL < 25 specificity: "
    f"{cl25_spec:.4f} "
    f"(95% CI {cl25_spec_ci[0]:.4f}-{cl25_spec_ci[1]:.4f})"
)

# --------------------------------------------------------
# Model vs CL < 25
# --------------------------------------------------------

(
    _,
    _,
    difference_25,
    p_25
) = paired_permutation_test(
    y_true,
    model_pred,
    cl_25_pred,
    n_permutations=N_PERMUTATIONS,
    seed=RANDOM_SEED
)

print(
    f"Model vs CL < 25: "
    f"Δ sensitivity = {difference_25:+.4f}, "
    f"p = {p_25:.4f}"
)

# --------------------------------------------------------
# Find subgroup-specific optimal CL cutoff
# --------------------------------------------------------

cl_optimal_cutoff = optimal_cl_cutoff(
    cl,
    y_true,
    specificity_target=0.85
)

cl_optimal_pred = cl < cl_optimal_cutoff

(
    cl_opt_sens,
    cl_opt_spec,
    cl_opt_sens_ci,
    cl_opt_spec_ci
) = sensitivity_specificity(
    y_true,
    cl_optimal_pred
)

print()
print(
    f"Optimal CL cutoff: "
    f"{cl_optimal_cutoff:.4f}"
)

print(
    f"Optimal CL sensitivity: "
    f"{cl_opt_sens:.4f} "
    f"(95% CI {cl_opt_sens_ci[0]:.4f}-{cl_opt_sens_ci[1]:.4f})"
)

print(
    f"Optimal CL specificity: "
    f"{cl_opt_spec:.4f} "
    f"(95% CI {cl_opt_spec_ci[0]:.4f}-{cl_opt_spec_ci[1]:.4f})"
)

# Bootstrap CI for optimal cutoff and sensitivity
(
    opt_sens_boot_ci,
    opt_cutoff_boot_ci
) = bootstrap_cl_sensitivity(
    cl,
    y_true,
    specificity_target=0.85,
    n_bootstrap=N_BOOTSTRAP,
    seed=RANDOM_SEED
)

print(
    f"Optimal CL sensitivity bootstrap 95% CI: "
    f"{opt_sens_boot_ci[0]:.4f}-{opt_sens_boot_ci[1]:.4f}"
)

print(
    f"Optimal CL cutoff bootstrap 95% CI: "
    f"{opt_cutoff_boot_ci[0]:.4f}-{opt_cutoff_boot_ci[1]:.4f}"
)

# --------------------------------------------------------
# Model vs optimal CL
# --------------------------------------------------------

(
    _,
    _,
    difference_opt,
    p_opt
) = paired_permutation_test(
    y_true,
    model_pred,
    cl_optimal_pred,
    n_permutations=N_PERMUTATIONS,
    seed=RANDOM_SEED
)

print(
    f"Model vs optimal CL: "
    f"Δ sensitivity = {difference_opt:+.4f}, "
    f"p = {p_opt:.4f}"
)

# --------------------------------------------------------
# Save result
# --------------------------------------------------------

print(f"\n{'=' * 60}")
print(f"GA < {cutoff} weeks")
print(f"{'=' * 60}")

print(f"N: {len(y_true)}")
print(f"Preterm: {int(y_true.sum())}")
print(f"Term: {int((~y_true).sum())}")

print("\nModel")
print(
    f"  Sensitivity: {model_sens:.3f} "
    f"(95% CI {model_sens_ci[0]:.3f}-{model_sens_ci[1]:.3f})"
)
print(
    f"  AUC: {model_auc:.3f} "
    f"(95% CI {model_auc_ci[0]:.3f}-{model_auc_ci[1]:.3f})"
)

print("\nCL < 25 mm")
print(
    f"  Sensitivity: {cl25_sens:.3f} "
    f"(95% CI {cl25_sens_ci[0]:.3f}-{cl25_sens_ci[1]:.3f})"
)
print(
    f"  Difference vs model: {difference_25:+.3f}"
)
print(
    f"  P-value: {p_25:.4f}"
)

print("\nCL optimal")
print(f"  Cutoff: {cl_optimal_cutoff:.2f} mm")
print(
    f"  Cutoff 95% CI: "
    f"{opt_cutoff_boot_ci[0]:.2f}-{opt_cutoff_boot_ci[1]:.2f} mm"
)
print(
    f"  Sensitivity: {cl_opt_sens:.3f} "
    f"(95% CI {opt_sens_boot_ci[0]:.3f}-{opt_sens_boot_ci[1]:.3f})"
)
print(
    f"  Difference vs model: {difference_opt:+.3f}"
)
print(
    f"  P-value: {p_opt:.4f}"
)

print("\nAUC: Model vs continuous CL")
print(
    f"  CL AUC: {cl_auc:.3f} "
    f"(95% CI {cl_auc_ci[0]:.3f}-{cl_auc_ci[1]:.3f})"
)
print(
    f"  Difference CL vs model: {auc_difference:+.3f}"
)
print(
    f"  P-value: {auc_p:.4f}"
)