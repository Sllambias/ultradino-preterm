#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:40:20 2026

@author: jacob
"""

import argparse

import polars as pl
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help="Name of the model/experiment",
)

args = parser.parse_args()

model = args.model_name

bias_df = pl.read_csv('/users/data/UCPH/DeepFetal/projects/preterm/Data/misc/bias_variables.csv')

cutoff = model.split('_')[-1]

output_path = f"/users/data/UCPH/DeepFetal/projects/preterm/misc/{model}/"

cutoff = int(model.split('_')[-1])
path = '/users/data/UCPH/DeepFetal/projects/preterm/training_runs/Running/'


def plot_variable_histograms(
    df: pl.DataFrame,
    df_CL: pl.DataFrame,
    variable: str,
    save_path: str,
    bins: int = 30,
    cl_cutoff=25):
    # ================================================================
    # 1. Define populations
    # ================================================================

    populations = {
        "all": pl.lit(True),
        "non_treated": ~pl.col("treatment"),
    }

    # ================================================================
    # 2. Define prediction groups
    # ================================================================
    
    comparisons = {
        "positives": {
            "all_patients": pl.lit(True),
            "true_positives": (
                (pl.col("preds_binary_model") == 1)
                & (pl.col("label") == 1)
            ),
            "false_positives": (
                (pl.col("preds_binary_model") == 1)
                & (pl.col("label") == 0)
            ),
        },
        "negatives": {
            "all_patients": pl.lit(True),
            "true_negatives": (
                (pl.col("preds_binary_model") == 0)
                & (pl.col("label") == 0)
            ),
            "false_negatives": (
                (pl.col("preds_binary_model") == 0)
                & (pl.col("label") == 1)
            ),
        },
    }
    
    # ================================================================
    # 3. Define clinical subgroups
    # ================================================================

    sub_groups = {
        "All": None,
        "PPROM": "pprom",
        "C-Section": "c-section_during_birth",
    }

    # ================================================================
    # 4. Prepare output directory
    # ================================================================

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 5. Prediction group histograms
    # ================================================================
    
    for population_name, population_filter in populations.items():
    
        population_df = df.filter(population_filter)
    
        if population_df.height == 0:
            continue
    
        # Define the reference populations using the actual GA cutoff
        actual_positives = pl.col("GA") // 7 < cutoff
        actual_negatives = pl.col("GA") // 7 >= cutoff
    
    
        print(population_df.select(
        (pl.col("preds_binary_model") == 0).sum().alias("model_negative"),
        (pl.col("label") == 1).sum().alias("actual_positive"),
        (
            (pl.col("preds_binary_model") == 0)
            & (pl.col("label") == 1)
        ).sum().alias("FN"),
    ))
    
        comparisons = {
            "TP": {
                "filter": (
                    (pl.col("preds_binary_model") == 1)
                    & (pl.col("label") == 1)
                ),
                "references": {
                    "all": pl.lit(True),
                    "actual_positives": actual_positives,
                },
            },
            "FN": {
                "filter": (
                    (pl.col("preds_binary_model") == 0)
                    & (pl.col("label") == 1)
                ),
                "references": {
                    "all": pl.lit(True),
                    "actual_positives": actual_positives,
                },
            },
            "TN": {
                "filter": (
                    (pl.col("preds_binary_model") == 0)
                    & (pl.col("label") == 0)
                ),
                "references": {
                    "all": pl.lit(True),
                    "actual_negatives": actual_negatives,
                },
            },
            "FP": {
                "filter": (
                    (pl.col("preds_binary_model") == 1)
                    & (pl.col("label") == 0)
                ),
                "references": {
                    "all": pl.lit(True),
                    "actual_negatives": actual_negatives,
                },
            },
        }
    
        for group_name, group_info in comparisons.items():
    
            group_filter = group_info["filter"]
    
            group_df = population_df.filter(group_filter)
            
            group_values = (
                group_df
                .get_column(variable)
                .drop_nulls()
                .to_numpy()
            )
            
            if group_df.height == 0:
                print(f"EMPTY: variable={variable}, population={population_name}, group={group_name}")
            
            if len(group_values) == 0:
                print(
                    f"{population_name}, {variable}: "
                    f"{group_df.height} FN patients exist, "
                    f"but all have null {variable}"
                )
                continue
    
            for reference_name, reference_filter in (
                group_info["references"].items()
            ):
    
                reference_df = population_df.filter(reference_filter)
    
                reference_values = (
                    reference_df
                    .get_column(variable)
                    .drop_nulls()
                    .to_numpy()
                )
    
                if len(reference_values) == 0:
                    continue
    
                # Use the reference distribution to define consistent bins
                bin_edges = np.histogram_bin_edges(
                    reference_values,
                    bins=bins,
                )
    
                fig, ax = plt.subplots(figsize=(9, 6))
    
                ax.hist(
                    reference_values,
                    bins=bin_edges,
                    alpha=0.4,
                    density=True,
                    edgecolor="black",
                    linewidth=0.5,
                    label=(
                        f"{reference_name.replace('_', ' ').title()} "
                        f"(n={len(reference_values):,})"
                    ),
                )
    
                ax.hist(
                    group_values,
                    bins=bin_edges,
                    alpha=0.5,
                    density=True,
                    edgecolor="black",
                    linewidth=0.5,
                    label=f"{group_name} (n={len(group_values):,})",
                )
    
                ax.set_title(
                    f"{variable} — "
                    f"{population_name.replace('_', ' ').title()} — "
                    f"{group_name} vs. {reference_name.replace('_', ' ').title()}"
                )
                ax.set_xlabel(variable)
                ax.set_ylabel("Probability density")
                ax.legend()
                ax.grid(axis="y", alpha=0.25)
    
                fig.tight_layout()
    
                filename = (
                    f"{variable}_{population_name}_"
                    f"{group_name}_vs_{reference_name}.png"
                )
    
                fig.savefig(
                    save_path / filename,
                    dpi=300,
                    bbox_inches="tight",
                )
    
                plt.close(fig)

    # ================================================================
    # 6. Compare model predictions against CL predictions
    # ================================================================
    
    for population_name, population_filter in populations.items():
    
        population_df = df_CL.filter(population_filter)
    
        comparisons = {
            "model_positive": {
                "filter": pl.col("preds_binary_model") == 1,
                "title": "Model-positive patients",
            },
            "model_negative": {
                "filter": pl.col("preds_binary_model") == 0,
                "title": "Model-negative patients",
            },
        }
    
        cl_groups = {
            "CL_positive": pl.col("CL") < cl_cutoff,
            "CL_negative": pl.col("CL") >= cl_cutoff,
        }
    
        for model_group_name, model_group_info in comparisons.items():
    
            model_df = population_df.filter(
                model_group_info["filter"]
            )
    
            group_values = {}
    
            for cl_group_name, cl_filter in cl_groups.items():
    
                values = (
                    model_df
                    .filter(cl_filter)
                    .get_column(variable)
                    .drop_nulls()
                    .to_numpy()
                )
    
                group_name = f"{model_group_name}_{cl_group_name}"
                group_values[group_name] = values
    
                print(
                    f"{variable} | {population_name} | "
                    f"{group_name}: n={len(values)}"
                )
    
            nonempty_values = [
                values for values in group_values.values()
                if len(values) > 0
            ]
    
            if not nonempty_values:
                print(
                    f"Skipping {variable} | {population_name} | "
                    f"{model_group_name}: both CL groups are empty."
                )
                continue
    
            all_values = np.concatenate(nonempty_values)
            bin_edges = np.histogram_bin_edges(all_values, bins=bins)
    
            fig, ax = plt.subplots(figsize=(9, 6))
    
            for group_name, values in group_values.items():
    
                if len(values) == 0:
                    continue
    
                ax.hist(
                    values,
                    bins=bin_edges,
                    alpha=0.5,
                    density=True,
                    edgecolor="black",
                    linewidth=0.5,
                    label=f"{group_name} (n={len(values):,})",
                )
    
            ax.set_title(
                f"{variable} — "
                f"{population_name.replace('_', ' ').title()} — "
                f"{model_group_info['title']}: CL-positive vs CL-negative"
            )
            ax.set_xlabel(variable)
            ax.set_ylabel("Probability density")
            ax.legend()
            ax.grid(axis="y", alpha=0.25)
    
            fig.tight_layout()
    
            filename = (
                f"{variable}_{population_name}_"
                f"{model_group_name}_CL_positive_vs_negative.png"
            )
    
            fig.savefig(
                save_path / filename,
                dpi=300,
                bbox_inches="tight",
            )
    
            plt.close(fig)


    # ================================================================
    # 7. Clinical subgroup histograms
    # ================================================================

    for population_name, population_filter in populations.items():

        population_df = df.filter(population_filter)

        for subgroup_name, subgroup_column in sub_groups.items():

            # Skip the "All" subgroup
            if subgroup_column is None:
                continue

            subgroup_df = population_df.filter(
                pl.col(subgroup_column).fill_null(False)
            )

            all_values = (
                population_df
                .get_column(variable)
                .drop_nulls()
                .to_numpy()
            )

            subgroup_values = (
                subgroup_df
                .get_column(variable)
                .drop_nulls()
                .to_numpy()
            )

            if len(all_values) == 0:
                continue

            bin_edges = np.histogram_bin_edges(
                all_values,
                bins=bins,
            )

            fig, ax = plt.subplots(figsize=(9, 6))

            ax.hist(
                all_values,
                bins=bin_edges,
                alpha=0.4,
                density=True,
                edgecolor="black",
                linewidth=0.5,
                label=f"All patients (n={len(all_values):,})",
            )

            if len(subgroup_values) > 0:
                ax.hist(
                    subgroup_values,
                    bins=bin_edges,
                    alpha=0.5,
                    density=True,
                    edgecolor="black",
                    linewidth=0.5,
                    label=(
                        f"{subgroup_name} "
                        f"(n={len(subgroup_values):,})"
                    ),
                )

            ax.set_title(
                f"{variable} — "
                f"{population_name.replace('_', ' ').title()} — "
                f"{subgroup_name}"
            )
            ax.set_xlabel(variable)
            ax.set_ylabel("Probability density")
            ax.legend()
            ax.grid(axis="y", alpha=0.25)

            fig.tight_layout()

            # Generate filenames
            subgroup_slug = subgroup_column.replace("-", "_")

            if population_name == "all":
                filename = f"{variable}_{subgroup_slug}.png"
            else:
                filename = (
                    f"{variable}_{population_name}_{subgroup_slug}.png"
                )

            fig.savefig(
                save_path / filename,
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)


# =====================================================================
# 1. Load data
# =====================================================================

preds = pl.read_parquet(
    path + model + f"/results/predictions/predictions_{cutoff}.parquet"
)

df = pl.read_parquet(
    "/users/data/UCPH/DeepFetal/projects/preterm/Data/dataset_v6/test2.parquet"
)

# =====================================================================
# 2. Aggregate clinical variables and image-level information
# =====================================================================

clinical_df = df.group_by("CPR_CHILD").agg([
    # Clinical variables
    pl.col("GA").first(),
    pl.col("induced").first(),
    pl.col("c-section").first(),
    pl.col("pprom").first(),
    pl.col("c-section_during_birth").first(),
    pl.col("contractions_with_preterm_birth").first(),
    pl.col("CL").first(),

    # Mean physical pixel spacing across images
    pl.mean_horizontal(
        pl.col("physical_delta_x").list.first(),
        pl.col("physical_delta_y").list.first()).mean().alias("physical_delta"),

    # Number of images per child
    pl.len().alias("number_of_imgs"),
])


# =====================================================================
# 3. Merge predictions with clinical variables
# =====================================================================

df_combined = preds.join(
    clinical_df,
    on="CPR_CHILD",
    how="left",
)

# Apply inclusion criteria
df_combined = df_combined.filter(
    (pl.col("GA") // 7 >= cutoff)
    | (~pl.col("induced") & ~pl.col("c-section"))
    | pl.col("pprom")
    | (
        pl.col("c-section")
        & (
            pl.col("c-section_during_birth")
            | pl.col("contractions_with_preterm_birth")
        )
        & ~pl.col("induced")
    )
)


# =====================================================================
# 4. Merge bias variables
# =====================================================================

assert not bias_df["b_cpr"].is_duplicated().any(), (
    "bias_df contains duplicate b_cpr values; "
    "the join could duplicate prediction rows."
)

df_combined = df_combined.join(
    bias_df,
    left_on="CPR_CHILD",
    right_on="b_cpr",
    how="left",
)

# Apply variable cutoffs
df_combined = df_combined.with_columns(
    pl.col("maternal_BMI").clip(upper_bound=50),
    pl.col("physical_delta").clip(upper_bound=0.05),
    pl.col("number_of_imgs").clip(upper_bound=20),
)

y_true = df_combined["label"].to_numpy()
y_score = df_combined["preds"].to_numpy()

fpr, tpr, thresholds = roc_curve(y_true, y_score)

specificity = 1 - fpr
valid = specificity >= 0.85

best_idx = np.argmax(np.where(valid, tpr, -1))
model_cutoff = thresholds[best_idx]

print(f"Threshold at 85% specificity: {model_cutoff:.4f}")
print(f"Sensitivity: {tpr[best_idx]:.3f}")
print(f"Specificity: {specificity[best_idx]:.3f}")

df_CL = df_combined.filter(pl.col("CL").is_not_null() & (pl.col("CL") != 0))

y_true = df_CL["label"].to_numpy()
y_score = -df_CL["CL"].to_numpy()

fpr, tpr, thresholds = roc_curve(y_true, y_score)

specificity = 1 - fpr
valid = specificity >= 0.85

best_idx = np.argmax(np.where(valid, tpr, -1))
threshold_85_spec = thresholds[best_idx]

# Convert back to the original CL scale
cl_cutoff = -threshold_85_spec

print(f"CL threshold at 85% specificity: {cl_cutoff:.4f}")
print(f"Sensitivity: {tpr[best_idx]:.3f}")
print(f"Specificity: {specificity[best_idx]:.3f}")

# Define binary predictions using the CL-based threshold

df_combined = df_combined.with_columns(
    (pl.col("preds") >= model_cutoff)
    .cast(pl.Int8)
    .alias("preds_binary_model")
)

df_CL = df_combined.filter(
    pl.col("CL").is_not_null() & (pl.col("CL") != 0)
).with_columns(
    (pl.col("CL") < cl_cutoff)
    .cast(pl.Int8)
    .alias("preds_binary_CL")
)


# =====================================================================
# 5. Validate
# =====================================================================

assert not df_combined["CPR_CHILD"].is_duplicated().any(), (
    "df_combined contains duplicate CPR_CHILD values."
)

print(f"Children after filtering: {df_combined.height:,}")

# =====================================================================
# 6. Model Insight — Variable Histograms
# =====================================================================

variables = [
    "maternal_age",
    "maternal_BMI",
    "physical_delta",
    "number_of_imgs",
    "CL"
]

histogram_path = Path(output_path) / "histograms"

for variable in variables:
    if variable not in df_combined.columns:
        print(f"Skipping {variable}: column not found in df_combined.")
        continue

    print(f"Creating histograms for {variable}...")

    plot_variable_histograms(
        df=df_combined,
        df_CL=df_CL,
        variable=variable,
        save_path=str(histogram_path),
        cl_cutoff=cl_cutoff)

print(f"Histograms saved to: {histogram_path}")






