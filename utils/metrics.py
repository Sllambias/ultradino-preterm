#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:03:57 2026

@author: jacob
"""

import torchmetrics.classification as tm
import polars as pl
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def aggregation_functions():
    """How the image predictions of a child are combined into one prediction."""
    return {"avg": pl.col("preds").mean(),
            "max": pl.col("preds").max(),
            "quantile": pl.col("preds").quantile(0.75)}


def patient_predictions(df, agg, tta, threshold):
    """One row per child: its image predictions combined by agg, the call at threshold and preds_normalized (threshold at 0.5)."""
    if tta == 'no-tta':
        df = df.filter(pl.col("img_path").str.ends_with("_1"))

    agg_fn = aggregation_functions()[agg]

    return (df.group_by("CPR_CHILD").agg([agg_fn.alias("preds"),
                                          pl.col("label").first().alias("label"),
                                          pl.col("treatment").first().alias("treatment")])
            .with_columns([(pl.col("preds") >= threshold).cast(pl.Int8).alias("preds_binary"),
                           (pl.when(pl.col("preds") < threshold)
                            .then(0.5 * pl.col("preds") / threshold)
                            .otherwise(0.5 + 0.5 * (pl.col("preds") - threshold) / (1 - threshold))
                            .alias("preds_normalized"))]))


def binary_metrics(device, t=0.5):
    """Sensitivity and specificity at threshold t, sensitivity at 85% specificity, and AUC."""
    return {'Sensitivity': tm.Recall(task='binary', threshold=t).to(device),
            'Specificity': tm.Specificity(task='binary', threshold=t).to(device),
            'SensAtSpec': tm.SensitivityAtSpecificity(min_specificity=0.85, task='binary').to(device),
            'AUC': tm.AUROC(task='binary').to(device)}


def metric_value(metric, preds, labels):
    """One torchmetrics value; SensAtSpec also returns its threshold, which is dropped."""
    metric.reset()
    output = metric(preds, labels)
    if isinstance(output, tuple):
        output = output[0]
    return output.item()


def evaluate_predictions(df, n_bootstrap, rng):
    """Metrics of child predictions, as in Metrics.log_model, with bootstrap 95% CIs if n_bootstrap > 0."""
    # group_by returns children in a random order, so sort them for the seeded bootstrap to be reproducible
    df = df.sort("CPR_CHILD")
    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.int32)
    scores = torch.tensor(df["preds_normalized"].to_numpy(), dtype=torch.float32)
    binary = torch.tensor(df["preds_binary"].to_numpy(), dtype=torch.float32)

    result = {"children": len(df), "preterm": int(df["label"].sum())}
    for name, metric in binary_metrics("cpu").items():
        # Sensitivity and specificity use the call at the validation threshold
        if name in ["Sensitivity", "Specificity"]:
            preds = binary
        else:
            preds = scores
        result[name] = round(metric_value(metric, preds, labels), 4)

        boot_values = []
        for _ in range(n_bootstrap):
            idx = torch.from_numpy(rng.integers(0, len(df), size=len(df)))
            if torch.unique(labels[idx]).numel() < 2:
                continue
            boot_values.append(metric_value(metric, preds[idx], labels[idx]))
        if boot_values:
            result[f"{name}_lower"] = round(np.percentile(boot_values, 2.5), 4)
            result[f"{name}_upper"] = round(np.percentile(boot_values, 97.5), 4)
    return result


def ensemble_predictions(folds_df):
    """Mean of the folds' preds_normalized per child, which puts every fold's threshold at 0.5."""
    return (folds_df.group_by("CPR_CHILD").agg([pl.col("preds_normalized").mean(),
                                                pl.col("label").first(),
                                                pl.col("treatment").first()])
            .with_columns([(pl.col("preds_normalized") >= 0.5).cast(pl.Int8).alias("preds_binary"),
                           pl.lit("ensemble").alias("model")]))


class Metrics():
    def __init__(self, cfg, save_path):
        self.cutoffs = cfg.tasks.preterm.cutoffs
        self.epoch = 0
        self.fold = 0
        self.device = cfg.device.type
        self.n_bootstrap = 10000
        # Validation metrics are also logged for the rows where each of these columns is True
        self.subgroups = list(cfg.data.get("val_subgroups", []))
        self.test_subgroup = cfg.data.get("test_subgroup")

        self.dfs = {str(c): [] for c in self.cutoffs}
        self.test_preds = {str(c): [] for c in self.cutoffs}
        self.save_path = Path(save_path)
        self.best_score = {str(c): {'score': -float("inf"),
                                    'threshold': None,
                                    'epoch': -1,
                                    'agg': None,
                                    'tta': None} for c in self.cutoffs}
        self.best_predictions = {str(c): None for c in self.cutoffs}
        self.best_df = {}
        
        self.pred_path = Path(self.save_path) / "predictions" 
        self.metrics_path = Path(self.save_path) / "metrics"
        self.plot_path = Path(self.save_path) / "plots"
        
        (self.pred_path / "validation").mkdir(parents=True, exist_ok=True)
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        self.plot_path.mkdir(parents=True, exist_ok=True)
        
        self.agg_fns = aggregation_functions()

        self.metrics = self.get_metrics()


    def update(self, outputs, data, cutoff=None):
        if cutoff is None:
            cutoffs = self.cutoffs
        else:
            cutoffs = [cutoff]
        for cutoff in cutoffs:
            mask = data["masks"][str(cutoff)].flatten().cpu().numpy()
            columns = {'CPR_CHILD': data['IDs'],
                       'img_path': data['img_paths'],
                       'preds': outputs['preterm'][str(cutoff)]['preds'].flatten().cpu().numpy(),
                       'label': (data['labels']['GA_weeks'] < float(cutoff)).flatten().cpu().numpy(),
                       'treatment': data['treatment']}
            for subgroup in self.subgroups:
                columns[subgroup] = data['subgroups'][subgroup]
            self.dfs[str(cutoff)].append(pl.DataFrame(columns).filter(mask))

    def compute_metrics(self, patient_dfs, agg, prefix=""):
        """Every metric on the patient predictions of one aggregation, with and without TTA."""
        metrics = {}
        for name, metric in self.metrics.items():
            for tta in ['tta', 'no-tta']:
                preds = torch.tensor(patient_dfs[tta][agg].to_numpy(), dtype=torch.float32)
                labels = torch.tensor(patient_dfs[tta]['label'].to_numpy(), dtype=torch.int32)

                metric.reset()
                output = metric(preds, labels)

                if isinstance(output, tuple):
                    metrics[f"{prefix}{name}_{agg}_{tta}"] = round(output[0].item(), 3)
                    metrics[f"{prefix}{name}_cutoff_{agg}_{tta}"] = round(output[1].item(), 3)
                else:
                    metrics[f"{prefix}{name}_{agg}_{tta}"] = round(output.item(), 3)
        return metrics

    def log_epoch(self, train_loss, val_loss):
        self.epoch += 1
        for cutoff in self.cutoffs:
            df = pl.concat(self.dfs[str(cutoff)])
        
            patient_aggs = ([expr.alias(name) for name, expr in self.agg_fns.items()]
                            + [pl.col("label").first().alias("label"),
                               pl.col("treatment").first().alias("treatment")]
                            + [pl.col(subgroup).first().alias(subgroup) for subgroup in self.subgroups])
            
            patient_df_tta = df.group_by("CPR_CHILD").agg(patient_aggs)
            
            patient_df_no_tta = (df.filter(pl.col("img_path").str.ends_with("_1"))
                                 .group_by("CPR_CHILD").agg(patient_aggs))
            
            patient_dfs = {'tta': patient_df_tta,
                           'no-tta': patient_df_no_tta}
            

            for agg in ['avg', 'quantile', 'max']:
                metrics = self.compute_metrics(patient_dfs, agg)

                # Also on each subgroup alone, e.g. parity_0_AUC_avg_tta. The best epoch is still picked on all rows
                for subgroup in self.subgroups:
                    subgroup_dfs = {tta: patient_df.filter(pl.col(subgroup)) for tta, patient_df in patient_dfs.items()}
                    metrics.update(self.compute_metrics(subgroup_dfs, agg, prefix=f"{subgroup}_"))

                best_tta = max(["tta", "no-tta"], key=lambda tta: metrics[f"SensAtSpec_{agg}_{tta}"])
                best_score = metrics[f"SensAtSpec_{agg}_{best_tta}"]
                best_threshold = metrics[f"SensAtSpec_cutoff_{agg}_{best_tta}"]
                # Keep predictions if this is the best epoch so far
                if best_score > self.best_score[str(cutoff)]['score']:
                    self.best_score[str(cutoff)] = {'score': best_score,
                                                    'threshold': best_threshold,
                                                    'epoch': self.epoch,
                                                    'agg': agg,
                                                    'tta': best_tta}
    
                    self.best_predictions[str(cutoff)] = (patient_dfs[best_tta].select(["CPR_CHILD",
                                                                                        agg,
                                                                                        "label",
                                                                                        "treatment"] + self.subgroups)
                                                          .rename({agg: "preds"}).with_columns([pl.lit(agg).alias("agg"),
                                                                                                pl.lit(best_tta).alias('tta'),
                                                                                                pl.lit(best_threshold).alias("threshold")]))
                    self.best_df[str(cutoff)] = df
           
                row = {'epoch': self.epoch,
                       'train_loss': round(sum(train_loss.values()), 4), 
                       'train_preterm_loss': round(train_loss['preterm'], 4),
                       'val_loss': round(sum(val_loss.values()), 4),
                       'val_preterm_loss': round(val_loss['preterm'], 4),
                       **dict(sorted(metrics.items()))}

                metrics_df = pl.DataFrame([row])

                path = self.metrics_path / f"metrics_{cutoff}_{agg}_fold_{self.fold}.csv"
            
                if path.exists():
                    existing = pl.read_csv(path)
                    metrics_df = pl.concat([existing, metrics_df], how="vertical")

                metrics_df.write_csv(path)

            
                self.plot_metrics(metrics_df, cutoff, agg)
        
        self.dfs = {str(c): [] for c in self.cutoffs}

    def plot_metrics(self, metrics_df, cutoff, agg):
        metric_cols = [col for col in metrics_df.columns if col not in ["epoch",
                                                                        "train_loss",
                                                                        "train_preterm_loss",
                                                                        "val_loss",
                                                                        "val_preterm_loss"]
                       and "_cutoff_" not in col and (col.endswith(f"_{agg}_tta") or col.endswith(f"_{agg}_no-tta"))]
    
        fig, ax = plt.subplots(figsize=(12, 6))
    
        ax.plot(metrics_df["epoch"], metrics_df["train_loss"],
                label="Train Total Loss", color="tab:blue")
        ax.plot(metrics_df["epoch"], metrics_df["train_preterm_loss"],
                label="Train Preterm Loss", color="tab:blue", linestyle="--")
    
        ax.plot(metrics_df["epoch"], metrics_df["val_loss"],
                label="Validation Total Loss", color="tab:orange")
        ax.plot(metrics_df["epoch"], metrics_df["val_preterm_loss"],
                label="Validation Preterm Loss", color="tab:orange", linestyle="--")
    
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][2:]
    
        for i, col in enumerate(metric_cols):
            color = colors[i // 2 % len(colors)]
            is_tta = col.endswith("_tta") and not col.endswith("_no-tta")
            linestyle = "--" if is_tta else "-"
            label = col.removesuffix(f"_{agg}_tta").removesuffix(f"_{agg}_no-tta")
            if is_tta:
                label += " (TTA)"
        
            ax.plot(metrics_df["epoch"], metrics_df[col], label=label, color=color, linestyle=linestyle)
    
        ax.set_title(f"Fold {self.fold} - GA {cutoff} - {agg.capitalize()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        fig.savefig(self.plot_path / f"Fold {self.fold} - GA {cutoff} - {agg.capitalize()}.png", dpi=300)
        plt.close(fig)
         
        
    def get_best_epoch(self):
        return {str(cutoff): str(self.best_score[str(cutoff)]['epoch']).zfill(3) for cutoff in self.cutoffs}


    def log_fold(self, cutoff):
        df = pl.concat(self.dfs[str(cutoff)])
        best = self.best_score[str(cutoff)]

        patient_df = (patient_predictions(df, best['agg'], best['tta'], best['threshold'])
                      .with_columns([pl.lit(self.fold).alias("fold"),
                                     pl.lit(best['agg']).alias('agg'),
                                     pl.lit(best['tta']).alias('tta'),
                                     pl.lit(best["threshold"]).alias('threshold')])
                      .select(["CPR_CHILD", "preds", "label", "treatment", "fold", "agg", "tta", "threshold",
                               "preds_binary", "preds_normalized"]))

        self.test_preds[str(cutoff)].append(patient_df)
        pl.concat(self.test_preds[str(cutoff)]).write_parquet(self.pred_path / f"predictions_{cutoff}.parquet")
    
    def reset(self):
        for cutoff in self.cutoffs:
            self.best_predictions[str(cutoff)].write_parquet(self.pred_path / f"validation/predictions_{cutoff}_fold_{self.fold}.parquet")        
            self.best_df[str(cutoff)].write_parquet(self.pred_path / f"validation/image_preds_{cutoff}_fold_{self.fold}.parquet")
        self.best_score = {str(c): {'score': -float("inf"),
                                    'threshold': None,
                                    'epoch': -1,
                                    'agg': None,
                                    'tta': None} for c in self.cutoffs}

        self.dfs = {str(c): [] for c in self.cutoffs}
        self.best_predictions = {str(c): None for c in self.cutoffs}
        self.epoch = 0
        self.fold += 1
        self.best_df = {}
    
    def get_val_preds_and_labels(self, cutoff, population, subgroup=None):
        temp_dfs = []

        for path in (self.pred_path / "validation").glob(f"predictions_{cutoff}*"):
            temp_df = pl.read_parquet(path)
            temp_df = temp_df.with_columns((pl.col("preds") - pl.col("threshold") + 0.50).alias("preds"))
            if population == 'Non-treated':
                temp_df = temp_df.filter(~pl.col("treatment"))
            if subgroup is not None:
                temp_df = temp_df.filter(pl.col(subgroup))
            temp_dfs.append(temp_df)
        df = pl.concat(temp_dfs)
        preds = torch.tensor(df["preds"].to_numpy(), dtype=torch.float32).to(self.device)
        labels = torch.tensor(df["label"].to_numpy(),dtype=torch.int32).to(self.device)
        preds = preds.clamp(0, 1)
        return preds, labels
        
    def log_model(self):
        for cutoff in self.cutoffs:
            if len(self.test_preds[str(cutoff)]) != 0:
                df = pl.concat(self.test_preds[str(cutoff)])
                df.write_parquet(self.pred_path / f"predictions_{cutoff}.parquet")
            else:
                df = pl.read_parquet(self.pred_path / f"predictions_{cutoff}.parquet")
            report = []
            if self.test_subgroup is not None:
                report.append(f"Test folds only hold rows where {self.test_subgroup} is True\n\n")
            for pop, pop_df in [("All Births", df), ("Non-treated", df.filter(~pl.col("treatment")))]:
                #Get fold variability
                for fold in range(df['fold'].max()):
                    fold_df = pop_df.filter(pl.col("fold") == fold)
                    
                    preds = torch.tensor(fold_df["preds"].to_numpy(), dtype=torch.float32).to(self.device)
                    labels = torch.tensor(fold_df["label"].to_numpy(),dtype=torch.int32).to(self.device)
    
                    metrics = self.get_metrics()
                    
                    if fold == 0:
                        results = {name: [] for name in metrics.keys()}
                        results['SensAtSpec_cutoff'] = []
                    
                    for name, metric in metrics.items():
                        output = metric(preds, labels)
        
                        if isinstance(output, tuple):
                            results[name].append(output[0].item())
                            results[f"{name}_cutoff"].append(output[1].item())
                        else:
                            results[name].append(output.item())
        
                fig, ax = plt.subplots(figsize=(max(10, len(results) * 1.2), 6))
    
                for i, (name, values) in enumerate(results.items()):
                    values = np.asarray(values, dtype=float)
    
                    if len(values) == 0:
                        continue
                    ymin, ymax = values.min(), values.max()
                    mean = values.mean()
                    cap_width = 0.1
                    color = f"C{i}"
                    
                    ax.vlines(i, ymin, ymax, linewidth=2, color=color)
                    ax.hlines([ymin, ymax], i - cap_width, i + cap_width, linewidth=2, color=color)
                    ax.scatter(i, mean, marker="o", s=60, color=color, zorder=3)
    
                ax.set_xticks(range(len(results)))
                ax.set_xticklabels(results.keys(), rotation=45, ha="right")
                ax.set_ylabel("Metric value")
                ax.set_title(f"Fold variability — GA {cutoff}")
                ax.grid(axis="y", alpha=0.3)
    
                plt.tight_layout()
                fig.savefig(self.save_path / f"Fold variability - {pop} - GA {cutoff}.png", dpi=300)
                plt.close(fig)
        
                #Get overall metrics and 95% CI
        
                results[pop] = {}
                n_preterm = int(pop_df["label"].sum())
                n_non_preterm = len(pop_df) - n_preterm
                
                preds = torch.tensor(pop_df["preds_normalized"].to_numpy(), dtype=torch.float32).to(self.device)
                preds_binary = torch.tensor(pop_df["preds_binary"].to_numpy(), dtype=torch.float32).to(self.device)
                labels = torch.tensor(pop_df["label"].to_numpy(), dtype=torch.int32).to(self.device)
                
                metrics = self.get_metrics()
                rng = np.random.default_rng()
                val_preds, val_labels = self.get_val_preds_and_labels(cutoff, pop)
                subgroup_val = {subgroup: self.get_val_preds_and_labels(cutoff, pop, subgroup) for subgroup in self.subgroups}

                for name, metric in metrics.items():
                    results[pop][name] = {}
                    # Use binary predictions for sensitivity and specificity
                    is_binary_metric = name in ["Sensitivity", "Specificity"]
    
                    metric_preds = preds_binary if is_binary_metric else preds
                    metric.reset()
                    output = metric(metric_preds, labels)

                    metric.reset()
                    val_output = metric(val_preds, val_labels)

                    if isinstance(output, tuple):
                        results[pop][name]['mean'] = output[0].item()
                        results[pop][name]['val'] = val_output[0].item()
                    else:
                        results[pop][name]['mean'] = output.item()
                        results[pop][name]['val'] = val_output.item()

                    for subgroup, (subgroup_preds, subgroup_labels) in subgroup_val.items():
                        metric.reset()
                        subgroup_output = metric(subgroup_preds, subgroup_labels)
                        if isinstance(subgroup_output, tuple):
                            results[pop][name][f'val_{subgroup}'] = subgroup_output[0].item()
                        else:
                            results[pop][name][f'val_{subgroup}'] = subgroup_output.item()
                    # Bootstrap 95% CI
                    boot_values = []
                
                    for _ in range(self.n_bootstrap):
                        idx = rng.integers(0, len(pop_df), size=len(pop_df))
                        boot_labels = labels[idx]
        
                        if torch.unique(boot_labels).numel() < 2:
                            continue
                
                        metric.reset()
                        boot_output = metric(metric_preds[idx], boot_labels)
                
                        value = boot_output[0] if isinstance(boot_output, tuple) else boot_output
                        boot_values.append(value.item())
                
    
                    results[pop][name]['lower'] = np.percentile(boot_values, 2.5)
                    results[pop][name]['upper'] = np.percentile(boot_values, 97.5)
                
                                
                agg_summary = (df.select(["fold", "agg", "tta"]).unique().group_by(["agg", "tta"])
                               .agg(pl.len().alias("n_folds")).sort(["agg", "tta"]))
                
                agg_summary = ", ".join(f"{row['agg']} {row['tta']} ({row['n_folds']} fold{'s' if row['n_folds'] != 1 else ''})"
                                        for row in agg_summary.iter_rows(named=True))
            
                report.append(f"-- {pop} --\n"
                              f"\tPreterm births: {n_preterm}\n"
                              f"\tNon-preterm births: {n_non_preterm}\n"
                              f"\tTotal births: {len(pop_df)}\n"
                              f"\n"
                              f"\tAggregation method: {agg_summary}\n"
                              f"\n"
                              f"\tMetrics:\n")
            
                for name, result in results[pop].items():
                    validation = f"Validation: {result['val']:.4f}"
                    for subgroup in self.subgroups:
                        validation += f", {subgroup}: {result[f'val_{subgroup}']:.4f}"
                    report.append(f"\t{name}: {result['mean']:.4f} "
                                  f"(95% CI: {result['lower']:.4f}–{result['upper']:.4f}) "
                                  f"({validation})\n")
                report.append('\n\n')
    
            report = "".join(report)
                
            with open(self.save_path / f"GA_{cutoff}.txt", "w") as f:
                f.write(report)
                            
                    
    def get_metrics(self, t=0.5):
        return binary_metrics(self.device, t)
    
    def restart(self):
        max_folds = []
    
        for cutoff in self.cutoffs:
            path = self.pred_path / f"predictions_{cutoff}.parquet"

            if not path.exists():
                return

            df = pl.read_parquet(path)
            if df.is_empty():
                return

            max_folds.append(df["fold"].max())
    
        self.fold = min(max_folds) + 1
        
        