#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from segbench.io import load_yaml


METRICS = [
    "mSA_50_95",
    "SA50",
    "SA75",
    "precision_iou50",
    "recall_iou50",
    "f1_iou50",
    "matched_iou_mean",
    "matched_dice_mean",
    "DQ",
    "SQ",
    "PQ",
    "AJI_plus",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    return parser.parse_args()


def paired_bootstrap(values: np.ndarray, iterations: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve()
    config = load_yaml(args.config)
    registry = load_yaml(args.registry)
    metrics_dir = project / "metrics"
    table = pd.read_csv(metrics_dir / "manual_roi_metrics.csv")
    iterations = int(config["bootstrap_iterations"])
    seed = int(config["bootstrap_seed"])
    model_keys = list(registry["models"])
    comparisons = list(combinations(model_keys, 2))
    rows: list[dict[str, object]] = []
    for baseline_key, comparison_key in comparisons:
        for metric_index, metric in enumerate(METRICS):
            pivot = table.pivot(index="region", columns="model_key", values=metric)
            paired = (pivot[comparison_key] - pivot[baseline_key]).dropna()
            estimate, low, high = paired_bootstrap(
                paired.to_numpy(),
                iterations,
                seed + metric_index,
            )
            rows.append(
                {
                    "comparison": f"{comparison_key}_minus_{baseline_key}",
                    "comparison_model": comparison_key,
                    "baseline_model": baseline_key,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_regions": int(len(paired)),
                    "regions_positive": int((paired > 0).sum()),
                    "regions_negative": int((paired < 0).sum()),
                }
            )
    paired_table = pd.DataFrame(rows)
    paired_table.to_csv(metrics_dir / "paired_model_deltas.csv", index=False)

    strata_rows: list[pd.DataFrame] = []
    for stratum_type, stratum_col in [("tissue", "tissue"), ("snr", "snr_bin")]:
        grouped = (
            table.groupby(["model_key", "display", stratum_col], observed=True)
            .agg(
                n_regions=("region", "nunique"),
                n_gt=("gt_instances", "sum"),
                mSA_50_95=("mSA_50_95", "mean"),
                SA50=("SA50", "mean"),
                SA75=("SA75", "mean"),
                f1_iou50=("f1_iou50", "mean"),
                PQ=("PQ", "mean"),
                AJI_plus=("AJI_plus", "mean"),
            )
            .reset_index()
            .rename(columns={stratum_col: "stratum"})
        )
        grouped["stratum_type"] = stratum_type
        strata_rows.append(grouped)
    pd.concat(strata_rows, ignore_index=True).to_csv(
        metrics_dir / "roi_stratified_summary.csv",
        index=False,
    )
    result = {
        "status": "PASS",
        "paired_comparisons": int(len(paired_table)),
        "model_pairs": [f"{right}_minus_{left}" for left, right in comparisons],
        "metrics": METRICS,
    }
    (metrics_dir / "stratified_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
