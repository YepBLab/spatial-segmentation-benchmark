#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from segbench.io import load_yaml, read_rois
from segbench.metrics import classify_errors, evaluate_pair
from segbench.stratify import (
    add_local_density,
    cluster_bootstrap_mean,
    quantile_labels,
    robust_snr,
)


CORE_METRICS = [
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


def add_ci_rows(
    table: pd.DataFrame,
    *,
    group_cols: list[str],
    value_cols: list[str],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in table.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for offset, metric in enumerate(value_cols):
            estimate, low, high = cluster_bootstrap_mean(
                group,
                value_col=metric,
                cluster_col="region",
                iterations=iterations,
                seed=seed + offset,
            )
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_regions": int(group["region"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve()
    config = load_yaml(args.config)
    registry = load_yaml(args.registry)
    metrics_dir = project / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    prepared_root = project / "prepared" / "manual_rois"
    rois = read_rois(config)
    thresholds = config["thresholds"]
    tolerances = config["boundary_tolerances_um"]
    pixel_size = float(config["pixel_size_um"])
    exclude_border = bool(config["exclude_border_objects"])

    roi_metadata: dict[str, dict[str, object]] = {}
    for roi in rois:
        manual = tifffile.imread(roi.label_path).squeeze()
        image = tifffile.imread(roi.image_path)
        snr = robust_snr(image, manual > 0)
        roi_metadata[roi.region] = {
            "region": roi.region,
            "tissue": roi.tissue,
            "medullary_overlap_fraction": roi.medullary_overlap_fraction,
            "snr_ch0_dapi": snr[0] if snr else np.nan,
            "snr_ch2_18s": snr[1] if len(snr) > 1 else np.nan,
            "snr_composite": float(np.nanmean(snr[:2])) if snr else np.nan,
        }
    roi_meta = pd.DataFrame(roi_metadata.values())
    roi_meta["snr_bin"] = quantile_labels(
        roi_meta["snr_composite"],
        ["low", "mid", "high"],
    ).astype(str)
    roi_meta.to_csv(metrics_dir / "roi_strata.csv", index=False)
    roi_meta_lookup = roi_meta.set_index("region").to_dict(orient="index")

    summary_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    boundary_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    gt_attribute_rows: list[dict[str, object]] = []

    for roi in rois:
        manual = tifffile.imread(roi.label_path).squeeze()
        gt_recorded = False
        for model_key, model in registry["models"].items():
            pred_path = prepared_root / model_key / f"{roi.region}_mask.tif"
            pred = tifffile.imread(pred_path).squeeze()
            evaluated = evaluate_pair(
                manual,
                pred,
                thresholds=thresholds,
                boundary_tolerances_um=tolerances,
                pixel_size_um=pixel_size,
                exclude_border=exclude_border,
            )
            meta = roi_meta_lookup[roi.region]
            summary_rows.append(
                {
                    "region": roi.region,
                    "model_key": model_key,
                    "display": model["display"],
                    **meta,
                    **evaluated["summary"],
                }
            )
            threshold_rows.extend(
                {
                    "region": roi.region,
                    "model_key": model_key,
                    "display": model["display"],
                    **row,
                }
                for row in evaluated["threshold_rows"]
            )
            boundary_rows.extend(
                {
                    "region": roi.region,
                    "model_key": model_key,
                    "display": model["display"],
                    **row,
                }
                for row in evaluated["boundary_rows"]
            )
            object_rows.extend(
                {
                    "region": roi.region,
                    "model_key": model_key,
                    "display": model["display"],
                    "tissue": meta["tissue"],
                    "snr_bin": meta["snr_bin"],
                    "snr_composite": meta["snr_composite"],
                    **row,
                }
                for row in evaluated["object_rows"]
            )
            if not gt_recorded:
                gt_attribute_rows.extend(
                    {
                        "region": roi.region,
                        "gt_label": row["gt_label"],
                        "gt_area_px": row["gt_area_px"],
                        "gt_area_um2": row["gt_area_um2"],
                        "gt_centroid_x_px": row["gt_centroid_x_px"],
                        "gt_centroid_y_px": row["gt_centroid_y_px"],
                    }
                    for row in evaluated["object_rows"]
                )
                gt_recorded = True

            for overlap_fraction in [
                float(config["error_overlap_fraction"]),
                float(config["error_overlap_sensitivity_fraction"]),
            ]:
                for event in classify_errors(
                    evaluated["gt_clean"],
                    evaluated["pred_clean"],
                    overlap_fraction=overlap_fraction,
                ):
                    x0, y0, x1, y1 = event.pop("bbox")
                    error_rows.append(
                        {
                            "region": roi.region,
                            "model_key": model_key,
                            "display": model["display"],
                            "overlap_fraction": overlap_fraction,
                            "error_type": event["error_type"],
                            "gt_labels_json": json.dumps(event["gt_labels"]),
                            "pred_labels_json": json.dumps(event["pred_labels"]),
                            "severity": event["severity"],
                            "bbox_x0": x0,
                            "bbox_y0": y0,
                            "bbox_x1": x1,
                            "bbox_y1": y1,
                            "global_x0": roi.x0 + x0,
                            "global_y0": roi.y0 + y0,
                            "global_x1": roi.x0 + x1,
                            "global_y1": roi.y0 + y1,
                        }
                    )

    summary = pd.DataFrame(summary_rows)
    threshold_table = pd.DataFrame(threshold_rows)
    boundary = pd.DataFrame(boundary_rows)
    objects = pd.DataFrame(object_rows)
    errors = pd.DataFrame(error_rows)
    gt_attributes = add_local_density(
        pd.DataFrame(gt_attribute_rows).drop_duplicates(["region", "gt_label"]),
        pixel_size_um=pixel_size,
        radius_um=float(config["local_density_radius_um"]),
    )
    gt_attributes["size_bin"] = quantile_labels(
        gt_attributes["gt_area_um2"],
        ["Q1 small", "Q2", "Q3", "Q4 large"],
    ).astype(str)
    gt_attributes["density_bin"] = quantile_labels(
        gt_attributes["local_neighbors_20um"],
        ["low", "mid", "high"],
    ).astype(str)
    objects = objects.merge(
        gt_attributes[
            ["region", "gt_label", "size_bin", "density_bin", "local_neighbors_20um"]
        ],
        on=["region", "gt_label"],
        how="left",
        validate="many_to_one",
    )

    summary.to_csv(metrics_dir / "manual_roi_metrics.csv", index=False)
    threshold_table.to_csv(metrics_dir / "sa_threshold_curve_by_roi.csv", index=False)
    boundary.to_parquet(metrics_dir / "matched_boundary_metrics.parquet", index=False)
    objects.to_parquet(metrics_dir / "object_outcomes.parquet", index=False)
    gt_attributes.to_parquet(metrics_dir / "gt_attributes.parquet", index=False)
    errors.to_parquet(metrics_dir / "error_events.parquet", index=False)

    iterations = int(config["bootstrap_iterations"])
    seed = int(config["bootstrap_seed"])
    macro = add_ci_rows(
        summary,
        group_cols=["model_key", "display"],
        value_cols=CORE_METRICS,
        iterations=iterations,
        seed=seed,
    )
    macro.to_csv(metrics_dir / "manual_macro_summary.csv", index=False)
    threshold_macro = add_ci_rows(
        threshold_table,
        group_cols=["model_key", "display", "threshold"],
        value_cols=["sa", "precision", "recall", "f1"],
        iterations=iterations,
        seed=seed + 100,
    )
    threshold_macro.to_csv(metrics_dir / "sa_threshold_curve_macro.csv", index=False)

    boundary_by_roi = (
        boundary.groupby(
            ["region", "model_key", "display", "tolerance_um"],
            observed=True,
        )[["boundary_precision", "boundary_recall", "boundary_f1", "nsd"]]
        .mean()
        .reset_index()
    )
    boundary_macro = add_ci_rows(
        boundary_by_roi,
        group_cols=["model_key", "display", "tolerance_um"],
        value_cols=["boundary_precision", "boundary_recall", "boundary_f1", "nsd"],
        iterations=iterations,
        seed=seed + 200,
    )
    boundary_macro.to_csv(metrics_dir / "boundary_sensitivity_macro.csv", index=False)

    object_strata = (
        objects.groupby(["model_key", "display", "size_bin"], observed=True)
        .agg(
            n_gt=("gt_label", "size"),
            recall_iou50=("detected_iou50", "mean"),
            mean_best_iou=("best_iou", "mean"),
            median_best_iou=("best_iou", "median"),
        )
        .reset_index()
    )
    density_strata = (
        objects.groupby(["model_key", "display", "density_bin"], observed=True)
        .agg(
            n_gt=("gt_label", "size"),
            recall_iou50=("detected_iou50", "mean"),
            mean_best_iou=("best_iou", "mean"),
            median_best_iou=("best_iou", "median"),
        )
        .reset_index()
    )
    object_strata["stratum_type"] = "gt_size"
    object_strata = object_strata.rename(columns={"size_bin": "stratum"})
    density_strata["stratum_type"] = "gt_density"
    density_strata = density_strata.rename(columns={"density_bin": "stratum"})
    pd.concat([object_strata, density_strata], ignore_index=True).to_csv(
        metrics_dir / "object_stratified_metrics.csv",
        index=False,
    )
    roi_strata = (
        summary.groupby(["model_key", "display", "tissue", "snr_bin"], observed=True)
        .agg(
            n_regions=("region", "nunique"),
            n_gt=("gt_instances", "sum"),
            mSA_50_95=("mSA_50_95", "mean"),
            SA50=("SA50", "mean"),
            f1_iou50=("f1_iou50", "mean"),
            PQ=("PQ", "mean"),
        )
        .reset_index()
    )
    roi_strata.to_csv(metrics_dir / "roi_stratified_metrics.csv", index=False)

    error_counts = (
        errors[errors["overlap_fraction"] == float(config["error_overlap_fraction"])]
        .groupby(["region", "model_key", "display", "error_type"], observed=True)
        .size()
        .rename("event_count")
        .reset_index()
        .merge(
            summary[["region", "model_key", "gt_instances", "pred_instances"]],
            on=["region", "model_key"],
            how="left",
            validate="many_to_one",
        )
    )
    error_counts["events_per_100_gt"] = (
        100.0 * error_counts["event_count"] / error_counts["gt_instances"].clip(lower=1)
    )
    error_counts.to_csv(metrics_dir / "error_summary_by_roi.csv", index=False)

    result = {
        "status": "PASS",
        "regions": int(summary["region"].nunique()),
        "models": int(summary["model_key"].nunique()),
        "roi_model_evaluations": int(len(summary)),
        "gt_instances_after_border_filter": int(gt_attributes.shape[0]),
        "threshold_rows": int(len(threshold_table)),
        "matched_boundary_rows": int(len(boundary)),
        "error_events": int(len(errors)),
        "boundary_mode": "provisional sensitivity only",
    }
    (metrics_dir / "manual_metrics_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
