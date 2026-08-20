#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from skimage.segmentation import find_boundaries

from segbench.io import load_yaml, read_rois
from segbench.metrics import prepare_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--manual-only", action="store_true")
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.8,
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#f8fafc",
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.facecolor": "#f8fafc",
            "savefig.bbox": "tight",
        }
    )


def model_meta(registry: dict) -> tuple[list[str], dict[str, str], dict[str, str]]:
    keys = list(registry["models"])
    names = {key: registry["models"][key]["display"] for key in keys}
    colors = {key: registry["models"][key]["color"] for key in keys}
    return keys, names, colors


def save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=220)
    plt.close(figure)


def draw_scorecard(
    macro: pd.DataFrame,
    keys: list[str],
    names: dict[str, str],
    colors: dict[str, str],
    output: Path,
) -> None:
    metrics = ["mSA_50_95", "SA50", "SA75", "f1_iou50", "PQ", "AJI_plus"]
    labels = ["mSA@[.50:.95]", "SA50", "SA75", "F1 @ IoU .50", "PQ", "AJI+"]
    pivot = macro[macro["metric"].isin(metrics)].pivot(
        index="model_key", columns="metric", values="estimate"
    )
    matrix = pivot.reindex(keys)[metrics].to_numpy()
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(metrics)), labels, rotation=22, ha="right")
    ax.set_yticks(np.arange(len(keys)), [names[k] for k in keys])
    ax.set_title("Manual-ROI accuracy scorecard (in-sample)")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value >= 0.55 else "#0f172a",
                fontweight="bold",
            )
        ax.get_yticklabels()[row].set_color(colors[keys[row]])
        ax.get_yticklabels()[row].set_fontweight("bold")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Score")
    fig.text(
        0.01,
        0.01,
        "Mean across 11 training ROIs; this measures in-sample fit, not generalization.",
        color="#64748b",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, output)


def draw_sa_curve(
    table: pd.DataFrame,
    keys: list[str],
    names: dict[str, str],
    colors: dict[str, str],
    output: Path,
) -> None:
    subset = table[table["metric"] == "sa"]
    fig, ax = plt.subplots(figsize=(9.6, 5.7))
    for key in keys:
        rows = subset[subset["model_key"] == key].sort_values("threshold")
        ax.plot(
            rows["threshold"],
            rows["estimate"],
            marker="o",
            markersize=4,
            linewidth=2.2,
            color=colors[key],
            label=names[key],
        )
        ax.fill_between(
            rows["threshold"],
            rows["ci_low"],
            rows["ci_high"],
            color=colors[key],
            alpha=0.10,
            linewidth=0,
        )
    ax.set(xlabel="IoU threshold", ylabel="Segmentation accuracy (SA)", xlim=(0.49, 0.96), ylim=(0, 1))
    ax.set_title("Segmentation accuracy across IoU thresholds")
    ax.grid(axis="y")
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout()
    save(fig, output)


def draw_detection(
    macro: pd.DataFrame,
    keys: list[str],
    names: dict[str, str],
    colors: dict[str, str],
    output: Path,
) -> None:
    metrics = ["precision_iou50", "recall_iou50", "f1_iou50"]
    labels = ["Precision", "Recall", "F1"]
    pivot = macro[macro["metric"].isin(metrics)].pivot(
        index="model_key", columns="metric", values="estimate"
    )
    fig, ax = plt.subplots(figsize=(10.4, 5.5))
    width = 0.12
    x = np.arange(len(metrics))
    for index, key in enumerate(keys):
        values = pivot.reindex(keys).loc[key, metrics].to_numpy()
        ax.bar(
            x + (index - (len(keys) - 1) / 2) * width,
            values,
            width=width,
            color=colors[key],
            label=names[key],
        )
    ax.set_xticks(x, labels)
    ax.set(ylabel="Score", ylim=(0, 1))
    ax.set_title("Detection metrics at IoU 0.50")
    ax.grid(axis="y")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    save(fig, output)


def draw_boundary(
    boundary: pd.DataFrame,
    keys: list[str],
    names: dict[str, str],
    colors: dict[str, str],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), sharey=True)
    for axis, metric, title in zip(
        axes,
        ["boundary_f1", "nsd"],
        ["Boundary F1", "Normalized Surface Dice"],
    ):
        subset = boundary[boundary["metric"] == metric]
        for key in keys:
            rows = subset[subset["model_key"] == key].sort_values("tolerance_um")
            axis.plot(
                rows["tolerance_um"],
                rows["estimate"],
                marker="o",
                linewidth=2.0,
                color=colors[key],
                label=names[key],
            )
            axis.fill_between(
                rows["tolerance_um"],
                rows["ci_low"],
                rows["ci_high"],
                color=colors[key],
                alpha=0.10,
                linewidth=0,
            )
        axis.set(xlabel="Tolerance (μm)", title=title, ylim=(0, 1))
        axis.grid(axis="y")
    axes[0].set_ylabel("Score")
    axes[1].legend(ncol=2, fontsize=8, loc="lower right")
    fig.suptitle("Boundary tolerance sensitivity (provisional)")
    fig.text(
        0.5,
        0.01,
        "No independent repeat annotations were available to calibrate a single primary tolerance.",
        ha="center",
        color="#64748b",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, output)


def draw_pq(
    macro: pd.DataFrame,
    keys: list[str],
    names: dict[str, str],
    colors: dict[str, str],
    output: Path,
) -> None:
    metrics = ["DQ", "SQ", "PQ"]
    pivot = macro[macro["metric"].isin(metrics)].pivot(
        index="model_key", columns="metric", values="estimate"
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    for axis, metric in zip(axes, metrics):
        values = pivot.reindex(keys)[metric]
        bars = axis.barh(
            np.arange(len(keys)),
            values,
            color=[colors[key] for key in keys],
            height=0.72,
        )
        axis.set(xlim=(0, 1), xlabel="Score", title=metric)
        axis.grid(axis="x")
        axis.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, fontsize=8)
        axis.set_yticks(np.arange(len(keys)), [names[k] for k in keys])
        axis.invert_yaxis()
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    fig.suptitle("Panoptic Quality decomposition: PQ = DQ × SQ")
    fig.tight_layout()
    save(fig, output)


def draw_errors(
    error_table: pd.DataFrame,
    keys: list[str],
    names: dict[str, str],
    colors: dict[str, str],
    output: Path,
) -> None:
    error_order = ["miss", "spurious", "split", "merge", "poor_overlap", "complex"]
    grouped = (
        error_table.groupby(["model_key", "error_type"], observed=True)["events_per_100_gt"]
        .mean()
        .unstack(fill_value=0)
        .reindex(keys)
        .reindex(columns=error_order, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    bottom = np.zeros(len(keys))
    error_colors = ["#dc2626", "#f59e0b", "#7c3aed", "#2563eb", "#64748b", "#111827"]
    for error_type, color in zip(error_order, error_colors):
        values = grouped[error_type].to_numpy()
        ax.bar(
            np.arange(len(keys)),
            values,
            bottom=bottom,
            label=error_type.replace("_", " "),
            color=color,
        )
        bottom += values
    ax.set_xticks(np.arange(len(keys)), [names[k] for k in keys], rotation=20, ha="right")
    for tick, key in zip(ax.get_xticklabels(), keys):
        tick.set_color(colors[key])
        tick.set_fontweight("bold")
    ax.set(ylabel="Events per 100 manual instances")
    ax.set_title("Error-type spectrum at 25% overlap graph threshold")
    ax.grid(axis="y")
    ax.legend(ncol=3)
    fig.tight_layout()
    save(fig, output)


def draw_strata(
    table: pd.DataFrame,
    keys: list[str],
    names: dict[str, str],
    colors: dict[str, str],
    output: Path,
) -> None:
    orders = {
        "gt_size": ["Q1 small", "Q2", "Q3", "Q4 large"],
        "gt_density": ["low", "mid", "high"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.1), sharey=True)
    for axis, stratum_type, title in zip(
        axes,
        ["gt_size", "gt_density"],
        ["Recall by manual-cell size", "Recall by local density"],
    ):
        subset = table[table["stratum_type"] == stratum_type]
        order = orders[stratum_type]
        for key in keys:
            rows = subset[subset["model_key"] == key].set_index("stratum").reindex(order)
            axis.plot(
                order,
                rows["recall_iou50"],
                marker="o",
                linewidth=2.0,
                color=colors[key],
                label=names[key],
            )
        axis.set(title=title, xlabel="Stratum", ylim=(0, 1))
        axis.grid(axis="y")
        axis.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Recall @ IoU 0.50")
    axes[1].legend(ncol=2, fontsize=8, loc="lower right")
    fig.suptitle("Manual-object stratification")
    fig.tight_layout()
    save(fig, output)


def _format_value(metric: str, value: float) -> str:
    if metric == "cells_per_mm2":
        return f"{value / 1000:.1f}k"
    if metric == "assignment_fraction":
        return f"{100 * value:.1f}%"
    return f"{value:.0f}"


def _utility_values(
    utility: pd.DataFrame,
    scan: pd.DataFrame,
    *,
    scope: str,
    metric: str,
    keys: list[str],
) -> np.ndarray:
    if metric == "assignment_fraction":
        column = (
            "assigned_fraction_q20_global"
            if scope == "global"
            else "assigned_fraction_q20_medullary"
        )
        return scan.set_index("model_key").reindex(keys)[column].to_numpy(dtype=float)
    return (
        utility[utility["scope"] == scope]
        .set_index("model_key")
        .reindex(keys)[metric]
        .to_numpy(dtype=float)
    )


def draw_utility_scope(
    utility: pd.DataFrame,
    scan: pd.DataFrame,
    keys: list[str],
    names: dict[str, str],
    colors: dict[str, str],
    *,
    scope: str,
    output: Path,
) -> None:
    metrics = [
        ("cells_per_mm2", "Cell density", "Cells / mm²"),
        ("transcripts_per_cell_median", "Median transcripts per cell", "Transcripts"),
        ("genes_per_cell_median", "Median genes per cell", "Genes"),
        ("assignment_fraction", "Q20 transcript assignment", "Assigned fraction"),
    ]
    short_names = {key: names[key] for key in keys}
    x = np.arange(len(keys))
    figure, axes = plt.subplots(2, 2, figsize=(13.4, 8.4))
    for axis, (metric, title, ylabel) in zip(axes.ravel(), metrics):
        values = _utility_values(
            utility,
            scan,
            scope=scope,
            metric=metric,
            keys=keys,
        )
        bars = axis.bar(
            x,
            values,
            color=[colors[key] for key in keys],
            edgecolor="#ffffff",
            linewidth=0.8,
            width=0.72,
        )
        comparison_values = np.concatenate(
            [
                _utility_values(
                    utility,
                    scan,
                    scope=scope_name,
                    metric=metric,
                    keys=keys,
                )
                for scope_name in ["global", "medullary"]
            ]
        )
        upper = 1.0 if metric == "assignment_fraction" else float(comparison_values.max() * 1.18)
        axis.set_ylim(0, upper)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y")
        axis.set_xticks(
            x,
            [short_names.get(key, names[key]) for key in keys],
            fontsize=8.5,
        )
        axis.bar_label(
            bars,
            labels=[_format_value(metric, value) for value in values],
            padding=3,
            fontsize=8,
            color="#334155",
        )
        if metric == "assignment_fraction":
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    scope_title = "GLOBAL — full tissue" if scope == "global" else "MEDULLARY — supplied GeoJSON"
    scope_color = "#153b5b" if scope == "global" else "#087d75"
    figure.suptitle(scope_title, color=scope_color, fontweight="bold", fontsize=18)
    figure.text(
        0.5,
        0.01,
        "Each figure contains one spatial scope only; identical metric panels use the same y-axis limits.",
        ha="center",
        color="#64748b",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    save(figure, output)


def draw_utility_contrast(
    utility: pd.DataFrame,
    scan: pd.DataFrame,
    keys: list[str],
    names: dict[str, str],
    colors: dict[str, str],
    output: Path,
) -> None:
    metrics = [
        ("cells_per_mm2", "Cell density ratio", "Medullary / global", "ratio"),
        (
            "transcripts_per_cell_median",
            "Median transcripts per cell",
            "Medullary − global",
            "delta",
        ),
        ("genes_per_cell_median", "Median genes per cell", "Medullary − global", "delta"),
        (
            "assignment_fraction",
            "Q20 transcript assignment",
            "Medullary − global (percentage points)",
            "percentage_points",
        ),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))
    y = np.arange(len(keys))
    for axis, (metric, title, xlabel, mode) in zip(axes.ravel(), metrics):
        global_values = _utility_values(
            utility,
            scan,
            scope="global",
            metric=metric,
            keys=keys,
        )
        medullary_values = _utility_values(
            utility,
            scan,
            scope="medullary",
            metric=metric,
            keys=keys,
        )
        if mode == "ratio":
            values = medullary_values / global_values
            reference = 1.0
            labels = [f"{value:.2f}×" for value in values]
        elif mode == "percentage_points":
            values = 100.0 * (medullary_values - global_values)
            reference = 0.0
            labels = [f"{value:+.1f} pp" for value in values]
        else:
            values = medullary_values - global_values
            reference = 0.0
            labels = [f"{value:+.0f}" for value in values]
        left = np.minimum(values, reference)
        widths = np.abs(values - reference)
        bars = axis.barh(
            y,
            widths,
            left=left,
            color=[colors[key] for key in keys],
            height=0.66,
        )
        axis.axvline(reference, color="#334155", linewidth=1.2)
        lower = float(min(values.min(), reference))
        upper = float(max(values.max(), reference))
        span = max(upper - lower, 1.0)
        axis.set_xlim(lower - 0.16 * span, upper + 0.16 * span)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.grid(axis="x")
        axis.set_yticks(y, [names[key] for key in keys], fontsize=8.5)
        axis.invert_yaxis()
        for bar, value, label in zip(bars, values, labels):
            x_value = value if mode != "ratio" else value
            offset = 4 if x_value >= reference else -4
            axis.annotate(
                label,
                (x_value, bar.get_y() + bar.get_height() / 2),
                xytext=(offset, 0),
                textcoords="offset points",
                ha="left" if offset > 0 else "right",
                va="center",
                fontsize=8,
                color="#334155",
            )
    figure.suptitle("Medullary versus global utility contrast", fontsize=17, fontweight="bold")
    figure.text(
        0.5,
        0.01,
        "These contrasts describe output yield and partitioning, not segmentation accuracy.",
        ha="center",
        color="#64748b",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    save(figure, output)


def normalize(channel: np.ndarray) -> np.ndarray:
    positive = channel[channel > 0]
    if positive.size == 0:
        return np.zeros(channel.shape, dtype=np.float32)
    low, high = np.percentile(positive, [1, 99.5])
    if high <= low:
        return np.zeros(channel.shape, dtype=np.float32)
    return np.clip((channel.astype(np.float32) - low) / (high - low), 0, 1) ** 0.75


def rgb_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        dapi = normalize(image)
        rna = np.zeros_like(dapi)
    else:
        dapi = normalize(image[0])
        rna = normalize(image[min(1, image.shape[0] - 1)])
    return np.stack((rna, 0.48 * rna, dapi), axis=-1)


def draw_failure_montage(
    selected_path: Path,
    project: Path,
    config: dict,
    registry: dict,
    output: Path,
) -> None:
    selected = pd.read_csv(selected_path)
    selected = selected[
        (selected["selection_type"] == "top_severity")
        & selected["error_type"].isin(["miss", "spurious", "split", "merge"])
    ]
    rois = {roi.region: roi for roi in read_rois(config)}
    keys, names, _ = model_meta(registry)
    error_types = ["miss", "spurious", "split", "merge"]
    fig, axes = plt.subplots(len(keys), len(error_types), figsize=(12.2, 16.5))
    for row_index, key in enumerate(keys):
        for col_index, error_type in enumerate(error_types):
            axis = axes[row_index, col_index]
            rows = selected[
                (selected["model_key"] == key) & (selected["error_type"] == error_type)
            ]
            if rows.empty:
                axis.text(0.5, 0.5, "No event", ha="center", va="center", color="#64748b")
                axis.axis("off")
                continue
            event = rows.sort_values("severity", ascending=False).iloc[0]
            roi = rois[str(event["region"])]
            image = tifffile.imread(roi.image_path)
            manual = tifffile.imread(roi.label_path).squeeze()
            pred = tifffile.imread(
                project / "prepared" / "manual_rois" / key / f"{roi.region}_mask.tif"
            ).squeeze()
            gt, prediction, _ = prepare_masks(
                manual,
                pred,
                exclude_border=bool(config["exclude_border_objects"]),
            )
            pad = 35
            x0 = max(0, int(event["bbox_x0"]) - pad)
            y0 = max(0, int(event["bbox_y0"]) - pad)
            x1 = min(gt.shape[1], int(event["bbox_x1"]) + pad)
            y1 = min(gt.shape[0], int(event["bbox_y1"]) + pad)
            panel = rgb_image(image)[y0:y1, x0:x1].copy()
            manual_boundary = find_boundaries(gt[y0:y1, x0:x1], mode="inner")
            pred_boundary = find_boundaries(prediction[y0:y1, x0:x1], mode="inner")
            panel[manual_boundary] = (0.05, 0.95, 0.95)
            panel[pred_boundary] = (1.0, 0.15, 0.60)
            axis.imshow(panel)
            axis.set_title(f"{error_type} · {roi.region}", fontsize=9)
            axis.axis("off")
            if col_index == 0:
                axis.text(
                    -0.06,
                    0.5,
                    names[key],
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                    fontweight="bold",
                )
    fig.suptitle("Representative highest-severity errors")
    fig.text(
        0.5,
        0.01,
        "Manual boundary: cyan · prediction boundary: magenta · event graph threshold: 25% overlap",
        ha="center",
        color="#64748b",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.025, 1, 0.975))
    save(fig, output)


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve()
    config = load_yaml(args.config)
    registry = load_yaml(args.registry)
    metrics_dir = project / "metrics"
    figures = project / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    set_style()
    keys, names, colors = model_meta(registry)

    macro = pd.read_csv(metrics_dir / "manual_macro_summary.csv")
    curve = pd.read_csv(metrics_dir / "sa_threshold_curve_macro.csv")
    boundary = pd.read_csv(metrics_dir / "boundary_sensitivity_macro.csv")
    errors = pd.read_csv(metrics_dir / "error_summary_by_roi.csv")
    strata = pd.read_csv(metrics_dir / "object_stratified_metrics.csv")
    outputs = {
        "scorecard": figures / "01_manual_accuracy_scorecard.png",
        "sa_curve": figures / "02_sa_threshold_curve.png",
        "detection": figures / "03_detection_iou50.png",
        "boundary": figures / "04_boundary_tolerance_sensitivity.png",
        "pq": figures / "05_pq_decomposition.png",
        "errors": figures / "06_error_spectrum.png",
        "strata": figures / "07_size_density_strata.png",
        "utility_global": figures / "08a_global_utility.png",
        "utility_medullary": figures / "08b_medullary_utility.png",
        "utility_contrast": figures / "08c_scope_utility_contrast.png",
        "failures": figures / "09_representative_failures.png",
    }
    draw_scorecard(macro, keys, names, colors, outputs["scorecard"])
    draw_sa_curve(curve, keys, names, colors, outputs["sa_curve"])
    draw_detection(macro, keys, names, colors, outputs["detection"])
    draw_boundary(boundary, keys, names, colors, outputs["boundary"])
    draw_pq(macro, keys, names, colors, outputs["pq"])
    draw_errors(errors, keys, names, colors, outputs["errors"])
    draw_strata(strata, keys, names, colors, outputs["strata"])
    if args.manual_only:
        for key in ["utility_global", "utility_medullary", "utility_contrast"]:
            outputs.pop(key)
    else:
        utility = pd.read_csv(metrics_dir / "global_medullary_utility.csv")
        scan_path = metrics_dir / "transcript_assignment_scan.csv"
        scan = pd.read_csv(scan_path) if scan_path.exists() else None
        if scan is None or scan.empty:
            raise ValueError("Transcript assignment scan is required for utility figures")
        draw_utility_scope(
            utility,
            scan,
            keys,
            names,
            colors,
            scope="global",
            output=outputs["utility_global"],
        )
        draw_utility_scope(
            utility,
            scan,
            keys,
            names,
            colors,
            scope="medullary",
            output=outputs["utility_medullary"],
        )
        draw_utility_contrast(
            utility,
            scan,
            keys,
            names,
            colors,
            outputs["utility_contrast"],
        )
    draw_failure_montage(
        project / "failure_cases" / "selected_failure_cases.csv",
        project,
        config,
        registry,
        outputs["failures"],
    )
    result = {
        "status": "PASS",
        "figures": {key: str(path) for key, path in outputs.items()},
        "count": len(outputs),
    }
    (figures / "figure_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
