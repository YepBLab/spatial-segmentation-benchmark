#!/usr/bin/env python3
"""Validate benchmark invariants and record caveats before sharing results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from segbench.io import load_yaml, read_rois


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str, checks: list[str], problems: list[str]) -> None:
    (checks if condition else problems).append(message)


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve()
    config = load_yaml(args.config)
    registry = load_yaml(args.registry)
    metrics = project / "metrics"
    figures = project / "figures"
    report = project / "report"
    report.mkdir(parents=True, exist_ok=True)

    checks: list[str] = []
    problems: list[str] = []
    warnings: list[str] = []
    expected_models = list(registry["models"])

    required_tables = [
        "manual_roi_metrics.csv",
        "sa_threshold_curve_by_roi.csv",
        "manual_macro_summary.csv",
        "sa_threshold_curve_macro.csv",
        "boundary_sensitivity_macro.csv",
    ]
    for filename in required_tables:
        require((metrics / filename).exists(), f"required table exists: {filename}", checks, problems)
    if problems:
        payload = {"status": "FAIL", "checks": checks, "problems": problems, "warnings": warnings}
        (report / "validation_report.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return 1

    manual = pd.read_csv(metrics / "manual_roi_metrics.csv")
    curve = pd.read_csv(metrics / "sa_threshold_curve_by_roi.csv")
    macro = pd.read_csv(metrics / "manual_macro_summary.csv")
    n_regions = int(manual["region"].nunique())
    expected_region_count = config.get("expected_manual_regions")
    if expected_region_count is None:
        expected_region_count = len(read_rois(config))
    require(
        n_regions == int(expected_region_count),
        f"manual ROI count matches configuration (observed {n_regions}; expected {expected_region_count})",
        checks,
        problems,
    )
    require(
        set(manual["model_key"]) == set(expected_models),
        "manual metrics contain every registered model and no unregistered model",
        checks,
        problems,
    )
    require(
        len(manual) == n_regions * len(expected_models),
        "one manual summary row exists per ROI × model",
        checks,
        problems,
    )

    bounded_columns = [
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
    bounded_columns = [column for column in bounded_columns if column in manual]
    values = manual[bounded_columns].to_numpy(dtype=float)
    require(
        bool(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()),
        "all manual-reference scores are finite and within [0, 1]",
        checks,
        problems,
    )
    require(
        bool(np.allclose(manual["PQ"], manual["DQ"] * manual["SQ"], atol=1e-10)),
        "PQ equals DQ × SQ for every ROI × model",
        checks,
        problems,
    )

    recomputed_msa = (
        curve.groupby(["region", "model_key"], observed=True)["sa"]
        .mean()
        .rename("mSA_recomputed")
        .reset_index()
    )
    comparison = manual.merge(recomputed_msa, on=["region", "model_key"], validate="one_to_one")
    require(
        bool(np.allclose(comparison["mSA_50_95"], comparison["mSA_recomputed"], atol=1e-12)),
        "mSA equals the mean of the stored threshold curve",
        checks,
        problems,
    )

    monotone = True
    invariant_counts = True
    for _, group in curve.groupby(["region", "model_key"], observed=True):
        ordered = group.sort_values("threshold")
        monotone &= bool(np.all(np.diff(ordered["sa"].to_numpy()) <= 1e-12))
        if {"n_gt", "n_pred"}.issubset(ordered.columns):
            invariant_counts &= ordered["n_gt"].nunique() == 1 and ordered["n_pred"].nunique() == 1
    require(monotone, "SA is non-increasing as IoU threshold increases", checks, problems)
    require(invariant_counts, "reference and prediction totals are threshold-invariant", checks, problems)

    expected_thresholds = np.asarray(config["thresholds"], dtype=float)
    observed_thresholds = np.sort(curve["threshold"].unique().astype(float))
    require(
        bool(np.array_equal(np.round(observed_thresholds, 8), np.round(expected_thresholds, 8))),
        "stored SA thresholds match the configured threshold grid",
        checks,
        problems,
    )

    macro_models = set(macro["model_key"])
    require(
        macro_models == set(expected_models),
        "macro summary contains every registered model",
        checks,
        problems,
    )

    core_figures = [
        "01_manual_accuracy_scorecard.png",
        "02_sa_threshold_curve.png",
        "03_detection_iou50.png",
        "04_boundary_tolerance_sensitivity.png",
        "05_pq_decomposition.png",
        "06_error_spectrum.png",
        "07_size_density_strata.png",
        "09_representative_failures.png",
    ]
    missing_figures = [name for name in core_figures if not (figures / name).exists()]
    require(not missing_figures, f"all core figures exist (missing: {missing_figures})", checks, problems)
    if (metrics / "global_medullary_utility.csv").exists():
        require(
            all(
                (figures / filename).exists()
                for filename in [
                    "08a_global_utility.png",
                    "08b_medullary_utility.png",
                    "08c_scope_utility_contrast.png",
                ]
            ),
            "all global/medullary utility figures exist when utility data are present",
            checks,
            problems,
        )

    qa_dir = figures / "coordinate_qa"
    qa_count = len(list(qa_dir.glob("*.png"))) if qa_dir.exists() else 0
    require(
        qa_count == n_regions,
        f"coordinate QA panel exists for every manual ROI (observed {qa_count}; expected {n_regions})",
        checks,
        problems,
    )
    require(
        (report / "segmentation_benchmark_report.html").exists(),
        "HTML report exists",
        checks,
        problems,
    )

    design = str(config.get("evaluation_design", "unspecified")).lower()
    if design in {"in_sample", "training", "train"}:
        warnings.append(
            "Reference regions overlap model-training data; results measure in-sample fit and do not establish generalization."
        )
    elif design == "unspecified":
        warnings.append("Evaluation design is unspecified; declare in_sample, held_out, or external before publication.")
    if config.get("primary_boundary_tolerance_um") is None:
        warnings.append(
            "No primary boundary tolerance is declared; boundary scores must be reported as a tolerance sensitivity curve."
        )

    status = "FAIL" if problems else ("PASS_WITH_CAVEATS" if warnings else "PASS")
    payload = {
        "status": status,
        "manual_rois": n_regions,
        "models": expected_models,
        "evaluation_design": design,
        "checks": checks,
        "problems": problems,
        "warnings": warnings,
    }
    (report / "validation_report.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
