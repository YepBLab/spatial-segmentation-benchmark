#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from segbench.io import load_yaml


PRIMARY_ERRORS = ["miss", "spurious", "split", "merge", "complex", "poor_overlap"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve()
    config = load_yaml(args.config)
    metrics_dir = project / "metrics"
    output_dir = project / "failure_cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = pd.read_parquet(metrics_dir / "error_events.parquet")
    errors = errors[
        errors["overlap_fraction"] == float(config["error_overlap_fraction"])
    ].copy()
    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    selected_rows = []
    for (model_key, error_type), group in errors.groupby(
        ["model_key", "error_type"],
        observed=True,
    ):
        if error_type not in PRIMARY_ERRORS or group.empty:
            continue
        top_index = group["severity"].idxmax()
        random_index = int(rng.choice(group.index.to_numpy()))
        for selection_type, row_index in [("top_severity", top_index), ("seeded_random", random_index)]:
            row = group.loc[row_index].to_dict()
            row["selection_type"] = selection_type
            selected_rows.append(row)
    selected = pd.DataFrame(selected_rows).drop_duplicates(
        ["model_key", "error_type", "region", "bbox_x0", "bbox_y0", "selection_type"]
    )
    selected.to_csv(output_dir / "selected_failure_cases.csv", index=False)
    summary = (
        errors.groupby(["model_key", "display", "error_type"], observed=True)
        .size()
        .rename("event_count")
        .reset_index()
    )
    summary.to_csv(output_dir / "error_type_counts.csv", index=False)
    result = {
        "status": "PASS",
        "primary_overlap_fraction": float(config["error_overlap_fraction"]),
        "total_primary_events": int(len(errors)),
        "selected_cases": int(len(selected)),
        "selection_rule": "top severity plus fixed-seed random per model and error type",
    }
    (output_dir / "failure_case_selection.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
