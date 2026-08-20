#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from segbench.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve()
    config = load_yaml(args.config)
    pixel_size_um = float(config["pixel_size_um"])
    output = project / "metrics"
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "PROVISIONAL_SENSITIVITY_ONLY",
        "primary_tolerance_um": None,
        "tested_tolerances_um": config["boundary_tolerances_um"],
        "reason": (
            "No independent repeat annotation was found. Boundary F1 and NSD are "
            "reported as a tolerance sensitivity analysis, not at a claimed "
            "annotation-calibrated primary tolerance."
        ),
        "future_calibration": (
            "Set primary tolerance to the median pair-level HD95 from repeat "
            f"annotations, rounded up to the {pixel_size_um:g} um pixel grid."
        ),
    }
    (output / "boundary_tolerance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
