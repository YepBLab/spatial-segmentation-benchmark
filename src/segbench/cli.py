from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from .metrics import classify_errors, evaluate_pair


def _float_list(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated number")
    return values


def _read_instance_mask(path: Path) -> np.ndarray:
    array = np.asarray(tifffile.imread(path)).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D instance mask at {path}, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Instance mask must have an integer dtype, got {array.dtype}")
    if np.any(array < 0):
        raise ValueError("Instance labels must be non-negative; 0 is reserved for background")
    return array


def _write_errors(path: Path, events: list[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for event in events:
        row = dict(event)
        row["gt_labels"] = json.dumps(row["gt_labels"])
        row["pred_labels"] = json.dumps(row["pred_labels"])
        x0, y0, x1, y1 = row.pop("bbox")
        row.update({"bbox_x0": x0, "bbox_y0": y0, "bbox_x1": x1, "bbox_y1": y1})
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one predicted 2D instance mask against one reference mask."
    )
    parser.add_argument("--reference", type=Path, required=True, help="Reference label TIFF")
    parser.add_argument("--prediction", type=Path, required=True, help="Predicted label TIFF")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixel-size-um", type=float, required=True)
    parser.add_argument(
        "--thresholds",
        type=_float_list,
        default=_float_list("0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"),
        help="Comma-separated IoU thresholds",
    )
    parser.add_argument(
        "--boundary-tolerances-um",
        type=_float_list,
        default=_float_list("0.5,1.0,1.5,2.0"),
        help="Comma-separated physical boundary tolerances",
    )
    parser.add_argument("--error-overlap-fraction", type=float, default=0.25)
    parser.add_argument(
        "--keep-border-objects",
        action="store_true",
        help="Keep instances touching the ROI border (excluded by default)",
    )
    return parser


def evaluate_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.pixel_size_um <= 0:
        raise ValueError("--pixel-size-um must be positive")
    if 0.50 not in args.thresholds or 0.75 not in args.thresholds:
        raise ValueError("--thresholds must include 0.50 and 0.75")
    if not 0 < args.error_overlap_fraction <= 1:
        raise ValueError("--error-overlap-fraction must be in (0, 1]")

    reference = _read_instance_mask(args.reference)
    prediction = _read_instance_mask(args.prediction)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    result = evaluate_pair(
        reference,
        prediction,
        thresholds=args.thresholds,
        boundary_tolerances_um=args.boundary_tolerances_um,
        pixel_size_um=args.pixel_size_um,
        exclude_border=not args.keep_border_objects,
    )
    summary = {
        "reference": str(args.reference.resolve()),
        "prediction": str(args.prediction.resolve()),
        "pixel_size_um": args.pixel_size_um,
        "exclude_border_objects": not args.keep_border_objects,
        "thresholds": args.thresholds,
        "boundary_tolerances_um": args.boundary_tolerances_um,
        "boundary_metrics_population": "IoU>=0.50 matched true-positive instances only",
        **result["summary"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(result["threshold_rows"]).to_csv(output / "threshold_metrics.csv", index=False)
    pd.DataFrame(result["object_rows"]).to_csv(output / "object_metrics.csv", index=False)
    pd.DataFrame(result["boundary_rows"]).to_csv(output / "boundary_metrics.csv", index=False)
    events = classify_errors(
        result["gt_clean"],
        result["pred_clean"],
        overlap_fraction=args.error_overlap_fraction,
    )
    _write_errors(output / "error_events.csv", events)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(evaluate_main())
