#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tifffile

from segbench.io import load_yaml, read_medullary_union, read_rois


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--skip-reimport-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve()
    config = load_yaml(args.config)
    registry = load_yaml(args.registry)
    prepared = project / "prepared"
    logs = project / "logs"
    prepared.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    expected_shape = tuple(config["full_shape_yx"])
    model_rows = []
    problems: list[str] = []
    for model_key, model in registry["models"].items():
        accuracy_path = Path(model["accuracy_source"])
        outs = Path(model["reimport_outs"])
        required_outs = [] if args.skip_reimport_check else [
            outs / "cells.parquet",
            outs / "cell_boundaries.parquet",
            outs / "transcripts.parquet",
            outs / "cell_feature_matrix.h5",
            outs / "metrics_summary.csv",
        ]
        missing = [str(path) for path in [accuracy_path, *required_outs] if not path.exists()]
        row = {
            "model_key": model_key,
            "display": model["display"],
            "kind": model["kind"],
            "accuracy_source": str(accuracy_path),
            "reimport_outs": str(outs),
            "missing": missing,
        }
        if accuracy_path.exists() and model["kind"] == "raster_mask":
            with tifffile.TiffFile(accuracy_path) as tif:
                page = tif.pages[0]
                row.update(
                    {
                        "shape_yx": list(page.shape),
                        "dtype": str(page.dtype),
                        "compression": str(page.compression),
                        "memory_mappable": bool(page.compression == 1),
                    }
                )
                if tuple(page.shape) != expected_shape:
                    problems.append(f"{model_key}: shape {page.shape} != {expected_shape}")
        if (outs / "cells.parquet").exists():
            row["cell_rows"] = int(pq.ParquetFile(outs / "cells.parquet").metadata.num_rows)
        if (outs / "transcripts.parquet").exists():
            row["transcript_rows"] = int(
                pq.ParquetFile(outs / "transcripts.parquet").metadata.num_rows
            )
        if missing:
            problems.append(f"{model_key}: missing {missing}")
        model_rows.append(row)

    rois = read_rois(config)
    roi_rows = []
    for roi in rois:
        label = tifffile.imread(roi.label_path).squeeze()
        image = tifffile.imread(roi.image_path)
        roi_row = {
            "region": roi.region,
            "x0": roi.x0,
            "y0": roi.y0,
            "x1": roi.x1,
            "y1": roi.y1,
            "shape_y": roi.shape[0],
            "shape_x": roi.shape[1],
            "label_shape": list(label.shape),
            "image_shape": list(image.shape),
            "manual_instances": int(np.unique(label[label > 0]).size),
            "medullary_overlap_fraction": roi.medullary_overlap_fraction,
            "tissue": roi.tissue,
        }
        if tuple(label.shape) != roi.shape:
            problems.append(f"{roi.region}: label shape {label.shape} != bbox shape {roi.shape}")
        if tuple(image.shape[-2:]) != roi.shape:
            problems.append(f"{roi.region}: image shape {image.shape} != bbox shape {roi.shape}")
        roi_rows.append(roi_row)

    medullary = read_medullary_union(config["medullary_geojson"])
    medullary_area_um2 = float(medullary.area * float(config["pixel_size_um"]) ** 2)
    pd.DataFrame(model_rows).to_csv(prepared / "model_registry_resolved.csv", index=False)
    pd.DataFrame(roi_rows).to_csv(prepared / "roi_manifest.csv", index=False)
    result = {
        "status": "PASS" if not problems else "FAIL",
        "host": platform.node(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "project_root": str(project),
        "models": model_rows,
        "rois": roi_rows,
        "n_manual_rois": len(rois),
        "medullary_area_um2": medullary_area_um2,
        "medullary_area_mm2": medullary_area_um2 / 1e6,
        "problems": problems,
    }
    (prepared / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
