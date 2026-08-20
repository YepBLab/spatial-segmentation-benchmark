#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from shapely import contains_xy
from shapely.affinity import scale

from segbench.io import load_yaml, read_medullary_union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--skip-transcript-scan", action="store_true")
    return parser.parse_args()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values]
    )


def genes_per_cell(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        barcodes = decode(handle["matrix/barcodes"][:])
        indptr = handle["matrix/indptr"][:]
    return pd.DataFrame(
        {
            "cell_id": barcodes,
            "genes_detected": np.diff(indptr).astype(np.int32),
        }
    )


def transcript_assignment_scan(
    path: Path,
    medullary_um,
) -> dict[str, float | int]:
    global_total = 0
    global_assigned = 0
    medullary_total = 0
    medullary_assigned = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=1_000_000,
        columns=["cell_id", "x_location", "y_location", "qv"],
    ):
        data = batch.to_pydict()
        x = np.asarray(data["x_location"], dtype=np.float64)
        y = np.asarray(data["y_location"], dtype=np.float64)
        qv = np.asarray(data["qv"], dtype=np.float64)
        cell_id = np.asarray(data["cell_id"], dtype=object)
        eligible = qv >= 20.0
        assigned = cell_id != "UNASSIGNED"
        in_medullary = contains_xy(medullary_um, x, y)
        global_total += int(eligible.sum())
        global_assigned += int(np.count_nonzero(eligible & assigned))
        medullary_total += int(np.count_nonzero(eligible & in_medullary))
        medullary_assigned += int(np.count_nonzero(eligible & in_medullary & assigned))
    return {
        "q20_transcripts_global": global_total,
        "q20_assigned_global": global_assigned,
        "assigned_fraction_q20_global": (
            global_assigned / global_total if global_total else np.nan
        ),
        "q20_transcripts_medullary": medullary_total,
        "q20_assigned_medullary": medullary_assigned,
        "assigned_fraction_q20_medullary": (
            medullary_assigned / medullary_total if medullary_total else np.nan
        ),
    }


def summarize_cells(
    cells: pd.DataFrame,
    *,
    area_um2: float,
    scope: str,
) -> dict[str, float | int | str]:
    transcript_counts = cells["transcript_counts"].to_numpy()
    control_counts = (
        cells["control_probe_counts"].to_numpy()
        + cells["control_codeword_counts"].to_numpy()
        + cells["genomic_control_counts"].to_numpy()
    )
    return {
        "scope": scope,
        "cell_count": int(len(cells)),
        "area_um2": float(area_um2),
        "cells_per_mm2": float(len(cells) / (area_um2 / 1e6)) if area_um2 else np.nan,
        "cell_area_median_um2": float(cells["cell_area"].median()),
        "cell_area_q25_um2": float(cells["cell_area"].quantile(0.25)),
        "cell_area_q75_um2": float(cells["cell_area"].quantile(0.75)),
        "transcripts_per_cell_median": float(np.median(transcript_counts)),
        "transcripts_per_cell_mean": float(np.mean(transcript_counts)),
        "empty_cell_fraction": float(np.mean(transcript_counts == 0)),
        "control_counts_per_cell_mean": float(np.mean(control_counts)),
        "genes_per_cell_median": (
            float(cells["genes_detected"].median())
            if "genes_detected" in cells
            else np.nan
        ),
    }


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve()
    config = load_yaml(args.config)
    registry = load_yaml(args.registry)
    metrics_dir = project / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pixel_size = float(config["pixel_size_um"])
    medullary_px = read_medullary_union(config["medullary_geojson"])
    medullary_um = scale(
        medullary_px,
        xfact=pixel_size,
        yfact=pixel_size,
        origin=(0, 0),
    )
    medullary_area_um2 = float(medullary_um.area)
    global_area_um2 = float(config["region_area_um2"])
    rows: list[dict[str, object]] = []
    scan_rows: list[dict[str, object]] = []

    for model_key, model in registry["models"].items():
        outs = Path(model["reimport_outs"])
        cells = pd.read_parquet(outs / "cells.parquet")
        gene_counts = genes_per_cell(outs / "cell_feature_matrix.h5")
        cells = cells.merge(gene_counts, on="cell_id", how="left", validate="one_to_one")
        if cells["genes_detected"].isna().any():
            raise ValueError(f"{model_key}: H5 barcode join has missing rows")
        in_medullary = contains_xy(
            medullary_um,
            cells["x_centroid"].to_numpy(),
            cells["y_centroid"].to_numpy(),
        )
        global_summary = summarize_cells(cells, area_um2=global_area_um2, scope="global")
        med_summary = summarize_cells(
            cells.loc[in_medullary],
            area_um2=medullary_area_um2,
            scope="medullary",
        )
        outside_summary = summarize_cells(
            cells.loc[~in_medullary],
            area_um2=max(global_area_um2 - medullary_area_um2, 0.0),
            scope="outside_medullary",
        )
        metrics_summary = pd.read_csv(outs / "metrics_summary.csv").iloc[0]
        common = {
            "model_key": model_key,
            "display": model["display"],
            "xenium_fraction_transcripts_assigned": float(
                metrics_summary["fraction_transcripts_assigned"]
            ),
            "xenium_median_genes_per_cell": float(metrics_summary["median_genes_per_cell"]),
            "xenium_median_transcripts_per_cell": float(
                metrics_summary["median_transcripts_per_cell"]
            ),
            "xenium_total_high_quality_decoded_transcripts": int(
                metrics_summary["total_high_quality_decoded_transcripts"]
            ),
        }
        rows.extend([{**common, **summary} for summary in [global_summary, med_summary, outside_summary]])
        if not args.skip_transcript_scan:
            scan_rows.append(
                {
                    "model_key": model_key,
                    "display": model["display"],
                    **transcript_assignment_scan(outs / "transcripts.parquet", medullary_um),
                }
            )

    utility = pd.DataFrame(rows)
    utility.to_parquet(metrics_dir / "global_medullary_utility.parquet", index=False)
    utility.to_csv(metrics_dir / "global_medullary_utility.csv", index=False)
    if scan_rows:
        transcript_scan = pd.DataFrame(scan_rows)
        transcript_scan.to_csv(metrics_dir / "transcript_assignment_scan.csv", index=False)
    else:
        transcript_scan = pd.DataFrame()

    result = {
        "status": "PASS",
        "models": len(registry["models"]),
        "scopes": ["global", "medullary", "outside_medullary"],
        "medullary_area_um2": medullary_area_um2,
        "medullary_area_mm2": medullary_area_um2 / 1e6,
        "transcript_scan_completed": bool(scan_rows),
        "utility_table": str(metrics_dir / "global_medullary_utility.parquet"),
    }
    (metrics_dir / "global_medullary_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
