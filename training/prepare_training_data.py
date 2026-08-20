#!/usr/bin/env python3
"""Create immutable, provenance-tracked CPSAM v2 training pairs from a CSV manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import tifffile


REQUIRED_COLUMNS = {"region", "image_path", "label_path"}


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else manifest_dir / path).resolve()


def relabel_consecutive(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    old_ids, counts = np.unique(mask, return_counts=True)
    keep = old_ids > 0
    old_ids = old_ids[keep]
    counts = counts[keep]
    output = np.zeros(mask.shape, dtype=np.uint32)
    foreground = mask > 0
    if old_ids.size:
        output[foreground] = np.searchsorted(old_ids, mask[foreground]).astype(np.uint32) + 1
    mapping = [
        (int(old_id), int(new_id), int(count))
        for new_id, (old_id, count) in enumerate(zip(old_ids, counts), start=1)
    ]
    return output, mapping


def atomic_tiff(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".part")
    tifffile.imwrite(
        temporary,
        array,
        photometric="minisblack",
        compression="zlib",
        compressionargs={"level": 1},
        metadata={"axes": "YX"},
    )
    os.replace(temporary, path)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Training manifest is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Training manifest contains no rows")
    regions = [row["region"].strip() for row in rows]
    if any(not region for region in regions):
        raise ValueError("Every manifest row must have a non-empty region")
    if len(regions) != len(set(regions)):
        raise ValueError("Training manifest region values must be unique")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--min-masks", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve()
    manifest_path = args.manifest.resolve()
    source_rows = read_manifest(manifest_path)
    output_root = project / "training_data"
    pair_dir = output_root / "all"
    mapping_dir = output_root / "label_maps"
    pair_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    total_instances = 0
    for source in source_rows:
        region = source["region"].strip()
        image_path = resolve_path(source["image_path"], manifest_path.parent)
        label_path = resolve_path(source["label_path"], manifest_path.parent)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not label_path.is_file():
            raise FileNotFoundError(label_path)

        image = tifffile.imread(image_path)
        label = np.asarray(tifffile.imread(label_path)).squeeze()
        if image.ndim != 3 or image.shape[0] != 2:
            raise ValueError(f"{region}: expected CYX=(2,Y,X), got {image.shape}")
        if label.ndim != 2 or tuple(image.shape[1:]) != tuple(label.shape):
            raise ValueError(f"{region}: image/label mismatch {image.shape} versus {label.shape}")
        if not np.issubdtype(label.dtype, np.integer) or np.any(label < 0):
            raise ValueError(f"{region}: label must contain non-negative integer instance IDs")

        derived_label, mapping = relabel_consecutive(label)
        if len(mapping) < args.min_masks:
            raise ValueError(
                f"{region}: found {len(mapping)} instances; minimum is {args.min_masks}"
            )
        total_instances += len(mapping)

        image_out = pair_dir / f"{region}_img.tif"
        label_out = pair_dir / f"{region}_masks.tif"
        if (image_out.exists() or label_out.exists()) and not args.overwrite:
            raise FileExistsError(
                f"Derived pair already exists; use --overwrite after reviewing it: {image_out}"
            )
        shutil.copy2(image_path, image_out)
        atomic_tiff(label_out, derived_label)

        map_path = mapping_dir / f"{region}_old_to_new.csv"
        temporary_map = map_path.with_name(map_path.name + ".part")
        with temporary_map.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["old_label_id", "new_label_id", "pixel_count"])
            writer.writerows(mapping)
        os.replace(temporary_map, map_path)

        rows.append(
            {
                "region": region,
                "split": "train",
                "source_image": str(image_path),
                "source_label": str(label_path),
                "source_image_sha256": sha256(image_path),
                "source_label_sha256": sha256(label_path),
                "derived_image": str(image_out),
                "derived_label": str(label_out),
                "derived_image_sha256": sha256(image_out),
                "derived_label_sha256": sha256(label_out),
                "shape_y": int(label.shape[0]),
                "shape_x": int(label.shape[1]),
                "image_dtype": str(image.dtype),
                "source_label_dtype": str(label.dtype),
                "derived_label_dtype": str(derived_label.dtype),
                "instance_count": len(mapping),
                "foreground_fraction": float(np.count_nonzero(label) / label.size),
                "label_map": str(map_path),
            }
        )
        print(f"{region}: shape={label.shape}, instances={len(mapping)}", flush=True)

    frozen_manifest = output_root / "training_manifest.csv"
    temporary_manifest = frozen_manifest.with_name(frozen_manifest.name + ".part")
    with temporary_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_manifest, frozen_manifest)

    summary = {
        "status": "PASS",
        "training_scope": "all_regions_no_validation",
        "channel_contract": {
            "array_axis": "CYX",
            "channel_0": "DAPI; physical morphology channel 0",
            "channel_1": "boundary/18S signal; physical morphology channel 2",
        },
        "regions": len(rows),
        "train_regions": [str(row["region"]) for row in rows],
        "validation_regions": [],
        "total_instances": total_instances,
        "source_labels_modified": False,
        "derived_labels_reindexed_only": True,
        "frozen_manifest": str(frozen_manifest),
    }
    summary_path = output_root / "training_data_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
