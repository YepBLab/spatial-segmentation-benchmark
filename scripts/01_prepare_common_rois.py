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
from skimage.segmentation import find_boundaries

from segbench.io import build_model_sources, load_yaml, read_rois


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    return parser.parse_args()


def normalize(channel: np.ndarray) -> np.ndarray:
    values = channel[channel > 0]
    if values.size == 0:
        return np.zeros(channel.shape, dtype=np.float32)
    lo, hi = np.percentile(values, [1, 99.5])
    if hi <= lo:
        return np.zeros(channel.shape, dtype=np.float32)
    return np.clip((channel.astype(np.float32) - lo) / (hi - lo), 0, 1) ** 0.75


def rgb_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        dapi = normalize(image)
        rna = np.zeros_like(dapi)
    else:
        dapi = normalize(image[0])
        rna = normalize(image[min(1, image.shape[0] - 1)])
    return np.stack((rna, 0.5 * rna, dapi), axis=-1)


def overlay_boundary(rgb: np.ndarray, mask: np.ndarray, color: tuple[float, ...]) -> np.ndarray:
    output = rgb.copy()
    output[find_boundaries(mask, mode="inner")] = color
    return output


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve()
    config = load_yaml(args.config)
    registry = load_yaml(args.registry)
    rois = read_rois(config)
    sources = build_model_sources(registry, float(config["pixel_size_um"]))
    manual_root = project / "prepared" / "manual_rois"
    qc_root = project / "prepared" / "coordinate_qc"
    manual_root.mkdir(parents=True, exist_ok=True)
    qc_root.mkdir(parents=True, exist_ok=True)
    for model_key in registry["models"]:
        (manual_root / model_key).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for roi in rois:
        manual = tifffile.imread(roi.label_path).squeeze()
        image = tifffile.imread(roi.image_path)
        rgb = rgb_image(image)
        predicted: dict[str, np.ndarray] = {}
        for model_key, source in sources.items():
            mask, source_qc = source.read_roi(roi)
            if mask.shape != manual.shape:
                raise ValueError(
                    f"{model_key}/{roi.region}: {mask.shape} != manual {manual.shape}"
                )
            tifffile.imwrite(
                manual_root / model_key / f"{roi.region}_mask.tif",
                mask.astype(np.uint32),
                compression="zlib",
            )
            predicted[model_key] = mask
            rows.append(
                {
                    "region": roi.region,
                    "model_key": model_key,
                    "display": registry["models"][model_key]["display"],
                    "shape_y": mask.shape[0],
                    "shape_x": mask.shape[1],
                    "foreground_fraction": float(np.mean(mask > 0)),
                    "instance_count": int(np.unique(mask[mask > 0]).size),
                    "tissue": roi.tissue,
                    "medullary_overlap_fraction": roi.medullary_overlap_fraction,
                    **source_qc,
                }
            )

        panels = [("DAPI + 18S", rgb), ("Manual", overlay_boundary(rgb, manual, (0, 1, 1)))]
        panels.extend(
            (
                registry["models"][model_key]["display"],
                overlay_boundary(rgb, predicted[model_key], (1.0, 0.25, 0.65)),
            )
            for model_key in registry["models"]
        )
        figure, axes = plt.subplots(2, 4, figsize=(15, 7.6), facecolor="#f8fafc")
        for axis, (title, panel) in zip(axes.ravel(), panels):
            axis.imshow(panel)
            axis.set_title(title, fontsize=10, color="#172033", fontweight="semibold")
            axis.axis("off")
        figure.suptitle(
            f"{roi.region} · {roi.tissue} · manual={int(np.unique(manual[manual > 0]).size)}",
            fontsize=14,
            color="#102a43",
            fontweight="bold",
        )
        figure.tight_layout()
        figure.savefig(
            qc_root / f"{roi.region}_coordinate_qc.png",
            dpi=180,
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )
        plt.close(figure)

    table = pd.DataFrame(rows)
    table.to_csv(project / "prepared" / "prediction_roi_manifest.csv", index=False)
    result = {
        "status": "PASS",
        "regions": len(rois),
        "models": len(registry["models"]),
        "prepared_masks": len(rows),
        "manifest": str(project / "prepared" / "prediction_roi_manifest.csv"),
        "coordinate_qc": str(qc_root),
    }
    (project / "prepared" / "roi_preparation_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
