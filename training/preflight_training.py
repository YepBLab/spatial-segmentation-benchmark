#!/usr/bin/env python3
"""Verify the Cellpose model, device, and prepared training-data contracts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from cellpose import models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pretrained-model", default="cpsam_v2")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--min-masks", type=int, default=5)
    parser.add_argument("--channel-axis", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve()
    version = importlib.metadata.version("cellpose")
    version_tuple = tuple(int(part) for part in version.split(".")[:2])
    if version_tuple < (4, 2):
        raise RuntimeError(f"Cellpose >=4.2 required, found {version}")
    if args.pretrained_model == "cpsam_v2" and "cpsam_v2" not in set(
        getattr(models, "MODEL_NAMES", ())
    ):
        raise RuntimeError("Installed Cellpose does not register cpsam_v2")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")

    train_dir = project / "training_data" / "train"
    image_paths = sorted(train_dir.glob("*_img.tif"))
    if not image_paths:
        raise RuntimeError(f"No prepared training images found in {train_dir}")
    pair_rows: list[dict[str, object]] = []
    for split in ("train", "validation"):
        directory = project / "training_data" / split
        for image_path in sorted(directory.glob("*_img.tif")):
            region = image_path.name.removesuffix("_img.tif")
            mask_path = directory / f"{region}_masks.tif"
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            image = tifffile.imread(image_path)
            mask = np.asarray(tifffile.imread(mask_path)).squeeze()
            if image.ndim != 3:
                raise RuntimeError(f"{region}: expected a 3D image, got {image.shape}")
            if not -image.ndim <= args.channel_axis < image.ndim:
                raise RuntimeError(
                    f"{region}: channel axis {args.channel_axis} is invalid for {image.shape}"
                )
            axis = args.channel_axis % image.ndim
            spatial_shape = tuple(
                int(size) for index, size in enumerate(image.shape) if index != axis
            )
            if spatial_shape != mask.shape:
                raise RuntimeError(f"{region}: incompatible shapes {image.shape} / {mask.shape}")
            instance_count = int(np.unique(mask[mask > 0]).size)
            if instance_count < args.min_masks:
                raise RuntimeError(
                    f"{region}: found {instance_count} instances; minimum is {args.min_masks}"
                )
            pair_rows.append(
                {
                    "region": region,
                    "split": split,
                    "image_shape": list(image.shape),
                    "mask_shape_yx": list(mask.shape),
                    "instances": instance_count,
                }
            )

    pilot_image_path = image_paths[0]
    pilot_region = pilot_image_path.name.removesuffix("_img.tif")
    pilot_image = tifffile.imread(pilot_image_path)
    pilot_mask = np.asarray(
        tifffile.imread(train_dir / f"{pilot_region}_masks.tif")
    ).squeeze()
    use_gpu = bool(torch.cuda.is_available())
    started = time.time()
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=args.pretrained_model)
    prediction = np.asarray(
        model.eval(
            pilot_image,
            channel_axis=args.channel_axis,
            batch_size=1,
            bsize=256,
            diameter=None,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            min_size=15,
        )[0]
    )
    if prediction.shape != pilot_mask.shape:
        raise RuntimeError(
            f"Pilot inference returned {prediction.shape}, expected {pilot_mask.shape}"
        )

    report = {
        "status": "PASS",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": os.sys.executable,
        "cellpose": version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pretrained_model": args.pretrained_model,
        "channel_axis": args.channel_axis,
        "training_regions": sum(row["split"] == "train" for row in pair_rows),
        "validation_regions": sum(row["split"] == "validation" for row in pair_rows),
        "pairs": pair_rows,
        "pilot_region": pilot_region,
        "pilot_prediction_instances": int(np.unique(prediction[prediction > 0]).size),
        "pilot_elapsed_seconds": time.time() - started,
    }
    output = project / "provenance" / "training_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
