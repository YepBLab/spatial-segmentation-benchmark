#!/usr/bin/env python3
"""Reload the trained model and run one in-memory shape/non-empty smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from cellpose import models


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--pilot-region")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--channel-axis", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve()
    metadata: dict[str, object] = {}
    if args.model_path is None:
        metadata_path = project / "training" / "final" / "training_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        model_path = Path(metadata["model_path"]).expanduser().resolve()
    else:
        model_path = args.model_path.expanduser().resolve()
    channel_axis = (
        args.channel_axis
        if args.channel_axis is not None
        else int(metadata.get("channel_axis", 0))
    )
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    train_dir = project / "training_data" / "train"
    image_paths = sorted(train_dir.glob("*_img.tif"))
    if not image_paths:
        raise RuntimeError(f"No prepared training images found in {train_dir}")
    if args.pilot_region:
        image_path = train_dir / f"{args.pilot_region}_img.tif"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
    else:
        image_path = image_paths[0]
    region = image_path.name.removesuffix("_img.tif")
    truth_path = train_dir / f"{region}_masks.tif"
    image = tifffile.imread(image_path)
    truth = np.asarray(tifffile.imread(truth_path)).squeeze()

    use_gpu = args.device == "cuda"
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    started = time.time()
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))
    prediction = np.asarray(
        model.eval(
            image,
            channel_axis=channel_axis,
            batch_size=args.batch_size,
            bsize=args.bsize,
            diameter=None,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            min_size=args.min_size,
        )[0]
    )
    if prediction.shape != truth.shape:
        raise RuntimeError(f"Smoke-test shape {prediction.shape} != {truth.shape}")
    predicted_instances = int(np.unique(prediction[prediction > 0]).size)
    if predicted_instances == 0:
        raise RuntimeError("Smoke-test inference produced no instances")

    result = {
        "status": "PASS",
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if use_gpu else None,
        "pilot_region": region,
        "pilot_shape_yx": list(truth.shape),
        "channel_axis": channel_axis,
        "pilot_instances": predicted_instances,
        "prediction_saved": False,
        "elapsed_seconds": time.time() - started,
    }
    output = project / "training" / "final" / "final_model_smoke_test.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
