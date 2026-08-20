#!/usr/bin/env python3
"""Fine-tune CPSAM v2 on every prepared region, with no validation split."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from cellpose import io, models, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pretrained-model", default="cpsam_v2")
    parser.add_argument("--model-name", default="cpsam_v2_finetuned_all_regions")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-train-masks", type=int, default=5)
    return parser.parse_args()


def load_all_pairs(directory: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    regions: list[str] = []
    for image_path in sorted(directory.glob("*_img.tif")):
        region = image_path.name.removesuffix("_img.tif")
        mask_path = directory / f"{region}_masks.tif"
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        image = tifffile.imread(image_path)
        mask = np.asarray(tifffile.imread(mask_path)).squeeze()
        if image.ndim != 3 or image.shape[0] != 2 or image.shape[1:] != mask.shape:
            raise ValueError(f"Invalid pair {image_path}: {image.shape} / {mask.shape}")
        images.append(image)
        masks.append(mask)
        regions.append(region)
    if not images:
        raise RuntimeError(f"No prepared training pairs found in {directory}")
    return images, masks, regions


def main() -> int:
    args = parse_args()
    if args.n_epochs <= 0 or args.batch_size <= 0 or args.bsize <= 0:
        raise ValueError("Epochs, batch size, and bsize must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")
    use_gpu = args.device == "cuda"
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")

    project = args.project_root.resolve()
    images, masks, regions = load_all_pairs(project / "training_data" / "all")
    output_dir = project / "training" / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    io.logger_setup()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)
    started = time.time()
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=args.pretrained_model)
    model_path, train_losses, _ = train.train_seg(
        model.net,
        train_data=images,
        train_labels=masks,
        test_data=None,
        test_labels=None,
        channel_axis=0,
        load_files=False,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        normalize=True,
        compute_flows=False,
        save_path=output_dir,
        save_every=args.save_every,
        save_each=False,
        rescale=False,
        bsize=args.bsize,
        min_train_masks=args.min_train_masks,
        model_name=args.model_name,
    )
    train_losses = np.asarray(train_losses, dtype=float)
    if train_losses.size == 0 or not np.isfinite(train_losses).all():
        raise RuntimeError("Training loss is empty or contains non-finite values")

    losses_path = output_dir / "losses.csv"
    with losses_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss"])
        for index, value in enumerate(train_losses, start=1):
            writer.writerow([index, f"{float(value):.10g}"])

    metadata = {
        "status": "PASS",
        "training_scope": "all_regions_no_validation",
        "host": socket.gethostname(),
        "python": os.sys.executable,
        "cellpose": importlib.metadata.version("cellpose"),
        "torch": torch.__version__,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if use_gpu else None,
        "pretrained_model": args.pretrained_model,
        "model_path": str(model_path),
        "model_name": args.model_name,
        "train_regions": regions,
        "validation_regions": [],
        "n_epochs_requested": args.n_epochs,
        "loss_rows_returned": int(train_losses.size),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "bsize": args.bsize,
        "normalize": True,
        "rescale": False,
        "seed": args.seed,
        "elapsed_seconds": time.time() - started,
        "losses": str(losses_path),
    }
    metadata_path = output_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
