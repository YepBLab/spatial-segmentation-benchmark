#!/usr/bin/env python3
"""Render a dark-theme, training-process-only loss history figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be at least 1")
    table = pd.read_csv(args.loss_csv)
    required = {"epoch", "train_loss"}
    if not required.issubset(table.columns):
        raise ValueError(f"Loss CSV must contain {sorted(required)}")
    if table.empty or not np.isfinite(table["train_loss"].to_numpy(dtype=float)).all():
        raise ValueError("Loss history is empty or contains non-finite values")

    epoch = table["epoch"].to_numpy(dtype=float)
    loss = table["train_loss"].to_numpy(dtype=float)
    smooth = (
        pd.Series(loss)
        .ewm(span=args.smooth_window, adjust=False, min_periods=1)
        .mean()
        .to_numpy()
    )
    delta = np.diff(loss, prepend=np.nan)
    delta_smooth = (
        pd.Series(delta)
        .rolling(args.smooth_window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    background = "#070B12"
    panel = "#0E1724"
    text = "#EAF2FA"
    muted = "#8EA2B8"
    grid = "#26384A"
    cyan = "#40E0D0"
    orange = "#FFB347"
    plt.rcParams.update(
        {
            "figure.facecolor": background,
            "axes.facecolor": panel,
            "axes.edgecolor": grid,
            "axes.labelcolor": text,
            "xtick.color": muted,
            "ytick.color": muted,
            "text.color": text,
            "font.family": "DejaVu Sans",
            "font.size": 11,
        }
    )
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.12},
    )

    axes[0].plot(epoch, loss, color=muted, linewidth=1.0, alpha=0.42, label="Raw loss")
    axes[0].plot(
        epoch,
        smooth,
        color=cyan,
        linewidth=2.7,
        label=f"EWMA (span={args.smooth_window})",
    )
    axes[0].scatter(epoch[-1], smooth[-1], color=orange, s=42, zorder=3)
    axes[0].annotate(
        f"final {loss[-1]:.4g}",
        (epoch[-1], smooth[-1]),
        xytext=(-8, 12),
        textcoords="offset points",
        ha="right",
        color=orange,
        fontsize=10,
    )
    axes[0].set_ylabel("Training loss")
    axes[0].set_title("CPSAM v2 fine-tuning history", fontsize=17, fontweight="bold", loc="left")
    axes[0].legend(frameon=False, labelcolor=text)

    axes[1].axhline(0, color=grid, linewidth=1.0)
    axes[1].plot(epoch, delta, color=muted, linewidth=0.9, alpha=0.35)
    axes[1].plot(epoch, delta_smooth, color=orange, linewidth=2.0)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Δ loss / epoch")
    axes[1].text(
        0.01,
        0.08,
        "Negative values indicate decreasing loss",
        transform=axes[1].transAxes,
        color=muted,
        fontsize=9,
    )
    for axis in axes:
        axis.grid(True, color=grid, linewidth=0.7, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=240, bbox_inches="tight", facecolor=background)
    plt.close(figure)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
