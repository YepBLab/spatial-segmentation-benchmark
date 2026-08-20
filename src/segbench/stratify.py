from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def robust_snr(image: np.ndarray, foreground: np.ndarray) -> list[float]:
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[None, ...]
    output: list[float] = []
    for channel in image:
        signal = channel[foreground]
        background = channel[~foreground]
        if signal.size == 0 or background.size < 8:
            output.append(float("nan"))
            continue
        bg_median = float(np.median(background))
        bg_mad = float(np.median(np.abs(background.astype(float) - bg_median)))
        noise = max(1.4826 * bg_mad, 1e-6)
        output.append((float(np.median(signal)) - bg_median) / noise)
    return output


def add_local_density(
    objects: pd.DataFrame,
    *,
    pixel_size_um: float,
    radius_um: float,
) -> pd.DataFrame:
    output = objects.copy()
    output["local_neighbors_20um"] = 0
    for region, indices in output.groupby("region").groups.items():
        subset = output.loc[indices]
        coords = subset[["gt_centroid_x_px", "gt_centroid_y_px"]].to_numpy() * pixel_size_um
        if coords.size == 0:
            continue
        tree = cKDTree(coords)
        neighbors = np.asarray([len(x) - 1 for x in tree.query_ball_point(coords, radius_um)])
        output.loc[indices, "local_neighbors_20um"] = neighbors
    return output


def quantile_labels(
    values: pd.Series,
    labels: list[str],
) -> pd.Series:
    ranked = values.rank(method="first")
    try:
        return pd.qcut(ranked, q=len(labels), labels=labels)
    except ValueError:
        return pd.Series([labels[0]] * len(values), index=values.index, dtype="object")


def cluster_bootstrap_mean(
    table: pd.DataFrame,
    *,
    value_col: str,
    cluster_col: str,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    clusters = np.asarray(sorted(table[cluster_col].dropna().unique()))
    if clusters.size == 0:
        return float("nan"), float("nan"), float("nan")
    by_cluster = table.groupby(cluster_col)[value_col].mean()
    estimate = float(by_cluster.mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        draw = rng.choice(clusters, size=clusters.size, replace=True)
        samples[index] = float(by_cluster.loc[draw].mean())
    low, high = np.quantile(samples, [0.025, 0.975])
    return estimate, float(low), float(high)
