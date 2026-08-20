from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries


@dataclass
class Overlap:
    intersections: np.ndarray
    gt_area: np.ndarray
    pred_area: np.ndarray
    iou: np.ndarray
    dice: np.ndarray
    matched_gt: np.ndarray
    matched_pred: np.ndarray
    matched_iou: np.ndarray


@dataclass
class InstanceMatch:
    """One-to-one instance matches, optionally restricted by an IoU threshold."""

    gt: np.ndarray
    pred: np.ndarray
    iou: np.ndarray


def relabel_contiguous(mask: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    mask = np.asarray(mask)
    labels = np.unique(mask)
    labels = labels[labels != 0]
    out = np.zeros(mask.shape, dtype=np.uint32)
    mapping: dict[int, int] = {}
    for new_label, old_label in enumerate(labels.tolist(), start=1):
        out[mask == old_label] = new_label
        mapping[int(old_label)] = new_label
    return out, mapping


def border_labels(mask: np.ndarray) -> set[int]:
    values = np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1]))
    return {int(v) for v in np.unique(values) if v != 0}


def drop_labels(mask: np.ndarray, labels: Iterable[int]) -> np.ndarray:
    labels = set(int(x) for x in labels)
    if not labels:
        return np.asarray(mask)
    out = np.asarray(mask).copy()
    out[np.isin(out, list(labels))] = 0
    return out


def prepare_masks(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    exclude_border: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    gt = np.asarray(gt).squeeze()
    pred = np.asarray(pred).squeeze()
    if gt.shape != pred.shape or gt.ndim != 2:
        raise ValueError(f"Expected matching 2D masks, got {gt.shape} and {pred.shape}")
    gt_border = border_labels(gt) if exclude_border else set()
    pred_border = border_labels(pred) if exclude_border else set()
    gt_clean, _ = relabel_contiguous(drop_labels(gt, gt_border))
    pred_clean, _ = relabel_contiguous(drop_labels(pred, pred_border))
    qc = {
        "gt_border_excluded": len(gt_border),
        "pred_border_excluded": len(pred_border),
        "gt_instances": int(gt_clean.max()),
        "pred_instances": int(pred_clean.max()),
    }
    return gt_clean, pred_clean, qc


def overlap_matrix(gt: np.ndarray, pred: np.ndarray) -> Overlap:
    n_gt = int(gt.max())
    n_pred = int(pred.max())
    gt_area_all = np.bincount(gt.ravel(), minlength=n_gt + 1).astype(np.float64)
    pred_area_all = np.bincount(pred.ravel(), minlength=n_pred + 1).astype(np.float64)
    pair_codes = gt.astype(np.int64).ravel() * (n_pred + 1) + pred.astype(np.int64).ravel()
    intersections_all = np.bincount(
        pair_codes,
        minlength=(n_gt + 1) * (n_pred + 1),
    ).reshape(n_gt + 1, n_pred + 1).astype(np.float64)
    intersections = intersections_all[1:, 1:]
    gt_area = gt_area_all[1:]
    pred_area = pred_area_all[1:]
    if n_gt and n_pred:
        union = gt_area[:, None] + pred_area[None, :] - intersections
        iou = np.divide(
            intersections,
            union,
            out=np.zeros_like(intersections),
            where=union > 0,
        )
        dice_denom = gt_area[:, None] + pred_area[None, :]
        dice = np.divide(
            2.0 * intersections,
            dice_denom,
            out=np.zeros_like(intersections),
            where=dice_denom > 0,
        )
        matched_gt, matched_pred = linear_sum_assignment(-iou)
        matched_iou = iou[matched_gt, matched_pred]
    else:
        iou = np.zeros((n_gt, n_pred), dtype=np.float64)
        dice = np.zeros_like(iou)
        matched_gt = np.asarray([], dtype=np.int64)
        matched_pred = np.asarray([], dtype=np.int64)
        matched_iou = np.asarray([], dtype=np.float64)
    return Overlap(
        intersections=intersections,
        gt_area=gt_area,
        pred_area=pred_area,
        iou=iou,
        dice=dice,
        matched_gt=matched_gt,
        matched_pred=matched_pred,
        matched_iou=matched_iou,
    )


def match_instances(overlap: Overlap, threshold: float | None = None) -> InstanceMatch:
    """Return one-to-one matches using Hungarian assignment.

    Without a threshold, the assignment maximizes total IoU. With a threshold,
    it first maximizes the number of pairs meeting ``IoU >= threshold`` and then
    uses IoU as a deterministic tie-breaker. The latter is required for correct
    thresholded detection/segmentation-accuracy counts.
    """

    n_gt, n_pred = overlap.iou.shape
    if n_gt == 0 or n_pred == 0:
        empty = np.asarray([], dtype=np.int64)
        return InstanceMatch(gt=empty, pred=empty.copy(), iou=np.asarray([], dtype=float))

    if threshold is None:
        cost = -overlap.iou
    else:
        threshold = float(threshold)
        cardinality_weight = (overlap.iou >= threshold).astype(np.float64)
        tie_breaker = overlap.iou / (2.0 * max(1, min(n_gt, n_pred)))
        cost = -(cardinality_weight + tie_breaker)

    matched_gt, matched_pred = linear_sum_assignment(cost)
    matched_iou = overlap.iou[matched_gt, matched_pred]
    if threshold is not None:
        keep = matched_iou >= threshold
        matched_gt = matched_gt[keep]
        matched_pred = matched_pred[keep]
        matched_iou = matched_iou[keep]
    return InstanceMatch(gt=matched_gt, pred=matched_pred, iou=matched_iou)


def threshold_metrics(overlap: Overlap, threshold: float) -> dict[str, float | int]:
    matches = match_instances(overlap, threshold)
    tp = int(matches.iou.size)
    fp = int(overlap.pred_area.size - tp)
    fn = int(overlap.gt_area.size - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    sa = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sa": sa,
    }


def boundary_pair_metrics(
    gt_binary: np.ndarray,
    pred_binary: np.ndarray,
    tolerances_um: Iterable[float],
    pixel_size_um: float,
) -> list[dict[str, float]]:
    gt_boundary = find_boundaries(gt_binary, mode="inner")
    pred_boundary = find_boundaries(pred_binary, mode="inner")
    gt_n = int(gt_boundary.sum())
    pred_n = int(pred_boundary.sum())
    if gt_n == 0 or pred_n == 0:
        return [
            {
                "tolerance_um": float(t),
                "boundary_precision": 0.0,
                "boundary_recall": 0.0,
                "boundary_f1": 0.0,
                "nsd": 0.0,
            }
            for t in tolerances_um
        ]
    distance_to_gt = distance_transform_edt(~gt_boundary, sampling=pixel_size_um)
    distance_to_pred = distance_transform_edt(~pred_boundary, sampling=pixel_size_um)
    pred_distances = distance_to_gt[pred_boundary]
    gt_distances = distance_to_pred[gt_boundary]
    rows: list[dict[str, float]] = []
    for tolerance in tolerances_um:
        tolerance = float(tolerance)
        precision = float(np.mean(pred_distances <= tolerance))
        recall = float(np.mean(gt_distances <= tolerance))
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        nsd = float(
            ((pred_distances <= tolerance).sum() + (gt_distances <= tolerance).sum())
            / (pred_n + gt_n)
        )
        rows.append(
            {
                "tolerance_um": tolerance,
                "boundary_precision": precision,
                "boundary_recall": recall,
                "boundary_f1": f1,
                "nsd": nsd,
            }
        )
    return rows


def object_properties(gt: np.ndarray, pixel_size_um: float) -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, float]] = {}
    for prop in regionprops(gt):
        output[int(prop.label)] = {
            "gt_area_px": float(prop.area),
            "gt_area_um2": float(prop.area * pixel_size_um**2),
            "gt_centroid_y_px": float(prop.centroid[0]),
            "gt_centroid_x_px": float(prop.centroid[1]),
        }
    return output


def evaluate_pair(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    thresholds: Iterable[float],
    boundary_tolerances_um: Iterable[float],
    pixel_size_um: float,
    exclude_border: bool = True,
) -> dict[str, object]:
    thresholds = [float(x) for x in thresholds]
    tolerances = [float(x) for x in boundary_tolerances_um]
    gt_clean, pred_clean, qc = prepare_masks(gt, pred, exclude_border=exclude_border)
    overlap = overlap_matrix(gt_clean, pred_clean)
    threshold_rows = [threshold_metrics(overlap, threshold) for threshold in thresholds]
    by_threshold = {round(float(row["threshold"]), 4): row for row in threshold_rows}
    row_50 = by_threshold.get(0.5)
    row_75 = by_threshold.get(0.75)
    if row_50 is None or row_75 is None:
        raise ValueError("Thresholds must include 0.50 and 0.75")
    match_50 = match_instances(overlap, 0.5)
    matched_gt = match_50.gt
    matched_pred = match_50.pred
    matched_ious = match_50.iou
    matched_dice = (
        overlap.dice[matched_gt, matched_pred]
        if matched_gt.size
        else np.asarray([], dtype=float)
    )
    tp = int(row_50["tp"])
    fp = int(row_50["fp"])
    fn = int(row_50["fn"])
    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn else 0.0
    sq = float(np.mean(matched_ious)) if matched_ious.size else 0.0
    pq = dq * sq

    positive_assignment = overlap.matched_iou > 0
    aji_gt = overlap.matched_gt[positive_assignment]
    aji_pred = overlap.matched_pred[positive_assignment]
    aji_intersection = (
        float(overlap.intersections[aji_gt, aji_pred].sum()) if aji_gt.size else 0.0
    )
    aji_union = (
        float(
            (
                overlap.gt_area[aji_gt]
                + overlap.pred_area[aji_pred]
                - overlap.intersections[aji_gt, aji_pred]
            ).sum()
        )
        if aji_gt.size
        else 0.0
    )
    unmatched_gt = np.setdiff1d(np.arange(overlap.gt_area.size), aji_gt)
    unmatched_pred = np.setdiff1d(np.arange(overlap.pred_area.size), aji_pred)
    aji_union += float(overlap.gt_area[unmatched_gt].sum())
    aji_union += float(overlap.pred_area[unmatched_pred].sum())
    aji_plus = aji_intersection / aji_union if aji_union else 0.0

    props = object_properties(gt_clean, pixel_size_um)
    object_rows: list[dict[str, float | int | bool]] = []
    detected_by_gt = {
        int(g + 1): (int(p + 1), float(i))
        for g, p, i in zip(
            matched_gt.tolist(),
            matched_pred.tolist(),
            matched_ious.tolist(),
        )
    }
    for gt_label in range(1, int(gt_clean.max()) + 1):
        if gt_label in detected_by_gt:
            pred_label, matched_iou = detected_by_gt[gt_label]
            detected_iou50 = True
        elif overlap.pred_area.size:
            pred_index = int(np.argmax(overlap.iou[gt_label - 1]))
            matched_iou = float(overlap.iou[gt_label - 1, pred_index])
            pred_label = pred_index + 1 if matched_iou > 0 else 0
            detected_iou50 = False
        else:
            pred_label, matched_iou, detected_iou50 = 0, 0.0, False
        matched_dice_value = (
            float(overlap.dice[gt_label - 1, pred_label - 1]) if pred_label else 0.0
        )
        object_rows.append(
            {
                "gt_label": gt_label,
                "pred_label": pred_label,
                "best_iou": matched_iou,
                "best_dice": matched_dice_value,
                "detected_iou50": detected_iou50,
                **props.get(gt_label, {}),
            }
        )

    boundary_rows: list[dict[str, float | int]] = []
    for gt_index, pred_index, iou_value, dice_value in zip(
        matched_gt.tolist(),
        matched_pred.tolist(),
        matched_ious.tolist(),
        matched_dice.tolist(),
    ):
        pair_union = (gt_clean == (gt_index + 1)) | (pred_clean == (pred_index + 1))
        ys, xs = np.where(pair_union)
        pad = int(np.ceil(max(tolerances, default=0.0) / pixel_size_um)) + 2
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(gt_clean.shape[0], int(ys.max()) + pad + 1)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(gt_clean.shape[1], int(xs.max()) + pad + 1)
        pair_metrics = boundary_pair_metrics(
            gt_clean[y0:y1, x0:x1] == (gt_index + 1),
            pred_clean[y0:y1, x0:x1] == (pred_index + 1),
            tolerances,
            pixel_size_um,
        )
        for metric in pair_metrics:
            boundary_rows.append(
                {
                    "gt_label": gt_index + 1,
                    "pred_label": pred_index + 1,
                    "matched_iou": float(iou_value),
                    "matched_dice": float(dice_value),
                    **metric,
                }
            )

    summary = {
        **qc,
        "mSA_50_95": float(np.mean([float(row["sa"]) for row in threshold_rows])),
        "SA50": float(row_50["sa"]),
        "SA75": float(row_75["sa"]),
        "precision_iou50": float(row_50["precision"]),
        "recall_iou50": float(row_50["recall"]),
        "f1_iou50": float(row_50["f1"]),
        "matched_iou_mean": float(np.mean(matched_ious)) if matched_ious.size else 0.0,
        "matched_iou_median": float(np.median(matched_ious)) if matched_ious.size else 0.0,
        "matched_dice_mean": float(np.mean(matched_dice)) if matched_dice.size else 0.0,
        "matched_dice_median": float(np.median(matched_dice)) if matched_dice.size else 0.0,
        "DQ": float(dq),
        "SQ": float(sq),
        "PQ": float(pq),
        "AJI_plus": float(aji_plus),
    }
    return {
        "summary": summary,
        "threshold_rows": threshold_rows,
        "object_rows": object_rows,
        "boundary_rows": boundary_rows,
        "gt_clean": gt_clean,
        "pred_clean": pred_clean,
        "overlap": overlap,
    }


def _component_bbox(
    gt: np.ndarray,
    pred: np.ndarray,
    gt_labels: list[int],
    pred_labels: list[int],
) -> tuple[int, int, int, int]:
    selected = np.zeros(gt.shape, dtype=bool)
    if gt_labels:
        selected |= np.isin(gt, gt_labels)
    if pred_labels:
        selected |= np.isin(pred, pred_labels)
    ys, xs = np.where(selected)
    if ys.size == 0:
        return 0, 0, 1, 1
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def classify_errors(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    overlap_fraction: float = 0.25,
) -> list[dict[str, object]]:
    overlap = overlap_matrix(gt, pred)
    n_gt = overlap.gt_area.size
    n_pred = overlap.pred_area.size
    gt_fraction = np.divide(
        overlap.intersections,
        overlap.gt_area[:, None],
        out=np.zeros_like(overlap.intersections),
        where=overlap.gt_area[:, None] > 0,
    )
    pred_fraction = np.divide(
        overlap.intersections,
        overlap.pred_area[None, :],
        out=np.zeros_like(overlap.intersections),
        where=overlap.pred_area[None, :] > 0,
    )
    edges = (gt_fraction >= overlap_fraction) | (pred_fraction >= overlap_fraction)
    gt_neighbors = {g: set(np.flatnonzero(edges[g]).tolist()) for g in range(n_gt)}
    pred_neighbors = {p: set(np.flatnonzero(edges[:, p]).tolist()) for p in range(n_pred)}
    events: list[dict[str, object]] = []

    for g in range(n_gt):
        if not gt_neighbors[g]:
            labels_gt = [g + 1]
            events.append(
                {
                    "error_type": "miss",
                    "gt_labels": labels_gt,
                    "pred_labels": [],
                    "severity": float(overlap.gt_area[g]),
                    "bbox": _component_bbox(gt, pred, labels_gt, []),
                }
            )
    for p in range(n_pred):
        if not pred_neighbors[p]:
            labels_pred = [p + 1]
            events.append(
                {
                    "error_type": "spurious",
                    "gt_labels": [],
                    "pred_labels": labels_pred,
                    "severity": float(overlap.pred_area[p]),
                    "bbox": _component_bbox(gt, pred, [], labels_pred),
                }
            )

    visited_gt: set[int] = set()
    visited_pred: set[int] = set()
    for start_gt in range(n_gt):
        if start_gt in visited_gt or not gt_neighbors[start_gt]:
            continue
        component_gt: set[int] = set()
        component_pred: set[int] = set()
        queue: list[tuple[str, int]] = [("g", start_gt)]
        while queue:
            kind, index = queue.pop()
            if kind == "g":
                if index in component_gt:
                    continue
                component_gt.add(index)
                queue.extend(("p", p) for p in gt_neighbors[index])
            else:
                if index in component_pred:
                    continue
                component_pred.add(index)
                queue.extend(("g", g) for g in pred_neighbors[index])
        visited_gt |= component_gt
        visited_pred |= component_pred
        ng = len(component_gt)
        npred = len(component_pred)
        gt_labels = [x + 1 for x in sorted(component_gt)]
        pred_labels = [x + 1 for x in sorted(component_pred)]
        if ng == 1 and npred == 1:
            g = next(iter(component_gt))
            p = next(iter(component_pred))
            if overlap.iou[g, p] >= 0.5:
                continue
            error_type = "poor_overlap"
            severity = float(1.0 - overlap.iou[g, p])
        elif ng == 1 and npred > 1:
            error_type = "split"
            severity = float(npred)
        elif ng > 1 and npred == 1:
            error_type = "merge"
            severity = float(ng)
        else:
            error_type = "complex"
            severity = float(ng + npred)
        events.append(
            {
                "error_type": error_type,
                "gt_labels": gt_labels,
                "pred_labels": pred_labels,
                "severity": severity,
                "bbox": _component_bbox(gt, pred, gt_labels, pred_labels),
            }
        )
    return events
