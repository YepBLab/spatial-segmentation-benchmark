from __future__ import annotations

import unittest

import numpy as np

from segbench.metrics import (
    Overlap,
    boundary_pair_metrics,
    classify_errors,
    evaluate_pair,
    match_instances,
    overlap_matrix,
    prepare_masks,
    threshold_metrics,
)


THRESHOLDS = np.arange(0.50, 0.951, 0.05)


def evaluate(gt: np.ndarray, pred: np.ndarray):
    return evaluate_pair(
        gt,
        pred,
        thresholds=THRESHOLDS,
        boundary_tolerances_um=[0.5, 1.0],
        pixel_size_um=0.2,
        exclude_border=False,
    )


class MetricTests(unittest.TestCase):
    def test_perfect_segmentation_is_one(self) -> None:
        gt = np.zeros((30, 30), dtype=np.uint32)
        gt[3:10, 3:10] = 1
        gt[16:25, 18:27] = 2
        result = evaluate(gt, gt.copy())
        for metric in [
            "mSA_50_95",
            "SA50",
            "SA75",
            "precision_iou50",
            "recall_iou50",
            "f1_iou50",
            "matched_iou_mean",
            "matched_dice_mean",
            "DQ",
            "SQ",
            "PQ",
            "AJI_plus",
        ]:
            self.assertTrue(np.isclose(result["summary"][metric], 1.0), metric)

    def test_empty_prediction_has_zero_accuracy(self) -> None:
        gt = np.zeros((20, 20), dtype=np.uint32)
        gt[4:12, 5:13] = 1
        pred = np.zeros_like(gt)
        result = evaluate(gt, pred)
        self.assertEqual(result["summary"]["SA50"], 0)
        self.assertEqual(result["summary"]["recall_iou50"], 0)
        self.assertEqual(result["summary"]["PQ"], 0)

    def test_border_objects_are_removed(self) -> None:
        gt = np.zeros((20, 20), dtype=np.uint32)
        pred = np.zeros_like(gt)
        gt[0:6, 2:8] = 9
        gt[10:16, 10:16] = 11
        pred[:] = gt
        gt_clean, pred_clean, qc = prepare_masks(gt, pred, exclude_border=True)
        self.assertEqual(qc["gt_border_excluded"], 1)
        self.assertEqual(qc["pred_border_excluded"], 1)
        self.assertEqual(gt_clean.max(), 1)
        self.assertEqual(pred_clean.max(), 1)

    def test_split_and_merge_are_classified(self) -> None:
        gt_split = np.zeros((20, 20), dtype=np.uint32)
        gt_split[4:16, 4:16] = 1
        pred_split = np.zeros_like(gt_split)
        pred_split[4:16, 4:10] = 1
        pred_split[4:16, 10:16] = 2
        split_types = {
            event["error_type"]
            for event in classify_errors(gt_split, pred_split, overlap_fraction=0.25)
        }
        self.assertIn("split", split_types)

        gt_merge = pred_split
        pred_merge = gt_split
        merge_types = {
            event["error_type"]
            for event in classify_errors(gt_merge, pred_merge, overlap_fraction=0.25)
        }
        self.assertIn("merge", merge_types)

    def test_boundary_metrics_perfect_pair_are_one(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[3:15, 5:17] = True
        rows = boundary_pair_metrics(mask, mask.copy(), [0.5, 1.0], 0.2)
        self.assertTrue(all(np.isclose(row["boundary_f1"], 1.0) for row in rows))
        self.assertTrue(all(np.isclose(row["nsd"], 1.0) for row in rows))

    def test_pq_identity_and_sa_monotonicity(self) -> None:
        gt = np.zeros((30, 30), dtype=np.uint32)
        pred = np.zeros_like(gt)
        gt[3:13, 3:13] = 1
        gt[16:26, 16:26] = 2
        pred[4:14, 3:13] = 1
        pred[16:25, 17:27] = 2
        result = evaluate(gt, pred)
        summary = result["summary"]
        self.assertTrue(np.isclose(summary["PQ"], summary["DQ"] * summary["SQ"]))
        sa = np.asarray([row["sa"] for row in result["threshold_rows"]])
        self.assertTrue(np.all(np.diff(sa) <= 0))
        overlap = overlap_matrix(result["gt_clean"], result["pred_clean"])
        self.assertEqual(overlap.iou.shape, (2, 2))

    def test_threshold_matching_maximizes_true_positive_count(self) -> None:
        # Maximizing total IoU alone would select the two 0.49 pairs and miss
        # the valid 0.90 pair. Threshold-aware matching must return one TP.
        iou = np.asarray([[0.90, 0.49], [0.49, 0.00]], dtype=float)
        overlap = Overlap(
            intersections=np.zeros((2, 2), dtype=float),
            gt_area=np.ones(2, dtype=float),
            pred_area=np.ones(2, dtype=float),
            iou=iou,
            dice=np.zeros((2, 2), dtype=float),
            matched_gt=np.asarray([0, 1]),
            matched_pred=np.asarray([1, 0]),
            matched_iou=np.asarray([0.49, 0.49]),
        )
        matches = match_instances(overlap, 0.50)
        self.assertEqual(matches.gt.tolist(), [0])
        self.assertEqual(matches.pred.tolist(), [0])
        row = threshold_metrics(overlap, 0.50)
        self.assertEqual(row["tp"], 1)
        self.assertEqual(row["fp"], 1)
        self.assertEqual(row["fn"], 1)


if __name__ == "__main__":
    unittest.main()
