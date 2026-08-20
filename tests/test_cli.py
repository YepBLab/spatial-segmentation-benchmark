from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from segbench.cli import evaluate_main


class CliTests(unittest.TestCase):
    def test_single_pair_cli_writes_complete_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = np.zeros((32, 32), dtype=np.uint32)
            reference[4:14, 4:14] = 1
            reference[18:28, 18:28] = 2
            prediction = reference.copy()
            reference_path = root / "reference.tif"
            prediction_path = root / "prediction.tif"
            output = root / "output"
            tifffile.imwrite(reference_path, reference)
            tifffile.imwrite(prediction_path, prediction)

            status = evaluate_main(
                [
                    "--reference",
                    str(reference_path),
                    "--prediction",
                    str(prediction_path),
                    "--pixel-size-um",
                    "0.2",
                    "--output-dir",
                    str(output),
                ]
            )

            self.assertEqual(status, 0)
            expected = {
                "summary.json",
                "threshold_metrics.csv",
                "object_metrics.csv",
                "boundary_metrics.csv",
                "error_events.csv",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(np.isclose(summary["mSA_50_95"], 1.0))
            threshold = pd.read_csv(output / "threshold_metrics.csv")
            self.assertEqual(len(threshold), 10)
            self.assertTrue(np.allclose(threshold["sa"], 1.0))


if __name__ == "__main__":
    unittest.main()
