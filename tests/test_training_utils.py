from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

from training.plot_training_history import main as plot_main
from training.prepare_training_data import main as prepare_main
from training.prepare_training_data import relabel_consecutive


class TrainingUtilityTests(unittest.TestCase):
    def test_relabel_consecutive_preserves_instances_and_background(self) -> None:
        mask = np.zeros((8, 9), dtype=np.uint16)
        mask[1:3, 1:4] = 7
        mask[4:7, 5:8] = 42

        relabeled, mapping = relabel_consecutive(mask)

        self.assertEqual(relabeled.dtype, np.uint32)
        self.assertEqual(set(np.unique(relabeled)), {0, 1, 2})
        self.assertEqual(mapping, [(7, 1, 6), (42, 2, 9)])
        self.assertTrue(np.array_equal(mask > 0, relabeled > 0))

    def test_prepare_training_data_from_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.zeros((2, 24, 24), dtype=np.uint16)
            mask = np.zeros((24, 24), dtype=np.uint16)
            for index in range(5):
                y0 = 2 + 4 * index
                mask[y0 : y0 + 3, 3:7] = 10 * (index + 1)
            image_path = root / "input_img.tif"
            label_path = root / "input_mask.tif"
            tifffile.imwrite(image_path, image, metadata={"axes": "CYX"})
            tifffile.imwrite(label_path, mask)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "region,image_path,label_path\n"
                f"pilot,{image_path},{label_path}\n"
            )
            project = root / "project"
            with patch(
                "sys.argv",
                [
                    "prepare_training_data.py",
                    "--project-root",
                    str(project),
                    "--manifest",
                    str(manifest),
                ],
            ):
                self.assertEqual(prepare_main(), 0)
            derived = tifffile.imread(project / "training_data/all/pilot_masks.tif")
            self.assertEqual(set(np.unique(derived)), {0, 1, 2, 3, 4, 5})
            self.assertTrue((project / "training_data/training_data_summary.json").exists())

    def test_plot_training_history(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            loss_path = root / "losses.csv"
            output = root / "training_history_dark.png"
            pd.DataFrame(
                {"epoch": np.arange(1, 21), "train_loss": np.linspace(1.0, 0.2, 20)}
            ).to_csv(loss_path, index=False)
            with patch(
                "sys.argv",
                [
                    "plot_training_history.py",
                    "--loss-csv",
                    str(loss_path),
                    "--output",
                    str(output),
                ],
            ):
                self.assertEqual(plot_main(), 0)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
