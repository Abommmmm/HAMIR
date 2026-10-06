from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcir.calibrate_thresholds import (  # noqa: E402
    calibrate,
    fit_thresholds,
    threshold_grid,
)
from dcir.evaluate_cloud import compute_metrics  # noqa: E402


class ThresholdCalibrationTests(unittest.TestCase):
    def test_compute_metrics_accepts_per_class_thresholds(self) -> None:
        target = np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.int8)
        probability = np.asarray(
            [[0.40, 0.30], [0.20, 0.70], [0.45, 0.25]],
            dtype=np.float64,
        )
        metrics, _, prediction = compute_metrics(
            target,
            probability,
            "multilabel",
            np.asarray([0.35, 0.60]),
            ["a", "b"],
        )

        self.assertEqual(metrics["threshold_mode"], "per_class")
        self.assertIsNone(metrics["threshold"])
        self.assertEqual(metrics["macro_f1"], 1.0)
        np.testing.assert_array_equal(prediction, target)

    def test_support_shrinkage_keeps_rare_class_near_global(self) -> None:
        target = np.asarray(
            [
                [1, 0],
                [1, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 1],
            ],
            dtype=np.int8,
        )
        probability = np.asarray(
            [
                [0.40, 0.10],
                [0.45, 0.20],
                [0.30, 0.30],
                [0.20, 0.40],
                [0.10, 0.50],
                [0.05, 0.60],
            ],
            dtype=np.float64,
        )
        grid = threshold_grid(0.05, 0.95, 0.05)
        global_threshold, thresholds, rows = fit_thresholds(
            target,
            probability,
            grid,
            shrinkage=20.0,
            min_support=2,
        )

        self.assertEqual(rows[1]["validation_support"], 1)
        self.assertEqual(rows[1]["shrinkage_weight"], 0.0)
        self.assertEqual(thresholds[1], global_threshold)
        self.assertGreater(rows[0]["shrinkage_weight"], 0.0)

    def test_calibrate_writes_frozen_threshold_outputs(self) -> None:
        validation_target = np.asarray(
            [[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.int8
        )
        validation_probability = np.asarray(
            [[0.45, 0.20], [0.40, 0.30], [0.25, 0.65], [0.20, 0.60]],
            dtype=np.float64,
        )
        test_target = validation_target[::-1].copy()
        test_probability = validation_probability[::-1].copy()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation_dir = root / "validation"
            test_dir = root / "test"
            output_dir = root / "calibrated"
            validation_dir.mkdir()
            test_dir.mkdir()
            np.savez_compressed(
                validation_dir / "predictions.npz",
                target=validation_target,
                probability=validation_probability,
                prediction=validation_probability >= 0.5,
            )
            np.savez_compressed(
                test_dir / "predictions.npz",
                target=test_target,
                probability=test_probability,
                prediction=test_probability >= 0.5,
            )

            calibrate(
                validation_dir,
                test_dir,
                output_dir,
                minimum=0.05,
                maximum=0.95,
                step=0.05,
                shrinkage=20.0,
                min_support=2,
            )

            self.assertTrue((output_dir / "thresholds.csv").is_file())
            self.assertTrue((output_dir / "metrics.json").is_file())
            with np.load(
                output_dir / "predictions.npz", allow_pickle=False
            ) as archive:
                self.assertEqual(archive["threshold"].shape, (2,))
                self.assertEqual(archive["prediction"].shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
