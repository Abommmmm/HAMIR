from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "evaluate_saved_predictions.py"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_saved_predictions", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SavedPredictionMetricTests(unittest.TestCase):
    def test_compute_full_metrics_at_fixed_threshold(self) -> None:
        target = np.asarray([[1, 0], [0, 1]], dtype=np.int8)
        probability = np.asarray(
            [[0.8, 0.2], [0.1, 0.9]], dtype=np.float64
        )

        metrics, per_class, prediction = MODULE.compute_full_metrics(
            target, probability, 0.5, ["1", "2"]
        )

        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["micro_f1"], 1.0)
        self.assertEqual(metrics["subset_accuracy"], 1.0)
        self.assertEqual(metrics["hamming_loss"], 0.0)
        self.assertEqual(len(per_class), 2)
        np.testing.assert_array_equal(prediction, target)

    def test_evaluate_writes_full_outputs_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary)
            np.savez_compressed(
                result_dir / "predictions.npz",
                target=np.asarray([[1, 0], [0, 1]], dtype=np.int8),
                probability=np.asarray(
                    [[0.8, 0.2], [0.1, 0.9]], dtype=np.float32
                ),
            )
            with (result_dir / "metrics.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        "baseline": "ssi_ddi",
                        "checkpoint_epoch": 85,
                    },
                    handle,
                )
            audit_path = result_dir / "audit.json"
            with audit_path.open("w", encoding="utf-8") as handle:
                json.dump({"type_values": ["a", "b"]}, handle)

            metrics = MODULE.evaluate_saved_predictions(
                result_dir, audit_path, 0.5
            )

            self.assertEqual(metrics["baseline"], "ssi_ddi")
            self.assertEqual(metrics["checkpoint_epoch"], 85)
            self.assertTrue((result_dir / "metrics_full.json").is_file())
            self.assertTrue(
                (result_dir / "per_class_metrics.csv").is_file()
            )
            self.assertTrue(
                (result_dir / "predictions_full.npz").is_file()
            )


if __name__ == "__main__":
    unittest.main()
