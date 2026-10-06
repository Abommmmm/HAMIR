from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcir.deepddi_baseline import (  # noqa: E402
    DeepDDI,
    DeepDDIPairDataset,
    load_pair_examples,
    load_pca50,
    macro_f1_at_half,
)


class DeepDDIBaselineTests(unittest.TestCase):
    def test_dataset_groups_labels_and_preserves_drug_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pca_path = root / "pca.csv"
            records_path = root / "records.csv"
            split_path = root / "split.csv"
            with pca_path.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["", "PC_1", "PC_2"])
                writer.writerow(["DB1", 1.0, 2.0])
                writer.writerow(["DB2", 3.0, 4.0])
            with records_path.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "record_id",
                        "d1",
                        "type_raw",
                        "type_internal",
                        "d2",
                        "unordered_pair",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "record_id": 0,
                        "d1": "DB1",
                        "type_raw": 1,
                        "type_internal": 0,
                        "d2": "DB2",
                        "unordered_pair": "DB1||DB2",
                    }
                )
                writer.writerow(
                    {
                        "record_id": 1,
                        "d1": "DB1",
                        "type_raw": 3,
                        "type_internal": 2,
                        "d2": "DB2",
                        "unordered_pair": "DB1||DB2",
                    }
                )
            with split_path.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["record_id", "split"]
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"record_id": 0, "split": "train"},
                        {"record_id": 1, "split": "train"},
                    ]
                )

            profiles, feature_count = load_pca50(pca_path)
            examples = load_pair_examples(
                records_path, split_path, "train", num_classes=3
            )
            dataset = DeepDDIPairDataset(
                examples, profiles, feature_count, num_classes=3
            )

            self.assertEqual(len(dataset), 1)
            features, target, index = dataset[0]
            torch.testing.assert_close(
                features, torch.tensor([1.0, 2.0, 3.0, 4.0])
            )
            torch.testing.assert_close(
                target, torch.tensor([1.0, 0.0, 1.0])
            )
            self.assertEqual(index, 0)
            self.assertEqual(examples[0].record_ids, (0, 1))

    def test_released_architecture_shape_and_layer_count(self) -> None:
        model = DeepDDI(
            input_dim=100,
            num_classes=86,
            hidden_dim=32,
            hidden_layers=9,
        )
        model.eval()
        output = model(torch.randn(4, 100))

        self.assertEqual(output.shape, (4, 86))
        self.assertEqual(len(model.linears), 9)
        self.assertEqual(len(model.normalizations), 9)
        self.assertTrue(
            all(
                isinstance(layer, nn.BatchNorm1d)
                for layer in model.normalizations
            )
        )

    def test_macro_f1_uses_fixed_half_threshold(self) -> None:
        target = np.asarray([[1, 0], [0, 1]], dtype=np.int8)
        probability = np.asarray(
            [[0.50, 0.49], [0.49, 0.50]], dtype=np.float32
        )
        self.assertEqual(macro_f1_at_half(target, probability), 1.0)


if __name__ == "__main__":
    unittest.main()
