from __future__ import annotations

import csv
import sqlite3
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcir.audit import load_dataset_rows, ryu_event_descriptions  # noqa: E402
from dcir.data import _radius_edges  # noqa: E402
from dcir.evaluate_cloud import compute_metrics  # noqa: E402
from dcir.losses import js_divergence_from_logits  # noqa: E402
from dcir.structures import adaptive_conformer_counts  # noqa: E402


class ProjectLayoutTests(unittest.TestCase):
    def test_root_directory_types(self) -> None:
        directories = {
            path.name
            for path in ROOT.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        self.assertTrue(
            {"configs", "data", "docs", "outputs", "scripts", "src", "test"}
            <= directories,
        )

    def test_source_package_location(self) -> None:
        self.assertTrue((ROOT / "src" / "dcir" / "__init__.py").is_file())

    def test_adaptive_conformer_counts(self) -> None:
        self.assertEqual(adaptive_conformer_counts(128, 20, 3), (20, 3))
        self.assertEqual(adaptive_conformer_counts(129, 20, 3), (10, 3))
        self.assertEqual(adaptive_conformer_counts(257, 20, 3), (1, 1))
        self.assertEqual(adaptive_conformer_counts(781, 1, 3), (1, 1))

    def test_single_atom_radius_graph_has_two_dimensional_empty_edges(self) -> None:
        edge_index, edge_features = _radius_edges(
            np.zeros((1, 3), dtype=np.float32),
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, 6), dtype=np.float32),
            cutoff=5.0,
        )
        self.assertEqual(edge_index.shape, (2, 0))
        self.assertEqual(edge_features.shape, (0, 7))
        self.assertTrue((ROOT / "scripts" / "dcir.py").is_file())

    def test_js_divergence_saturated_logits_have_finite_gradients(self) -> None:
        logits_a = torch.tensor([[1000.0, -1000.0]], requires_grad=True)
        logits_b = torch.tensor([[-1000.0, 1000.0]], requires_grad=True)

        loss = js_divergence_from_logits(
            logits_a, logits_b, "multilabel"
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits_a.grad).all())
        self.assertTrue(torch.isfinite(logits_b.grad).all())

    def test_multilabel_test_metrics_include_supported_auc_counts(self) -> None:
        target = np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
        probability = np.asarray(
            [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]], dtype=np.float32
        )

        metrics, per_class, prediction = compute_metrics(
            target, probability, "multilabel", 0.5, ["1", "2"]
        )

        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["auprc_supported_classes"], 2)
        self.assertEqual(metrics["auroc_supported_classes"], 2)
        self.assertEqual(len(per_class), 2)
        np.testing.assert_array_equal(prediction, target)


class OriginalDatasetTests(unittest.TestCase):
    def test_ddimdl_author_database_counts(self) -> None:
        path = ROOT / "data" / "raw" / "DDIMDL-master" / "event.db"
        with sqlite3.connect(path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM drug").fetchone()[0], 572)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM event").fetchone()[0], 37264)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM event_number").fetchone()[0],
                65,
            )

    def test_deepddi_author_dataset_counts(self) -> None:
        deepddi_data = ROOT / "data" / "raw" / "DeepDDI" / "data"
        path = deepddi_data / "KnownDDI.csv"
        row_count = 0
        drugs: set[str] = set()
        labels: set[str] = set()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row_count += 1
                drugs.update((row["Drug1"], row["Drug2"]))
                labels.add(row["Label"])
        self.assertEqual(row_count, 192284)
        self.assertEqual(len(drugs), 1710)
        self.assertEqual(len(labels), 86)
        self.assertEqual(
            len(list((deepddi_data / "DrugBank5.0_Approved_drugs").glob("*.sdf"))),
            2159,
        )

    def test_default_loaders_use_author_data_where_labels_are_native(self) -> None:
        rows, source_paths = load_dataset_rows(ROOT, "ryu")
        self.assertEqual(len(rows), 192284)
        deepddi_data = ROOT / "data" / "raw" / "DeepDDI" / "data"
        self.assertEqual(source_paths, [deepddi_data / "KnownDDI.csv"])
        descriptions = ryu_event_descriptions(
            deepddi_data / "Interaction_information.csv"
        )
        self.assertEqual(len(descriptions), 86)

    def test_ddimdl_labels_are_rebuilt_without_secondary_data(self) -> None:
        rows, source_paths = load_dataset_rows(ROOT, "deng")
        self.assertEqual(len(rows), 37264)
        self.assertEqual(len({row.type_raw for row in rows}), 65)
        self.assertEqual(
            source_paths,
            [ROOT / "data" / "raw" / "DDIMDL-master" / "event.db"],
        )


if __name__ == "__main__":
    unittest.main()
