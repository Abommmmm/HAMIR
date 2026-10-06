from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcir.deepddi_baseline import PairExample  # noqa: E402
from dcir.graph_ddi_baselines import (  # noqa: E402
    ATOM_FEATURE_DIM,
    DSNDDI,
    GraphPairCollator,
    GraphPairDataset,
    SSIDDI,
    molecule_graph_from_smiles,
)


class GraphDDIBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graphs = {
            "A": molecule_graph_from_smiles("CCO"),
            "B": molecule_graph_from_smiles("CC(=O)O"),
            "C": molecule_graph_from_smiles("c1ccccc1"),
        }
        self.examples = [
            PairExample("A", "B", (0, 2), (1, 2)),
            PairExample("C", "A", (1,), (3,)),
        ]
        self.dataset = GraphPairDataset(self.examples, num_classes=3)

    def test_official_atom_feature_dimension(self) -> None:
        self.assertEqual(
            self.graphs["A"]["x"].shape[1], ATOM_FEATURE_DIM
        )
        self.assertEqual(ATOM_FEATURE_DIM, 55)

    def test_pair_collator_builds_multilabel_and_bipartite_graph(self) -> None:
        batch = GraphPairCollator(
            self.dataset, self.graphs, include_bipartite=True
        )([0, 1])

        self.assertEqual(batch.target.shape, (2, 3))
        torch.testing.assert_close(
            batch.target,
            torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        )
        expected_edges = (
            len(self.graphs["A"]["x"]) * len(self.graphs["B"]["x"])
            + len(self.graphs["C"]["x"]) * len(self.graphs["A"]["x"])
        )
        self.assertEqual(
            batch.bipartite_edge_index.shape, (2, expected_edges)
        )

    def test_ssi_ddi_outputs_all_relation_logits(self) -> None:
        batch = GraphPairCollator(
            self.dataset, self.graphs, include_bipartite=False
        )([0, 1])
        model = SSIDDI(
            num_classes=3,
            kge_dim=16,
            head_output_features=[8, 8],
            heads=[2, 2],
        )
        output = model(batch.head, batch.tail)
        self.assertEqual(output.shape, (2, 3))
        self.assertTrue(torch.isfinite(output).all())

    def test_dsn_ddi_outputs_all_relation_logits(self) -> None:
        batch = GraphPairCollator(
            self.dataset, self.graphs, include_bipartite=True
        )([0, 1])
        model = DSNDDI(
            num_classes=3,
            kge_dim=128,
            head_output_features=[64, 64],
            heads=[2, 2],
        )
        output = model(
            batch.head, batch.tail, batch.bipartite_edge_index
        )
        self.assertEqual(output.shape, (2, 3))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
