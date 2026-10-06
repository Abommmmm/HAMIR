from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcir.deepddi_baseline import PairExample  # noqa: E402
from dcir.graph_ddi_baselines import (  # noqa: E402
    GraphPairDataset,
    molecule_graph_from_smiles,
)
from dcir.recent_ddi_baselines import (  # noqa: E402
    HDNDDI,
    HLNDDI,
    MRCGNN,
    HierarchicalPairCollator,
    MRCPairCollator,
    hierarchical_graph_from_smiles,
)


class RecentDDIBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = [
            PairExample("A", "B", (0, 2), (1, 2)),
            PairExample("C", "A", (1,), (3,)),
        ]
        self.dataset = GraphPairDataset(self.examples, num_classes=3)
        self.hierarchical = {
            "A": hierarchical_graph_from_smiles("CCOC(=O)N"),
            "B": hierarchical_graph_from_smiles("CC(=O)Oc1ccccc1C(=O)O"),
            "C": hierarchical_graph_from_smiles("c1ccccc1O"),
        }

    def test_brics_hierarchy_has_all_three_levels(self) -> None:
        graph = self.hierarchical["B"]
        self.assertEqual(set(graph["node_type"].tolist()), {0, 1, 2})
        self.assertEqual(len(graph["x"]), len(graph["node_type"]))
        self.assertEqual(graph["x"].shape[1], 55)

    def test_hdn_and_hln_emit_multilabel_logits(self) -> None:
        batch = HierarchicalPairCollator(
            self.dataset, self.hierarchical
        )([0, 1])
        hdn = HDNDDI(
            num_classes=3,
            hidden_dim=16,
            blocks=2,
            heads=2,
            dropout=0.0,
        )
        hln = HLNDDI(
            num_classes=3,
            hidden_dim=16,
            layers=2,
            heads=2,
            dropout=0.0,
        )
        self.assertEqual(hdn(batch).shape, (2, 3))
        batch = HierarchicalPairCollator(
            self.dataset, self.hierarchical
        )([0, 1])
        self.assertEqual(hln(batch).shape, (2, 3))

    def test_mrcgnn_uses_relation_context_and_auxiliary_losses(self) -> None:
        molecular = {
            "A": molecule_graph_from_smiles("CCO"),
            "B": molecule_graph_from_smiles("CC(=O)O"),
            "C": molecule_graph_from_smiles("c1ccccc1"),
        }
        drug_ids = ["A", "B", "C"]
        context = {
            "initial_features": torch.stack(
                [molecular[item]["x"].mean(0) for item in drug_ids]
            ),
            "edge_index": torch.tensor(
                [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
            ),
            "edge_type": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        }
        model = MRCGNN(
            num_classes=3,
            context=context,
            hidden1=8,
            hidden2=4,
            decoder_hidden=16,
            dropout=0.0,
        )
        batch = MRCPairCollator(
            self.dataset, {item: i for i, item in enumerate(drug_ids)}
        )([0, 1])
        logits, auxiliary = model(batch, compute_auxiliary=True)
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(
            set(auxiliary), {"node_contrastive", "relation_contrastive"}
        )
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(
            all(torch.isfinite(value) for value in auxiliary.values())
        )


if __name__ == "__main__":
    unittest.main()
