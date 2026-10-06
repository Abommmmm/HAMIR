from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcir.models import DCIR, DirectionalGatedBilinearFusion  # noqa: E402


class DirectionalGatedBilinearFusionTests(unittest.TestCase):
    @staticmethod
    def _two_atom_molecule(offset: float = 0.0) -> dict[str, torch.Tensor]:
        return {
            "atomic_numbers": torch.tensor([6, 8], dtype=torch.long),
            "atom_features": torch.zeros(2, 7),
            "pos": torch.tensor(
                [[offset, 0.0, 0.0], [offset + 1.2, 0.0, 0.0]]
            ),
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "edge_features": torch.zeros(2, 7),
            "batch": torch.zeros(2, dtype=torch.long),
            "ptr": torch.tensor([0, 2], dtype=torch.long),
        }

    def test_zero_initialized_residual_recovers_base_logits(self) -> None:
        torch.manual_seed(17)
        module = DirectionalGatedBilinearFusion(
            hidden_dim=8,
            pair_dim=6,
            rank=4,
            fusion_dim=8,
            num_classes=3,
            dropout=0.0,
            alpha_init=0.0,
        )
        base = torch.randn(2, 3)
        logits, residual = module(
            torch.randn(2, 8),
            torch.randn(2, 8),
            torch.randn(2, 6),
            torch.randn(2, 6),
            base,
        )

        torch.testing.assert_close(logits, base, rtol=0.0, atol=0.0)
        self.assertEqual(residual.shape, base.shape)
        self.assertEqual(module.residual_scale.item(), 0.0)

    def test_a2_selects_dcia_and_dgbf_without_crdm(self) -> None:
        model = DCIR(
            num_classes=5,
            hidden_dim=8,
            painn_layers=1,
            num_rbf=4,
            interaction_dim=8,
            pair_dim=8,
            class_dim=4,
            heads=2,
            dropout=0.0,
            ablation="A2",
            dgbf_rank=4,
            dgbf_dim=8,
        )

        self.assertTrue(model.use_dcia)
        self.assertTrue(model.use_dgbf)
        self.assertFalse(model.use_crdm)
        self.assertTrue(hasattr(model, "dgbf"))
        self.assertFalse(any(p.requires_grad for p in model.crdm1.parameters()))

    def test_a2_complete_forward_is_finite(self) -> None:
        torch.manual_seed(17)
        model = DCIR(
            num_classes=5,
            hidden_dim=8,
            painn_layers=1,
            num_rbf=4,
            interaction_dim=8,
            pair_dim=8,
            class_dim=4,
            heads=2,
            dropout=0.0,
            ablation="A2",
            dgbf_rank=4,
            dgbf_dim=8,
        ).eval()

        output = model(
            self._two_atom_molecule(),
            self._two_atom_molecule(offset=3.0),
        )

        self.assertEqual(output["interaction_logits"].shape, (1, 5))
        self.assertEqual(output["dgbf_residual_logits"].shape, (1, 5))
        self.assertTrue(torch.isfinite(output["interaction_logits"]).all())
        self.assertEqual(output["dgbf_residual_scale"].item(), 0.0)

    def test_legacy_m1_state_has_no_dgbf_parameters(self) -> None:
        model = DCIR(
            num_classes=5,
            hidden_dim=8,
            painn_layers=1,
            num_rbf=4,
            interaction_dim=8,
            pair_dim=8,
            class_dim=4,
            heads=2,
            dropout=0.0,
            ablation="M1",
        )

        self.assertFalse(any(key.startswith("dgbf.") for key in model.state_dict()))


if __name__ == "__main__":
    unittest.main()
