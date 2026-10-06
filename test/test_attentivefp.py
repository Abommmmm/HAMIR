from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcir.models import AttentiveFPMolecularReadout, DCIR  # noqa: E402


def two_atom_molecule(offset: float = 0.0) -> dict[str, torch.Tensor]:
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


class AttentiveFPReadoutTests(unittest.TestCase):
    def test_attention_is_normalized_for_each_molecule(self) -> None:
        torch.manual_seed(17)
        readout = AttentiveFPMolecularReadout(
            hidden_dim=8,
            num_timesteps=2,
            dropout=0.0,
        ).eval()
        features = torch.randn(5, 8)
        batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        ptr = torch.tensor([0, 2, 5], dtype=torch.long)

        graph, attention = readout(
            features,
            batch,
            ptr,
            return_attention=True,
        )

        self.assertEqual(graph.shape, (2, 8))
        torch.testing.assert_close(attention[:2].sum(), torch.tensor(1.0))
        torch.testing.assert_close(attention[2:].sum(), torch.tensor(1.0))

    def test_readout_is_invariant_to_atom_order_within_molecules(self) -> None:
        torch.manual_seed(17)
        readout = AttentiveFPMolecularReadout(
            hidden_dim=8,
            num_timesteps=2,
            dropout=0.0,
        ).eval()
        features = torch.randn(5, 8)
        batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        ptr = torch.tensor([0, 2, 5], dtype=torch.long)
        permutation = torch.tensor([1, 0, 4, 2, 3], dtype=torch.long)

        original = readout(features, batch, ptr)
        permuted = readout(features[permutation], batch, ptr)

        torch.testing.assert_close(original, permuted, rtol=1e-5, atol=1e-6)

    def test_complete_attentivefp_variant_forward_is_finite(self) -> None:
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
            ablation="A2_ATTENTIVEFP",
            attentivefp_timesteps=2,
        ).eval()

        output = model(
            two_atom_molecule(),
            two_atom_molecule(offset=3.0),
        )
        output["interaction_logits"].sum().backward()

        self.assertEqual(output["interaction_logits"].shape, (1, 5))
        self.assertTrue(torch.isfinite(output["interaction_logits"]).all())
        self.assertTrue(model.use_dcia)
        self.assertTrue(model.use_attentivefp)
        self.assertFalse(model.use_crdm)
        self.assertFalse(model.use_dgbf)
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.dcia.message1.parameters())
        )
        message_gradient = model.dcia.message1.weight.grad
        attention_gradient = model.attentive_readout.attention[0].weight.grad
        self.assertIsNotNone(message_gradient)
        self.assertIsNotNone(attention_gradient)
        self.assertTrue(torch.isfinite(message_gradient).all())
        self.assertTrue(torch.isfinite(attention_gradient).all())

    def test_legacy_m1_state_has_no_attentivefp_parameters(self) -> None:
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

        state_keys = model.state_dict()
        self.assertFalse(
            any(key.startswith("attentive_") for key in state_keys)
        )
        self.assertFalse(
            any(parameter.requires_grad for parameter in model.dcia.message1.parameters())
        )


if __name__ == "__main__":
    unittest.main()
