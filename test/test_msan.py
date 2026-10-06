from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcir.models import (  # noqa: E402
    DCIR,
    MSANSubstructureExtractor,
    MSANSubstructureReadout,
)


def molecule(offset: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "atomic_numbers": torch.tensor([6, 8, 7], dtype=torch.long),
        "atom_features": torch.zeros(3, 7),
        "pos": torch.tensor(
            [
                [offset, 0.0, 0.0],
                [offset + 1.2, 0.0, 0.0],
                [offset, 1.1, 0.0],
            ]
        ),
        "edge_index": torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]],
            dtype=torch.long,
        ),
        "edge_features": torch.zeros(6, 7),
        "batch": torch.zeros(3, dtype=torch.long),
        "ptr": torch.tensor([0, 3], dtype=torch.long),
    }


def test_msan_se_shapes_and_atom_assignments() -> None:
    torch.manual_seed(17)
    extractor = MSANSubstructureExtractor(
        hidden_dim=8, num_patterns=4
    )
    node = torch.randn(5, 8)
    ptr = torch.tensor([0, 2, 5])
    patterns, assignment, attention, valid = extractor(node, ptr)

    assert patterns.shape == (2, 4, 8)
    assert assignment.shape == (2, 3)
    assert attention.shape == (2, 4, 3)
    assert valid.sum().item() == 5
    torch.testing.assert_close(
        attention[0, :, :2].sum(dim=0),
        torch.ones(2),
    )
    torch.testing.assert_close(
        attention[1].sum(dim=0),
        torch.ones(3),
    )


def test_msan_si_is_bounded_cosine_similarity() -> None:
    torch.manual_seed(17)
    readout = MSANSubstructureReadout(
        hidden_dim=8,
        num_classes=5,
        num_patterns=4,
        prediction_layers=2,
        dropout=0.0,
        substructure_drop_probability=0.0,
    ).eval()
    node1 = torch.randn(5, 8)
    node2 = torch.randn(5, 8)
    batch = torch.tensor([0, 0, 1, 1, 1])
    ptr = torch.tensor([0, 2, 5])
    output = readout(node1, batch, ptr, node2, batch, ptr)

    assert output["logits"].shape == (2, 5)
    assert output["similarity"].shape == (2, 4, 4)
    assert output["similarity"].max() <= 1.0 + 1e-6
    assert output["similarity"].min() >= -1.0 - 1e-6


def test_msan_sd_preserves_gradient_for_kept_atoms() -> None:
    torch.manual_seed(17)
    readout = MSANSubstructureReadout(
        hidden_dim=8,
        num_classes=5,
        num_patterns=4,
        prediction_layers=2,
        dropout=0.0,
        substructure_drop_probability=1.0,
    ).train()
    node1 = torch.randn(5, 8, requires_grad=True)
    node2 = torch.randn(5, 8, requires_grad=True)
    batch = torch.tensor([0, 0, 1, 1, 1])
    ptr = torch.tensor([0, 2, 5])
    output = readout(node1, batch, ptr, node2, batch, ptr)
    output["logits"].sum().backward()

    assert output["dropped1"].any()
    assert output["dropped2"].any()
    assert node1.grad is not None
    assert node2.grad is not None
    assert node1.grad[~output["dropped1"]].abs().sum() > 0
    assert node2.grad[~output["dropped2"]].abs().sum() > 0


def test_complete_msan_variant_reaches_dcia_messages() -> None:
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
        ablation="A2_MSAN",
        msan_num_patterns=4,
        msan_prediction_layers=2,
        msan_dropout=0.0,
        msan_substructure_drop_probability=0.0,
    ).eval()
    output = model(molecule(), molecule(offset=3.0))
    output["interaction_logits"].sum().backward()

    assert output["interaction_logits"].shape == (1, 5)
    assert output["msan_similarity"].shape == (1, 4, 4)
    assert torch.isfinite(output["interaction_logits"]).all()
    assert model.use_dcia
    assert model.use_msan
    assert not model.use_crdm
    assert not model.use_dgbf
    assert not model.use_attentivefp
    assert model.dcia.message1.weight.grad is not None
    assert model.msan_readout.extractor.patterns.grad is not None
