from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcir.three_d_ddi_baselines import (  # noqa: E402
    Meta3DDI,
    ThreeDGTDDI,
)


def molecule_batch() -> dict[str, torch.Tensor]:
    return {
        "atomic_numbers": torch.tensor([6, 8, 6, 7, 8]),
        "atom_features": torch.zeros(5, 7),
        "pos": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        "batch": torch.tensor([0, 0, 1, 1, 1]),
        "ptr": torch.tensor([0, 2, 5]),
        "edge_index": torch.tensor(
            [
                [0, 1, 2, 2, 3, 3, 4, 4],
                [1, 0, 3, 4, 2, 4, 2, 3],
            ]
        ),
        "edge_features": torch.zeros(8, 7),
    }


def test_3dgt_forward_and_rotation_invariance() -> None:
    model = ThreeDGTDDI(
        num_classes=4,
        hidden_channels=16,
        num_layers=2,
        num_filters=16,
        num_gaussians=8,
        cutoff=5.0,
        graph_channels=8,
        pair_hidden_channels=8,
    ).eval()
    first = molecule_batch()
    second = molecule_batch()
    original = model(first, second)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated = dict(first)
    rotated["pos"] = first["pos"] @ rotation.T + 3.0
    transformed = model(rotated, second)
    assert original.shape == (2, 4)
    torch.testing.assert_close(original, transformed)


def test_meta3d_forward_and_rotation_invariance() -> None:
    model = Meta3DDI(
        num_classes=4,
        hidden_channels=16,
        num_layers=2,
        num_gaussians=8,
        cutoff=5.0,
        dropout=0.0,
        decoder_hidden_channels=16,
    ).eval()
    first = molecule_batch()
    second = molecule_batch()
    original = model(first, second)
    rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    rotated = dict(second)
    rotated["pos"] = second["pos"] @ rotation.T - 2.0
    transformed = model(first, rotated)
    assert original.shape == (2, 4)
    torch.testing.assert_close(original, transformed)
