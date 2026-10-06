from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from dcir.reaction_site_diagnostics import (
    _mapped_atom_predictions,
    _occlude,
    _paired_mean_test,
    _pool_atoms,
    _topk_target_hits,
    _transform_positions,
)


def test_group_pooling_max_and_mean():
    probabilities = torch.tensor([0.2, 0.8, 0.4, 0.6])
    motif_index = torch.tensor([0, 0, 1, 1])
    maximum = _pool_atoms(probabilities, motif_index, 2, "max")
    mean = _pool_atoms(probabilities, motif_index, 2, "mean")
    assert torch.allclose(maximum, torch.tensor([0.8, 0.6]))
    assert torch.allclose(mean, torch.tensor([0.5, 0.5]))


def test_occlusion_marks_atoms_without_changing_chemical_graph():
    molecule = {
        "atomic_numbers": torch.tensor([6, 6, 8]),
        "atom_features": torch.ones(3, 7),
        "pos": torch.zeros(3, 3),
        "batch": torch.zeros(3, dtype=torch.long),
        "ptr": torch.tensor([0, 3]),
        "edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        "edge_features": torch.ones(4, 7),
    }
    mask = torch.tensor([False, True, False])
    changed = _occlude(molecule, mask)
    assert changed["atomic_numbers"].tolist() == [6, 6, 8]
    assert bool(changed["atom_features"][1].all())
    assert torch.equal(changed["edge_index"], molecule["edge_index"])
    assert torch.equal(changed["occlusion_mask"], mask)


def test_rotation_translation_preserve_distances():
    molecule = {
        "pos": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]]
        ),
        "batch": torch.zeros(3, dtype=torch.long),
        "ptr": torch.tensor([0, 3]),
    }
    original = torch.cdist(molecule["pos"], molecule["pos"])
    changed = _transform_positions(
        molecule, rotate=True, translate=True, noise_std=0.0
    )
    transformed = torch.cdist(changed["pos"], changed["pos"])
    assert torch.allclose(original, transformed, atol=1e-5)


def test_paired_mean_test_detects_larger_candidate_drop():
    candidate = torch.linspace(0.2, 0.4, 100).numpy()
    random = torch.linspace(0.0, 0.1, 100).numpy()
    result = _paired_mean_test(
        candidate, random, iterations=1000, seed=17
    )
    assert result["paired_difference"] > 0
    assert result["ci_95_low"] > 0
    assert result["significant"]


def test_swapped_predictions_are_mapped_back_to_original_roles():
    batch = {
        "target1": torch.tensor([1.0, 0.0]),
        "target2": torch.tensor([0.0, 1.0]),
        "d1": {"ptr": torch.tensor([0, 2])},
        "d2": {"ptr": torch.tensor([0, 2])},
    }
    output = {
        "logits1": torch.tensor([-4.0, 4.0]),
        "logits2": torch.tensor([4.0, -4.0]),
    }
    targets, probabilities = _mapped_atom_predictions(
        output, batch, ("2", "1")
    )
    assert targets.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert probabilities[0] > 0.9
    assert probabilities[-1] > 0.9
    hits, total = _topk_target_hits(
        output, batch, ("2", "1"), k=1
    )
    assert (hits, total) == (2, 2)
