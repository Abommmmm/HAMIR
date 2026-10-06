from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcir.models import DCIR  # noqa: E402


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


def options(ablation: str) -> dict:
    return {
        "num_classes": 5,
        "hidden_dim": 8,
        "painn_layers": 1,
        "num_rbf": 4,
        "interaction_dim": 8,
        "pair_dim": 8,
        "class_dim": 4,
        "heads": 2,
        "dropout": 0.0,
        "ablation": ablation,
        "msan_num_patterns": 4,
        "msan_prediction_layers": 2,
        "msan_dropout": 0.0,
        "msan_substructure_drop_probability": 0.0,
    }


def test_zero_residual_is_exactly_a1() -> None:
    torch.manual_seed(17)
    a1 = DCIR(**options("M1")).eval()
    residual = DCIR(**options("A2_MSAN_RESIDUAL")).eval()
    missing, unexpected = residual.load_state_dict(
        a1.state_dict(), strict=False
    )
    assert not unexpected
    assert missing
    assert all(
        name.startswith(("msan_",))
        for name in missing
    )

    first = molecule()
    second = molecule(offset=3.0)
    with torch.inference_mode():
        a1_logits = a1(first, second)["interaction_logits"]
        output = residual(first, second)

    torch.testing.assert_close(
        output["msan_residual_scale"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        output["interaction_logits"], a1_logits
    )


def test_residual_scale_and_branch_receive_gradients() -> None:
    torch.manual_seed(17)
    # Evaluation mode avoids BatchNorm's batch-size-one guard while still
    # retaining autograd for this focused residual-gradient test.
    model = DCIR(**options("A2_MSAN_RESIDUAL")).eval()
    first = molecule()
    second = molecule(offset=3.0)

    initial = model(first, second)
    initial["interaction_logits"].sum().backward()
    assert model.msan_residual_alpha.grad is not None
    assert torch.isfinite(model.msan_residual_alpha.grad)

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.msan_residual_alpha.fill_(0.2)
    active = model(first, second)
    active["interaction_logits"].sum().backward()
    pattern_gradient = model.msan_readout.extractor.patterns.grad
    assert pattern_gradient is not None
    assert torch.isfinite(pattern_gradient).all()
    assert pattern_gradient.abs().sum() > 0
