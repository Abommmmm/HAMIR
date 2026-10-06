from test_a2_core import molecule
import torch
from dcir.hamir_2d import BondEncoder, collate_covalent

def test_2d_encoder_ignores_positions_and_backpropagates():
    model = BondEncoder(hidden_dim=32, layers=2).eval()
    inputs = molecule("CC(=O)O")
    first = collate_covalent([inputs])
    inputs["pos"] = inputs["pos"] + 123.0
    second = collate_covalent([inputs])
    left, right = model(first), model(second)
    assert left.shape == (4, 32)
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    left.square().mean().backward()
    assert any(p.grad is not None for p in model.parameters())
