from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from dcir.data import collate_molecules
from dcir.reaction_site_baselines import build_reaction_site_baseline
from dcir.reaction_sites import site_loss
from dcir.structures import atom_features, bond_features


def _molecule(smiles: str) -> dict[str, np.ndarray]:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    AllChem.Compute2DCoords(molecule)
    numbers, atom_attr = atom_features(molecule)
    bond_index, bond_attr = bond_features(molecule)
    return {
        "atomic_numbers": numbers,
        "atom_features": atom_attr,
        "bond_index": bond_index,
        "bond_features": bond_attr,
        "pos": np.asarray(
            molecule.GetConformer().GetPositions(), dtype=np.float32
        ),
    }


def _pair():
    d1 = collate_molecules(
        [_molecule("CCBr"), _molecule("CCO")], 5.0
    )
    d2 = collate_molecules(
        [_molecule("CN"), _molecule("NC=O")], 5.0
    )
    return d1, d2


def test_all_adapted_baselines_forward_and_backward():
    d1, d2 = _pair()
    configurations = [
        {
            "family": "dracon_atom",
            "hidden_dim": 32,
            "layers": 2,
            "attention_heads": 4,
        },
        {
            "family": "rmechrp_site",
            "hidden_dim": 32,
            "layers": 2,
            "projection_dim": 16,
        },
        {
            "family": "reactaivate_rai",
            "hidden_dim": 32,
            "layers": 2,
            "attention_heads": 4,
            "num_reaction_classes": 11,
        },
        {"family": "eac_2hop", "hidden_dim": 32},
        {
            "family": "spaba_style_7_7",
            "hidden_dim": 32,
            "layers": 8,
            "attention_heads": 4,
        },
    ]
    for config in configurations:
        model = build_reaction_site_baseline(
            {**config, "dropout": 0.0, "cutoff": 5.0}
        )
        output = model(d1, d2)
        assert output["logits1"].shape == d1["atomic_numbers"].shape
        assert output["logits2"].shape == d2["atomic_numbers"].shape
        loss = (
            output["logits1"].square().mean()
            + output["logits2"].square().mean()
        )
        loss.backward()
        assert any(
            parameter.grad is not None for parameter in model.parameters()
        )


def test_reactaivate_class_and_rmechrp_contrastive_losses():
    d1, d2 = _pair()
    batch = {
        "target1": torch.zeros(len(d1["atomic_numbers"])),
        "target2": torch.zeros(len(d2["atomic_numbers"])),
        "reaction_class": torch.tensor([1, 1]),
    }
    batch["target1"][0] = 1.0
    batch["target2"][0] = 1.0

    reactaivate = build_reaction_site_baseline(
        {
            "family": "reactaivate_rai",
            "hidden_dim": 32,
            "layers": 1,
            "attention_heads": 4,
            "num_reaction_classes": 11,
            "dropout": 0.0,
        }
    )
    reactaivate_loss = site_loss(
        reactaivate(d1, d2),
        batch,
        pos_weight=torch.tensor(4.0),
        focal_gamma=2.0,
        dice_weight=0.5,
        class_weight=0.1,
    )
    reactaivate_loss.backward()
    assert reactaivate.classifier[-1].weight.grad is not None

    rmechrp = build_reaction_site_baseline(
        {
            "family": "rmechrp_site",
            "hidden_dim": 32,
            "layers": 1,
            "projection_dim": 16,
            "dropout": 0.0,
        }
    )
    rmechrp_loss = site_loss(
        rmechrp(d1, d2),
        batch,
        pos_weight=torch.tensor(4.0),
        focal_gamma=2.0,
        dice_weight=0.5,
        contrastive_weight=0.05,
        contrastive_temperature=0.1,
    )
    rmechrp_loss.backward()
    assert rmechrp.projection[-1].weight.grad is not None

