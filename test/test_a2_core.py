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

from dcir.a2_data import collate_molecules
from dcir.a2_structures import atom_features, bond_features
from dcir.reaction_sites import (
    ReactionSitePaiNN,
    brics_motif_ids,
    reaction_site_collate,
    site_loss,
)


def molecule(smiles: str) -> dict[str, np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    AllChem.Compute2DCoords(mol)
    numbers, features = atom_features(mol)
    bond_index, bond_attr = bond_features(mol)
    return {
        "atomic_numbers": numbers,
        "atom_features": features,
        "bond_index": bond_index,
        "bond_features": bond_attr,
        "pos": np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32),
    }


def test_a2_hierarchical_motif_forward_and_backward():
    items = []
    for index, (first, second) in enumerate(
        [("CC(=O)O", "CN"), ("CCNC", "NC=O")]
    ):
        mol1, mol2 = Chem.MolFromSmiles(first), Chem.MolFromSmiles(second)
        assert mol1 is not None and mol2 is not None
        target1 = np.zeros(mol1.GetNumAtoms(), dtype=np.float32)
        target2 = np.zeros(mol2.GetNumAtoms(), dtype=np.float32)
        target1[0] = target2[0] = 1.0
        items.append(
            {
                "d1": molecule(first),
                "d2": molecule(second),
                "target1": target1,
                "target2": target2,
                "motif_ids1": np.asarray(brics_motif_ids(mol1)),
                "motif_ids2": np.asarray(brics_motif_ids(mol2)),
                "sample_id": str(index),
            }
        )
    batch = reaction_site_collate(5.0)(items)
    model = ReactionSitePaiNN(
        hidden_dim=32,
        painn_layers=1,
        num_rbf=8,
        cutoff=5.0,
        dropout=0.0,
        architecture="hierarchical_motif",
        attention_heads=4,
        attention_topk=2,
    )
    output = model(batch["d1"], batch["d2"])
    assert output["logits1"].shape == batch["target1"].shape
    assert output["logits2"].shape == batch["target2"].shape
    assert "group_logits1" in output and "group_logits2" in output
    loss = site_loss(
        output,
        batch,
        pos_weight=torch.tensor(4.0),
        focal_gamma=2.0,
        dice_weight=0.5,
        group_pos_weight=torch.tensor(2.0),
        group_weight=0.2,
        consistency_weight=0.1,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.motif_attention.in_proj_weight.grad is not None
    assert model.group_head[-1].weight.grad is not None


def test_collate_molecules_for_a2():
    batch = collate_molecules([molecule("CCO"), molecule("CN")], 5.0)
    assert batch["ptr"].tolist() == [0, 3, 5]
    assert batch["atomic_numbers"].shape[0] == 5
