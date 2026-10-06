from __future__ import annotations

import json
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
from dcir.reaction_sites import (
    ReactionSitePaiNN,
    _canonical_component,
    _load_full_checkpoint,
    augment_motif_index,
    brics_motif_ids,
    changed_atom_maps,
    prepare_split,
    reaction_site_collate,
    site_loss,
)
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


def test_changed_maps_and_canonical_label_alignment():
    reactants = Chem.MolFromSmiles(
        "[CH3:1][Br:2].[NH2:3][CH3:4]"
    )
    products = Chem.MolFromSmiles("[CH3:1][NH:3][CH3:4]")
    assert reactants is not None and products is not None
    changed = changed_atom_maps(reactants, products)
    assert {1, 2, 3}.issubset(changed)

    components = Chem.GetMolFrags(reactants, asMols=True)
    first = _canonical_component(components[0], changed)
    second = _canonical_component(components[1], changed)
    assert first["positive_indices"]
    assert second["positive_indices"]
    assert max(first["positive_indices"]) < first["n_atoms"]
    assert max(second["positive_indices"]) < second["n_atoms"]
    assert len(first["motif_ids"]) == first["n_atoms"]
    assert len(second["motif_ids"]) == second["n_atoms"]


def test_brics_motif_partition():
    molecule = Chem.MolFromSmiles("CC(=O)NCC1=CC=CC=C1")
    assert molecule is not None
    motif_ids = brics_motif_ids(molecule)
    assert len(motif_ids) == molecule.GetNumAtoms()
    assert min(motif_ids) == 0
    assert len(set(motif_ids)) >= 2


def test_augment_existing_index_without_raw_csv(tmp_path):
    index = tmp_path / "train.jsonl"
    molecule = Chem.MolFromSmiles("CC(=O)NCC1=CC=CC=C1")
    assert molecule is not None
    item = {
        "key": "unchanged-key",
        "smiles": Chem.MolToSmiles(molecule),
        "n_atoms": molecule.GetNumAtoms(),
        "positive_indices": [1],
        "atom_maps": [0] * molecule.GetNumAtoms(),
    }
    record = {
        "sample_id": "unchanged-sample",
        "reaction_class": 1,
        "d1": dict(item),
        "d2": dict(item),
    }
    index.write_text(json.dumps(record) + "\n", encoding="utf-8")
    report = augment_motif_index(index)
    updated = json.loads(index.read_text(encoding="utf-8"))
    assert report["rows"] == 1
    assert updated["sample_id"] == record["sample_id"]
    assert updated["d1"]["key"] == item["key"]
    assert updated["d1"]["positive_indices"] == item["positive_indices"]
    assert len(updated["d1"]["motif_ids"]) == item["n_atoms"]


def test_prepare_missing_source_does_not_truncate_existing_index(tmp_path):
    index = tmp_path / "train.jsonl"
    index.write_text("existing-index\n", encoding="utf-8")
    try:
        prepare_split(
            tmp_path / "missing.csv",
            index,
            max_atoms=128,
            component_policy="exactly-two",
            require_positive_both=True,
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing source should raise FileNotFoundError")
    assert index.read_text(encoding="utf-8") == "existing-index\n"


def test_reaction_site_painn_forward_and_loss():
    d1 = collate_molecules([_molecule("CCBr"), _molecule("CCO")], 5.0)
    d2 = collate_molecules([_molecule("CN"), _molecule("NC=O")], 5.0)
    model = ReactionSitePaiNN(
        hidden_dim=32,
        painn_layers=1,
        num_rbf=8,
        cutoff=5.0,
        dropout=0.0,
    )
    output = model(d1, d2)
    assert output["logits1"].shape == d1["atomic_numbers"].shape
    assert output["logits2"].shape == d2["atomic_numbers"].shape

    batch = {
        "target1": torch.zeros_like(output["logits1"]),
        "target2": torch.zeros_like(output["logits2"]),
    }
    batch["target1"][0] = 1.0
    batch["target2"][0] = 1.0
    loss = site_loss(
        output,
        batch,
        pos_weight=torch.tensor(4.0),
        focal_gamma=2.0,
        dice_weight=0.5,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.site_head[-1].weight.grad is not None


def test_full_checkpoint_initialization_loads_weights_only(tmp_path):
    source = ReactionSitePaiNN(
        hidden_dim=32,
        painn_layers=1,
        num_rbf=8,
        cutoff=5.0,
        dropout=0.0,
    )
    destination = ReactionSitePaiNN(
        hidden_dim=32,
        painn_layers=1,
        num_rbf=8,
        cutoff=5.0,
        dropout=0.0,
    )
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "epoch": 7,
            "validation": {"atom_f1": 0.5},
            "model_state": source.state_dict(),
            "optimizer_state": {"must_not_be_loaded": True},
        },
        checkpoint,
    )
    _load_full_checkpoint(destination, checkpoint)
    for source_parameter, destination_parameter in zip(
        source.parameters(), destination.parameters()
    ):
        assert torch.equal(source_parameter, destination_parameter)


def test_sparse_cross_attention_uses_preliminary_logits():
    d1 = collate_molecules([_molecule("CCBr"), _molecule("CCO")], 5.0)
    d2 = collate_molecules([_molecule("CN"), _molecule("NC=O")], 5.0)
    model = ReactionSitePaiNN(
        hidden_dim=32,
        painn_layers=1,
        num_rbf=8,
        cutoff=5.0,
        dropout=0.0,
        architecture="sparse_cross_attention",
        attention_heads=4,
        attention_topk=2,
    )
    output = model(d1, d2)
    assert output["preliminary_logits1"].shape == d1["atomic_numbers"].shape
    assert output["preliminary_logits2"].shape == d2["atomic_numbers"].shape
    batch = {
        "target1": torch.zeros_like(output["logits1"]),
        "target2": torch.zeros_like(output["logits2"]),
    }
    batch["target1"][0] = 1.0
    batch["target2"][0] = 1.0
    loss = site_loss(
        output,
        batch,
        pos_weight=torch.tensor(4.0),
        focal_gamma=2.0,
        dice_weight=0.5,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.cross_attention.in_proj_weight.grad is not None


def test_hierarchical_motif_forward_and_losses():
    raw_items = []
    for index, (first, second) in enumerate(
        [("CC(=O)O", "CN"), ("CCNC", "NC=O")]
    ):
        molecule1 = Chem.MolFromSmiles(first)
        molecule2 = Chem.MolFromSmiles(second)
        assert molecule1 is not None and molecule2 is not None
        target1 = np.zeros(molecule1.GetNumAtoms(), dtype=np.float32)
        target2 = np.zeros(molecule2.GetNumAtoms(), dtype=np.float32)
        target1[0] = 1.0
        target2[0] = 1.0
        raw_items.append(
            {
                "d1": _molecule(first),
                "d2": _molecule(second),
                "target1": target1,
                "target2": target2,
                "motif_ids1": np.asarray(
                    brics_motif_ids(molecule1), dtype=np.int64
                ),
                "motif_ids2": np.asarray(
                    brics_motif_ids(molecule2), dtype=np.int64
                ),
                "sample_id": str(index),
            }
        )
    batch = reaction_site_collate(5.0)(raw_items)
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
    assert output["group_logits1"].shape == batch["motif_target1"].shape
    assert output["group_logits2"].shape == batch["motif_target2"].shape
    loss = site_loss(
        output,
        batch,
        pos_weight=torch.tensor(4.0),
        group_pos_weight=torch.tensor(2.0),
        focal_gamma=2.0,
        dice_weight=0.5,
        group_weight=0.2,
        consistency_weight=0.1,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.motif_attention.in_proj_weight.grad is not None
    assert model.group_head[-1].weight.grad is not None
