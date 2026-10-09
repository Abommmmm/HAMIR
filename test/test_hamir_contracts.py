import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

from hamir import HAMIR, load_config
from hamir.predict import predict_pair
from hamir.reaction_sites import _build_model, brics_motif_ids, reaction_site_collate
from test_hamir_core import molecule


def pair_batch():
    item = {"sample_id": "test", "reaction_class": 0}
    for role, smiles in enumerate(("CC(=O)O", "CN"), 1):
        mol = Chem.MolFromSmiles(smiles)
        item[f"d{role}"] = molecule(smiles)
        item[f"target{role}"] = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        item[f"motif_ids{role}"] = np.asarray(brics_motif_ids(mol))
    return reaction_site_collate(5.0)([item])


def test_geometry_invariance_and_reactant_swap():
    torch.manual_seed(17)
    model = HAMIR(hidden_dim=32, painn_layers=1, num_rbf=8, dropout=0).eval()
    batch = pair_batch()
    transformed = copy.deepcopy(batch)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for role in ("d1", "d2"):
        transformed[role]["pos"] = transformed[role]["pos"] @ rotation + 3.0
    with torch.no_grad():
        original = model(batch["d1"], batch["d2"])
        rotated = model(transformed["d1"], transformed["d2"])
        swapped = model(batch["d2"], batch["d1"])
    for key in original:
        torch.testing.assert_close(original[key], rotated[key], rtol=1e-5, atol=1e-6)
    for prefix in ("logits", "preliminary_logits", "group_logits"):
        torch.testing.assert_close(original[prefix + "1"], swapped[prefix + "2"])
        torch.testing.assert_close(original[prefix + "2"], swapped[prefix + "1"])


def test_all_release_configs_resolve_and_build():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "configs/reaction_sites").glob("*.yaml"))
    assert len(paths) == 6
    for path in paths:
        config = load_config(path)
        assert Path(config["_project_root"]) == root
        assert all(Path(p).is_absolute() for p in config["paths"].values())
        assert isinstance(_build_model(config), HAMIR)


@pytest.mark.parametrize(
    "kwargs", [{"attention_topk": 0}, {"attention_heads": 0}, {"cutoff": 0}]
)
def test_invalid_model_parameters_fail_early(kwargs):
    with pytest.raises(ValueError):
        HAMIR(**kwargs)


def test_smiles_prediction_without_product_or_labels():
    config = {"model": {"encoder_type": "bond_mpnn", "cutoff": 5.0}}
    model = HAMIR(hidden_dim=32, painn_layers=1, encoder_type="bond_mpnn").eval()
    result = predict_pair(model, "CCO", "CN", config)
    assert len(result["reactant1"]["atoms"]) == 3
    assert len(result["reactant2"]["atoms"]) == 2
    assert result["threshold"] == 0.5
    assert all(0 <= a["probability"] <= 1 for a in result["reactant1"]["atoms"])
    with pytest.raises(ValueError):
        predict_pair(model, "C.C", "CN", config)
