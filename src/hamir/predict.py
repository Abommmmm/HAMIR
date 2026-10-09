"""Predict reaction-site probabilities from two reactant SMILES."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from .config import load_config
from .features import atom_features, bond_features
from .reaction_sites import (
    _build_model,
    _geometry_payload,
    _move_batch,
    brics_motif_ids,
    reaction_site_collate,
)


def prepare_reactant(smiles, config):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or not molecule.GetNumAtoms():
        raise ValueError(f"Invalid reactant SMILES: {smiles}")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("Supply one connected molecule per reactant")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    molecule = Chem.MolFromSmiles(canonical)
    limit = int(config.get("data", {}).get("max_atoms_per_molecule", 128))
    if molecule.GetNumAtoms() > limit:
        raise ValueError(f"Reactant exceeds configured atom limit {limit}")
    if config["model"].get("encoder_type", "painn") == "bond_mpnn":
        numbers, features = atom_features(molecule)
        edges, edge_features = bond_features(molecule)
        payload = dict(
            atomic_numbers=numbers,
            atom_features=features,
            bond_index=edges,
            bond_features=edge_features,
        )
        geometry = "covalent-graph"
    else:
        key = hashlib.sha256(canonical.encode()).hexdigest()[:24]
        _, payload, geometry = _geometry_payload(
            (
                key,
                canonical,
                int(config.get("seed", 17)),
                int(config.get("data", {}).get("conformer_max_iters", 100)),
            )
        )
    return canonical, molecule, payload, geometry


def predict_pair(model, smiles1, smiles2, config, device="cpu"):
    prepared = [prepare_reactant(s, config) for s in (smiles1, smiles2)]
    item = {"sample_id": "prediction", "reaction_class": 0}
    for role, (_, molecule, payload, _) in enumerate(prepared, 1):
        item[f"d{role}"] = payload
        item[f"target{role}"] = np.zeros(molecule.GetNumAtoms(), dtype=np.float32)
        item[f"motif_ids{role}"] = np.asarray(brics_motif_ids(molecule), dtype=np.int64)
    geometry = config["model"].get("encoder_type", "painn") != "bond_mpnn"
    batch = reaction_site_collate(
        float(config["model"].get("cutoff", 5.0)), geometry=geometry
    )([item])
    model = model.to(device).eval()
    with torch.inference_mode():
        batch = _move_batch(batch, torch.device(device))
        output = model(batch["d1"], batch["d2"])
    result = {
        "threshold": 0.5,
        "atom_indexing": "zero-based RDKit order of canonical_smiles",
    }
    for role, (canonical, molecule, _, geometry_name) in enumerate(prepared, 1):
        probabilities = output[f"logits{role}"].sigmoid().cpu().tolist()
        result[f"reactant{role}"] = {
            "canonical_smiles": canonical,
            "geometry": geometry_name,
            "atoms": [
                {
                    "index": atom.GetIdx(),
                    "element": atom.GetSymbol(),
                    "motif_id": int(item[f"motif_ids{role}"][atom.GetIdx()]),
                    "probability": probabilities[atom.GetIdx()],
                    "predicted_site": probabilities[atom.GetIdx()] >= 0.5,
                }
                for atom in molecule.GetAtoms()
            ],
            "motif_probabilities": output[f"group_logits{role}"]
            .sigmoid()
            .cpu()
            .tolist(),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--smiles1", required=True)
    parser.add_argument("--smiles2", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    model = _build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=True)
    result = predict_pair(model, args.smiles1, args.smiles2, config, args.device)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
