from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .data import MoleculeStore, collate_molecules, move_molecule_to_device
from .explain import (
    aggregate_cross_fragments,
    aggregate_fragment_response,
    fragment_membership_from_smiles,
)
from .models import DCIR


def load_type_mapping(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            int(row["type_internal"]): row["type_raw"]
            for row in csv.DictReader(handle)
        }


def smiles_from_npz(path: Path) -> str:
    with np.load(path, allow_pickle=False) as archive:
        return str(archive["canonical_smiles"].item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Inference from a cloud-trained model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--d1", required=True)
    parser.add_argument("--d2", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = {key: Path(value) for key, value in config["paths"].items()}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    num_classes = int(checkpoint["num_classes"])
    task_mode = checkpoint["task_mode"]
    model = DCIR(num_classes=num_classes, **config["model"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    store = MoleculeStore(paths["conformers"], seed=int(config.get("seed", 17)))
    cutoff = float(config.get("data", {}).get("cutoff", 5.0))
    mol1 = move_molecule_to_device(
        collate_molecules([store.load(args.d1)], cutoff), device
    )
    mol2 = move_molecule_to_device(
        collate_molecules([store.load(args.d2)], cutoff), device
    )
    with torch.no_grad():
        output = model.predict(
            mol1, mol2, task_mode, return_explanations=True
        )
    probability = output["interaction_probability"][0].cpu().numpy()
    type_map_path = paths["audit_report"].parent / "type_mapping.csv"
    type_mapping = load_type_mapping(type_map_path)
    if task_mode == "multiclass":
        selected = [int(output["predicted_type"][0])]
    else:
        selected = (
            torch.where(output["predicted_type"][0])[0].cpu().tolist()
        )
    conformer_dir = paths["conformers"]
    fragments1 = fragment_membership_from_smiles(
        smiles_from_npz(conformer_dir / f"{args.d1}.npz")
    )
    fragments2 = fragment_membership_from_smiles(
        smiles_from_npz(conformer_dir / f"{args.d2}.npz")
    )
    explanations = {}
    for class_index in selected:
        response1 = (
            output["d1_atom_response"][:, class_index].cpu().numpy()
        )
        response2 = (
            output["d2_atom_response"][:, class_index].cpu().numpy()
        )
        cross12 = (
            output["cross_atom_contributions_12"][0][class_index]
            .cpu()
            .numpy()
        )
        explanations[type_mapping[class_index]] = {
            "d1_atom_response": response1.tolist(),
            "d2_atom_response": response2.tolist(),
            "cross_atom_contributions": cross12.tolist(),
            "fragment_response": {
                "d1": aggregate_fragment_response(response1, fragments1),
                "d2": aggregate_fragment_response(response2, fragments2),
                "cross": aggregate_cross_fragments(
                    cross12, fragments1, fragments2
                ),
            },
        }
    result = {
        "d1": args.d1,
        "d2": args.d2,
        "task_mode": task_mode,
        "interaction_probability": {
            type_mapping[index]: float(value)
            for index, value in enumerate(probability)
        },
        "predicted_type": [type_mapping[index] for index in selected],
        "prediction_confidence": float(
            output["prediction_confidence"][0].cpu()
        ),
        "explanations": explanations,
        "interpretation_notice": (
            "Responses are conditional representation differences, not physical "
            "bond changes or verified mechanisms."
        ),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
