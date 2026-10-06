from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def fragment_membership_from_smiles(smiles: str) -> dict[str, list[int]]:
    """Return deterministic ring and BRICS fragment memberships.

    The resulting groups are structural annotations for aggregating model
    responses. They are not reaction centers or experimentally verified sites.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import BRICS
    except ImportError as exc:
        raise RuntimeError("RDKit is required for fragment aggregation.") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    groups: dict[str, list[int]] = {}
    for index, ring in enumerate(Chem.GetSymmSSSR(mol)):
        groups[f"ring_{index}"] = sorted(int(atom) for atom in ring)

    brics_bonds = [
        mol.GetBondBetweenAtoms(int(a), int(b)).GetIdx()
        for (a, b), _ in BRICS.FindBRICSBonds(mol)
    ]
    if brics_bonds:
        fragmented = Chem.FragmentOnBonds(mol, brics_bonds, addDummies=False)
        for index, atoms in enumerate(Chem.GetMolFrags(fragmented)):
            groups[f"brics_{index}"] = sorted(int(atom) for atom in atoms)
    else:
        groups["brics_0"] = list(range(mol.GetNumAtoms()))
    return groups


def aggregate_fragment_response(
    atom_response: np.ndarray,
    fragments: dict[str, Iterable[int]],
) -> dict[str, dict[str, float | list[int]]]:
    atom_response = np.asarray(atom_response, dtype=float)
    total = float(atom_response.sum())
    result = {}
    for name, members_iter in fragments.items():
        members = sorted(set(int(index) for index in members_iter))
        valid = [index for index in members if 0 <= index < len(atom_response)]
        values = atom_response[valid] if valid else np.asarray([], dtype=float)
        result[name] = {
            "atom_indices": valid,
            "mean_response": float(values.mean()) if len(values) else 0.0,
            "response_mass": float(values.sum() / total) if total > 0 else 0.0,
        }
    return result


def aggregate_cross_fragments(
    contributions: np.ndarray,
    fragments1: dict[str, Iterable[int]],
    fragments2: dict[str, Iterable[int]],
) -> dict[str, float]:
    contributions = np.asarray(contributions, dtype=float)
    result: dict[str, float] = {}
    for name1, members1 in fragments1.items():
        idx1 = [int(index) for index in members1]
        for name2, members2 in fragments2.items():
            idx2 = [int(index) for index in members2]
            if idx1 and idx2:
                result[f"{name1}::{name2}"] = float(
                    contributions[np.ix_(idx1, idx2)].sum()
                )
    return result

