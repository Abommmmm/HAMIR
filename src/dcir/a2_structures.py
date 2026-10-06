from __future__ import annotations

import numpy as np


def atom_features(mol) -> tuple[np.ndarray, np.ndarray]:
    atomic_numbers: list[int] = []
    features: list[list[float]] = []
    for atom in mol.GetAtoms():
        atomic_numbers.append(atom.GetAtomicNum())
        features.append(
            [
                float(atom.GetFormalCharge()),
                float(atom.GetIsAromatic()),
                float(atom.GetTotalDegree()),
                float(atom.GetTotalNumHs()),
                float(int(atom.GetHybridization())),
                float(atom.IsInRing()),
                float(int(atom.GetChiralTag())),
            ]
        )
    return np.asarray(atomic_numbers, dtype=np.int64), np.asarray(features, dtype=np.float32)


def bond_features(mol) -> tuple[np.ndarray, np.ndarray]:
    indices: list[tuple[int, int]] = []
    features: list[list[float]] = []
    bond_types = {
        "SINGLE": 0,
        "DOUBLE": 1,
        "TRIPLE": 2,
        "AROMATIC": 3,
    }
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        one_hot = [0.0] * 6
        one_hot[bond_types.get(str(bond.GetBondType()), 4)] = 1.0
        one_hot[4] = float(bond.GetIsConjugated())
        one_hot[5] = float(bond.IsInRing())
        for source, target in ((begin, end), (end, begin)):
            indices.append((source, target))
            features.append(one_hot)
    if not indices:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 6), dtype=np.float32)
    return np.asarray(indices, dtype=np.int64).T, np.asarray(features, dtype=np.float32)
