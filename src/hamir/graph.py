from __future__ import annotations

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required in the cloud training environment."
        ) from exc
    return torch


def _radius_edges(
    pos: np.ndarray,
    bond_index: np.ndarray,
    bond_features: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_atoms = len(pos)
    distances = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    source, target = np.where((distances <= cutoff) & (distances > 1e-8))
    spatial = {(int(i), int(j)) for i, j in zip(source, target)}
    bond_lookup = {
        (int(bond_index[0, idx]), int(bond_index[1, idx])): bond_features[idx]
        for idx in range(bond_index.shape[1])
    }
    spatial.update(bond_lookup)
    ordered = sorted(spatial)
    if ordered:
        edge_index = np.asarray(ordered, dtype=np.int64).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
    edge_features = np.zeros((len(ordered), 7), dtype=np.float32)
    for idx, edge in enumerate(ordered):
        if edge in bond_lookup:
            edge_features[idx, :6] = bond_lookup[edge]
        else:
            edge_features[idx, 6] = 1.0
    return edge_index, edge_features


def collate_molecules(molecules: list[dict[str, np.ndarray]], cutoff: float) -> dict:
    torch = _torch()
    atomic_numbers = []
    atom_features = []
    positions = []
    batch = []
    ptr = [0]
    edge_indices = []
    edge_features = []
    offset = 0
    for batch_index, molecule in enumerate(molecules):
        n_atoms = len(molecule["atomic_numbers"])
        local_index, local_features = _radius_edges(
            molecule["pos"],
            molecule["bond_index"],
            molecule["bond_features"],
            cutoff,
        )
        atomic_numbers.append(molecule["atomic_numbers"])
        atom_features.append(molecule["atom_features"])
        positions.append(molecule["pos"])
        batch.append(np.full(n_atoms, batch_index, dtype=np.int64))
        edge_indices.append(local_index + offset)
        edge_features.append(local_features)
        offset += n_atoms
        ptr.append(offset)

    return {
        "atomic_numbers": torch.as_tensor(
            np.concatenate(atomic_numbers), dtype=torch.long
        ),
        "atom_features": torch.as_tensor(
            np.concatenate(atom_features), dtype=torch.float32
        ),
        "pos": torch.as_tensor(np.concatenate(positions), dtype=torch.float32),
        "batch": torch.as_tensor(np.concatenate(batch), dtype=torch.long),
        "ptr": torch.as_tensor(ptr, dtype=torch.long),
        "edge_index": torch.as_tensor(
            np.concatenate(edge_indices, axis=1), dtype=torch.long
        ),
        "edge_features": torch.as_tensor(
            np.concatenate(edge_features), dtype=torch.float32
        ),
    }


def move_molecule_to_device(molecule: dict, device) -> dict:

    return {
        key: (value if key == "ptr" else value.to(device, non_blocking=True))
        for key, value in molecule.items()
    }
