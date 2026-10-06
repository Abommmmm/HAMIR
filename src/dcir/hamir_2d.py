"""Geometry-free encoder and batching for the internal HAMIR control."""
import numpy as np
import torch
from torch import nn
from .a2_bond_layers import BondMessageLayer


def collate_covalent(molecules):
    ptr = np.cumsum([0] + [len(m["atomic_numbers"]) for m in molecules])
    return {
        "atomic_numbers": torch.as_tensor(np.concatenate([m["atomic_numbers"] for m in molecules]), dtype=torch.long),
        "atom_features": torch.as_tensor(np.concatenate([m["atom_features"] for m in molecules]), dtype=torch.float32),
        "edge_index": torch.as_tensor(np.concatenate([m["bond_index"] + ptr[i] for i, m in enumerate(molecules)], axis=1), dtype=torch.long),
        "edge_features": torch.as_tensor(np.concatenate([m["bond_features"] for m in molecules]), dtype=torch.float32),
        "batch": torch.repeat_interleave(torch.arange(len(molecules)), torch.as_tensor(np.diff(ptr))),
        "ptr": torch.as_tensor(ptr, dtype=torch.long),
    }


class BondEncoder(nn.Module):
    def __init__(self, hidden_dim=128, layers=5):
        super().__init__()
        # Same atom input transformation as PaiNN. PaiNN has no encoder dropout.
        self.atom_embedding = nn.Embedding(119, hidden_dim)
        self.atom_projection = nn.Sequential(nn.Linear(7, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(BondMessageLayer(hidden_dim, 0.0) for _ in range(layers))

    def forward(self, molecule):
        atoms = self.atom_embedding(molecule["atomic_numbers"].clamp(0, 118)) + self.atom_projection(molecule["atom_features"])
        features = molecule["edge_features"]
        # Standard 2D batches contain exactly the cached covalent graph. Also
        # accept legacy radius batches for invariance tests, filtering spatial edges.
        if features.shape[1] == 7:
            mask = features[:, 6] == 0
            edges, features = molecule["edge_index"][:, mask], features[mask, :6]
        else:
            if features.shape[1] != 6:
                raise ValueError("Expected six cached chemical bond features")
            edges = molecule["edge_index"]
        for layer in self.layers:
            atoms = layer(atoms, edges, features)
        return atoms
