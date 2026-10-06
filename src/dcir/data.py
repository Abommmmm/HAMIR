from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required in the cloud training environment.") from exc
    return torch


@dataclass
class PairExample:
    record_ids: list[int]
    d1: str
    d2: str
    labels: list[int]


class MoleculeStore:
    def __init__(self, root: str | Path, seed: int = 17):
        self.root = Path(root)
        self.rng = random.Random(seed)

    def load(self, drug_id: str, *, alternate: bool = False) -> dict[str, np.ndarray]:
        path = self.root / f"{drug_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing conformer file: {path}")
        with np.load(path, allow_pickle=False) as archive:
            positions = np.asarray(archive["positions"], dtype=np.float32)
            if len(positions) == 1:
                conf_id = 0
            elif alternate:
                conf_id = self.rng.randrange(1, len(positions))
            else:
                conf_id = self.rng.randrange(len(positions))
            return {
                "atomic_numbers": np.asarray(archive["atomic_numbers"], dtype=np.int64),
                "atom_features": np.asarray(archive["atom_features"], dtype=np.float32),
                "bond_index": np.asarray(archive["bond_index"], dtype=np.int64),
                "bond_features": np.asarray(archive["bond_features"], dtype=np.float32),
                "pos": positions[conf_id],
            }

    def load_pair(
        self, drug_id: str
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Load two distinct conformers with a single NPZ read when possible."""
        path = self.root / f"{drug_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing conformer file: {path}")
        with np.load(path, allow_pickle=False) as archive:
            positions = np.asarray(archive["positions"], dtype=np.float32)
            primary_id = self.rng.randrange(len(positions))
            if len(positions) == 1:
                alternate_id = primary_id
            else:
                alternate_id = (
                    primary_id + self.rng.randrange(1, len(positions))
                ) % len(positions)
            static = {
                "atomic_numbers": np.asarray(
                    archive["atomic_numbers"], dtype=np.int64
                ),
                "atom_features": np.asarray(
                    archive["atom_features"], dtype=np.float32
                ),
                "bond_index": np.asarray(archive["bond_index"], dtype=np.int64),
                "bond_features": np.asarray(
                    archive["bond_features"], dtype=np.float32
                ),
            }
            primary = dict(static)
            primary["pos"] = positions[primary_id]
            alternate = dict(static)
            alternate["pos"] = positions[alternate_id]
            return primary, alternate


def _load_examples(records_path: Path, split_path: Path, split: str) -> list[PairExample]:
    split_by_id: dict[int, str] = {}
    with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split_by_id[int(row["record_id"])] = row["split"].strip()

    grouped: dict[tuple[str, str], PairExample] = {}
    with records_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            record_id = int(row["record_id"])
            if split_by_id.get(record_id) != split:
                continue
            key = (row["d1"].strip(), row["d2"].strip())
            label = int(row["type_internal"])
            if key not in grouped:
                grouped[key] = PairExample([record_id], key[0], key[1], [label])
            else:
                grouped[key].record_ids.append(record_id)
                if label not in grouped[key].labels:
                    grouped[key].labels.append(label)
    return list(grouped.values())


class DDIPairDataset:
    def __init__(
        self,
        records_path: str | Path,
        split_path: str | Path,
        conformer_dir: str | Path,
        split: str,
        num_classes: int,
        task_mode: str,
        seed: int = 17,
        include_alternates: bool = False,
    ):
        self.examples = _load_examples(Path(records_path), Path(split_path), split)
        self.store = MoleculeStore(conformer_dir, seed=seed)
        self.num_classes = int(num_classes)
        self.task_mode = task_mode
        self.include_alternates = bool(include_alternates)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        if self.task_mode == "multiclass":
            target: Any = example.labels[0]
        else:
            target = np.zeros(self.num_classes, dtype=np.float32)
            target[example.labels] = 1.0
        if self.include_alternates:
            d1, d1_alt = self.store.load_pair(example.d1)
            d2, d2_alt = self.store.load_pair(example.d2)
        else:
            d1 = self.store.load(example.d1)
            d2 = self.store.load(example.d2)
        result = {
            "d1": d1,
            "d2": d2,
            "target": target,
            "drug_ids": (example.d1, example.d2),
            "record_ids": example.record_ids,
        }
        if self.include_alternates:
            result["d1_alt"] = d1_alt
            result["d2_alt"] = d2_alt
        return result


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
        "atomic_numbers": torch.as_tensor(np.concatenate(atomic_numbers), dtype=torch.long),
        "atom_features": torch.as_tensor(np.concatenate(atom_features), dtype=torch.float32),
        "pos": torch.as_tensor(np.concatenate(positions), dtype=torch.float32),
        "batch": torch.as_tensor(np.concatenate(batch), dtype=torch.long),
        "ptr": torch.as_tensor(ptr, dtype=torch.long),
        "edge_index": torch.as_tensor(np.concatenate(edge_indices, axis=1), dtype=torch.long),
        "edge_features": torch.as_tensor(np.concatenate(edge_features), dtype=torch.float32),
    }


def make_collate_fn(task_mode: str, cutoff: float = 5.0):
    def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        torch = _torch()
        if task_mode == "multiclass":
            target = torch.as_tensor([item["target"] for item in items], dtype=torch.long)
        else:
            target = torch.as_tensor(
                np.stack([item["target"] for item in items]), dtype=torch.float32
            )
        result = {
            "d1": collate_molecules([item["d1"] for item in items], cutoff),
            "d2": collate_molecules([item["d2"] for item in items], cutoff),
            "target": target,
            "drug_ids": [item["drug_ids"] for item in items],
            "record_ids": [item["record_ids"] for item in items],
        }
        if "d1_alt" in items[0]:
            result["d1_alt"] = collate_molecules(
                [item["d1_alt"] for item in items], cutoff
            )
            result["d2_alt"] = collate_molecules(
                [item["d2_alt"] for item in items], cutoff
            )
        return result

    return collate


def move_molecule_to_device(molecule: dict, device) -> dict:
    # ptr is used only for Python slicing. Keeping it on CPU avoids one
    # device-wide synchronization for every boundary converted with int().
    return {
        key: (
            value
            if key == "ptr"
            else value.to(device, non_blocking=True)
        )
        for key, value in molecule.items()
    }


def move_batch_to_device(batch: dict, device, include_alt: bool = True) -> dict:
    result = dict(batch)
    result["d1"] = move_molecule_to_device(batch["d1"], device)
    result["d2"] = move_molecule_to_device(batch["d2"], device)
    if include_alt and "d1_alt" in batch:
        result["d1_alt"] = move_molecule_to_device(batch["d1_alt"], device)
        result["d2_alt"] = move_molecule_to_device(batch["d2_alt"], device)
    result["target"] = batch["target"].to(device, non_blocking=True)
    return result
