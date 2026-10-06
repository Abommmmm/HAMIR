from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import load_config
from .data import collate_molecules, move_molecule_to_device
from .models import PaiNNEncoder, segment_mean
from .structures import atom_features, bond_features


def _rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for reaction-site preparation.") from exc
    return Chem, AllChem


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _bond_map(mol, allowed_maps: set[int] | None = None) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for bond in mol.GetBonds():
        first = int(bond.GetBeginAtom().GetAtomMapNum())
        second = int(bond.GetEndAtom().GetAtomMapNum())
        if first <= 0 or second <= 0:
            continue
        if allowed_maps is not None and not ({first, second} & allowed_maps):
            continue
        result[tuple(sorted((first, second)))] = float(bond.GetBondTypeAsDouble())
    return result


def _atom_map(mol) -> dict[int, tuple[int, int, int, int]]:
    result = {}
    for atom in mol.GetAtoms():
        atom_map = int(atom.GetAtomMapNum())
        if atom_map <= 0:
            continue
        result[atom_map] = (
            int(atom.GetAtomicNum()),
            int(atom.GetFormalCharge()),
            int(atom.GetIsAromatic()),
            int(atom.GetChiralTag()),
        )
    return result


def changed_atom_maps(reactants, products) -> set[int]:
    """Return reactant atom-map numbers whose local chemistry changes.

    Whole components absent from the product are treated as reagents rather
    than marking every atom as reactive. A disappearing bond is counted only
    when at least one endpoint remains in the product.
    """

    product_atoms = _atom_map(products)
    product_maps = set(product_atoms)
    reactant_atoms = _atom_map(reactants)
    left_bonds = _bond_map(reactants, allowed_maps=product_maps)
    right_bonds = _bond_map(products)
    changed: set[int] = set()
    for key in set(left_bonds) | set(right_bonds):
        if not math.isclose(
            left_bonds.get(key, -1.0),
            right_bonds.get(key, -1.0),
            abs_tol=1e-6,
        ):
            changed.update(key)
    for atom_map in set(reactant_atoms) & product_maps:
        if reactant_atoms[atom_map] != product_atoms[atom_map]:
            changed.add(atom_map)
    return changed


def brics_motif_ids(molecule) -> list[int]:
    """Partition atoms into BRICS-connected motifs without adding dummy atoms."""

    try:
        from rdkit.Chem import BRICS
    except ImportError as exc:
        raise RuntimeError("RDKit BRICS support is required.") from exc
    cut_edges = {
        tuple(sorted((int(first), int(second))))
        for (first, second), _ in BRICS.FindBRICSBonds(molecule)
    }
    adjacency: list[list[int]] = [[] for _ in range(molecule.GetNumAtoms())]
    for bond in molecule.GetBonds():
        first = int(bond.GetBeginAtomIdx())
        second = int(bond.GetEndAtomIdx())
        if tuple(sorted((first, second))) in cut_edges:
            continue
        adjacency[first].append(second)
        adjacency[second].append(first)
    motif_ids = [-1] * molecule.GetNumAtoms()
    motif = 0
    for start in range(molecule.GetNumAtoms()):
        if motif_ids[start] >= 0:
            continue
        stack = [start]
        motif_ids[start] = motif
        while stack:
            atom = stack.pop()
            for neighbor in adjacency[atom]:
                if motif_ids[neighbor] < 0:
                    motif_ids[neighbor] = motif
                    stack.append(neighbor)
        motif += 1
    return motif_ids


def _canonical_component(component, positive_maps: set[int]) -> dict[str, Any]:
    """Canonicalize a component and align atom labels to canonical atom order."""

    Chem, _ = _rdkit()
    source = Chem.Mol(component)
    source_labels = [
        int(atom.GetAtomMapNum()) in positive_maps for atom in source.GetAtoms()
    ]
    source_maps = [int(atom.GetAtomMapNum()) for atom in source.GetAtoms()]
    for atom in source.GetAtoms():
        atom.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(source, canonical=True, isomericSmiles=True)
    canonical = Chem.MolFromSmiles(smiles)
    if canonical is None:
        raise ValueError("Canonical component could not be parsed")
    match = canonical.GetSubstructMatch(source, useChirality=True)
    if len(match) != source.GetNumAtoms():
        match = canonical.GetSubstructMatch(source, useChirality=False)
    if len(match) != source.GetNumAtoms():
        raise ValueError("Could not align component to canonical atom order")
    labels = np.zeros(canonical.GetNumAtoms(), dtype=np.uint8)
    atom_maps = np.zeros(canonical.GetNumAtoms(), dtype=np.int64)
    for source_index, canonical_index in enumerate(match):
        labels[canonical_index] = source_labels[source_index]
        atom_maps[canonical_index] = source_maps[source_index]
    key = hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:24]
    motif_ids = brics_motif_ids(canonical)
    return {
        "key": key,
        "smiles": smiles,
        "n_atoms": int(canonical.GetNumAtoms()),
        "positive_indices": np.flatnonzero(labels).astype(int).tolist(),
        "atom_maps": atom_maps.astype(int).tolist(),
        "motif_ids": motif_ids,
        "n_motifs": max(motif_ids, default=-1) + 1,
    }


def _iter_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def prepare_split(
    source: Path,
    destination: Path,
    *,
    max_atoms: int,
    component_policy: str,
    require_positive_both: bool,
) -> dict[str, int]:
    Chem, _ = _rdkit()
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing raw USPTO split; existing index was not modified: {source}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.prepare-tmp")
    counters = {
        "source_rows": 0,
        "written": 0,
        "parse_failed": 0,
        "wrong_component_count": 0,
        "too_many_atoms": 0,
        "no_changed_atoms": 0,
        "missing_positive_component": 0,
    }
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row_index, row in enumerate(
            tqdm(_iter_csv(source), desc=f"prepare {source.stem}", dynamic_ncols=True)
        ):
            counters["source_rows"] += 1
            reaction = (row.get("rxn_smiles") or row.get("reactions") or "").strip()
            parts = reaction.split(">")
            if len(parts) != 3:
                counters["parse_failed"] += 1
                continue
            reactants = Chem.MolFromSmiles(parts[0])
            products = Chem.MolFromSmiles(parts[2])
            if reactants is None or products is None:
                counters["parse_failed"] += 1
                continue
            product_maps = {
                int(atom.GetAtomMapNum())
                for atom in products.GetAtoms()
                if atom.GetAtomMapNum() > 0
            }
            changed = changed_atom_maps(reactants, products)
            if not changed:
                counters["no_changed_atoms"] += 1
                continue
            components = []
            for component in Chem.GetMolFrags(
                reactants, asMols=True, sanitizeFrags=True
            ):
                maps = {
                    int(atom.GetAtomMapNum())
                    for atom in component.GetAtoms()
                    if atom.GetAtomMapNum() > 0
                }
                if maps & product_maps:
                    components.append(component)
            if component_policy == "exactly-two":
                if len(components) != 2:
                    counters["wrong_component_count"] += 1
                    continue
            elif component_policy == "top-two":
                if len(components) < 2:
                    counters["wrong_component_count"] += 1
                    continue
                components.sort(
                    key=lambda mol: (
                        sum(
                            int(atom.GetAtomMapNum()) in changed
                            for atom in mol.GetAtoms()
                        ),
                        mol.GetNumHeavyAtoms(),
                    ),
                    reverse=True,
                )
                components = components[:2]
            else:
                raise ValueError(f"Unsupported component policy: {component_policy}")
            try:
                first = _canonical_component(components[0], changed)
                second = _canonical_component(components[1], changed)
            except (ValueError, RuntimeError):
                counters["parse_failed"] += 1
                continue
            if max(first["n_atoms"], second["n_atoms"]) > max_atoms:
                counters["too_many_atoms"] += 1
                continue
            if require_positive_both and (
                not first["positive_indices"] or not second["positive_indices"]
            ):
                counters["missing_positive_component"] += 1
                continue
            record = {
                "sample_id": str(row.get("id") or f"{source.stem}:{row_index}"),
                "reaction_class": int(row.get("class") or 0),
                "d1": first,
                "d2": second,
            }
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            counters["written"] += 1
    temporary.replace(destination)
    return counters


def prepare_command(config: dict[str, Any]) -> None:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    data_config = config.get("data", {})
    max_atoms = int(data_config.get("max_atoms_per_molecule", 128))
    component_policy = str(data_config.get("component_policy", "exactly-two"))
    require_positive_both = bool(data_config.get("require_positive_both", True))
    report: dict[str, Any] = {
        "max_atoms_per_molecule": max_atoms,
        "component_policy": component_policy,
        "require_positive_both": require_positive_both,
        "splits": {},
    }
    for split in ("train", "valid", "test"):
        report["splits"][split] = prepare_split(
            paths[f"uspto_{split}"],
            paths[f"{split}_index"],
            max_atoms=max_atoms,
            component_policy=component_policy,
            require_positive_both=require_positive_both,
        )
    report_path = paths["preparation_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Preparation report written to {report_path}")


def augment_motif_index(path: Path) -> dict[str, int]:
    """Add BRICS metadata to an existing index without needing the raw CSV."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing reaction-site index: {path}")
    Chem, _ = _rdkit()
    temporary = path.with_name(f"{path.name}.motif-tmp")
    rows = molecules = motifs = 0
    try:
        with (
            path.open("r", encoding="utf-8") as source,
            temporary.open("w", encoding="utf-8", newline="\n") as output,
        ):
            for line_number, line in enumerate(
                tqdm(source, desc=f"augment {path.stem}", dynamic_ncols=True),
                start=1,
            ):
                record = json.loads(line)
                for role in ("d1", "d2"):
                    item = record[role]
                    molecule = Chem.MolFromSmiles(item["smiles"])
                    if molecule is None:
                        raise ValueError(
                            f"Invalid SMILES in {path}:{line_number}:{role}"
                        )
                    if molecule.GetNumAtoms() != int(item["n_atoms"]):
                        raise ValueError(
                            "Canonical atom count changed for "
                            f"{path}:{line_number}:{role}"
                        )
                    motif_ids = brics_motif_ids(molecule)
                    item["motif_ids"] = motif_ids
                    item["n_motifs"] = max(motif_ids, default=-1) + 1
                    molecules += 1
                    motifs += item["n_motifs"]
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
                rows += 1
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"rows": rows, "molecules": molecules, "motifs": motifs}


def augment_motifs_command(config: dict[str, Any]) -> None:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    report = {
        split: augment_motif_index(paths[f"{split}_index"])
        for split in ("train", "valid", "test")
    }
    report_path = paths["preparation_report"].with_name(
        "motif_augmentation_report.json"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Motif augmentation report written to {report_path}")


def _geometry_payload(item: tuple[str, str, int, int]) -> tuple[str, dict, str]:
    key, smiles, seed, max_opt_iters = item
    Chem, AllChem = _rdkit()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse cached molecule {key}")
    molecule_h = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(
        int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16)
        & 0x7FFFFFFF
    )
    params.enforceChirality = True
    params.useRandomCoords = False
    status = AllChem.EmbedMolecule(molecule_h, params)
    geometry = "ETKDGv3"
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(molecule_h, params)
        geometry = "ETKDGv3-random"
    if status == 0:
        try:
            if AllChem.MMFFHasAllMoleculeParams(molecule_h):
                AllChem.MMFFOptimizeMolecule(
                    molecule_h, mmffVariant="MMFF94s", maxIters=max_opt_iters
                )
                geometry += "+MMFF94s"
            else:
                AllChem.UFFOptimizeMolecule(molecule_h, maxIters=max_opt_iters)
                geometry += "+UFF"
            heavy = Chem.RemoveHs(molecule_h)
            positions = np.asarray(
                heavy.GetConformer().GetPositions(), dtype=np.float32
            )
        except Exception:
            status = -1
    if status != 0:
        fallback = Chem.Mol(molecule)
        AllChem.Compute2DCoords(fallback)
        positions = np.asarray(
            fallback.GetConformer().GetPositions(), dtype=np.float32
        )
        geometry = "2D-fallback"
        heavy = fallback
    positions -= positions.mean(axis=0, keepdims=True)
    numbers, atom_attr = atom_features(heavy)
    bond_index, bond_attr = bond_features(heavy)
    payload = {
        "atomic_numbers": numbers,
        "atom_features": atom_attr,
        "bond_index": bond_index,
        "bond_features": bond_attr,
        "pos": positions,
    }
    return key, payload, geometry


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_3d_command(
    config: dict[str, Any],
    *,
    workers: int,
    overwrite: bool,
    splits: tuple[str, ...] = ("train", "valid", "test"),
) -> None:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    molecules: dict[str, str] = {}
    for split in splits:
        for record in _load_jsonl(paths[f"{split}_index"]):
            for role in ("d1", "d2"):
                item = record[role]
                previous = molecules.setdefault(item["key"], item["smiles"])
                if previous != item["smiles"]:
                    raise ValueError(f"Molecule key collision: {item['key']}")
    cache = paths["molecule_cache"]
    cache.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 17))
    max_opt_iters = int(config.get("data", {}).get("conformer_max_iters", 100))
    pending = [
        (key, smiles, seed, max_opt_iters)
        for key, smiles in molecules.items()
        if overwrite or not (cache / f"{key}.npz").is_file()
    ]
    counts: dict[str, int] = {}
    if workers <= 1:
        iterator = map(_geometry_payload, pending)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_geometry_payload, pending, chunksize=16)
    try:
        for key, payload, geometry in tqdm(
            iterator, total=len(pending), desc="build 3D", dynamic_ncols=True
        ):
            np.savez_compressed(cache / f"{key}.npz", **payload)
            counts[geometry] = counts.get(geometry, 0) + 1
    finally:
        if executor is not None:
            executor.shutdown()
    manifest = {
        "unique_molecules": len(molecules),
        "generated_now": len(pending),
        "geometry_counts": counts,
        "cache": str(cache),
    }
    manifest_path = paths["geometry_report"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


class ReactionSiteDataset(Dataset):
    def __init__(self, index_path: Path, molecule_cache: Path, *, geometry: bool = True):
        self.records = _load_jsonl(index_path)
        self.molecule_cache = molecule_cache
        self.geometry = geometry

    def __len__(self) -> int:
        return len(self.records)

    def _molecule(
        self, item: dict[str, Any]
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        path = self.molecule_cache / f"{item['key']}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing molecule cache: {path}. Run build-3d first."
            )
        with np.load(path, allow_pickle=False) as archive:
            molecule = {
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
            if self.geometry:
                molecule["pos"] = np.asarray(archive["pos"], dtype=np.float32)
        target = np.zeros(len(molecule["atomic_numbers"]), dtype=np.float32)
        target[np.asarray(item["positive_indices"], dtype=np.int64)] = 1.0
        motif_ids = np.asarray(
            item.get("motif_ids", np.zeros(len(target), dtype=np.int64)),
            dtype=np.int64,
        )
        if len(motif_ids) != len(target):
            raise ValueError(
                f"Motif/atom length mismatch for molecule {item['key']}"
            )
        return molecule, target, motif_ids

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        d1, target1, motif_ids1 = self._molecule(record["d1"])
        d2, target2, motif_ids2 = self._molecule(record["d2"])
        return {
            "d1": d1,
            "d2": d2,
            "target1": target1,
            "target2": target2,
            "motif_ids1": motif_ids1,
            "motif_ids2": motif_ids2,
            "reaction_class": int(record.get("reaction_class", 0)),
            "sample_id": record["sample_id"],
        }

    def positive_weights(self, cap: float) -> tuple[float, float]:
        atom_positives = 0
        atoms = 0
        motif_positives = 0
        motifs = 0
        for record in self.records:
            for role in ("d1", "d2"):
                item = record[role]
                atom_positives += len(item["positive_indices"])
                atoms += int(item["n_atoms"])
                motif_ids = item.get("motif_ids")
                if motif_ids is None:
                    motif_ids = [0] * int(item["n_atoms"])
                positive_indices = set(item["positive_indices"])
                positive_motifs = {
                    int(motif_ids[index]) for index in positive_indices
                }
                motif_positives += len(positive_motifs)
                motifs += max((int(value) for value in motif_ids), default=-1) + 1
        atom_negatives = atoms - atom_positives
        motif_negatives = motifs - motif_positives
        return (
            min(float(cap), atom_negatives / max(atom_positives, 1)),
            min(float(cap), motif_negatives / max(motif_positives, 1)),
        )


def reaction_site_collate(cutoff: float, *, geometry: bool = True):
    def motif_metadata(
        items: list[dict[str, Any]], role: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_ids = []
        motif_batches = []
        motif_targets = []
        offset = 0
        for batch_index, item in enumerate(items):
            local_ids = np.asarray(item[f"motif_ids{role}"], dtype=np.int64)
            local_target = np.asarray(item[f"target{role}"], dtype=np.float32)
            count = int(local_ids.max()) + 1 if len(local_ids) else 0
            global_ids.append(local_ids + offset)
            motif_batches.append(
                np.full(count, batch_index, dtype=np.int64)
            )
            target = np.zeros(count, dtype=np.float32)
            np.maximum.at(target, local_ids, local_target)
            motif_targets.append(target)
            offset += count
        return (
            torch.as_tensor(np.concatenate(global_ids), dtype=torch.long),
            torch.as_tensor(np.concatenate(motif_batches), dtype=torch.long),
            torch.as_tensor(np.concatenate(motif_targets), dtype=torch.float32),
        )

    def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        if geometry:
            d1 = collate_molecules([item["d1"] for item in items], cutoff)
            d2 = collate_molecules([item["d2"] for item in items], cutoff)
        else:
            from .hamir_2d import collate_covalent
            d1 = collate_covalent([item["d1"] for item in items])
            d2 = collate_covalent([item["d2"] for item in items])
        motif_index1, motif_batch1, motif_target1 = motif_metadata(items, "1")
        motif_index2, motif_batch2, motif_target2 = motif_metadata(items, "2")
        d1["motif_index"] = motif_index1
        d1["motif_batch"] = motif_batch1
        d2["motif_index"] = motif_index2
        d2["motif_batch"] = motif_batch2
        return {
            "d1": d1,
            "d2": d2,
            "target1": torch.as_tensor(
                np.concatenate([item["target1"] for item in items]),
                dtype=torch.float32,
            ),
            "target2": torch.as_tensor(
                np.concatenate([item["target2"] for item in items]),
                dtype=torch.float32,
            ),
            "motif_target1": motif_target1,
            "motif_target2": motif_target2,
            "reaction_class": torch.as_tensor(
                [item.get("reaction_class", 0) for item in items],
                dtype=torch.long,
            ),
            "sample_ids": [item["sample_id"] for item in items],
        }

    return collate


class ReactionSitePaiNN(nn.Module):
    """Shared PaiNN with other-molecule-conditioned atom participation heads."""

    def __init__(
        self,
        hidden_dim: int = 128,
        painn_layers: int = 5,
        num_rbf: int = 20,
        cutoff: float = 5.0,
        dropout: float = 0.1,
        architecture: str = "conditioned",
        attention_heads: int = 4,
        attention_topk: int = 4,
        encoder_type: str = "painn",
    ):
        super().__init__()
        if architecture not in {
            "conditioned",
            "sparse_cross_attention",
            "hierarchical_motif",
        }:
            raise ValueError(
                "architecture must be 'conditioned', "
                "'sparse_cross_attention', or 'hierarchical_motif'"
            )
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.architecture = architecture
        self.attention_topk = int(attention_topk)
        if encoder_type not in {"painn", "bond_mpnn"}:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        self.encoder_type = encoder_type
        self.encoder = PaiNNEncoder(
            hidden_dim=hidden_dim,
            layers=painn_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
        )
        self.invariant = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        if encoder_type == "bond_mpnn":
            from .hamir_2d import BondEncoder
            self.encoder = BondEncoder(hidden_dim, painn_layers)
            self.invariant = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
            )
            self.baseline_variant = "hamir_2d"
        self.site_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        if self.architecture in {
            "sparse_cross_attention",
            "hierarchical_motif",
        }:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(hidden_dim)
            self.cross_scale_raw = nn.Parameter(torch.tensor(-2.0))
        if self.architecture == "hierarchical_motif":
            self.motif_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.motif_norm = nn.LayerNorm(hidden_dim)
            self.motif_scale_raw = nn.Parameter(torch.tensor(-2.0))
            self.motif_to_atom = nn.Linear(hidden_dim, hidden_dim)
            self.atom_motif_norm = nn.LayerNorm(hidden_dim)
            self.group_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

    def _encode(self, molecule: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.encoder_type == "bond_mpnn":
            embedding = self.invariant(self.encoder(molecule))
        else:
            scalar, vector = self.encoder(molecule)
            vector_norm = torch.linalg.vector_norm(vector, dim=1)
            embedding = self.invariant(torch.cat([scalar, vector_norm], dim=-1))
        occlusion_mask = molecule.get("occlusion_mask")
        if occlusion_mask is not None:
            embedding = embedding.clone()
            atom_batch = molecule["batch"]
            for index in range(int(len(molecule["ptr"]) - 1)):
                local = atom_batch == index
                masked = local & occlusion_mask
                available = local & ~occlusion_mask
                if not bool(masked.any()):
                    continue
                source = embedding[available]
                if not len(source):
                    source = embedding[local]
                replacement = source.mean(dim=0)
                embedding[masked] = replacement
        return embedding

    def _conditioned_logits(
        self,
        atoms: torch.Tensor,
        atom_batch: torch.Tensor,
        other_global: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        context = other_global[atom_batch]
        features = torch.cat(
            [atoms, context, atoms * context, torch.abs(atoms - context)], dim=-1
        )
        if deterministic:
            output = features
            for layer in self.site_head:
                if not isinstance(layer, nn.Dropout):
                    output = layer(output)
            return output.squeeze(-1)
        return self.site_head(features).squeeze(-1)

    def _sparse_bidirectional_attention(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        d1: dict,
        d2: dict,
        preliminary1: torch.Tensor,
        preliminary2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        updated1 = h1.clone()
        updated2 = h2.clone()
        ptr1 = d1["ptr"].tolist()
        ptr2 = d2["ptr"].tolist()
        for index in range(len(ptr1) - 1):
            start1, end1 = ptr1[index], ptr1[index + 1]
            start2, end2 = ptr2[index], ptr2[index + 1]
            count1 = min(self.attention_topk, end1 - start1)
            count2 = min(self.attention_topk, end2 - start2)
            local1 = torch.topk(
                preliminary1[start1:end1], k=count1
            ).indices + start1
            local2 = torch.topk(
                preliminary2[start2:end2], k=count2
            ).indices + start2
            query1 = h1[local1].unsqueeze(0)
            query2 = h2[local2].unsqueeze(0)
            message1, _ = self.cross_attention(
                query1, query2, query2, need_weights=False
            )
            message2, _ = self.cross_attention(
                query2, query1, query1, need_weights=False
            )
            scale = torch.sigmoid(self.cross_scale_raw)
            refined1 = self.cross_norm(
                h1[local1] + scale * message1.squeeze(0)
            )
            refined2 = self.cross_norm(
                h2[local2] + scale * message2.squeeze(0)
            )
            updated1.index_copy_(0, local1, refined1)
            updated2.index_copy_(0, local2, refined2)
        return updated1, updated2

    def _hierarchical_motif_update(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        d1: dict,
        d2: dict,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        motif1 = segment_mean(
            h1, d1["motif_index"], int(len(d1["motif_batch"]))
        )
        motif2 = segment_mean(
            h2, d2["motif_index"], int(len(d2["motif_batch"]))
        )
        updated1 = motif1.clone()
        updated2 = motif2.clone()
        scale = torch.sigmoid(self.motif_scale_raw)
        for index in range(batch_size):
            local1 = torch.nonzero(
                d1["motif_batch"] == index, as_tuple=False
            ).flatten()
            local2 = torch.nonzero(
                d2["motif_batch"] == index, as_tuple=False
            ).flatten()
            query1 = motif1[local1].unsqueeze(0)
            query2 = motif2[local2].unsqueeze(0)
            message1, _ = self.motif_attention(
                query1, query2, query2, need_weights=False
            )
            message2, _ = self.motif_attention(
                query2, query1, query1, need_weights=False
            )
            updated1.index_copy_(
                0,
                local1,
                self.motif_norm(motif1[local1] + scale * message1.squeeze(0)),
            )
            updated2.index_copy_(
                0,
                local2,
                self.motif_norm(motif2[local2] + scale * message2.squeeze(0)),
            )
        h1 = self.atom_motif_norm(
            h1 + scale * self.motif_to_atom(updated1[d1["motif_index"]])
        )
        h2 = self.atom_motif_norm(
            h2 + scale * self.motif_to_atom(updated2[d2["motif_index"]])
        )
        return (
            h1,
            h2,
            self.group_head(updated1).squeeze(-1),
            self.group_head(updated2).squeeze(-1),
        )

    def forward(self, d1: dict, d2: dict) -> dict[str, torch.Tensor]:
        h1 = self._encode(d1)
        h2 = self._encode(d2)
        batch_size = int(len(d1["ptr"]) - 1)
        g1 = segment_mean(h1, d1["batch"], batch_size)
        g2 = segment_mean(h2, d2["batch"], batch_size)
        preliminary1 = self._conditioned_logits(
            h1, d1["batch"], g2, deterministic=True
        )
        preliminary2 = self._conditioned_logits(
            h2, d2["batch"], g1, deterministic=True
        )
        result: dict[str, torch.Tensor] = {}
        if self.architecture in {
            "sparse_cross_attention",
            "hierarchical_motif",
        }:
            h1, h2 = self._sparse_bidirectional_attention(
                h1, h2, d1, d2, preliminary1, preliminary2
            )
            result["preliminary_logits1"] = preliminary1
            result["preliminary_logits2"] = preliminary2
        if self.architecture == "hierarchical_motif":
            h1, h2, group1, group2 = self._hierarchical_motif_update(
                h1, h2, d1, d2, batch_size
            )
            result["group_logits1"] = group1
            result["group_logits2"] = group2
        if self.architecture != "conditioned":
            g1 = segment_mean(h1, d1["batch"], batch_size)
            g2 = segment_mean(h2, d2["batch"], batch_size)
        if getattr(self, "return_embeddings", False):
            result["reaction_embedding"] = torch.cat(
                [g1, g2, torch.abs(g1 - g2), g1 * g2], dim=-1
            )
        result.update({
            "logits1": self._conditioned_logits(h1, d1["batch"], g2),
            "logits2": self._conditioned_logits(h2, d2["batch"], g1),
        })
        return result


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        **batch,
        "d1": move_molecule_to_device(batch["d1"], device),
        "d2": move_molecule_to_device(batch["d2"], device),
        "target1": batch["target1"].to(device, non_blocking=True),
        "target2": batch["target2"].to(device, non_blocking=True),
        "motif_target1": batch["motif_target1"].to(
            device, non_blocking=True
        ),
        "motif_target2": batch["motif_target2"].to(
            device, non_blocking=True
        ),
        "reaction_class": batch["reaction_class"].to(
            device, non_blocking=True
        ),
    }


def _supervised_contrastive_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Supervised contrastive loss over reactions in one mini-batch."""
    if len(embedding) < 2:
        return embedding.new_zeros(())
    embedding = F.normalize(embedding.float(), dim=-1)
    similarity = embedding @ embedding.T / temperature
    identity = torch.eye(
        len(embedding), dtype=torch.bool, device=embedding.device
    )
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    valid = positive.any(dim=1)
    if not bool(valid.any()):
        return embedding.new_zeros(())
    similarity = similarity.masked_fill(identity, -torch.inf)
    log_probability = similarity - torch.logsumexp(
        similarity, dim=1, keepdim=True
    )
    per_example = -(
        log_probability.masked_fill(~positive, 0.0).sum(dim=1)
        / positive.sum(dim=1).clamp_min(1)
    )
    return per_example[valid].mean().to(embedding.dtype)


def site_loss(
    output: dict[str, torch.Tensor],
    batch: dict,
    *,
    pos_weight: torch.Tensor,
    focal_gamma: float,
    dice_weight: float,
    group_pos_weight: torch.Tensor | None = None,
    group_weight: float = 0.0,
    consistency_weight: float = 0.0,
    class_weight: float = 0.0,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.1,
) -> torch.Tensor:
    logits = torch.cat([output["logits1"], output["logits2"]])
    target = torch.cat([batch["target1"], batch["target2"]])
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )
    probability = torch.sigmoid(logits)
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    focal = ((1.0 - pt).pow(focal_gamma) * bce).mean()
    intersection = (probability * target).sum()
    dice = 1.0 - (
        2.0 * intersection + 1.0
    ) / (probability.sum() + target.sum() + 1.0)
    loss = focal + dice_weight * dice
    if group_weight and "group_logits1" in output:
        group_logits = torch.cat(
            [output["group_logits1"], output["group_logits2"]]
        )
        group_target = torch.cat(
            [batch["motif_target1"], batch["motif_target2"]]
        )
        group_loss = F.binary_cross_entropy_with_logits(
            group_logits,
            group_target,
            pos_weight=group_pos_weight,
        )
        loss = loss + group_weight * group_loss
    if consistency_weight and "group_logits1" in output:
        consistency = logits.new_zeros(())
        for role in ("1", "2"):
            atom_probability = torch.sigmoid(output[f"logits{role}"])
            motif_index = batch[f"d{role}"]["motif_index"]
            group_probability = torch.sigmoid(output[f"group_logits{role}"])
            atom_group_probability = torch.zeros_like(group_probability)
            atom_group_probability.scatter_reduce_(
                0,
                motif_index,
                atom_probability,
                reduce="amax",
                include_self=True,
            )
            consistency = consistency + F.mse_loss(
                group_probability, atom_group_probability
            )
        loss = loss + consistency_weight * consistency / 2.0
    if class_weight and "class_logits" in output:
        class_logits = output["class_logits"]
        reaction_class = batch["reaction_class"]
        if int(reaction_class.max()) >= class_logits.shape[-1]:
            raise ValueError(
                "reaction_class exceeds the configured number of classes"
            )
        loss = loss + class_weight * F.cross_entropy(
            class_logits, reaction_class
        )
    if contrastive_weight and "reaction_embedding" in output:
        contrastive = _supervised_contrastive_loss(
            output["reaction_embedding"],
            batch["reaction_class"],
            contrastive_temperature,
        )
        loss = loss + contrastive_weight * contrastive
    return loss


def _binary_counts(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float
) -> tuple[int, int, int]:
    prediction = torch.sigmoid(logits) >= threshold
    truth = targets >= 0.5
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    return tp, fp, fn


def _topk_hits(
    logits: torch.Tensor, target: torch.Tensor, ptr: torch.Tensor, k: int
) -> tuple[int, int]:
    hits = 0
    total = 0
    for start, end in zip(ptr[:-1].tolist(), ptr[1:].tolist()):
        local_target = target[start:end]
        if not bool((local_target > 0.5).any()):
            continue
        local_logits = logits[start:end]
        count = min(k, len(local_logits))
        selected = torch.topk(local_logits, k=count).indices
        hits += int(bool((local_target[selected] > 0.5).any()))
        total += 1
    return hits, total


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_dtype: torch.dtype | None,
    threshold: float,
    save_predictions: Path | None = None,
) -> dict[str, float | int]:
    model.eval()
    tp = fp = fn = 0
    group_tp = group_fp = group_fn = 0
    top3_hits = top3_total = 0
    exact_pairs = examples = 0
    saved_logits: list[np.ndarray] = []
    saved_targets: list[np.ndarray] = []
    for batch in tqdm(loader, desc="evaluate", leave=False, dynamic_ncols=True):
        batch = _move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None,
        ):
            output = model(batch["d1"], batch["d2"])
        for role in ("1", "2"):
            counts = _binary_counts(
                output[f"logits{role}"], batch[f"target{role}"], threshold
            )
            tp += counts[0]
            fp += counts[1]
            fn += counts[2]
            hits, total = _topk_hits(
                output[f"logits{role}"],
                batch[f"target{role}"],
                batch[f"d{role}"]["ptr"],
                3,
            )
            top3_hits += hits
            top3_total += total
            if f"group_logits{role}" in output:
                counts = _binary_counts(
                    output[f"group_logits{role}"],
                    batch[f"motif_target{role}"],
                    threshold,
                )
                group_tp += counts[0]
                group_fp += counts[1]
                group_fn += counts[2]
        prediction1 = torch.sigmoid(output["logits1"]) >= threshold
        prediction2 = torch.sigmoid(output["logits2"]) >= threshold
        truth1 = batch["target1"] >= 0.5
        truth2 = batch["target2"] >= 0.5
        ptr1 = batch["d1"]["ptr"].tolist()
        ptr2 = batch["d2"]["ptr"].tolist()
        for index in range(len(ptr1) - 1):
            first_exact = torch.equal(
                prediction1[ptr1[index] : ptr1[index + 1]],
                truth1[ptr1[index] : ptr1[index + 1]],
            )
            second_exact = torch.equal(
                prediction2[ptr2[index] : ptr2[index + 1]],
                truth2[ptr2[index] : ptr2[index + 1]],
            )
            exact_pairs += int(first_exact and second_exact)
            examples += 1
        if save_predictions is not None:
            saved_logits.append(
                torch.cat([output["logits1"], output["logits2"]])
                .float()
                .cpu()
                .numpy()
            )
            saved_targets.append(
                torch.cat([batch["target1"], batch["target2"]])
                .float()
                .cpu()
                .numpy()
            )
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    atom_f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    metrics: dict[str, float | int] = {
        "atom_f1": float(atom_f1),
        "atom_precision": float(precision),
        "atom_recall": float(recall),
        "top3_atom_hit_rate": float(top3_hits / max(top3_total, 1)),
        "exact_pair_center_accuracy": float(exact_pairs / max(examples, 1)),
        "examples": examples,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "threshold": threshold,
    }
    if group_tp + group_fp + group_fn:
        group_precision = group_tp / max(group_tp + group_fp, 1)
        group_recall = group_tp / max(group_tp + group_fn, 1)
        metrics.update(
            {
                "group_f1": float(
                    2.0
                    * group_precision
                    * group_recall
                    / max(group_precision + group_recall, 1e-12)
                ),
                "group_precision": float(group_precision),
                "group_recall": float(group_recall),
            }
        )
    if save_predictions is not None:
        save_predictions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_predictions,
            logits=np.concatenate(saved_logits),
            targets=np.concatenate(saved_targets),
            threshold=np.asarray(threshold),
        )
    return metrics


class ReactionSiteCollator:
    """Picklable wrapper for Windows DataLoader workers."""
    def __init__(self, cutoff, geometry=True):
        self.cutoff, self.geometry = cutoff, geometry

    def __call__(self, items):
        return reaction_site_collate(self.cutoff, geometry=self.geometry)(items)


def _loader(
    dataset: ReactionSiteDataset,
    *,
    batch_size: int,
    workers: int,
    cutoff: float,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=ReactionSiteCollator(cutoff, getattr(dataset, "geometry", True)),
    )


def _amp_dtype(name: str, device: torch.device) -> torch.dtype | None:
    normalized = name.lower()
    if normalized == "none":
        return None
    if device.type != "cuda":
        raise ValueError("Mixed precision requires a CUDA device")
    if normalized == "fp16":
        return torch.float16
    if normalized == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is not supported by this GPU")
        return torch.bfloat16
    raise ValueError("mixed_precision must be one of: none, fp16, bf16")


def _build_model(config: dict[str, Any]) -> nn.Module:
    model_config = dict(config["model"])
    family = str(model_config.get("family", "painn")).lower()
    if family == "painn":
        model_config.pop("family", None)
        return ReactionSitePaiNN(**model_config)
    from .reaction_site_baselines import build_reaction_site_baseline

    return build_reaction_site_baseline(model_config)


def _load_ddi_encoder(model: nn.Module, checkpoint_path: Path) -> None:
    if not isinstance(model, ReactionSitePaiNN):
        raise ValueError(
            "--init-ddi-checkpoint is supported only for the PaiNN model"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model_state", checkpoint)
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in source.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(f"No encoder.* weights found in {checkpoint_path}")
    result = model.encoder.load_state_dict(encoder_state, strict=True)
    print(f"Loaded DDI PaiNN encoder from {checkpoint_path}: {result}")


def _load_full_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    """Load model weights only for transfer learning.

    Optimizer, scheduler, epoch, and early-stopping state are intentionally
    not inherited from the source-domain run.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    source = checkpoint.get("model_state", checkpoint)
    result = model.load_state_dict(source, strict=True)
    validation = checkpoint.get("validation")
    print(
        f"Loaded transfer initialization from {checkpoint_path}: {result}; "
        f"source_epoch={checkpoint.get('epoch', 'unknown')} "
        f"source_validation={validation}"
    )


def train_command(
    config: dict[str, Any],
    *,
    resume: Path | None,
    init_checkpoint: Path | None,
    init_ddi_checkpoint: Path | None,
) -> None:
    initialization_options = sum(
        value is not None
        for value in (resume, init_checkpoint, init_ddi_checkpoint)
    )
    if initialization_options > 1:
        raise ValueError(
            "Use only one of --resume, --init-checkpoint, or "
            "--init-ddi-checkpoint"
        )
    paths = {key: Path(value) for key, value in config["paths"].items()}
    training = config["training"]
    loss_config = config.get("loss", {})
    seed = int(config.get("seed", 17))
    set_seed(seed)
    device_name = str(training.get("device", "cuda"))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but no CUDA GPU is available")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    torch.set_float32_matmul_precision("high")
    cutoff = float(config["model"].get("cutoff", 5.0))
    cache = paths["molecule_cache"]
    geometry = config["model"].get("encoder_type", "painn") != "bond_mpnn"
    train_set = ReactionSiteDataset(paths["train_index"], cache, geometry=geometry)
    valid_set = ReactionSiteDataset(paths["valid_index"], cache, geometry=geometry)
    batch_size = int(training.get("batch_size", 8))
    workers = int(training.get("num_workers", 4))
    train_loader = _loader(
        train_set,
        batch_size=batch_size,
        workers=workers,
        cutoff=cutoff,
        shuffle=True,
    )
    valid_loader = _loader(
        valid_set,
        batch_size=batch_size,
        workers=workers,
        cutoff=cutoff,
        shuffle=False,
    )
    model = _build_model(config)
    if init_checkpoint is not None:
        _load_full_checkpoint(model, init_checkpoint)
    elif init_ddi_checkpoint is not None:
        _load_ddi_encoder(model, init_ddi_checkpoint)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training.get("lr_factor", 0.5)),
        patience=int(training.get("lr_patience", 3)),
        min_lr=float(training.get("min_learning_rate", 1e-6)),
    )
    amp_dtype = _amp_dtype(str(training.get("mixed_precision", "fp16")), device)
    scaler = torch.amp.GradScaler(
        device.type, enabled=amp_dtype == torch.float16
    )
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    gradient_clip = float(training.get("gradient_clip", 1.0))
    pos_weight_value, group_pos_weight_value = train_set.positive_weights(
        float(loss_config.get("max_pos_weight", 20.0))
    )
    pos_weight = torch.tensor(pos_weight_value, device=device)
    group_pos_weight = torch.tensor(group_pos_weight_value, device=device)
    print(
        f"train={len(train_set):,} valid={len(valid_set):,} "
        f"parameters={sum(p.numel() for p in model.parameters()):,} "
        f"pos_weight={pos_weight_value:.4f} "
        f"group_pos_weight={group_pos_weight_value:.4f}"
    )
    checkpoints = paths["checkpoints"]
    checkpoints.mkdir(parents=True, exist_ok=True)
    history_path = checkpoints / "history.csv"
    start_epoch = 0
    best_f1 = -1.0
    stale = 0
    initialization_checkpoint_value = (
        str(init_checkpoint.resolve())
        if init_checkpoint is not None
        else None
    )
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"])
        best_f1 = float(state.get("best_atom_f1", -1.0))
        initialization_checkpoint_value = state.get(
            "initialization_checkpoint"
        )
        print(f"Resumed from {resume} at completed epoch {start_epoch}")
    max_epochs = int(training.get("epochs", 60))
    patience = int(training.get("patience", 8))
    threshold = float(training.get("threshold", 0.5))
    header_needed = not history_path.is_file() or start_epoch == 0
    with history_path.open(
        "a" if start_epoch else "w", encoding="utf-8", newline=""
    ) as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=[
                "epoch",
                "loss",
                "val_atom_f1",
                "val_precision",
                "val_recall",
                "learning_rate",
                "seconds",
            ],
        )
        if header_needed:
            writer.writeheader()
        for epoch in range(start_epoch, max_epochs):
            started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            progress = tqdm(
                train_loader,
                desc=f"epoch {epoch + 1:03d}/{max_epochs:03d}",
                dynamic_ncols=True,
            )
            for step, batch in enumerate(progress, start=1):
                batch = _move_batch(batch, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype or torch.float32,
                    enabled=amp_dtype is not None,
                ):
                    output = model(batch["d1"], batch["d2"])
                    loss = site_loss(
                        output,
                        batch,
                        pos_weight=pos_weight,
                        group_pos_weight=group_pos_weight,
                        focal_gamma=float(loss_config.get("focal_gamma", 2.0)),
                        dice_weight=float(loss_config.get("dice_weight", 0.5)),
                        group_weight=float(
                            loss_config.get("group_weight", 0.0)
                        ),
                        consistency_weight=float(
                            loss_config.get("consistency_weight", 0.0)
                        ),
                        class_weight=float(
                            loss_config.get("class_weight", 0.0)
                        ),
                        contrastive_weight=float(
                            loss_config.get("contrastive_weight", 0.0)
                        ),
                        contrastive_temperature=float(
                            loss_config.get(
                                "contrastive_temperature", 0.1
                            )
                        ),
                    )
                    scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()
                if step % accumulation == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                running_loss += float(loss.detach())
                progress.set_postfix(loss=f"{running_loss / step:.6f}")
            metrics = evaluate_loader(
                model,
                valid_loader,
                device,
                amp_dtype=amp_dtype,
                threshold=threshold,
            )
            scheduler.step(float(metrics["atom_f1"]))
            epoch_loss = running_loss / max(len(train_loader), 1)
            elapsed = time.perf_counter() - started
            row = {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "val_atom_f1": metrics["atom_f1"],
                "val_precision": metrics["atom_precision"],
                "val_recall": metrics["atom_recall"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": elapsed,
            }
            writer.writerow(row)
            history_file.flush()
            state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "validation": metrics,
                "best_atom_f1": max(best_f1, float(metrics["atom_f1"])),
                "threshold": threshold,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "config": config,
                "initialization_checkpoint": initialization_checkpoint_value,
            }
            torch.save(state, checkpoints / "latest.pt")
            if float(metrics["atom_f1"]) > best_f1:
                best_f1 = float(metrics["atom_f1"])
                stale = 0
                state["best_atom_f1"] = best_f1
                torch.save(state, checkpoints / "best.pt")
            else:
                stale += 1
            print(
                f"epoch={epoch + 1:03d} loss={epoch_loss:.6f} "
                f"val_atom_f1={metrics['atom_f1']:.6f} "
                f"precision={metrics['atom_precision']:.6f} "
                f"recall={metrics['atom_recall']:.6f} stale={stale}/{patience}"
            )
            if stale >= patience:
                print("Early stopping.")
                break


def evaluate_command(
    config: dict[str, Any], *, checkpoint: Path, split: str
) -> None:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    training = config["training"]
    device = torch.device(str(training.get("device", "cuda")))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = _build_model(config)
    model.load_state_dict(state["model_state"])
    model.to(device)
    dataset = ReactionSiteDataset(
        paths[f"{split}_index"], paths["molecule_cache"],
        geometry=config["model"].get("encoder_type", "painn") != "bond_mpnn",
    )
    loader = _loader(
        dataset,
        batch_size=int(training.get("batch_size", 8)),
        workers=int(training.get("num_workers", 4)),
        cutoff=float(config["model"].get("cutoff", 5.0)),
        shuffle=False,
    )
    results = paths["results"]
    results.mkdir(parents=True, exist_ok=True)
    threshold = float(state.get("threshold", training.get("threshold", 0.5)))
    metrics = evaluate_loader(
        model,
        loader,
        device,
        amp_dtype=_amp_dtype(
            str(training.get("mixed_precision", "fp16")), device
        ),
        threshold=threshold,
        save_predictions=results / f"{split}_predictions.npz",
    )
    metrics.update(
        {
            "split": split,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_epoch": int(state["epoch"]),
            "primary_metric": "atom_f1",
            "model_variant": getattr(
                model, "baseline_variant", "painn"
            ),
        }
    )
    destination = results / f"{split}_metrics.json"
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Metrics written to {destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="USPTO-50K two-reactant atom participation training."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prepare",
        "augment-motifs",
        "build-3d",
        "train",
        "evaluate",
    ):
        child = subparsers.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
        if name == "build-3d":
            child.add_argument("--workers", type=int, default=4)
            child.add_argument("--overwrite", action="store_true")
            child.add_argument(
                "--splits",
                nargs="+",
                choices=["train", "valid", "test"],
                default=["train", "valid", "test"],
                help=(
                    "Indexes whose molecule geometries should be built. Use "
                    "--splits test for zero-shot evaluation."
                ),
            )
        if name == "train":
            child.add_argument("--resume", type=Path)
            child.add_argument(
                "--init-checkpoint",
                type=Path,
                help=(
                    "Load all model weights for transfer learning without "
                    "loading optimizer, scheduler, or epoch state."
                ),
            )
            child.add_argument("--init-ddi-checkpoint", type=Path)
        if name == "evaluate":
            child.add_argument("--checkpoint", type=Path, required=True)
            child.add_argument(
                "--split", choices=["train", "valid", "test"], default="test"
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        prepare_command(config)
    elif args.command == "augment-motifs":
        augment_motifs_command(config)
    elif args.command == "build-3d":
        build_3d_command(
            config,
            workers=args.workers,
            overwrite=args.overwrite,
            splits=tuple(args.splits),
        )
    elif args.command == "train":
        train_command(
            config,
            resume=args.resume,
            init_checkpoint=args.init_checkpoint,
            init_ddi_checkpoint=args.init_ddi_checkpoint,
        )
    elif args.command == "evaluate":
        evaluate_command(config, checkpoint=args.checkpoint, split=args.split)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
