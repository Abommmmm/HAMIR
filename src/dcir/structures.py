from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class StructureResult:
    drug_id: str
    status: str
    conformer_count: int = 0
    atom_count: int = 0
    force_field: str = ""
    message: str = ""


def stable_seed(drug_id: str, global_seed: int) -> int:
    digest = hashlib.sha256(f"{global_seed}:{drug_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def read_smiles_tables(paths: Iterable[Path]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            has_header = "drug_id" in sample.splitlines()[0].lower()
            if has_header:
                for row in csv.DictReader(handle):
                    drug_id = str(row.get("drug_id", "")).strip()
                    smiles = str(row.get("smiles", "")).strip()
                    if drug_id and smiles:
                        mapping[drug_id] = smiles
            else:
                for row in csv.reader(handle):
                    if len(row) >= 2 and row[0].strip() and row[1].strip():
                        mapping[row[0].strip()] = row[1].strip()
    return mapping


def read_sdf_directory(path: Path) -> dict[str, str]:
    """Convert the author-released DrugBank SDF files to canonical SMILES."""
    Chem, _, _ = _rdkit_modules()
    mapping: dict[str, str] = {}
    if not path.is_dir():
        return mapping
    for sdf_path in sorted(path.glob("*.sdf")):
        molecule = Chem.MolFromMolFile(str(sdf_path), sanitize=True, removeHs=True)
        if molecule is None:
            continue
        mapping[sdf_path.stem] = Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )
    return mapping


def _rdkit_modules():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for 3D preparation. Install "
            "configs/requirements-data.txt "
            "locally or run this preprocessing step in the cloud."
        ) from exc
    return Chem, AllChem, rdMolStandardize


def standardize_molecule(smiles: str):
    Chem, _, rdMolStandardize = _rdkit_modules()
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES")

    cleanup = rdMolStandardize.Cleanup(mol)
    parent = rdMolStandardize.FragmentParent(cleanup)
    normalizer = rdMolStandardize.Normalizer()
    parent = normalizer.normalize(parent)
    Chem.SanitizeMol(parent)
    canonical = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    for atom in parent.GetAtoms():
        atom.SetIntProp("_OriginalAtomIndex", atom.GetIdx())
    return parent, canonical


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


def _optimize_conformers(mol_h, conf_ids: list[int], max_iters: int):
    _, AllChem, _ = _rdkit_modules()
    energies: list[float] = []
    statuses: list[int] = []
    if AllChem.MMFFHasAllMoleculeParams(mol_h):
        properties = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
        force_field = "MMFF94s"
        for conf_id in conf_ids:
            ff = AllChem.MMFFGetMoleculeForceField(
                mol_h, properties, confId=conf_id
            )
            status = int(ff.Minimize(maxIts=max_iters))
            statuses.append(status)
            energies.append(float(ff.CalcEnergy()))
    else:
        force_field = "UFF"
        for conf_id in conf_ids:
            ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=conf_id)
            status = int(ff.Minimize(maxIts=max_iters))
            statuses.append(status)
            energies.append(float(ff.CalcEnergy()))
    return force_field, np.asarray(energies), np.asarray(statuses)


def adaptive_conformer_counts(
    atom_count: int,
    requested_num_confs: int,
    requested_keep_confs: int,
) -> tuple[int, int]:
    """Bound ETKDG work for large drugs while retaining up to three conformers."""
    if atom_count > 256:
        candidate_cap = 1
    elif atom_count > 128:
        candidate_cap = 10
    else:
        candidate_cap = requested_num_confs
    num_confs = max(1, min(requested_num_confs, candidate_cap))
    keep_confs = max(1, min(requested_keep_confs, num_confs))
    return num_confs, keep_confs


def generate_conformers(
    drug_id: str,
    smiles: str,
    num_confs: int,
    keep_confs: int,
    global_seed: int,
    prune_rms: float,
    max_iters: int,
) -> tuple[dict[str, np.ndarray | str], StructureResult]:
    Chem, AllChem, _ = _rdkit_modules()
    mol, canonical = standardize_molecule(smiles)
    if mol.GetNumAtoms() < 1:
        raise ValueError("Molecule has no atoms after standardization")
    requested_num_confs = num_confs
    requested_max_iters = max_iters
    num_confs, keep_confs = adaptive_conformer_counts(
        mol.GetNumAtoms(),
        num_confs,
        keep_confs,
    )
    if mol.GetNumAtoms() > 256:
        max_iters = min(max_iters, 100)

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = stable_seed(drug_id, global_seed)
    params.pruneRmsThresh = float(prune_rms)
    params.timeout = 20
    params.maxIterations = 1000
    params.useRandomCoords = False
    params.enforceChirality = True
    conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params))
    embedding_method = "ETKDGv3"
    if not conf_ids:
        retry_params = AllChem.ETKDGv3()
        retry_params.randomSeed = (
            stable_seed(drug_id, global_seed) + 104729
        ) & 0x7FFFFFFF
        retry_params.pruneRmsThresh = min(float(prune_rms), 0.5)
        retry_params.timeout = 20
        retry_params.useRandomCoords = True
        retry_params.boxSizeMult = 4.0
        retry_params.maxIterations = 2000
        retry_params.enforceChirality = True
        retry_params.clearConfs = True
        conf_ids = list(
            AllChem.EmbedMultipleConfs(
                mol_h,
                numConfs=num_confs,
                params=retry_params,
            )
        )
        embedding_method = "ETKDGv3-random-coordinates"
    if not conf_ids:
        rescue_params = AllChem.ETKDGv3()
        rescue_params.randomSeed = (
            stable_seed(drug_id, global_seed) + 209759
        ) & 0x7FFFFFFF
        rescue_params.pruneRmsThresh = -1.0
        rescue_params.timeout = 20
        rescue_params.useRandomCoords = True
        rescue_params.boxSizeMult = 8.0
        rescue_params.maxIterations = 5000
        rescue_params.ignoreSmoothingFailures = True
        rescue_params.enforceChirality = True
        rescue_params.clearConfs = True
        conf_ids = list(
            AllChem.EmbedMultipleConfs(
                mol_h,
                numConfs=num_confs,
                params=rescue_params,
            )
        )
        embedding_method = "ETKDGv3-random-rescue"
    if not conf_ids:
        raise RuntimeError("ETKDGv3 generated no conformers after two fallbacks")

    force_field, energies, statuses = _optimize_conformers(
        mol_h, conf_ids, max_iters
    )
    order = np.argsort(energies)
    selected_indices = order[: min(keep_confs, len(order))]
    selected_ids = [conf_ids[int(index)] for index in selected_indices]
    selected_energies = energies[selected_indices]
    selected_statuses = statuses[selected_indices]

    heavy = Chem.RemoveHs(mol_h)
    positions: list[np.ndarray] = []
    for conf_id in selected_ids:
        conformer = heavy.GetConformer(conf_id)
        coords = np.asarray(conformer.GetPositions(), dtype=np.float32)
        coords -= coords.mean(axis=0, keepdims=True)
        positions.append(coords)

    numbers, atom_attr = atom_features(heavy)
    bond_index, bond_attr = bond_features(heavy)
    payload: dict[str, np.ndarray | str] = {
        "atomic_numbers": numbers,
        "atom_features": atom_attr,
        "bond_index": bond_index,
        "bond_features": bond_attr,
        "positions": np.stack(positions, axis=0),
        "energies": selected_energies.astype(np.float32),
        "optimization_status": selected_statuses.astype(np.int64),
        "canonical_smiles": canonical,
        "embedding_method": embedding_method,
        "force_field": force_field,
        "original_atom_index": np.arange(heavy.GetNumAtoms(), dtype=np.int64),
    }
    result = StructureResult(
        drug_id=drug_id,
        status="ok",
        conformer_count=len(positions),
        atom_count=heavy.GetNumAtoms(),
        force_field=force_field,
        message=(
            f"embedding={embedding_method}; "
            f"candidate_conformers={num_confs}/{requested_num_confs}; "
            f"optimization_steps={max_iters}/{requested_max_iters}"
        ),
    )
    return payload, result


def save_payload(path: Path, payload: dict[str, np.ndarray | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _prepare_one(
    drug_id: str,
    smiles: str | None,
    output_dir: str,
    num_confs: int,
    keep_confs: int,
    seed: int,
    prune_rms: float,
    max_iters: int,
    overwrite: bool,
) -> StructureResult:
    target = Path(output_dir) / f"{drug_id}.npz"
    if target.exists() and not overwrite:
        try:
            with np.load(target, allow_pickle=False) as archive:
                return StructureResult(
                    drug_id=drug_id,
                    status="existing",
                    conformer_count=int(archive["positions"].shape[0]),
                    atom_count=int(archive["positions"].shape[1]),
                    force_field=str(archive["force_field"]),
                )
        except (OSError, ValueError, KeyError):
            target.unlink(missing_ok=True)
    if not smiles:
        return StructureResult(
            drug_id=drug_id,
            status="missing_smiles",
            message="No SMILES in the versioned structure tables.",
        )
    try:
        payload, result = generate_conformers(
            drug_id,
            smiles,
            num_confs,
            keep_confs,
            seed,
            prune_rms,
            max_iters,
        )
        save_payload(target, payload)
        return result
    except Exception as exc:  # per-drug failure must not abort the whole corpus
        return StructureResult(
            drug_id=drug_id,
            status="failed",
            message=f"{type(exc).__name__}: {exc}",
        )


def prepare_store(
    drug_ids: Iterable[str],
    smiles_map: dict[str, str],
    output_dir: Path,
    *,
    num_confs: int = 20,
    keep_confs: int = 3,
    seed: int = 17,
    prune_rms: float = 0.75,
    max_iters: int = 500,
    overwrite: bool = False,
    num_workers: int = 1,
) -> list[StructureResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_ids = sorted(set(drug_ids))
    worker_count = max(1, int(num_workers))
    results: list[StructureResult] = []
    jobs = [
        (
            drug_id,
            smiles_map.get(drug_id),
            str(output_dir),
            num_confs,
            keep_confs,
            seed,
            prune_rms,
            max_iters,
            overwrite,
        )
        for drug_id in ordered_ids
    ]

    if worker_count == 1:
        for index, job in enumerate(jobs, start=1):
            result = _prepare_one(*job)
            results.append(result)
            print(
                f"[{index}/{len(jobs)}] {result.drug_id}: {result.status}",
                flush=True,
            )
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(_prepare_one, *job): job[0] for job in jobs
            }
            for index, future in enumerate(as_completed(futures), start=1):
                drug_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = StructureResult(
                        drug_id,
                        status="failed",
                        message=f"WorkerError: {type(exc).__name__}: {exc}",
                    )
                results.append(result)
                print(
                    f"[{index}/{len(jobs)}] {result.drug_id}: {result.status}",
                    flush=True,
                )
    results.sort(key=lambda item: item.drug_id)
    with (output_dir / "structure_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump([asdict(item) for item in results], handle, indent=2)
    return results
