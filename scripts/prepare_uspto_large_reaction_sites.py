#!/usr/bin/env python
"""Normalize USPTO-MIT/FULL and build leakage-audited A2 indexes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
source_path = str(SOURCE)
if source_path in sys.path:
    sys.path.remove(source_path)
sys.path.insert(0, source_path)

from dcir.reaction_sites import _rdkit, prepare_split  # noqa: E402


def mapped_reaction_identity(reaction: str) -> str | None:
    """Canonical identity using product-contributing precursors and products."""

    Chem, _ = _rdkit()
    parts = reaction.strip().split(">")
    if len(parts) != 3:
        return None
    reactants = Chem.MolFromSmiles(".".join(x for x in parts[:2] if x))
    products = Chem.MolFromSmiles(parts[2])
    if reactants is None or products is None:
        return None
    product_maps = {
        int(atom.GetAtomMapNum())
        for atom in products.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    if not product_maps:
        return None

    def canonical(molecule) -> str:
        copy = Chem.Mol(molecule)
        for atom in copy.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True)

    left = []
    for fragment in Chem.GetMolFrags(reactants, asMols=True, sanitizeFrags=True):
        maps = {
            int(atom.GetAtomMapNum())
            for atom in fragment.GetAtoms()
            if atom.GetAtomMapNum() > 0
        }
        if maps & product_maps:
            left.append(canonical(fragment))
    right = [
        canonical(fragment)
        for fragment in Chem.GetMolFrags(products, asMols=True, sanitizeFrags=True)
    ]
    payload = ".".join(sorted(left)) + ">>" + ".".join(sorted(right))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reactions_from_csv(path: Path) -> Iterable[tuple[str, str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            reaction = (row.get("rxn_smiles") or row.get("reactions") or "").strip()
            sample_id = str(row.get("id") or f"{path.stem}:{index}")
            yield sample_id, reaction, str(row.get("PatentNumber") or "")


def reactions_from_mit(path: Path) -> Iterable[tuple[str, str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            reaction = line.strip().split(None, 1)[0] if line.strip() else ""
            yield f"{path.stem}:{index}", reaction, ""


def source_uspto50k_identities(directory: Path) -> set[str]:
    identities: set[str] = set()
    for split in ("train", "valid", "test"):
        path = directory / f"atom_mapped_{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing USPTO-50K source split: {path}")
        for _, reaction, _ in reactions_from_csv(path):
            identity = mapped_reaction_identity(reaction)
            if identity:
                identities.add(identity)
    return identities


def reaction_site_identity(record: dict[str, object]) -> str:
    components = []
    for role in ("d1", "d2"):
        item = record[role]
        if not isinstance(item, dict):
            raise ValueError(f"Invalid reaction-site component: {role}")
        components.append(
            {
                "smiles": str(item["smiles"]),
                "positive_indices": sorted(
                    int(value) for value in item["positive_indices"]
                ),
            }
        )
    components.sort(
        key=lambda item: (item["smiles"], item["positive_indices"])
    )
    payload = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_uspto50k_site_identities(directory: Path) -> set[str]:
    identities: set[str] = set()
    for split in ("train", "valid", "test"):
        path = directory / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing USPTO-50K source index: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    identities.add(reaction_site_identity(json.loads(line)))
    return identities


def filter_index_site_overlap(
    path: Path, source_identities: set[str]
) -> dict[str, int]:
    temporary = path.with_suffix(path.suffix + ".overlap-tmp")
    counters = {"before": 0, "excluded_uspto50k_overlap": 0, "after": 0}
    with (
        path.open("r", encoding="utf-8") as source,
        temporary.open("w", encoding="utf-8", newline="\n") as output,
    ):
        for line in source:
            if not line.strip():
                continue
            counters["before"] += 1
            record = json.loads(line)
            if reaction_site_identity(record) in source_identities:
                counters["excluded_uspto50k_overlap"] += 1
                continue
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            counters["after"] += 1
    temporary.replace(path)
    return counters


def choose_full_split(identity: str) -> str:
    bucket = int(identity[:8], 16) % 100
    return "train" if bucket < 80 else "valid" if bucket < 90 else "test"


def write_normalized_splits(
    *,
    dataset: str,
    raw_root: Path,
    normalized_dir: Path,
    source_identities: set[str],
) -> dict[str, object]:
    normalized_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (normalized_dir / f"atom_mapped_{split}.csv").open(
            "w", encoding="utf-8", newline=""
        )
        for split in ("train", "valid", "test")
    }
    writers = {split: csv.writer(handle) for split, handle in handles.items()}
    for writer in writers.values():
        writer.writerow(["id", "rxn_smiles", "class"])
    counters: Counter[str] = Counter()
    seen: set[str] = set()
    try:
        if dataset == "mit":
            inputs = [
                (split, reactions_from_mit(raw_root / f"mapped_{split}.txt"))
                for split in ("train", "valid", "test")
            ]
        else:
            path = raw_root / "atom_mapped_all.csv"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing mapped USPTO-FULL file: {path}. Run "
                    "scripts/map_uspto_full.py first."
                )
            inputs = [("hash", reactions_from_csv(path))]
        for declared_split, rows in inputs:
            for sample_id, reaction, _ in rows:
                counters["source_rows"] += 1
                if not reaction:
                    counters["empty_or_mapping_failed"] += 1
                    continue
                try:
                    identity = mapped_reaction_identity(reaction)
                except Exception:
                    identity = None
                if identity is None:
                    counters["identity_failed"] += 1
                    continue
                if identity in source_identities:
                    counters["excluded_uspto50k_overlap"] += 1
                    continue
                if identity in seen:
                    counters["duplicate_reaction"] += 1
                    continue
                seen.add(identity)
                split = (
                    declared_split
                    if declared_split != "hash"
                    else choose_full_split(identity)
                )
                writers[split].writerow([sample_id, reaction, 0])
                counters[f"normalized_{split}"] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return {
        "dataset": f"USPTO-{dataset.upper()}",
        "split_protocol": (
            "official Jin et al. fixed split"
            if dataset == "mit"
            else "deterministic reaction-identity hash 80/10/10"
        ),
        "deduplication": "canonical mapped reaction identity",
        "excluded_source": "all exact USPTO-50K reaction identities",
        "source_uspto50k_identities": len(source_identities),
        "counts": dict(sorted(counters.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["mit", "full"])
    parser.add_argument("--max-atoms", type=int, default=128)
    parser.add_argument(
        "--uspto50k-raw",
        type=Path,
        default=Path("data/raw/reactions/uspto_50k"),
    )
    parser.add_argument(
        "--uspto50k-index",
        type=Path,
        default=Path("outputs/reaction_sites/uspto50k/index"),
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--normalized-dir", type=Path)
    args = parser.parse_args()
    raw_root = args.raw_root or Path(f"data/raw/reactions/uspto_{args.dataset}")
    output_root = args.output_root or Path(
        f"outputs/reaction_sites/uspto_{args.dataset}"
    )
    normalized_dir = args.normalized_dir or Path(
        f"data/processed/reactions/uspto_{args.dataset}"
    )
    raw_splits = [
        args.uspto50k_raw / f"atom_mapped_{split}.csv"
        for split in ("train", "valid", "test")
    ]
    if all(path.is_file() for path in raw_splits):
        source_identities = source_uspto50k_identities(args.uspto50k_raw)
        overlap_audit_mode = "canonical mapped reaction identity"
    else:
        source_identities = set()
        overlap_audit_mode = "processed reaction-site identity fallback"
    normalization = write_normalized_splits(
        dataset=args.dataset,
        raw_root=raw_root,
        normalized_dir=normalized_dir,
        source_identities=source_identities,
    )
    index_dir = output_root / "index"
    preparation = {}
    for split in ("train", "valid", "test"):
        preparation[split] = prepare_split(
            normalized_dir / f"atom_mapped_{split}.csv",
            index_dir / f"{split}.jsonl",
            max_atoms=args.max_atoms,
            component_policy="exactly-two",
            require_positive_both=True,
        )
    site_overlap_filter = None
    if overlap_audit_mode == "processed reaction-site identity fallback":
        source_site_identities = source_uspto50k_site_identities(
            args.uspto50k_index
        )
        site_overlap_filter = {
            "source_uspto50k_site_identities": len(source_site_identities),
            "splits": {
                split: filter_index_site_overlap(
                    index_dir / f"{split}.jsonl", source_site_identities
                )
                for split in ("train", "valid", "test")
            },
        }
    report = {
        "overlap_audit_mode": overlap_audit_mode,
        "normalization": normalization,
        "reaction_site_filtering": {
            "component_policy": "exactly-two product-participating reactants",
            "require_positive_both": True,
            "max_atoms_per_molecule": args.max_atoms,
            "splits": preparation,
        },
    }
    if site_overlap_filter is not None:
        report["post_filter_uspto50k_overlap"] = site_overlap_filter
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Preparation report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
