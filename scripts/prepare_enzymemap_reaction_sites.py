#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from dcir.reaction_sites import (  # noqa: E402
    _canonical_component,
    _load_jsonl,
    _rdkit,
    changed_atom_maps,
)


PROCESSED_MEMBER = "enzymemap-main/data/processed_reactions.csv.gz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows_from_archive(
    archive_path: Path, member: str
) -> Iterator[dict[str, str]]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with (
        ZipFile(archive_path) as archive,
        archive.open(member) as compressed,
        gzip.GzipFile(fileobj=compressed) as uncompressed,
        io.TextIOWrapper(
            uncompressed, encoding="utf-8", newline=""
        ) as text,
    ):
        yield from csv.DictReader(text)


def _source_keys(index_dir: Path) -> tuple[set[str], set[tuple[str, str]]]:
    molecule_keys: set[str] = set()
    pair_keys: set[tuple[str, str]] = set()
    for split in ("train", "valid", "test"):
        path = index_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing USPTO source index: {path}")
        for record in _load_jsonl(path):
            first = str(record["d1"]["key"])
            second = str(record["d2"]["key"])
            molecule_keys.update((first, second))
            pair_keys.add(tuple(sorted((first, second))))
    return molecule_keys, pair_keys


def _ec_class(value: str) -> int:
    first = value.strip().split(".", 1)[0]
    return int(first) if first.isdigit() else 0


def _truth(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def _reaction_record(
    row: dict[str, str],
    row_index: int,
    *,
    max_atoms: int,
) -> tuple[dict[str, Any] | None, str]:
    Chem, _ = _rdkit()
    reaction = row.get("mapped", "").strip()
    parts = reaction.split(">>")
    if len(parts) != 2:
        return None, "invalid_reaction"
    reactants = Chem.MolFromSmiles(parts[0])
    products = Chem.MolFromSmiles(parts[1])
    if reactants is None or products is None:
        return None, "parse_failed"
    changed = changed_atom_maps(reactants, products)
    if not changed:
        return None, "no_changed_atoms"

    product_maps = {
        int(atom.GetAtomMapNum())
        for atom in products.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    participating = []
    try:
        fragments = Chem.GetMolFrags(
            reactants, asMols=True, sanitizeFrags=True
        )
    except Exception:
        return None, "parse_failed"
    for component in fragments:
        maps = {
            int(atom.GetAtomMapNum())
            for atom in component.GetAtoms()
            if atom.GetAtomMapNum() > 0
        }
        if maps & product_maps and maps & changed:
            participating.append(component)
    if len(participating) != 2:
        return None, "not_two_participating_reactants"

    try:
        components = [
            _canonical_component(component, changed)
            for component in participating
        ]
    except (ValueError, RuntimeError):
        return None, "canonicalization_failed"
    if any(not item["positive_indices"] for item in components):
        return None, "missing_positive_component"
    if max(item["n_atoms"] for item in components) > max_atoms:
        return None, "too_many_atoms"

    components.sort(key=lambda item: (item["key"], item["smiles"]))
    signature_payload = [
        {
            "smiles": item["smiles"],
            "positive_indices": item["positive_indices"],
        }
        for item in components
    ]
    signature = hashlib.sha256(
        json.dumps(
            signature_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    record = {
        "sample_id": f"enzymemap:{row.get('rxn_idx', row_index)}:{signature[:12]}",
        "reaction_class": _ec_class(row.get("ec_num", "")),
        "d1": components[0],
        "d2": components[1],
        "metadata": {
            "dataset": "EnzymeMap",
            "rxn_idx": str(row.get("rxn_idx", "")),
            "ec_num": str(row.get("ec_num", "")),
            "quality": float(row.get("quality") or 0.0),
            "natural": _truth(row.get("natural", "")),
            "source": str(row.get("source", "")),
            "steps": str(row.get("steps", "")),
            "signature": signature,
        },
    }
    return record, signature


def prepare(
    *,
    archive_path: Path,
    output_dir: Path,
    source_index_dir: Path,
    member: str,
    min_quality: float,
    max_atoms: int,
    limit: int | None,
) -> dict[str, Any]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing EnzymeMap archive: {archive_path}")
    source_molecules, source_pairs = _source_keys(source_index_dir)
    index_dir = output_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    temporary = index_dir / "test.jsonl.prepare-tmp"
    cold_temporary = index_dir / "test_cold_both.jsonl.prepare-tmp"
    counters: Counter[str] = Counter()
    seen_mapped_inputs: set[bytes] = set()
    seen_signatures: set[str] = set()
    unique_molecules: dict[str, str] = {}
    ec_classes: Counter[int] = Counter()
    primary_pairs: set[tuple[str, str]] = set()
    total_atoms = 0
    positive_atoms = 0

    try:
        with (
            temporary.open("w", encoding="utf-8", newline="\n") as output,
            cold_temporary.open(
                "w", encoding="utf-8", newline="\n"
            ) as cold_output,
        ):
            for row_index, row in enumerate(
                _rows_from_archive(archive_path, member)
            ):
                counters["source_rows"] += 1
                if not _truth(row.get("natural", "")):
                    counters["filtered_not_natural"] += 1
                    continue
                if row.get("steps", "").strip().lower() != "single":
                    counters["filtered_not_single_step"] += 1
                    continue
                if row.get("source", "").strip().lower() != "direct":
                    counters["filtered_not_direct"] += 1
                    continue
                try:
                    quality = float(row.get("quality") or 0.0)
                except ValueError:
                    counters["filtered_invalid_quality"] += 1
                    continue
                if quality < min_quality:
                    counters["filtered_low_quality"] += 1
                    continue
                counters["eligible_rows"] += 1
                mapped_digest = hashlib.sha256(
                    row.get("mapped", "").strip().encode("utf-8")
                ).digest()
                if mapped_digest in seen_mapped_inputs:
                    counters["duplicate_mapped_input"] += 1
                    continue
                seen_mapped_inputs.add(mapped_digest)
                record, status = _reaction_record(
                    row, row_index, max_atoms=max_atoms
                )
                if record is None:
                    counters[status] += 1
                    continue
                signature = status
                if signature in seen_signatures:
                    counters["duplicate_reaction"] += 1
                    continue
                seen_signatures.add(signature)
                pair = tuple(
                    sorted((record["d1"]["key"], record["d2"]["key"]))
                )
                if pair in source_pairs:
                    counters["excluded_uspto_pair_overlap"] += 1
                    continue
                molecule_overlap = sum(
                    item["key"] in source_molecules
                    for item in (record["d1"], record["d2"])
                )
                record["metadata"]["uspto_molecule_overlap"] = molecule_overlap
                line = json.dumps(record, separators=(",", ":")) + "\n"
                output.write(line)
                counters["written_primary"] += 1
                primary_pairs.add(pair)
                if molecule_overlap == 0:
                    cold_output.write(line)
                    counters["written_cold_both"] += 1
                else:
                    counters[
                        f"written_with_{molecule_overlap}_source_molecules"
                    ] += 1
                ec_classes[int(record["reaction_class"])] += 1
                for role in ("d1", "d2"):
                    item = record[role]
                    unique_molecules[item["key"]] = item["smiles"]
                    total_atoms += int(item["n_atoms"])
                    positive_atoms += len(item["positive_indices"])
                if limit is not None and counters["written_primary"] >= limit:
                    counters["stopped_at_limit"] = 1
                    break
        temporary.replace(index_dir / "test.jsonl")
        cold_temporary.replace(index_dir / "test_cold_both.jsonl")
    except BaseException:
        temporary.unlink(missing_ok=True)
        cold_temporary.unlink(missing_ok=True)
        raise

    for split in ("train", "valid"):
        (index_dir / f"{split}.jsonl").write_text("", encoding="utf-8")
    molecule_manifest = output_dir / "molecules.jsonl"
    with molecule_manifest.open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for key, smiles in sorted(unique_molecules.items()):
            handle.write(
                json.dumps(
                    {"key": key, "smiles": smiles}, separators=(",", ":")
                )
                + "\n"
            )

    report: dict[str, Any] = {
        "dataset": "EnzymeMap",
        "archive": str(archive_path.resolve()),
        "archive_sha256": _sha256(archive_path),
        "archive_member": member,
        "selection": {
            "natural_only": True,
            "single_step_only": True,
            "source": "direct",
            "min_quality": min_quality,
            "exactly_two_participating_reactants": True,
            "positive_atoms_required_in_both": True,
            "max_atoms_per_molecule": max_atoms,
            "exclude_exact_uspto_pair_overlap": True,
        },
        "source_uspto": {
            "index_dir": str(source_index_dir.resolve()),
            "unique_molecules": len(source_molecules),
            "unique_pairs": len(source_pairs),
        },
        "counts": dict(sorted(counters.items())),
        "unique_external_molecules": len(unique_molecules),
        "primary_statistics": {
            "unique_reactant_pairs": len(primary_pairs),
            "atoms": total_atoms,
            "positive_atoms": positive_atoms,
            "positive_atom_rate": (
                positive_atoms / total_atoms if total_atoms else 0.0
            ),
        },
        "ec_class_counts_primary": {
            str(key): value for key, value in sorted(ec_classes.items())
        },
        "primary_index": str((index_dir / "test.jsonl").resolve()),
        "strict_cold_index": str(
            (index_dir / "test_cold_both.jsonl").resolve()
        ),
        "molecule_manifest": str(molecule_manifest.resolve()),
        "output_sha256": {
            "test": _sha256(index_dir / "test.jsonl"),
            "test_cold_both": _sha256(
                index_dir / "test_cold_both.jsonl"
            ),
            "molecule_manifest": _sha256(molecule_manifest),
        },
    }
    report_path = output_dir / "preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.warning")
    except ImportError:
        pass
    parser = argparse.ArgumentParser(
        description=(
            "Prepare high-quality EnzymeMap reactions for zero-shot "
            "two-reactant atom-site evaluation."
        )
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reaction_sites/enzymemap"),
    )
    parser.add_argument(
        "--source-index-dir",
        type=Path,
        default=Path("outputs/reaction_sites/uspto50k/index"),
    )
    parser.add_argument("--member", default=PROCESSED_MEMBER)
    parser.add_argument("--min-quality", type=float, default=0.9)
    parser.add_argument("--max-atoms", type=int, default=128)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 0.0 <= args.min_quality <= 1.0:
        raise ValueError("--min-quality must be in [0, 1]")
    if args.max_atoms < 1:
        raise ValueError("--max-atoms must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    report = prepare(
        archive_path=args.archive,
        output_dir=args.output_dir,
        source_index_dir=args.source_index_dir,
        member=args.member,
        min_quality=args.min_quality,
        max_atoms=args.max_atoms,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2))
    print(
        "Preparation report written to "
        f"{args.output_dir / 'preparation_report.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
