#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _pair_key(record: dict[str, Any]) -> tuple[str, str]:
    return tuple(sorted((record["d1"]["key"], record["d2"]["key"])))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grouped_split(
    records: list[dict[str, Any]],
    *,
    seed: int,
    train_fraction: float,
    valid_fraction: float,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_pair_key(record)].append(record)

    by_class: dict[int, list[tuple[tuple[str, str], list[dict[str, Any]]]]] = (
        defaultdict(list)
    )
    for pair, items in groups.items():
        counts = Counter(int(item.get("reaction_class", 0)) for item in items)
        dominant_class = min(
            counts, key=lambda value: (-counts[value], value)
        )
        by_class[dominant_class].append((pair, items))

    assignment: dict[tuple[str, str], str] = {}
    for reaction_class, class_groups in sorted(by_class.items()):
        rng = random.Random(seed * 1009 + reaction_class)
        rng.shuffle(class_groups)
        total = sum(len(items) for _, items in class_groups)
        train_target = total * train_fraction
        valid_target = total * valid_fraction
        assigned_train = 0
        assigned_valid = 0
        for pair, items in class_groups:
            if assigned_train < train_target:
                split = "train"
                assigned_train += len(items)
            elif assigned_valid < valid_target:
                split = "valid"
                assigned_valid += len(items)
            else:
                split = "test"
            assignment[pair] = split

    result = {"train": [], "valid": [], "test": []}
    for record in records:
        result[assignment[_pair_key(record)]].append(record)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic reactant-pair-disjoint EnzymeMap "
            "fine-tuning splits."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("outputs/reaction_sites/enzymemap/index/test.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reaction_sites/enzymemap_finetune/index"),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if args.train_fraction <= 0 or args.valid_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive")
    if args.train_fraction + args.valid_fraction >= 1:
        raise ValueError("Train + validation fractions must be below 1")

    records = _load(args.source)
    splits = grouped_split(
        records,
        seed=args.seed,
        train_fraction=args.train_fraction,
        valid_fraction=args.valid_fraction,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    pair_sets: dict[str, set[tuple[str, str]]] = {}
    report: dict[str, Any] = {
        "source": str(args.source.resolve()),
        "source_sha256": _sha256(args.source),
        "seed": args.seed,
        "grouping_unit": "canonical_unordered_reactant_pair",
        "requested_fractions": {
            "train": args.train_fraction,
            "valid": args.valid_fraction,
            "test": 1.0 - args.train_fraction - args.valid_fraction,
        },
        "splits": {},
    }
    for split, items in splits.items():
        path = args.output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in items:
                handle.write(
                    json.dumps(record, separators=(",", ":")) + "\n"
                )
        paths[split] = path
        pairs = {_pair_key(record) for record in items}
        pair_sets[split] = pairs
        classes = Counter(
            int(record.get("reaction_class", 0)) for record in items
        )
        report["splits"][split] = {
            "examples": len(items),
            "unique_pairs": len(pairs),
            "ec_class_counts": {
                str(key): value for key, value in sorted(classes.items())
            },
            "sha256": _sha256(path),
        }

    overlap = {}
    for first, second in (
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ):
        overlap[f"{first}_{second}"] = len(
            pair_sets[first] & pair_sets[second]
        )
    if any(overlap.values()):
        raise AssertionError(f"Reactant-pair leakage detected: {overlap}")
    report["pair_overlap"] = overlap
    report["total_examples"] = sum(len(items) for items in splits.values())
    report_path = args.output_dir.parent / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Split report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
