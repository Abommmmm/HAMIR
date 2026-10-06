from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Record:
    record_id: int
    d1: str
    d2: str
    label: int

    @property
    def group(self) -> tuple[str, str]:
        return tuple(sorted((self.d1, self.d2)))


def read_records(path: Path) -> list[Record]:
    rows: list[Record] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                Record(
                    record_id=int(row["record_id"]),
                    d1=row["d1"].strip(),
                    d2=row["d2"].strip(),
                    label=int(row["type_internal"]),
                )
            )
    return rows


def _group_records(records: Iterable[Record]) -> dict[tuple[str, str], list[Record]]:
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for record in records:
        groups[record.group].append(record)
    return groups


def grouped_pair_split(
    records: list[Record],
    seed: int,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[int, str]:
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to 1")
    rng = random.Random(seed)
    groups = _group_records(records)
    total_labels = Counter(record.label for record in records)
    targets = {
        split: {
            label: count * fraction
            for label, count in total_labels.items()
        }
        for split, fraction in zip(("train", "val", "test"), fractions)
    }
    size_targets = {
        split: len(records) * fraction
        for split, fraction in zip(("train", "val", "test"), fractions)
    }
    assigned_counts = {split: Counter() for split in targets}
    assigned_sizes = Counter()
    assignment: dict[int, str] = {}

    items = list(groups.items())
    rng.shuffle(items)
    items.sort(
        key=lambda item: (
            -len({record.label for record in item[1]}),
            -len(item[1]),
        )
    )
    for _, group_records in items:
        group_counts = Counter(record.label for record in group_records)
        scores: dict[str, float] = {}
        for split in ("train", "val", "test"):
            # Compare normalized fill ratios, not absolute residuals. Absolute
            # residuals favor the smaller validation/test targets and can invert
            # an intended 80/10/10 split.
            size_fill = (
                assigned_sizes[split] + len(group_records)
            ) / max(size_targets[split], 1.0)
            relevant_labels = list(group_counts)
            label_fill = sum(
                (
                    assigned_counts[split][label]
                    + group_counts[label]
                )
                / max(targets[split][label], 1.0)
                for label in relevant_labels
            ) / max(len(relevant_labels), 1)
            overfill = max(
                0.0,
                assigned_sizes[split]
                + len(group_records)
                - size_targets[split],
            )
            scores[split] = size_fill + 0.20 * label_fill + 10.0 * overfill
        best_score = min(scores.values())
        candidates = [split for split, score in scores.items() if score == best_score]
        chosen = rng.choice(candidates)
        assigned_counts[chosen].update(group_counts)
        assigned_sizes[chosen] += len(group_records)
        for record in group_records:
            assignment[record.record_id] = chosen
    return assignment


def _partition_drugs(
    records: list[Record], seed: int, fractions=(0.7, 0.15, 0.15)
) -> tuple[set[str], set[str], set[str]]:
    drugs = sorted({record.d1 for record in records} | {record.d2 for record in records})
    rng = random.Random(seed)
    rng.shuffle(drugs)
    n_train = int(len(drugs) * fractions[0])
    n_val = int(len(drugs) * fractions[1])
    train = set(drugs[:n_train])
    val = set(drugs[n_train : n_train + n_val])
    test = set(drugs[n_train + n_val :])
    return train, val, test


def cold_start_split(
    records: list[Record], seed: int, scenario: str
) -> dict[int, str]:
    train_drugs, val_drugs, test_drugs = _partition_drugs(records, seed)
    assignment: dict[int, str] = {}
    scenario = scenario.lower()
    for record in records:
        pair = {record.d1, record.d2}
        if pair <= train_drugs:
            assignment[record.record_id] = "train"
            continue
        if scenario == "one_cold":
            if (
                (record.d1 in val_drugs and record.d2 in train_drugs)
                or (record.d2 in val_drugs and record.d1 in train_drugs)
            ):
                assignment[record.record_id] = "val"
            elif (
                (record.d1 in test_drugs and record.d2 in train_drugs)
                or (record.d2 in test_drugs and record.d1 in train_drugs)
            ):
                assignment[record.record_id] = "test"
        elif scenario == "two_cold":
            if pair <= val_drugs:
                assignment[record.record_id] = "val"
            elif pair <= test_drugs:
                assignment[record.record_id] = "test"
        else:
            raise ValueError(f"Unknown cold-start scenario: {scenario}")
    return assignment


def validate_assignment(records: list[Record], assignment: dict[int, str]) -> dict:
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    split_counts = Counter()
    label_counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        split = assignment.get(record.record_id, "unused")
        if split != "unused":
            group_splits[record.group].add(split)
            split_counts[split] += 1
            label_counts[split][record.label] += 1
    leaks = {
        "||".join(group): sorted(splits)
        for group, splits in group_splits.items()
        if len(splits) > 1
    }
    if leaks:
        raise RuntimeError(
            f"Unordered/reverse pair leakage detected in {len(leaks)} groups"
        )
    return {
        "record_counts": dict(split_counts),
        "label_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in label_counts.items()
        },
        "unused_record_count": len(records) - sum(split_counts.values()),
        "leakage_groups": 0,
    }


def write_assignment(
    records: list[Record],
    assignment: dict[int, str],
    output_path: Path,
    metadata: dict,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = validate_assignment(records, assignment)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["record_id", "split"],
        )
        writer.writeheader()
        for record in records:
            split = assignment.get(record.record_id)
            if split is not None:
                writer.writerow({"record_id": record.record_id, "split": split})
    summary.update(metadata)
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
