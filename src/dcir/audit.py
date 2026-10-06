from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


REQUIRED_COLUMNS = ("d1", "type", "d2")


@dataclass(frozen=True)
class DDIRow:
    d1: str
    type_raw: str
    d2: str
    source_row: int

    @property
    def triple(self) -> tuple[str, str, str]:
        return self.d1, self.type_raw, self.d2

    @property
    def directed_pair(self) -> tuple[str, str]:
        return self.d1, self.d2

    @property
    def unordered_pair(self) -> tuple[str, str]:
        return tuple(sorted((self.d1, self.d2)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: object) -> str:
    return str(value).strip().replace("\ufeff", "")


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    attempted = "\n  - ".join(str(path) for path in paths)
    raise FileNotFoundError(f"None of the expected paths exists:\n  - {attempted}")


def ddimdl_database_path(project_root: Path) -> Path:
    return _first_existing(
        project_root / "data" / "raw" / "DDIMDL-master" / "event.db",
        project_root / "data" / "raw" / "deng" / "event.db",
    )


def deepddi_data_path(project_root: Path) -> Path:
    return _first_existing(
        project_root / "data" / "raw" / "DeepDDI" / "data",
        project_root / "data" / "raw" / "ryu",
    )


def read_ddi_csv(path: Path) -> list[DDIRow]:
    rows: list[DDIRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        normalized = {_clean(name): name for name in reader.fieldnames}
        missing = [name for name in REQUIRED_COLUMNS if name not in normalized]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        for index, row in enumerate(reader, start=2):
            d1 = _clean(row[normalized["d1"]])
            d2 = _clean(row[normalized["d2"]])
            label = _clean(row[normalized["type"]])
            if not d1 or not d2 or not label:
                raise ValueError(f"Missing d1/type/d2 at {path}:{index}")
            rows.append(DDIRow(d1=d1, type_raw=label, d2=d2, source_row=index))
    return rows


def read_deepddi_csv(path: Path) -> list[DDIRow]:
    """Read the author-released DeepDDI gold-standard interaction table."""
    rows: list[DDIRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Drug1", "Drug2", "Label"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} is missing columns: {sorted(required)}")
        for index, row in enumerate(reader, start=2):
            d1 = _clean(row["Drug1"])
            d2 = _clean(row["Drug2"])
            label = _clean(row["Label"])
            if not d1 or not d2 or not label:
                raise ValueError(f"Missing Drug1/Drug2/Label at {path}:{index}")
            rows.append(DDIRow(d1=d1, type_raw=label, d2=d2, source_row=index))
    return rows


def _normalize_ddimdl_template(
    interaction: str, name1: str, name2: str
) -> str:
    text = interaction.strip()
    for name in sorted({name1, name2}, key=len, reverse=True):
        if name:
            text = re.sub(re.escape(name), "name", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text.rstrip(".") + ".").lower()


def read_ddimdl_database(path: Path) -> list[DDIRow]:
    """Rebuild DDIMDL labels from the author database's event_number order."""
    with sqlite3.connect(path) as connection:
        event_types = connection.execute(
            "SELECT rowid, event FROM event_number ORDER BY rowid"
        ).fetchall()
        type_by_template = {
            re.sub(r"\s+", " ", str(template).strip().rstrip(".") + ".").lower():
            str(int(rowid) - 1)
            for rowid, template in event_types
        }
        events = connection.execute(
            'SELECT id1, name1, id2, name2, interaction FROM event ORDER BY "index"'
        ).fetchall()

    rows: list[DDIRow] = []
    for index, (d1, name1, d2, name2, interaction) in enumerate(events, start=1):
        template = _normalize_ddimdl_template(
            str(interaction), str(name1), str(name2)
        )
        if template not in type_by_template:
            raise ValueError(f"Unknown DDIMDL event template at database row {index}")
        rows.append(
            DDIRow(
                d1=_clean(d1),
                type_raw=type_by_template[template],
                d2=_clean(d2),
                source_row=index,
            )
        )
    return rows


def load_dataset_rows(project_root: Path, dataset: str) -> tuple[list[DDIRow], list[Path]]:
    dataset = dataset.lower()
    if dataset == "deng":
        path = ddimdl_database_path(project_root)
        return read_ddimdl_database(path), [path]
    if dataset == "ryu":
        path = deepddi_data_path(project_root) / "KnownDDI.csv"
        return read_deepddi_csv(path), [path]
    raise ValueError(f"Unknown dataset: {dataset}")


def _numeric_type_sort(values: Iterable[str]) -> list[str]:
    values = list(set(values))
    try:
        return sorted(values, key=lambda item: (int(item), item))
    except ValueError:
        return sorted(values)


def _gini(counts: list[int]) -> float:
    if not counts or sum(counts) == 0:
        return 0.0
    ordered = sorted(counts)
    n = len(ordered)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2 * weighted) / (n * sum(ordered)) - (n + 1) / n


def _normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0 or len(counts) <= 1:
        return 0.0
    entropy = -sum(
        (value / total) * math.log(value / total)
        for value in counts
        if value > 0
    )
    return entropy / math.log(len(counts))


def _continuous_numeric(types: list[str]) -> tuple[bool, list[int]]:
    try:
        values = sorted(int(value) for value in types)
    except ValueError:
        return False, []
    expected = set(range(values[0], values[-1] + 1))
    missing = sorted(expected - set(values))
    return not missing, missing


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_event_template(interaction: str, name1: str, name2: str) -> str:
    text = interaction.strip()
    replacements = sorted(
        [(name1, "<d1>"), (name2, "<d2>")],
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for name, replacement in replacements:
        if name:
            text = re.sub(re.escape(name), replacement, text, flags=re.IGNORECASE)
    return text


def deng_event_descriptions(
    db_path: Path, rows: Iterable[DDIRow]
) -> dict[str, str]:
    if not db_path.is_file():
        return {}
    by_pair: dict[tuple[str, str], tuple[str, str, str]] = {}
    with sqlite3.connect(db_path) as connection:
        for id1, name1, id2, name2, interaction in connection.execute(
            "SELECT id1, name1, id2, name2, interaction FROM event"
        ):
            by_pair[(str(id1), str(id2))] = (
                str(name1 or ""),
                str(name2 or ""),
                str(interaction or ""),
            )

    templates: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        event = by_pair.get(row.directed_pair)
        if event is None:
            continue
        name1, name2, interaction = event
        templates[row.type_raw][
            _normalize_event_template(interaction, name1, name2)
        ] += 1
    return {
        type_raw: counts.most_common(1)[0][0]
        for type_raw, counts in templates.items()
        if counts
    }


def ryu_event_descriptions(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.is_file():
        return mapping
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_type = _clean(row.get("DDI type", ""))
            match = re.search(r"(\d+)$", raw_type)
            description = _clean(row.get("Description", ""))
            if not match or not description:
                continue
            mapping[match.group(1)] = description
    return mapping


def event_descriptions(
    project_root: Path, dataset: str, rows: list[DDIRow]
) -> dict[str, str]:
    if dataset == "deng":
        return deng_event_descriptions(
            ddimdl_database_path(project_root), rows
        )
    return ryu_event_descriptions(
        deepddi_data_path(project_root) / "Interaction_information.csv"
    )


def audit_dataset(project_root: Path, dataset: str, output_dir: Path) -> dict:
    dataset = dataset.lower()
    raw_rows, source_paths = load_dataset_rows(project_root, dataset)
    exact_counts = Counter(row.triple for row in raw_rows)
    unique_rows: list[DDIRow] = []
    seen: set[tuple[str, str, str]] = set()
    for row in raw_rows:
        if row.triple not in seen:
            seen.add(row.triple)
            unique_rows.append(row)

    types = _numeric_type_sort(row.type_raw for row in unique_rows)
    type_to_internal = {value: index for index, value in enumerate(types)}
    continuous, missing_numeric = _continuous_numeric(types)
    pair_to_types: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in unique_rows:
        pair_to_types[row.directed_pair].add(row.type_raw)

    reverse_groups: set[tuple[str, str]] = set()
    reverse_same = reverse_partial = reverse_disjoint = 0
    for pair, labels in pair_to_types.items():
        a, b = pair
        reverse = (b, a)
        if reverse not in pair_to_types or a == b:
            continue
        group = tuple(sorted((a, b)))
        if group in reverse_groups:
            continue
        reverse_groups.add(group)
        reverse_labels = pair_to_types[reverse]
        if labels == reverse_labels:
            reverse_same += 1
        elif labels & reverse_labels:
            reverse_partial += 1
        else:
            reverse_disjoint += 1

    class_counts = Counter(row.type_raw for row in unique_rows)
    counts = [class_counts[value] for value in types]
    drugs = sorted({row.d1 for row in unique_rows} | {row.d2 for row in unique_rows})
    multi_pairs = {
        pair: labels for pair, labels in pair_to_types.items() if len(labels) > 1
    }
    task_mode = "multilabel" if multi_pairs else "multiclass"
    descriptions = event_descriptions(project_root, dataset, unique_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "type_mapping.csv",
        ["type_raw", "type_internal"],
        (
            {"type_raw": value, "type_internal": type_to_internal[value]}
            for value in types
        ),
    )
    _write_csv(
        output_dir / "event_mapping.csv",
        [
            "type_raw",
            "type_internal",
            "event_text",
            "source",
            "semantic_direction",
            "inverse_type_raw",
            "review_status",
        ],
        (
            {
                "type_raw": value,
                "type_internal": type_to_internal[value],
                "event_text": descriptions.get(value, ""),
                "source": (
                    "DDIMDL author event.db"
                    if dataset == "deng"
                    else "DeepDDI author Interaction_information.csv"
                ),
                "semantic_direction": "unreviewed",
                "inverse_type_raw": "",
                "review_status": "pending_manual_review",
            }
            for value in types
        ),
    )
    _write_csv(
        output_dir / "class_distribution.csv",
        ["type_raw", "type_internal", "count", "fraction"],
        (
            {
                "type_raw": value,
                "type_internal": type_to_internal[value],
                "count": class_counts[value],
                "fraction": class_counts[value] / len(unique_rows),
            }
            for value in types
        ),
    )
    _write_csv(
        output_dir / "exact_duplicates.csv",
        ["d1", "type_raw", "d2", "count"],
        (
            {"d1": key[0], "type_raw": key[1], "d2": key[2], "count": count}
            for key, count in exact_counts.items()
            if count > 1
        ),
    )
    _write_csv(
        output_dir / "directed_pair_multilabel.csv",
        ["d1", "d2", "label_count", "types"],
        (
            {
                "d1": pair[0],
                "d2": pair[1],
                "label_count": len(labels),
                "types": "|".join(_numeric_type_sort(labels)),
            }
            for pair, labels in sorted(multi_pairs.items())
        ),
    )
    _write_csv(
        output_dir / "records.csv",
        [
            "record_id",
            "d1",
            "type_raw",
            "type_internal",
            "d2",
            "unordered_pair",
        ],
        (
            {
                "record_id": index,
                "d1": row.d1,
                "type_raw": row.type_raw,
                "type_internal": type_to_internal[row.type_raw],
                "d2": row.d2,
                "unordered_pair": "||".join(row.unordered_pair),
            }
            for index, row in enumerate(unique_rows)
        ),
    )

    report = {
        "dataset": dataset,
        "task_mode": task_mode,
        "raw_record_count": len(raw_rows),
        "deduplicated_record_count": len(unique_rows),
        "exact_duplicate_extra_rows": len(raw_rows) - len(unique_rows),
        "unique_drug_count": len(drugs),
        "unique_directed_pair_count": len(pair_to_types),
        "unique_unordered_pair_count": len(
            {row.unordered_pair for row in unique_rows}
        ),
        "type_count": len(types),
        "type_values": types,
        "type_numeric_continuous": continuous,
        "missing_numeric_types": missing_numeric,
        "directed_pairs_with_multiple_types": len(multi_pairs),
        "reverse_unordered_pair_groups": len(reverse_groups),
        "reverse_same_label_groups": reverse_same,
        "reverse_partial_label_groups": reverse_partial,
        "reverse_disjoint_label_groups": reverse_disjoint,
        "self_pair_count": sum(1 for pair in pair_to_types if pair[0] == pair[1]),
        "imbalance_ratio": max(counts) / min(counts) if counts else None,
        "gini": _gini(counts),
        "normalized_entropy": _normalized_entropy(counts),
        "event_descriptions_available": sum(
            bool(descriptions.get(value)) for value in types
        ),
        "source_files": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
    }
    with (output_dir / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def iter_checksum_rows(paths: Iterable[Path]) -> Iterator[dict[str, str | int]]:
    for path in paths:
        if path.is_file():
            yield {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
