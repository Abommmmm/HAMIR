from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def mapper_input(reaction: str) -> str:
    parts = reaction.strip().split(">")
    if len(parts) != 3:
        return ""
    left = ".".join(value for value in parts[:2] if value)
    return f"{left}>>{parts[2]}"


def completed_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/reactions/uspto_full/USPTO_FULL.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/reactions/uspto_full/atom_mapped_all.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        from rxnmapper import BatchedMapper
    except ImportError as exc:
        raise RuntimeError(
            'Install the mapper first: pip install "rxnmapper[rdkit]"'
        ) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_rows(args.output)
    mapper = BatchedMapper(batch_size=args.batch_size)
    fields = [
        "id",
        "PatentNumber",
        "Year",
        "rxn_smiles",
        "mapping_status",
    ]
    mode = "a" if done else "w"
    counters = {"source_seen": 0, "mapped": 0, "failed": 0, "resumed_at": done}
    with (
        args.input.open("r", encoding="utf-8-sig", newline="") as source,
        args.output.open(mode, encoding="utf-8", newline="") as destination,
    ):
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=fields)
        if not done:
            writer.writeheader()
        batch: list[tuple[int, dict[str, str], str]] = []

        def flush() -> None:
            if not batch:
                return
            inputs = [item[2] for item in batch]
            try:
                mapped = list(mapper.map_reactions(inputs))
            except Exception:
                mapped = [""] * len(inputs)
            if len(mapped) != len(batch):
                mapped = [""] * len(batch)
            for (row_index, row, _), reaction in zip(batch, mapped):
                ok = bool(reaction and reaction != ">>")
                writer.writerow(
                    {
                        "id": f"uspto_full:{row_index}",
                        "PatentNumber": row.get("PatentNumber", ""),
                        "Year": row.get("Year", ""),
                        "rxn_smiles": reaction if ok else "",
                        "mapping_status": "ok" if ok else "failed",
                    }
                )
                counters["mapped" if ok else "failed"] += 1
            destination.flush()
            batch.clear()

        for row_index, row in enumerate(reader):
            counters["source_seen"] += 1
            if row_index < done:
                continue
            reaction = mapper_input(row.get("reactions", ""))
            if not reaction:
                writer.writerow(
                    {
                        "id": f"uspto_full:{row_index}",
                        "PatentNumber": row.get("PatentNumber", ""),
                        "Year": row.get("Year", ""),
                        "rxn_smiles": "",
                        "mapping_status": "invalid",
                    }
                )
                counters["failed"] += 1
                continue
            batch.append((row_index, row, reaction))
            if len(batch) >= args.batch_size:
                flush()
            if args.limit is not None and row_index + 1 >= done + args.limit:
                break
        flush()
    report = args.output.with_suffix(".mapping_report.json")
    report.write_text(json.dumps(counters, indent=2), encoding="utf-8")
    print(json.dumps(counters, indent=2))
    print(f"Mapped reactions written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
