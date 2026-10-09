from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


MIT_URL = (
    "https://raw.githubusercontent.com/wengong-jin/"
    "nips17-rexgen/master/USPTO/data.zip"
)
MIT_SHA256 = "6d94a136e11f76fe464430cb95d1ae6db37b6ca352161ca4edddd9e6fe76a88a"
FULL_URL = "https://deepchemdata.s3.us-west-1.amazonaws.com/" "datasets/USPTO_FULL.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        actual = sha256(destination)
        if expected_sha256 is None or actual == expected_sha256:
            print(f"Reusing {destination} (sha256={actual})")
            return
        raise RuntimeError(
            f"Existing file has unexpected SHA256: {destination}: {actual}"
        )
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "HAMIR-dataset/1.0"})
    with urllib.request.urlopen(request) as source, temporary.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
    actual = sha256(temporary)
    if expected_sha256 is not None and actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"SHA256 mismatch for {url}: {actual}")
    temporary.replace(destination)
    print(f"Downloaded {destination} (sha256={actual})")


def download_mit(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "nips17_rexgen_uspto_data.zip"
    download(MIT_URL, archive, MIT_SHA256)
    members = {
        "train": "data/train.txt",
        "valid": "data/valid.txt",
        "test": "data/test.txt",
    }
    with zipfile.ZipFile(archive) as bundle:
        for split, member in members.items():
            destination = root / f"mapped_{split}.txt"
            with bundle.open(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
    return {
        "dataset": "USPTO-MIT",
        "source_url": MIT_URL,
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "splits": {split: str(root / f"mapped_{split}.txt") for split in members},
        "note": "Fixed mapped splits from nips17-rexgen.",
    }


def download_full(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "USPTO_FULL.csv"
    download(FULL_URL, raw, None)
    return {
        "dataset": "USPTO-FULL",
        "source_url": FULL_URL,
        "raw_csv": str(raw),
        "raw_sha256": sha256(raw),
        "note": (
            "This mirror is not atom mapped. Run scripts/map_uspto_full.py "
            "before preparing reaction-site labels."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["mit", "full", "all"])
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/reactions"))
    args = parser.parse_args()
    reports = []
    if args.dataset in {"mit", "all"}:
        reports.append(download_mit(args.raw_root / "uspto_mit"))
    if args.dataset in {"full", "all"}:
        reports.append(download_full(args.raw_root / "uspto_full"))
    report_path = args.raw_root / "uspto_large_download_report.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    print(f"Download report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
