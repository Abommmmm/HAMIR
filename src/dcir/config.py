from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Install configs/requirements-data.txt or "
            "configs/requirements-cloud.txt."
        ) from exc

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")

    root = next(
        (
            parent
            for parent in config_path.parents
            if (parent / "src" / "dcir").is_dir()
            and (parent / "configs").is_dir()
        ),
        config_path.parents[1],
    )
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(root)
    for key, value in config.get("paths", {}).items():
        candidate = Path(value)
        if not candidate.is_absolute():
            config["paths"][key] = str((root / candidate).resolve())
    return config
