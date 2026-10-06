#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
# ``scripts/dcir.py`` is a CLI module, but Python can otherwise mistake it for
# the ``src/dcir`` package when this wrapper is launched by file path. Always
# put src first, even when PYTHONPATH already contains it later in sys.path.
source_path = str(SOURCE)
if source_path in sys.path:
    sys.path.remove(source_path)
sys.path.insert(0, source_path)

from dcir.reaction_sites import main


if __name__ == "__main__":
    raise SystemExit(main())
