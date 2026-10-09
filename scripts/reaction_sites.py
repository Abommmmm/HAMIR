from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
source_path = str(SOURCE)
if source_path in sys.path:
    sys.path.remove(source_path)
sys.path.insert(0, source_path)

from hamir.reaction_sites import main


if __name__ == "__main__":
    raise SystemExit(main())
