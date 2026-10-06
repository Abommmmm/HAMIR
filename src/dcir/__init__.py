"""DCIR: double-molecular conditional interaction prediction package."""

from .config import load_config

__all__ = ["DCIR", "load_config"]
__version__ = "0.1.0"


def __getattr__(name: str):
    """Load the neural model lazily so data-only commands do not require PyTorch."""
    if name == "DCIR":
        from .models import DCIR

        return DCIR
    raise AttributeError(name)
