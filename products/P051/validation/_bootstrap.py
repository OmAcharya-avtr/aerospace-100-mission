"""Path bootstrap shared by the validation scripts and the examples.

The package is importable from ``src`` without an install step, matching the
``pythonpath = ["src"]`` setting in ``pyproject.toml`` that lets
``python -m pytest tests/ -q`` run from a cold clone.
"""

from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path() -> Path:
    """Insert ``<repo>/src`` at the front of ``sys.path`` and return the repo root."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root
