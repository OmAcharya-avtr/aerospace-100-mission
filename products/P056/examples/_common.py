"""Shared path handling for the example scripts.

Each example writes exactly one PNG into ``../screenshots/`` and prints the
absolute path it wrote, so the README's screenshots can never drift from the
code that produced them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS = ROOT / "screenshots"
sys.path.insert(0, str(ROOT / "src"))


def save(fig, name: str) -> Path:
    """Save ``fig`` as ``screenshots/<name>.png`` and return the path."""
    SCREENSHOTS.mkdir(exist_ok=True)
    path = SCREENSHOTS / f"{name}.png"
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")
    return path
