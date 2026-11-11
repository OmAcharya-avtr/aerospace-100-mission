"""Shared plotting setup for the examples. Agg backend, never ``show()``.

Paths are anchored on ``__file__`` and the working directory is changed to the
repository root, so every example runs identically from any directory and
every figure lands in ``screenshots/`` beside the repository root rather than
beside the caller.

One palette across every figure, so the same thing is always the same colour:
live is blue, recorded green, simulated amber, a refusal red, truth black.
That mapping is the product's whole subject, so it is declared once here and
never chosen per figure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

LIVE = "#1f4e79"
RECORDED = "#3f8f4f"
SIMULATED = "#d9850b"
REFUSED = "#b0344f"
TRUTH = "#1a1a1a"
GRID = "#d6d6d6"
NEUTRAL = "#7a7a7a"

PROVENANCE_COLOURS = {
    "live": LIVE,
    "recorded": RECORDED,
    "simulated": SIMULATED,
}

LABEL_COLOURS = {
    "MEASURED": LIVE,
    "REPLAYED": RECORDED,
    "SYNTHETIC": SIMULATED,
}


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path``, change to the repository root, return it."""
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(ROOT)
    return ROOT


def style() -> None:
    """Set the shared rcParams. Called once per example before plotting."""
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.autolayout": False,
        }
    )


def relative(path: str | Path) -> str:
    """Render ``path`` relative to the repository root, never absolutely."""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return candidate.name


def save(figure, name: str) -> str:
    """Save ``figure`` to ``screenshots/<name>.png`` and return a relative path."""
    target = ROOT / "screenshots" / f"{name}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return relative(target)
