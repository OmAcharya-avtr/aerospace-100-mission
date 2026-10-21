"""Shared plotting setup for the examples. Agg backend, never ``show()``.

Paths are anchored on ``__file__`` and the working directory is changed to the
repository root, so every example runs identically from any directory.

One palette is used across every figure so that the same thing is always the
same colour: the quadrature reference is black, Foster-Estes blue, Chan orange,
Alfano green, the small-hard-body approximation red, and Monte Carlo grey.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

REFERENCE = "#1a1a1a"
FOSTER = "#1f6fb4"
CHAN = "#d9700b"
ALFANO = "#4f8f3f"
SMALL_BODY = "#b0344f"
MONTE_CARLO = "#8c8c8c"
LEARNED = "#1f6fb4"
MISS_SCREEN = "#8c8c8c"
MAHALANOBIS_SCREEN = "#d9700b"
GRID = "#d6d6d6"
SHADE = "#f0e2d2"

METHOD_COLOURS = {
    "foster": FOSTER,
    "chan": CHAN,
    "alfano": ALFANO,
    "small_body": SMALL_BODY,
    "quadrature": REFERENCE,
}


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path``, change to the repository root, return it."""
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(ROOT)
    return ROOT


def screenshots_dir() -> Path:
    """``<repo>/screenshots``, created if missing."""
    path = ROOT / "screenshots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def style() -> None:
    """Apply the shared rcParams. Called once at the top of every example."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 110,
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


def save(figure, name: str) -> Path:
    """Save ``figure`` into ``screenshots/<name>`` and return the relative path."""
    target = screenshots_dir() / name
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return target.relative_to(ROOT)
