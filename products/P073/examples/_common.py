"""Shared plotting setup for the examples. Agg backend, never ``show()``.

Paths are anchored on ``__file__`` and the working directory is changed to the
repository root, so every example runs identically from any directory and every
figure lands in ``screenshots/`` beside the repository root rather than beside
the caller. The release gate runs each of these scripts with its working
directory set to ``examples/``, which is why this matters.

One palette is used across every figure so that the same thing is always the
same colour: the greedy baseline is orange, the exact optimum blue, the
fairness-constrained optimum green, the learned predictor purple, a
contention-ratio baseline grey, and anything starved or violated is red.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

GREEDY = "#d9700b"
EXACT = "#1f6fb4"
FAIR = "#4f8f3f"
LEARNED = "#7b4fa8"
BASELINE = "#8c8c8c"
STARVED = "#b0344f"
GRID = "#d6d6d6"
WINDOW = "#c9d9e8"

STATION_COLOURS = (
    "#1f6fb4",
    "#d9700b",
    "#4f8f3f",
    "#7b4fa8",
    "#b0344f",
    "#5a5a5a",
)


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


def hours(seconds: float) -> float:
    """Convert a second offset to hours, for axis labels."""
    return float(seconds) / 3600.0
