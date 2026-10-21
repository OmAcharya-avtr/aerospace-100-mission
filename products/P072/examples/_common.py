"""Shared plotting setup for the examples. Agg backend, never ``show()``.

Paths are anchored on ``__file__`` and the working directory is changed to the
repository root, so every example runs identically from any directory and every
figure lands in ``screenshots/`` beside the repository root rather than beside
the caller.

One palette is used across every figure so that the same thing is always the
same colour: the exact solver is blue, the greedy baseline orange, the oracle
black, the learned components green, the analytic baselines red, and anything
left behind grey.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

EXACT = "#1f6fb4"
GREEDY = "#d9700b"
ORACLE = "#1a1a1a"
LEARNED = "#4f8f3f"
ANALYTIC = "#b0344f"
LEFT_BEHIND = "#8c8c8c"
GRID = "#d6d6d6"
SHADE = "#f0e2d2"

REASON_COLOURS = {
    "expired": "#b0344f",
    "oversized": "#d9700b",
    "dependency": "#7a5195",
    "capacity": "#1f6fb4",
}


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path``, change to the repository root, return it."""
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(ROOT)
    return ROOT


def style() -> None:
    """Set the shared rcParams. Called once per example before plotting.

    ``figure.autolayout`` is off: several figures use an explicit
    :meth:`matplotlib.figure.Figure.add_gridspec` with width ratios, which
    ``tight_layout`` cannot reconcile and warns about. Each figure sets its own
    spacing and is saved with ``bbox_inches="tight"``.
    """
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
    import matplotlib.pyplot as pyplot

    pyplot.close(figure)
    return relative(target)
