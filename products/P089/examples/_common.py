"""Shared plotting setup for the examples. Agg backend, never ``show()``.

Paths are anchored on ``__file__`` and the working directory is changed to the
repository root, so every example runs identically from any directory and every
figure lands in ``screenshots/`` beside the repository root rather than beside
the caller.

One palette across every figure, so the same thing is always the same colour:
an observable degree of freedom is blue, a marginal one amber, an unobservable
one red; the batch solution is dark blue, the regularised variant teal, the
recursive filter purple, the learned estimator green and the trivial baseline
grey. A Cramer-Rao bound is always a dashed red line.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

OBSERVABLE = "#1f6fb4"
MARGINAL = "#d9850b"
UNOBSERVABLE = "#b0344f"
BATCH = "#1f4e79"
MAP = "#2e8b8b"
RECURSIVE = "#6f3f8f"
LEARNED = "#3f8f4f"
TRIVIAL = "#9a9a9a"
BOUND = "#b0344f"
FLOOR = "#7a7a7a"
GRID = "#d6d6d6"

VERDICT_COLOURS = {
    "observable": OBSERVABLE,
    "marginal": MARGINAL,
    "unobservable": UNOBSERVABLE,
}

PROFILE_COLOURS = {
    "single_axis": "#b0344f",
    "near_single_axis": "#d9850b",
    "two_axis": "#1f6fb4",
    "three_axis": "#3f8f4f",
    "isotropic": "#6f3f8f",
}

ESTIMATOR_COLOURS = {
    "batch": BATCH,
    "batch MAP": MAP,
    "recursive": RECURSIVE,
    "learned": LEARNED,
    "trivial": TRIVIAL,
}

ARCSEC_PER_RAD = 206264.80624709636


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path``, change to the repository root, return it."""
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(ROOT)
    return ROOT


def style() -> None:
    """Set the shared rcParams. Called once per example before plotting.

    ``figure.autolayout`` is off: several figures use an explicit gridspec with
    height ratios, which ``tight_layout`` cannot reconcile and warns about. Each
    figure sets its own spacing and is saved with ``bbox_inches="tight"``.
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
    plt.close(figure)
    return relative(target)
