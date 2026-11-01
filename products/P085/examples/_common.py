"""Shared plotting setup for the examples. Agg backend, never ``show()``.

Paths are anchored on ``__file__`` and the working directory is changed to the
repository root, so every example runs identically from any directory and every
figure lands in ``screenshots/`` beside the repository root rather than beside
the caller.

One palette across every figure, so the same thing is always the same colour:
least squares is dark blue, Huber green, RANSAC purple, the truth black, a
refusal red, a determined direction green, a bound red.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

LEAST = "#1f4e79"
HUBER = "#3f8f4f"
RANSAC = "#6f3f8f"
TRUTH = "#1a1a1a"
REFUSED = "#b0344f"
DETERMINED = "#3f8f4f"
MARGINAL = "#d9850b"
BOUND = "#b0344f"
INDEPENDENT = "#6f3f8f"
GRID = "#d6d6d6"
NEUTRAL = "#7a7a7a"

GEOMETRY_COLOURS = {
    "full_sky": "#1f6fb4",
    "two_az": "#3f8f4f",
    "low_el": "#d9850b",
    "narrow_el": "#b0344f",
    "single_az": "#6f3f8f",
}

TERM_BLOCK = {
    "IA": "cross-elevation",
    "CA": "cross-elevation",
    "NPAE": "cross-elevation",
    "IE": "elevation",
    "TF": "elevation",
    "TS": "elevation",
    "RF": "elevation",
    "AN": "both",
    "AW": "both",
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

    ``figure.autolayout`` is off: several figures use an explicit gridspec with
    height ratios, which ``tight_layout`` cannot reconcile and warns about.
    Each figure sets its own spacing and is saved with ``bbox_inches="tight"``.
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
