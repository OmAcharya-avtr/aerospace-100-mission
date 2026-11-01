"""Shared plotting setup for the examples. Agg backend, never ``show()``.

Paths are anchored on ``__file__`` and the working directory is changed to the
repository root, so every example runs identically from any directory and
every figure lands in ``screenshots/`` beside the repository root rather than
beside the caller.

One palette across every figure, so the same thing is always the same colour:
a distribution-free bound is blue, a resampling bound green, a parametric
bound amber, the learned model purple, the conformal repair dark purple, the
truth black, a refusal red.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

DISTRIBUTION_FREE = "#1f4e79"
RESAMPLE = "#3f8f4f"
PARAMETRIC = "#d9850b"
LEARNED = "#8f5fb0"
LEARNED_CONFORMAL = "#5a2d78"
TRUTH = "#1a1a1a"
REFUSED = "#b0344f"
GRID = "#d6d6d6"
NEUTRAL = "#7a7a7a"

METHOD_COLOURS = {
    "hoeffding": DISTRIBUTION_FREE,
    "empirical_bernstein": "#2f7fb4",
    "bootstrap_percentile": RESAMPLE,
    "bootstrap_basic": "#6fb07f",
    "wilks_tolerance": DISTRIBUTION_FREE,
    "split_conformal": RESAMPLE,
    "normal_tolerance": PARAMETRIC,
    "learned_quantile": LEARNED,
    "learned_quantile_conformal": LEARNED_CONFORMAL,
    "learned_mean_plugin": LEARNED,
}

METHOD_MARKERS = {
    "hoeffding": "o",
    "empirical_bernstein": "s",
    "bootstrap_percentile": "^",
    "bootstrap_basic": "v",
    "wilks_tolerance": "o",
    "split_conformal": "^",
    "normal_tolerance": "D",
    "learned_quantile": "*",
    "learned_quantile_conformal": "P",
    "learned_mean_plugin": "*",
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
