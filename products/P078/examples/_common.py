"""Shared plotting setup for the examples. Agg backend, never ``show()``.

Paths are anchored on ``__file__`` and the working directory is changed to the
repository root, so every example runs identically from any directory and every
figure lands in ``screenshots/`` beside the repository root rather than beside
the caller.

One palette across every figure, so that the same thing is always the same
colour: the forward walk (ground truth) is blue, the static lint orange, the
learned ranker green, the free orderings purple and red, the random control
grey, and anything the gate refuses is dark red.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

FORWARD = "#1f6fb4"
LINT = "#d9700b"
LEARNED = "#4f8f3f"
FLAG = "#7a5195"
RESOURCE = "#b0344f"
RANDOM = "#8c8c8c"
REFUSE = "#8c1c13"
GRID = "#d6d6d6"
SHADE = "#f3e2e0"
OK = "#4f8f3f"

KIND_COLOURS = {
    "precondition_false": "#b0344f",
    "precondition_unknown": "#d9700b",
    "irreversible_unverified": "#8c1c13",
    "mode_gate": "#7a5195",
    "illegal_transition": "#5c3f8f",
    "timing_gap": "#1f6fb4",
    "timing_interval": "#3f8fb4",
    "mutual_exclusion": "#4f8f3f",
    "resource_limit": "#9f7f1f",
    "undeclared_command": "#4a4a4a",
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
    height ratios that ``tight_layout`` cannot reconcile and warns about. Each
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
