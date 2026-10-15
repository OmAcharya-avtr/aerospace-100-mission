"""Shared plotting setup for the examples. Agg backend, never show().

A small colour-blind-safe palette is defined here so that every figure in the
repository uses the same colours for the same things: the greedy baseline is
always grey, the exact optimum always black, the learned component always blue,
and the robust variant always orange.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

GREEDY = "#8c8c8c"
EXACT = "#1a1a1a"
LEARNED = "#1f6fb4"
ROBUST = "#d9700b"
ANALYTIC = "#6a9f58"
BATTERY = "#b0344f"
RECORDER = "#1f6fb4"
SOLAR = "#d9a40b"
LOAD = "#5a5a5a"
GRID = "#d6d6d6"


def add_src_to_path() -> Path:
    """Insert ``<repo>/src`` at the front of ``sys.path`` and return the repo root."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def screenshots_dir() -> Path:
    """``<repo>/screenshots``, created if missing."""
    path = add_src_to_path() / "screenshots"
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


def save(fig, name: str) -> Path:
    """Save ``fig`` into ``screenshots/<name>`` and return the path."""
    target = screenshots_dir() / name
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target
