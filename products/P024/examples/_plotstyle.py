"""Shared matplotlib setup for the examples (Agg backend, never show())."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SHOTS = ROOT / "screenshots"

COLORS = {
    "bdot": "#1f4e79",
    "cross": "#c0504d",
    "fixed": "#7f7f7f",
    "sized": "#9467bd",
    "powerlaw": "#2e7d32",
    "learned": "#e07b00",
    "analytic": "#000000",
    "sat": "#f4c542",
    "weak": "#c0504d",
    "strong": "#1f4e79",
}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def save(fig, name: str) -> Path:
    """Save ``fig`` into ``screenshots/`` and return the path."""
    SHOTS.mkdir(exist_ok=True)
    path = SHOTS / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
