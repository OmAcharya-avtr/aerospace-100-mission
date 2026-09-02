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
    "chi2_short": "#1f4e79",
    "chi2_long": "#4f81bd",
    "cusum": "#c0504d",
    "glr": "#2e7d32",
    "learned": "#e07b00",
    "analytic": "#000000",
    "onset": "#7f7f7f",
    "threshold": "#b03060",
    "channel0": "#1f4e79",
    "channel1": "#c0504d",
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
