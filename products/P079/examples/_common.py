"""Path bootstrap and plot style shared by every example.

Each example anchors on ``__file__`` and changes the working directory to the
product root, so it runs from anywhere and writes its PNG to
``screenshots/<name>.png`` relative to that root rather than to the caller's
directory. Matplotlib is forced onto the Agg backend before ``pyplot`` is
imported anywhere, because an example that tries to open a window on a
headless machine fails in a way that looks like a code defect.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

INK = "#1b1b1b"
GRID = "#d8d8d8"
SERIES = ("#1f4e79", "#c0504d", "#4f8a3b", "#8064a2", "#e8a33d")


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path``, change to the product root, return it."""
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(ROOT)
    (ROOT / "screenshots").mkdir(exist_ok=True)
    return ROOT


def style() -> None:
    """A plain, legible style. No colour map tricks, no shadows."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 110,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> str:
    """Save ``fig`` to ``screenshots/<name>.png`` and return the relative path."""
    target = ROOT / "screenshots" / f"{name}.png"
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return str(target.relative_to(ROOT))
