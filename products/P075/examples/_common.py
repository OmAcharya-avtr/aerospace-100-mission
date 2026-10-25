"""Shared setup for the example scripts.

Every example anchors its paths on ``__file__`` and changes the working
directory to the repository root, for the same reason the validation scripts
do: the script has to work whatever directory it is invoked from, and the
release gate runs it with its working directory set to ``examples/``. Each
example writes one PNG into ``screenshots/`` and prints the numbers that appear
on it, so the figure in the README and the text a reader can reproduce come
from the same run.

Matplotlib is forced onto the Agg backend before ``pyplot`` is imported.
Nothing here calls ``show()``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS = ROOT / "screenshots"

INK = "#1b1b1b"
GRID = "#d4d4d4"
SERIES = ("#1f4e79", "#c0504d", "#4f6228", "#7f6000", "#5f497a", "#31859b")


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path``, change to the repository root, return it."""
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(ROOT)
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    return ROOT


def relative(path: str | Path) -> str:
    """Render ``path`` relative to the repository root, never absolutely."""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return candidate.name


def style_axes(axes: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply one consistent look to every figure in this directory."""
    axes.set_title(title, color=INK, fontsize=11, loc="left")
    axes.set_xlabel(xlabel, color=INK, fontsize=9)
    axes.set_ylabel(ylabel, color=INK, fontsize=9)
    axes.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(colors=INK, labelsize=8)
    for spine in axes.spines.values():
        spine.set_color(GRID)


def save(figure: plt.Figure, name: str) -> str:
    """Write ``figure`` into ``screenshots/`` and return its relative path."""
    path = SCREENSHOTS / name
    figure.tight_layout()
    figure.savefig(path, dpi=130, facecolor="white")
    plt.close(figure)
    return relative(path)
