"""Shared setup for the example scripts: path, backend, style and output path.

Each example sets ``sys.dont_write_bytecode = True`` as its own first statement
before importing this module, because CPython writes a module's bytecode cache
BEFORE executing it and this module cannot set the flag on its own behalf.

The matplotlib backend is Agg and ``show`` is never called: these scripts run
unattended in continuous integration and on the release gate, where there is no
display. Every figure is written into ``screenshots/`` with a path derived from
``__file__``, so an example works from any working directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

#: Declared figure colours. Chosen for legibility in print and when converted
#: to greyscale; no colour carries meaning on its own in any figure here, and
#: every series is also distinguished by marker or line style.
ANALYTIC = "#1f4e79"
LEARNED = "#b7472a"
REFUSED = "#7a7a7a"
ACCENT = "#2e7d32"
WARN = "#c28400"


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path``, change to the repository root, return it."""
    source = ROOT / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    os.chdir(ROOT)
    return ROOT


def relative(path: str | Path) -> str:
    """Render ``path`` relative to the repository root, never absolutely."""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return candidate.name


def save(figure, name: str) -> str:
    """Write ``figure`` into ``screenshots/`` and return its relative path."""
    directory = ROOT / "screenshots"
    directory.mkdir(exist_ok=True)
    target = directory / name
    figure.savefig(target, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return relative(target)


def style() -> None:
    """Declared plot style. No grid-heavy defaults, no coloured backgrounds."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "legend.frameon": False,
        }
    )
