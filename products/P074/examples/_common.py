"""Path bootstrap and plot style shared by every example.

Each example anchors on ``__file__`` and changes the working directory to the
product root, so it runs from anywhere and writes its PNG to
``screenshots/<name>.png`` relative to that root rather than to the caller's
directory. Matplotlib is forced onto the Agg backend before ``pyplot`` is
imported anywhere, because an example that tries to open a window on a
headless machine fails in a way that looks like a code defect.

Examples load the committed ``fixtures/divergence_dataset.npz`` rather than
regenerating it; regeneration is ``validation/regenerate_dataset.py`` and takes
about three minutes on two cores.
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
SERIES = ("#1f4e79", "#c0504d", "#4f8a3b", "#8064a2", "#e8a33d", "#3f8f8f")


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path``, change to the product root, return it."""
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.chdir(ROOT)
    (ROOT / "screenshots").mkdir(exist_ok=True)
    return ROOT


FAST_ENV = "ORBITDRIFT_FAST"
FAST_OBJECT_LIMIT = 24


def is_fast() -> bool:
    """``True`` when the example is running as an exit-status probe.

    ``validation/validate_cwd_independence.py`` runs every example from four
    working directories to check that none of them depends on the caller's
    directory. Those runs only check the exit status, so the examples shrink
    their input to keep the whole matrix inside the per-script budget. A probe
    run still writes its PNG -- from fewer objects -- and the committed
    screenshots are regenerated from the full dataset before hand-off.
    """
    return os.environ.get(FAST_ENV, "") not in ("", "0", "false", "False")


def load_dataset():
    """Load the committed divergence table, reduced in probe mode."""
    import numpy as np

    from orbitdrift import DivergenceTable

    target = ROOT / "fixtures" / "divergence_dataset.npz"
    if not target.exists():
        raise SystemExit(
            "fixtures/divergence_dataset.npz is missing from this clone. "
            "Regenerate it with python validation/regenerate_dataset.py"
        )
    table = DivergenceTable.load(target)
    if not is_fast():
        return table
    keep = np.flatnonzero(table.column("object_index") < FAST_OBJECT_LIMIT)
    return DivergenceTable(
        values=table.values[keep],
        columns=table.columns,
        tle_text=table.tle_text,
        detail=table.detail + f" [PROBE SUBSET: first {FAST_OBJECT_LIMIT} objects]",
    )


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


def save(fig, name: str) -> str:
    """Save ``fig`` to ``screenshots/<name>.png`` and return the relative path."""
    target = ROOT / "screenshots" / f"{name}.png"
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return str(target.relative_to(ROOT))
