"""Shared plotting setup for the examples. Agg backend, never ``show()``.

Paths are anchored on ``__file__`` and the working directory is changed to the
repository root, so every example runs identically from any directory and every
figure lands in ``screenshots/`` beside the repository root rather than beside
the caller.

One palette across every figure, so the same thing is always the same colour:
missing is grey, corrupt is red, misparsed is amber, undecided is blue, a
refusal is red and a permission is green, the rules are dark blue and the
learned model green.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

MISSING = "#7a7a7a"
CORRUPT = "#b0344f"
MISPARSED = "#d9850b"
UNDECIDED = "#1f6fb4"
CLEAN = "#3f8f4f"
REFUSE = "#b0344f"
PERMIT = "#3f8f4f"
RULES = "#1f4e79"
RULES_LIGHT = "#6f9fc8"
LEARNED = "#3f8f4f"
MAJORITY = "#9a9a9a"
BOUND = "#b0344f"
INDEPENDENT = "#6f3f8f"
GRID = "#d6d6d6"

VERDICT_COLOURS = {
    "missing": MISSING,
    "corrupt": CORRUPT,
    "misparsed": MISPARSED,
    "undecided": UNDECIDED,
    "clean": CLEAN,
}

CHANNEL_COLOURS = {
    "battery_voltage": "#1f6fb4",
    "valve_current": "#d9850b",
    "gyro_rate": "#8f3f8f",
    "tank_pressure": "#3f8f4f",
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


def tolerances() -> dict[str, float]:
    """The declared tolerances every figure in this repository uses."""
    from telemgap.__main__ import DEFAULT_TOLERANCES

    return dict(DEFAULT_TOLERANCES)
