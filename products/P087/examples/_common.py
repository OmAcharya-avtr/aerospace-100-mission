"""Shared plotting setup for the examples. Agg backend, never ``show()``.

Paths are anchored on ``__file__`` and the working directory is changed to the
repository root, so every example runs identically from any directory and every
figure lands in ``screenshots/`` beside the repository root rather than beside
the caller.

One palette across every figure, so the same thing is always the same colour:
the ray trace is dark blue, the plane-parallel model red, the Oriani series
purple, Bennett amber and Saemundsson green. A refusal is red, a degraded
verdict amber and a usable verdict green.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

RAYTRACE = "#1f4e79"
PLANE_PARALLEL = "#b0344f"
ORIANI = "#6f3f8f"
BENNETT = "#d9850b"
SAEMUNDSSON = "#3f8f4f"
ENVELOPE = "#7a7a7a"
TOLERANCE = "#b0344f"
REFUSE = "#b0344f"
DEGRADED = "#d9850b"
USABLE = "#3f8f4f"
GRID = "#d6d6d6"

METHOD_COLOURS = {
    "raytrace": RAYTRACE,
    "plane_parallel": PLANE_PARALLEL,
    "oriani_series": ORIANI,
    "bennett_1982": BENNETT,
    "saemundsson_1986": SAEMUNDSSON,
}

VERDICT_COLOURS = {
    "usable": USABLE,
    "degraded": DEGRADED,
    "refuse": REFUSE,
}

FORMULATION_COLOURS = {
    "iau_explanatory_supplement": "#b0344f",
    "edlen_1966": "#d9850b",
    "birch_downs_1994": "#1f6fb4",
    "ciddor_1996": "#3f8f4f",
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


def published_state() -> object:
    """1010 hPa, 10 C, dry, 550 nm: the conditions equations (15) and (16) assume."""
    from refractpath import AtmosphericState

    return AtmosphericState(
        pressure_hpa=1010.0, temperature_c=10.0, relative_humidity=0.0, wavelength_um=0.55
    )


def tolerances_arcsec() -> tuple[float, ...]:
    """The declared pointing tolerances every figure in this repository uses."""
    return (1.0, 5.0, 10.0, 30.0, 60.0)
