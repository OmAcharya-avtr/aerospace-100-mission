"""Shared setup for the examples: paths, the Agg backend, and the origin stamp.

Two rules every example here follows.

**The origin is stamped on the figure itself, not only in the caption.**
:func:`load_for_example` prefers the campaign's MEASURED traces and falls back to
the SYNTHETIC generator when they are absent. The fallback is deliberate for an
example -- a reader with a cold clone must be able to run every example and get a
picture -- and it is safe only because the fallback is impossible to miss: the
figure gets a diagonal SYNTHETIC watermark across the axes and the title says so.
A validation script, by contrast, has no fallback at all and refuses.

**Nothing writes outside ``screenshots/``.** :func:`save` puts the PNG there and
prints its path relative to the repository root, so no committed output carries
an absolute path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where every example writes its PNG.
SCREENSHOTS = ROOT / "screenshots"

#: The campaign's measured traces, relative to the repository root.
MEASURED_TRACES = "data/traces"

#: Seeds, matching validation/_bootstrap.py so a figure and a number agree.
SEED = 910091

#: Resamples for intervals drawn on a figure. Lower than the validation scripts'
#: 20000 because a figure is read at a few significant figures and every example
#: must finish well inside the release gate's 600 s budget.
RESAMPLES = 4000

WINDOW_S = 120.0
THRESHOLD_C_PER_S = 0.005

#: Colours. One palette for every figure so the four read as one set.
MEASURED_COLOUR = "#1b3a5c"
SYNTHETIC_COLOUR = "#7a4b2a"
ACCENT = "#b03030"
GRID = "#d8d8d8"


def bootstrap() -> Path:
    """Put ``src`` and ``tests`` on ``sys.path``, chdir to the root, return it."""
    for directory in (ROOT / "src", ROOT / "tests"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    os.chdir(ROOT)
    SCREENSHOTS.mkdir(exist_ok=True)
    return ROOT


def pyplot():
    """``matplotlib.pyplot`` with the Agg backend forced. Never ``show()``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "font.size": 9,
            "axes.grid": True,
            "axes.grid.axis": "both",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.edgecolor": "#444444",
            "axes.titlesize": 10,
            "legend.frameon": False,
            "figure.autolayout": False,
        }
    )
    return plt


def relative(path: str | Path) -> str:
    """Render ``path`` relative to the repository root, never absolutely."""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return candidate.name


def measured_present() -> bool:
    """Whether the campaign's measured traces are on disk."""
    directory = ROOT / MEASURED_TRACES
    return directory.is_dir() and any(directory.glob("*.csv"))


def load_for_example(regime: str = "no_throttle"):
    """Load MEASURED traces if present, else SYNTHETIC ones. Returns the pair.

    Returns
    -------
    (campaign, is_measured)
        ``is_measured`` is what every caller stamps on the figure.
    """
    from thermalbudget.provenance import Origin
    from thermalbudget.trace import load_campaign

    if measured_present():
        return load_campaign(ROOT / MEASURED_TRACES, origin=Origin.MEASURED), True
    import tempfile

    from fixtures import synthetic_trace

    directory = Path(tempfile.mkdtemp(prefix=f"thermalbudget-example-{regime}-"))
    return synthetic_trace.load_synthetic_campaign(
        directory, regime=regime, seed=SEED
    ), False


def stamp(figure, axes_list, is_measured: bool, detail: str = "") -> None:
    """Put the origin on the figure so a reader cannot miss it.

    A synthetic figure gets a diagonal watermark across every axes it is given
    plus a footer line. A measured figure gets the footer naming the power mode
    and the single ambient, because a measured figure that omits those two
    qualifications is overclaimed.
    """
    if is_measured:
        footer = (
            "MEASURED on an NVIDIA Jetson Orin Nano Engineering Reference Developer Kit "
            "Super at power mode MAXN_SUPER, one uninstrumented ambient, one board, the "
            "declared EKF-plus-GEMM payload."
        )
        colour = MEASURED_COLOUR
    else:
        footer = (
            "SYNTHETIC fixture from tests/fixtures/synthetic_trace.py. NOT a measurement "
            "of any board. The measured figures appear here only once "
            f"{MEASURED_TRACES}/ holds the campaign's traces."
        )
        colour = SYNTHETIC_COLOUR
        for axes in axes_list:
            axes.text(
                0.5,
                0.5,
                "SYNTHETIC",
                transform=axes.transAxes,
                fontsize=34,
                color="#000000",
                alpha=0.08,
                ha="center",
                va="center",
                rotation=28,
                zorder=0,
            )
    if detail:
        footer = f"{footer}  {detail}"
    figure.text(0.01, 0.012, footer, fontsize=6.6, color=colour, wrap=True)


def save(figure, name: str) -> str:
    """Write ``figure`` into ``screenshots/`` and return its relative path."""
    SCREENSHOTS.mkdir(exist_ok=True)
    path = SCREENSHOTS / name
    figure.savefig(path, bbox_inches="tight")
    return relative(path)
