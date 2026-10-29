"""Shared plotting setup for the examples. Agg backend, never ``show()``.

Paths are anchored on ``__file__`` and the working directory is changed to the
repository root, so every example runs identically from any directory and every
figure lands in ``screenshots/`` beside the repository root rather than beside
the caller.

One palette across every figure, so the same thing is always the same colour:
benign is green, silently wrong is red, a loud failure is amber, the rule
baseline is dark blue, the learned model green, the majority baseline grey, and
anything the screen cannot determine is drawn in violet.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

OBJECTS = int(os.environ.get("TLECHECK_VALIDATION_OBJECTS", "420"))
"""Objects per split for every figure. 420 unless the probe shrinks it.

``validation/probe_working_directories.py`` runs every example from five
different working directories to test a one-line path contract, and five
full-size runs of nine figures is not an affordable way to do that. The
committed PNGs are produced at the default.
"""

HISTORY_LENGTH = 12

BENIGN = "#3f8f4f"
SILENT = "#b0344f"
LOUD = "#d9850b"
RULES = "#1f4e79"
RULES_LIGHT = "#6f9fc8"
LEARNED = "#3f8f4f"
MAJORITY = "#9a9a9a"
UNDETERMINABLE = "#6f3f8f"
THRESHOLD = "#b0344f"
GRID = "#d6d6d6"

CLASS_COLOURS = {
    "benign": BENIGN,
    "silently_wrong": SILENT,
    "will_fail_loudly": LOUD,
}

FAMILY_COLOURS = {
    "leo_sun_synchronous": "#1f6fb4",
    "leo_low_inclination": "#d9850b",
    "leo_decaying": "#b0344f",
    "meo_navigation": "#3f8f4f",
    "geo_stationary": "#8f3f8f",
    "heo_critical_inclination": "#6a6a6a",
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
    height ratios, which ``tight_layout`` cannot reconcile and warns about. Each
    figure sets its own spacing and is saved with ``bbox_inches="tight"``.
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
    """Save ``figure`` to ``screenshots/<name>.png`` and return a relative path.

    ``TLECHECK_FIGURE_DIR`` redirects the destination. It exists for the same
    caller and the same reason as ``TLECHECK_OUTPUT_DIR`` in
    ``validation/_bootstrap.py``: the working-directory probe re-runs every
    example at a reduced corpus size and must not overwrite the committed
    figures with small-corpus ones.
    """
    directory = os.environ.get("TLECHECK_FIGURE_DIR")
    target = (Path(directory) if directory else ROOT / "screenshots") / f"{name}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return relative(target)


def splits(objects: int = OBJECTS, history_length: int = HISTORY_LENGTH) -> dict[str, object]:
    """The declared splits, screened, measured and featurised.

    Imported from the validation bootstrap so that an example and a validation
    script can never be looking at different corpora.
    """
    sys.path.insert(0, str(ROOT / "validation"))
    from _bootstrap import prepare_splits

    return prepare_splits(history_length=history_length, objects=objects)
