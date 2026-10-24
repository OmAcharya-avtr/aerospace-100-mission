"""Shared plotting setup for the examples. Agg backend, never ``show()``.

Paths are anchored on ``__file__`` and the working directory is changed to the
repository root, so every example runs identically from any directory and every
figure lands in ``screenshots/`` beside the repository root rather than beside
the caller.

One palette across every figure, so the same thing is always the same colour:
the declared bound is red, ``previous`` grey, ``linear`` blue, ``quadratic``
and ``cubic`` progressively lighter blue-grey, the learned predictors green,
the standard-library baselines orange, and the lossless references black.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

BOUND = "#b0344f"
PREVIOUS = "#7a7a7a"
LINEAR = "#1f6fb4"
QUADRATIC = "#6f9fc8"
CUBIC = "#a8c4dc"
LEARNED_LINEAR = "#4f8f3f"
LEARNED_MLP = "#2f6f2f"
BASELINE = "#d9700b"
LOSSLESS = "#1a1a1a"
GRID = "#d6d6d6"
SHADE = "#f2e6e9"

PREDICTOR_COLOURS = {
    "previous": PREVIOUS,
    "linear": LINEAR,
    "quadratic": QUADRATIC,
    "cubic": CUBIC,
    "learned_linear": LEARNED_LINEAR,
    "learned_mlp": LEARNED_MLP,
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
    width ratios, which ``tight_layout`` cannot reconcile and warns about. Each
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
    """Save ``figure`` to ``screenshots/<name>.png`` and return a relative path."""
    target = ROOT / "screenshots" / f"{name}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return relative(target)


def fit_models() -> dict:
    """Fit both learned predictors on the disjoint training realisations."""
    from boundcompress import dataset
    from boundcompress.learned import KINDS, LearnedPredictor
    from boundcompress.quantise import make_spec, quantise

    arrays = []
    for signal in dataset.training_signals().values():
        span = float(signal.values.max() - signal.values.min())
        for fraction in (1e-3, 1e-4):
            spec = make_spec(signal.values, "abs", fraction * span)
            symbols, _, _ = quantise(signal.values, spec)
            arrays.append(symbols)
    return {kind: LearnedPredictor(kind=kind, members=3).fit(arrays) for kind in KINDS}
