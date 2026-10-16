"""Shared matplotlib setup. Agg only; no figure is ever shown."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

#: Colours used across the example figures, chosen for legibility in print.
NOMINAL = "#4a6a8a"
CONTINGENCY = "#c08a2e"
MARGIN = "#2d6a2d"
OVERRUN = "#a00000"
LINEAR = "#a00000"
RSS = "#2d6a2d"
EXACT = "#6a3d9a"
MUTED = "#999999"

FOOTER = (
    "budgetroll, educational-grade (validation level 1). Arithmetic on a declared "
    "budget; no statement about any real system."
)


def finish(fig, path, dpi: int = 150):
    """Add the standing footer, tighten the layout, save and close.

    Returns the path written.
    """
    fig.text(0.5, 0.012, FOOTER, ha="center", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
