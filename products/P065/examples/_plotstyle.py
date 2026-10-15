"""Shared matplotlib setup. Agg only; no figure is ever shown."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

#: Colours used across the example figures. Chosen for legibility in print.
GOOD = "#2d6a2d"
BAD = "#a00000"
NEUTRAL = "#4a6a8a"
MUTED = "#999999"

FOOTER = (
    "hazardtrace, research-grade. Completeness of a declared argument; "
    "no safety obligation is discharged."
)


def finish(fig, path, dpi: int = 150):
    """Add the standing footer, tighten and save. Returns ``path``."""
    fig.text(0.5, 0.012, FOOTER, ha="center", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
