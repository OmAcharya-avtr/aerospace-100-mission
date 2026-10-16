"""Shared matplotlib setup. Agg only; no figure is ever shown."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

#: Colours used across the example figures, chosen for legibility in print.
MET = "#2d6a2d"
MISSED = "#a00000"
NEUTRAL = "#2b3a55"
ACCENT = "#c08a00"
MUTED = "#8d99ae"

FOOTER = (
    "watchdogspec, research-grade. Arithmetic over a declared specification; "
    "no running system is measured."
)


def finish(fig, path, dpi: int = 150):
    """Add the standing footer, tighten and save. Returns ``path``."""
    fig.text(0.5, 0.012, FOOTER, ha="center", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
