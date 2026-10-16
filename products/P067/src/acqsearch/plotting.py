"""Matplotlib figures for the examples. Agg backend only; nothing is shown.

Importing this module sets the Agg backend, so it is safe on a headless
machine and it never opens a window.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .acquisition import DwellOptimum  # noqa: E402
from .benchmark import PolicyScore  # noqa: E402
from .coverage import CoverageCurve  # noqa: E402


def plot_coverage_curves(
    curves: dict[str, CoverageCurve],
    path: str | Path,
    closed_form: tuple[float, float] | None = None,
    title: str = "Covered prior mass against dwell index",
) -> Path:
    """Covered prior mass against dwell index, one line per pattern.

    ``closed_form``, when given, is ``(dwell_index, coverage)`` for the spiral
    geometric closed form and is drawn as a single marker.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=130)
    for name, curve in curves.items():
        ax.plot(curve.dwell_index, curve.covered_mass, linewidth=1.4, label=name)
    if closed_form is not None:
        ax.plot(
            [closed_form[0]],
            [closed_form[1]],
            marker="o",
            markersize=7,
            markerfacecolor="none",
            color="black",
            linestyle="none",
            label="spiral closed form",
        )
    ax.set_xlabel("dwell index")
    ax.set_ylabel("covered prior mass")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    if curves or closed_form is not None:
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_dwell_tradeoff(
    optimum: DwellOptimum,
    path: str | Path,
    title: str = "Dwell-time trade",
) -> Path:
    """Expected acquisition time, mean cell detection probability and false alarm rate."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.3), dpi=130)
    left.plot(optimum.dwell_grid_s * 1e3, optimum.expected_time_s, linewidth=1.6)
    left.axvline(optimum.best_dwell_s * 1e3, color="black", linestyle="--", linewidth=1.0)
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("dwell time (ms)")
    left.set_ylabel("expected acquisition time (s)")
    left.set_title(
        f"minimum {optimum.best_expected_time_s:.4f} s at {optimum.best_dwell_s * 1e3:.3f} ms"
    )
    left.grid(alpha=0.3, which="both")
    right.plot(
        optimum.dwell_grid_s * 1e3,
        optimum.detection_probability,
        linewidth=1.6,
        label="mean cell detection probability",
    )
    right.plot(
        optimum.dwell_grid_s * 1e3,
        optimum.false_alarm_probability,
        linewidth=1.6,
        label="per-dwell false alarm probability",
    )
    right.axvline(optimum.best_dwell_s * 1e3, color="black", linestyle="--", linewidth=1.0)
    right.set_xscale("log")
    right.set_xlabel("dwell time (ms)")
    right.set_ylabel("probability")
    right.set_ylim(-0.02, 1.02)
    right.grid(alpha=0.3, which="both")
    right.legend(loc="center left", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_policy_comparison(
    scores: dict[str, PolicyScore],
    path: str | Path,
    title: str = "Held-out policy comparison",
) -> Path:
    """Mean censored acquisition time with standard-error bars, and acquisition rate."""
    ordered = sorted(scores.values(), key=lambda s: s.mean_time_s)
    names = [s.name for s in ordered]
    means = np.array([s.mean_time_s for s in ordered])
    errors = np.array([s.standard_error_s for s in ordered])
    rates = np.array([s.acquisition_rate for s in ordered])
    y = np.arange(len(ordered))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.4), dpi=130)
    left.barh(y, means, xerr=errors, color="#4c72b0", height=0.6, capsize=3)
    left.set_yticks(y)
    left.set_yticklabels(names, fontsize=8)
    left.invert_yaxis()
    left.set_xlabel("mean time to acquire, censored at the deadline (s)")
    left.grid(alpha=0.3, axis="x")
    right.barh(y, rates, color="#55a868", height=0.6)
    right.set_yticks(y)
    right.set_yticklabels([])
    right.set_xlim(0.0, 1.05)
    right.set_xlabel("fraction acquired before the deadline")
    right.grid(alpha=0.3, axis="x")
    fig.suptitle(title)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
