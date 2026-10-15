"""Figures. Agg backend only; nothing in this module calls ``show()``.

Matplotlib is an optional dependency: importing this module without it raises
:class:`ImportError` with the install hint, and the rest of the package does
not import it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .benchmark import MutationPoint, StrategyScore  # noqa: E402

__all__ = ["plot_safety_loss_vs_cost", "plot_fault_detection", "plot_feature_importances"]

_STYLE = {
    "dependency": ("tab:blue", "o", "Baseline 1: import graph"),
    "history": ("tab:orange", "s", "Baseline 2: co-failure history"),
    "learned": ("tab:green", "^", "Learned selector"),
}


def _finish(fig: plt.Figure, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_safety_loss_vs_cost(
    scores: Mapping[str, StrategyScore], out_path: str | Path, title: str = ""
) -> Path:
    """Safety loss (dimensionless) against realised cost (fraction of suite)."""
    fig, (ax_loss, ax_apfd) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for name, score in scores.items():
        colour, marker, label = _STYLE.get(name, ("tab:gray", "x", name))
        xs = [p.tests_run_fraction for p in score.points]
        ys = [p.safety_loss for p in score.points]
        ax_loss.plot(xs, ys, marker=marker, color=colour, label=label)
    ax_loss.set_xlabel("fraction of the suite run")
    ax_loss.set_ylabel("safety loss (fraction of real failures missed)")
    ax_loss.set_title("Safety loss against cost")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(fontsize=8)

    names = list(scores)
    ax_apfd.bar(
        range(len(names)),
        [scores[n].apfd_full_order for n in names],
        color=[_STYLE.get(n, ("tab:gray",))[0] for n in names],
    )
    ax_apfd.set_xticks(range(len(names)))
    ax_apfd.set_xticklabels(names, rotation=15)
    ax_apfd.set_ylabel("APFD (full order)")
    ax_apfd.set_ylim(0.0, 1.0)
    ax_apfd.set_title("APFD, Rothermel et al. 2001")
    ax_apfd.grid(alpha=0.3, axis="y")
    if title:
        fig.suptitle(title, fontsize=10)
    return _finish(fig, out_path)


def plot_fault_detection(
    points: Mapping[str, Sequence[MutationPoint]], out_path: str | Path, title: str = ""
) -> Path:
    """Seeded faults detected against tests run, with the full-suite bound."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    upper = 0.0
    for name, series in points.items():
        colour, marker, label = _STYLE.get(name, ("tab:gray", "x", name))
        xs = [p.tests_run_fraction for p in series]
        ys = [p.faults_detected_all for p in series]
        upper = max(upper, max(p.full_suite_upper_bound for p in series))
        ax.plot(xs, ys, marker=marker, color=colour, label=label)
    ax.axhline(
        upper,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"full suite upper bound = {upper:.4f}",
    )
    ax.set_xlabel("fraction of the suite run")
    ax.set_ylabel("fraction of seeded faults detected")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(title or "Seeded-fault detection against cost")
    return _finish(fig, out_path)


def plot_feature_importances(
    importances: Mapping[str, float], out_path: str | Path, title: str = ""
) -> Path:
    """Horizontal bar chart of the learned selector's feature importances."""
    items = sorted(importances.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.barh([k for k, _ in items], [v for _, v in items], color="tab:green")
    ax.set_xlabel("gradient-boosting feature importance (dimensionless)")
    ax.grid(alpha=0.3, axis="x")
    ax.set_title(title or "Learned selector feature importances")
    return _finish(fig, out_path)
