"""Matplotlib figures, Agg backend only.

Every figure here is produced by a script in ``examples/`` and written into
``screenshots/``. The backend is forced to Agg at import time: this package is
used in headless build environments and must never try to open a window.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .benchmark import BenchmarkResult  # noqa: E402
from .occupancy import CellOccupancy  # noqa: E402
from .report import CampaignReport  # noqa: E402

__all__ = [
    "plot_coverage_curves",
    "plot_occupancy_slices",
    "plot_report_summary",
]

_COLOURS: Mapping[str, str] = {
    "uniform_random": "#767676",
    "greedy_maxmin": "#1f77b4",
    "learned_gapfinder": "#d62728",
    "coverage_greedy_reference": "#2ca02c",
}


def plot_coverage_curves(result: BenchmarkResult, path: str | Path) -> Path:
    """Coverage-per-scenario curves with the across-seed spread drawn, not just the mean.

    Left panel: mean curve per strategy with a shaded band of plus and minus
    one standard deviation across seeds. Right panel: the per-seed coverage
    gain of equation (6), as a strip plot, so that the overlap between
    strategies is visible rather than averaged away.

    Args:
        result: a :class:`BenchmarkResult`.
        path: output PNG path.

    Returns:
        The path written.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    steps = np.arange(result.n_steps + 1)
    for name, curves in result.by_strategy.items():
        colour = _COLOURS.get(name, "#333333")
        mean = curves.mean
        sd = curves.sd
        left.plot(steps, mean, label=name, color=colour, linewidth=1.6)
        left.fill_between(steps, mean - sd, mean + sd, color=colour, alpha=0.18, linewidth=0)
    left.set_xlabel(f"scenarios added beyond the {result.n_start} already executed")
    left.set_ylabel(f"{result.t}-way coverage fraction")
    left.set_title(
        f"mean over {len(result.seeds)} seeds, band = +/- 1 sd",
        fontsize=10,
    )
    left.grid(alpha=0.3)
    left.legend(fontsize=8, loc="lower right")

    names = list(result.by_strategy)
    for i, name in enumerate(names):
        gains = result.by_strategy[name].gain_per_scenario
        jitter = np.linspace(-0.16, 0.16, len(gains))
        right.plot(
            np.full(len(gains), i) + jitter,
            gains,
            "o",
            color=_COLOURS.get(name, "#333333"),
            markersize=4,
            alpha=0.8,
        )
        right.plot([i - 0.28, i + 0.28], [gains.mean()] * 2, "-", color="black", linewidth=1.4)
    right.set_xticks(range(len(names)))
    right.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    right.set_ylabel("coverage gain per scenario, eq. (6)")
    right.set_title("one point per seed; bar is the mean", fontsize=10)
    right.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_occupancy_slices(
    occupancy: CellOccupancy,
    path: str | Path,
    axis_pair: tuple[int, int] = (0, 1),
) -> Path:
    """Heat map of scenario counts over two grid axes, marginalised over the rest.

    Args:
        occupancy: a :class:`CellOccupancy`.
        path: output PNG path.
        axis_pair: which two grid parameters to show.

    Returns:
        The path written.
    """
    i, j = axis_pair
    if i == j or not all(0 <= a < len(occupancy.shape) for a in axis_pair):
        raise ValueError("axis_pair must be two distinct valid grid axes")
    other = tuple(a for a in range(len(occupancy.shape)) if a not in axis_pair)
    counts = occupancy.counts.sum(axis=other) if other else occupancy.counts
    empty = (occupancy.counts == 0).sum(axis=other) if other else (occupancy.counts == 0)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for axis, data, title in (
        (left, counts, "scenarios per (marginalised) cell"),
        (right, empty, "EMPTY cells behind each column"),
    ):
        image = axis.imshow(data.T, origin="lower", cmap="viridis", aspect="auto")
        axis.set_xlabel(f"{occupancy.parameters[i]} bin")
        axis.set_ylabel(f"{occupancy.parameters[j]} bin")
        axis.set_title(title, fontsize=10)
        fig.colorbar(image, ax=axis)
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                axis.text(
                    x, y, str(int(data[x, y])), ha="center", va="center",
                    color="white", fontsize=8,
                )
    fig.suptitle(
        f"grid occupancy {occupancy.occupied}/{occupancy.n_cells} = {occupancy.fraction:.4f}; "
        f"discretisation {occupancy.shape}",
        fontsize=10,
    )
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_report_summary(report: CampaignReport, path: str | Path, top: int = 10) -> Path:
    """Per-parameter-pair coverage bars and the enumerated uncovered tuples.

    Args:
        report: a :class:`CampaignReport`.
        path: output PNG path.
        top: how many of the worst parameter pairs to draw.

    Returns:
        The path written.
    """
    worst = report.tway.worst_combinations(top)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    labels = [" x ".join(c.parameters) for c in worst]
    values = [c.fraction for c in worst]
    positions = np.arange(len(worst))
    left.barh(positions, values, color="#1f77b4")
    left.set_yticks(positions)
    left.set_yticklabels(labels, fontsize=8)
    left.set_xlim(0.0, 1.05)
    left.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    left.set_xlabel(f"{report.strength}-way coverage of that parameter pair")
    left.set_title(
        f"worst {len(worst)} pairs; overall {report.tway.covered}/{report.tway.total} "
        f"= {report.tway.fraction:.4f}",
        fontsize=10,
    )
    left.grid(alpha=0.3, axis="x")
    left.invert_yaxis()

    right.axis("off")
    lines = [f"uncovered {report.strength}-way tuples: {report.tway.n_uncovered}", ""]
    lines += [f"  {u.as_text()}" for u in report.tway.uncovered[:12]]
    if report.classes is not None:
        lines += ["", f"uncovered declared classes: {len(report.classes.uncovered)}"]
        lines += [f"  {c.parameter}/{c.name}" for c in report.classes.uncovered]
    missed = [b for b in report.boundaries if not b.covered]
    lines += ["", f"missed declared boundaries: {len(missed)}"]
    for b in missed:
        lines.append(
            f"  {b.spec.parameter}/{b.spec.name}: nearest {b.nearest_distance:.4g}, "
            f"tol {b.spec.tolerance:g}"
        )
    right.text(
        0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.5,
        transform=right.transAxes,
    )
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
