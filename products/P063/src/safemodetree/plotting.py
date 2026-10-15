"""Plot helpers. Agg backend only; nothing here calls ``show()``.

Every function takes already-computed results and returns a Matplotlib figure.
No function fits a model or runs a benchmark, so a plot can never disagree with
the number it is plotting.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .calibration import CalibrationResult  # noqa: E402
from .evaluate import DetectorScore, PolicyScore  # noqa: E402
from .reach import ReachabilityReport  # noqa: E402
from .spec import SafeModeSpec  # noqa: E402

__all__ = [
    "plot_ambiguity_costs",
    "plot_conservatism_sweep",
    "plot_detector_sweep",
    "plot_reachability",
    "plot_reliability",
]


def plot_reliability(
    results: dict[str, CalibrationResult], title: str = "Reliability"
) -> plt.Figure:
    """Reliability curves for several named calibration results.

    Args:
        results: name -> :class:`CalibrationResult`.
        title: axes title.
    """
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    axes[0].plot([0, 1], [0, 1], color="0.4", linestyle="--", linewidth=1.0, label="ideal")
    for name, result in results.items():
        mask = result.bin_counts > 0
        axes[0].plot(
            result.bin_confidence[mask],
            result.bin_accuracy[mask],
            marker="o",
            linewidth=1.4,
            label=f"{name} (ECE {result.ece:.4f})",
        )
        centres = 0.5 * (result.bin_edges[:-1] + result.bin_edges[1:])
        axes[1].plot(centres[mask], result.bin_counts[mask], marker="s", linewidth=1.2, label=name)
    axes[0].set_xlabel("mean confidence in bin")
    axes[0].set_ylabel("fraction correct in bin")
    axes[0].set_title(title)
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("confidence bin centre")
    axes[1].set_ylabel("predictions in bin")
    axes[1].set_yscale("log")
    axes[1].set_title("bin occupancy (read the curve with this)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    return figure


def plot_detector_sweep(
    noise_scales: tuple[float, ...],
    baseline: tuple[DetectorScore, ...],
    learned: tuple[DetectorScore, ...],
    ceiling: float,
) -> plt.Figure:
    """Identification accuracy, false-alarm rate and ECE against noise scale."""
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    scales = np.asarray(noise_scales, dtype=float)
    axes[0].axhline(ceiling, color="0.4", linestyle="--", linewidth=1.0, label="declared ceiling")
    axes[0].plot(
        scales,
        [s.identification_accuracy for s in baseline],
        marker="o",
        label="threshold logic",
    )
    axes[0].plot(
        scales,
        [s.identification_accuracy for s in learned],
        marker="s",
        label="learned, matched FAR",
    )
    axes[0].set_ylabel("identification accuracy")
    axes[0].set_title("at an equal false-alarm rate")
    axes[1].plot(
        scales, [s.false_alarm_rate for s in baseline], marker="o", label="threshold logic"
    )
    axes[1].plot(scales, [s.false_alarm_rate for s in learned], marker="s", label="learned")
    axes[1].set_ylabel("false-alarm rate on the test split")
    axes[1].set_title("the match is not exact; this is the residual")
    axes[2].plot(scales, [s.calibration.ece for s in baseline], marker="o", label="threshold logic")
    axes[2].plot(scales, [s.calibration.ece for s in learned], marker="s", label="learned")
    axes[2].set_ylabel("expected calibration error")
    axes[2].set_title("calibration of the confidence")
    for axis in axes:
        axis.set_xlabel("declared noise scale (dimensionless)")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def plot_conservatism_sweep(rows: tuple[PolicyScore, ...]) -> plt.Figure:
    """Mean resolution cost and mis-route count against the routing threshold."""
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    thresholds = np.array([r.confidence_threshold for r in rows], dtype=float)
    costs = np.array([r.mean_cost_min for r in rows], dtype=float)
    misroutes = np.array([r.n_misrouted for r in rows], dtype=float)
    axes[0].plot(thresholds, costs, marker="o", color="tab:red")
    axes[0].set_xlabel("confidence routing threshold (dimensionless)")
    axes[0].set_ylabel("mean resolution cost [min]")
    axes[0].set_title("what conservatism costs")
    twin = axes[0].twinx()
    twin.plot(thresholds, misroutes, marker="s", color="tab:blue", linestyle="--")
    twin.set_ylabel("mis-routed events", color="tab:blue")
    axes[1].plot(misroutes, costs, marker="o", color="0.2")
    for threshold, misroute, cost in zip(thresholds, misroutes, costs, strict=True):
        axes[1].annotate(f"{threshold:.1f}", (misroute, cost), fontsize=7, xytext=(3, 3),
                         textcoords="offset points")
    axes[1].set_xlabel("mis-routed events")
    axes[1].set_ylabel("mean resolution cost [min]")
    axes[1].set_title("the trade, labelled by threshold")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def plot_reachability(report: ReachabilityReport, spec: SafeModeSpec) -> plt.Figure:
    """Cheapest cost to a declared safe mode for every fault state."""
    rows = report.rows
    labels = [f"{r.state.mode}/{r.state.fault}" for r in rows]
    costs = [r.cost_min if math.isfinite(r.cost_min) else 0.0 for r in rows]
    colours = ["tab:blue" if r.reachable else "tab:red" for r in rows]
    figure, axis = plt.subplots(figsize=(8.5, max(3.2, 0.3 * len(rows) + 1.4)))
    positions = np.arange(len(rows))
    axis.barh(positions, costs, color=colours)
    for position, row in zip(positions, rows, strict=True):
        if not row.reachable:
            axis.text(0.4, position, "NO SAFE MODE REACHABLE", va="center", fontsize=8,
                      color="white")
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=7)
    axis.invert_yaxis()
    axis.set_xlabel("cheapest cost to a declared safe mode [min]")
    axis.set_title(
        f"{spec.name}: {report.assumption.describe()}\n"
        f"{len(report.unreachable_rows)} of {len(rows)} fault states cannot reach one"
    )
    axis.grid(alpha=0.3, axis="x")
    figure.tight_layout()
    return figure


def plot_ambiguity_costs(classes: tuple, title: str = "Ambiguity classes") -> plt.Figure:
    """Conservative against fault-specific cost for each ambiguity class."""
    figure, axis = plt.subplots(figsize=(8.8, 4.6))
    labels: list[str] = []
    conservative: list[float] = []
    specific_low: list[float] = []
    specific_high: list[float] = []
    for item in classes:
        finite = [c for c in item.best_specific_cost_min.values() if math.isfinite(c)]
        if not finite or not math.isfinite(item.conservative_cost_min):
            continue
        labels.append(f"{'+'.join(f[:11] for f in item.faults)}\n{item.mode}")
        conservative.append(item.conservative_cost_min)
        specific_low.append(min(finite))
        specific_high.append(max(finite))
    positions = np.arange(len(labels))
    axis.bar(positions - 0.18, specific_low, width=0.34, label="cheapest fault-specific")
    axis.bar(positions + 0.18, conservative, width=0.34, label="class-conservative")
    axis.plot(positions - 0.18, specific_high, "k_", markersize=14,
              label="dearest fault-specific")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=7)
    axis.set_ylabel("resolution cost [min]")
    axis.set_title(title)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    return figure
