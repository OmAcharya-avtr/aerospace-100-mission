"""Matplotlib figures. Agg backend only; no interactive display anywhere.

Every figure in ``screenshots/`` is produced by a script in ``examples/`` or
``validation/`` that calls one of these functions, so a figure cannot drift away
from the code that made it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)
import numpy as np  # noqa: E402

__all__ = [
    "plot_detector_traces",
    "plot_operating_point_spread",
    "plot_stream_panel",
    "plot_tradeoff",
    "plot_transient_response",
]

_MARKERS = ("o", "s", "^", "D", "v", "P")


def _save(fig, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_tradeoff(curves: dict[str, list[tuple[float, float, float]]], path, title: str,
                  target_arl0: float | None = None):
    """Delay-versus-false-alarm trade-off curve. The headline figure.

    Parameters
    ----------
    curves:
        ``label -> [(arl0, arl1, arl1_sem), ...]`` with points in increasing
        ARL0 order.
    target_arl0:
        If given, a vertical line marking the operating point at which the
        detectors are compared in the tables.
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for i, (label, pts) in enumerate(curves.items()):
        if not pts:
            continue
        arr = np.asarray(pts, dtype=float)
        order = np.argsort(arr[:, 0])
        arr = arr[order]
        ax.errorbar(
            arr[:, 0],
            arr[:, 1],
            yerr=arr[:, 2],
            marker=_MARKERS[i % len(_MARKERS)],
            markersize=5,
            capsize=2.5,
            linewidth=1.4,
            label=label,
        )
    if target_arl0 is not None:
        ax.axvline(target_arl0, color="0.4", linestyle=":", linewidth=1.2)
        ax.text(target_arl0, ax.get_ylim()[1], f" compared at ARL0={target_arl0:.0f}",
                color="0.3", fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("ARL0: mean samples to false alarm on a stationary stream (log)")
    ax.set_ylabel("ARL1: mean detection delay, samples")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    fig.text(0.01, -0.03, "Error bars are Monte Carlo standard errors of the mean delay. "
             "Lower and further right is better.", fontsize=7.5, color="0.35")
    return _save(fig, path)


def plot_operating_point_spread(labels, arl0s, sems, defaults_text, path, target: float):
    """Measured ARL0 at each detector's own default threshold, log scale."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, arl0s, yerr=sems, capsize=3, color="#5b7fa6", edgecolor="0.2", linewidth=0.6)
    ax.axhline(target, color="#b4452f", linestyle="--", linewidth=1.3,
               label=f"common operating point used for comparison (ARL0={target:.0f})")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("measured ARL0 at default threshold, samples (log)")
    ax.set_title("Default thresholds do not land on one false-alarm rate", fontsize=11)
    for xi, (a, txt) in enumerate(zip(arl0s, defaults_text, strict=True)):
        ax.text(xi, a * 1.12, txt, ha="center", fontsize=7.5, color="0.25")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, path)


def plot_stream_panel(streams: dict[str, tuple[np.ndarray, int]], path):
    """The four change types plus the transient negative control."""
    n = len(streams)
    fig, axes = plt.subplots(n, 1, figsize=(7.4, 1.55 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (label, (stream, idx)) in zip(axes, streams.items(), strict=True):
        ax.plot(stream, linewidth=0.55, color="#31506e")
        ax.axvline(idx, color="#b4452f", linestyle="--", linewidth=1.1)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("sample index (dashed line: declared change index)")
    axes[0].set_title("Declared change types, seeded and regenerable", fontsize=11)
    return _save(fig, path)


def plot_detector_traces(stream, change_index, ratios: dict[str, np.ndarray],
                         path, title: str, alarms: dict[str, int] | None = None):
    """Channel above, every detector's alarm ratio below, one shared alarm line.

    Parameters
    ----------
    ratios:
        ``label -> alarm-ratio series`` from
        :func:`telemdrift.detectors.alarm_ratio_trace`. An alarm is raised when
        a series exceeds 1.0, so five incomparable statistics share one axis
        without any rescaling chosen by hand.
    alarms:
        Optional ``label -> first alarm index`` (``-1`` for no alarm), marked
        with a vertical tick on the lower panel.
    """
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(7.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
    )
    ax0.plot(stream, linewidth=0.55, color="#31506e")
    ax0.axvline(change_index, color="#b4452f", linestyle="--", linewidth=1.1)
    ax0.set_ylabel("channel", fontsize=8)
    ax0.grid(alpha=0.25)
    ax0.tick_params(labelsize=7)
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    colours = ["#31506e", "#b4452f", "#4f7a4a", "#7a4f8a", "#9c6b4f", "#2f7f8f"]
    for i, (label, series) in enumerate(ratios.items()):
        ax1.plot(np.asarray(series, dtype=float), linewidth=1.1, label=label,
                 linestyle=styles[i % len(styles)], color=colours[i % len(colours)],
                 alpha=0.92)
        if alarms and alarms.get(label, -1) >= 0:
            ax1.plot([alarms[label]], [1.0], marker="v", markersize=7,
                     color=colours[i % len(colours)])
    ax1.axhline(1.0, color="0.15", linewidth=1.4)
    ax1.text(0, 1.03, "alarm line (ratio = 1)", fontsize=7.5, color="0.15")
    ax1.axvline(change_index, color="#b4452f", linestyle="--", linewidth=1.1)
    ax1.set_ylabel("detector statistic / its own threshold", fontsize=8)
    ax1.set_xlabel("sample index (dashed line: declared change index)")
    ax1.set_ylim(0, 2.4)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=7.5, ncol=3, loc="upper left")
    ax1.tick_params(labelsize=7)
    ax0.set_title(title, fontsize=11)
    return _save(fig, path)


def plot_transient_response(labels, fire_rates, cis, path, title: str):
    """False-alarm rate on the transient negative control, with binomial CIs."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    y = np.arange(len(labels))
    lo = [max(0.0, r - c[0]) for r, c in zip(fire_rates, cis, strict=True)]
    hi = [max(0.0, c[1] - r) for r, c in zip(fire_rates, cis, strict=True)]
    ax.barh(y, fire_rates, xerr=[lo, hi], capsize=3, color="#9c6b4f",
            edgecolor="0.2", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("fraction of transient-spike streams that raised an alarm")
    ax.set_xlim(0, 1.02)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3, axis="x")
    for yi, r in zip(y, fire_rates, strict=True):
        ax.text(min(r + 0.02, 0.95), yi, f"{100 * r:.1f} %", va="center", fontsize=8)
    fig.text(0.01, -0.04, "Every alarm here is a false alarm: the channel recovers. "
             "Error bars are Wilson 95 % intervals.", fontsize=7.5, color="0.35")
    return _save(fig, path)
