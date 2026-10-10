"""Matplotlib figures for the audit. Agg backend only; nothing is shown.

Every function returns a ``matplotlib.figure.Figure`` and writes nothing. The
example scripts in ``examples/`` are what save the PNGs under
``screenshots/``, so a figure in the README always came from a committed,
runnable script.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .decomposition import BrierDecomposition  # noqa: E402
from .ece import ECEBiasCurve  # noqa: E402
from .recalibration import SampleSizeSweep  # noqa: E402
from .reliability import ReliabilityCurve  # noqa: E402

__all__ = [
    "plot_decomposition_bars",
    "plot_ece_bias_curve",
    "plot_reliability",
    "plot_sample_size_sweep",
]


def plot_reliability(
    curve: ReliabilityCurve,
    *,
    title: str = "reliability diagram",
    truth_x: np.ndarray | None = None,
    truth_y: np.ndarray | None = None,
) -> plt.Figure:
    """Reliability diagram with its bootstrap band and a bin-count histogram.

    ``truth_x`` / ``truth_y`` optionally draw the exact calibration curve of a
    synthetic spec, which is the thing the band is supposed to cover.
    """
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6.4, 6.8), height_ratios=[3, 1], constrained_layout=True
    )
    ax.plot([0, 1], [0, 1], color="0.4", lw=1.0, ls="--", label="perfect calibration")
    if truth_x is not None and truth_y is not None:
        ax.plot(truth_x, truth_y, color="tab:green", lw=1.4, label="exact E[o | f]")
    if curve.lower is not None and curve.upper is not None:
        ax.fill_between(
            curve.mean_forecast,
            curve.lower,
            curve.upper,
            color="tab:blue",
            alpha=0.25,
            label=f"{curve.level:.0%} pointwise bootstrap band",
        )
    ax.plot(
        curve.mean_forecast,
        curve.observed_frequency,
        "o-",
        color="tab:blue",
        ms=4,
        lw=1.2,
        label="observed frequency",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("observed relative frequency")
    ax.set_title(f"{title}\nn = {curve.n_samples}, {curve.strategy}, {curve.n_bins} bins")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    widths = np.diff(curve.edges)[curve.bin_index]
    ax2.bar(
        curve.mean_forecast,
        curve.counts,
        width=np.maximum(widths * 0.85, 1e-3),
        color="0.6",
        edgecolor="black",
        lw=0.4,
    )
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("mean forecast probability")
    ax2.set_ylabel("samples per bin")
    ax2.grid(alpha=0.3)
    return fig


def plot_ece_bias_curve(
    curve: ECEBiasCurve, *, title: str = "ECE estimator bias"
) -> plt.Figure:
    """Measured ECE against bin count, one line per sample size, plus the
    collapse onto ``sqrt(B / n)``."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
    sizes = sorted({r.n_samples for r in curve.rows})
    cmap = plt.get_cmap("viridis")
    for i, n in enumerate(sizes):
        rows = sorted((r for r in curve.rows if r.n_samples == n), key=lambda r: r.n_bins)
        color = cmap(i / max(len(sizes) - 1, 1))
        bins = [r.n_bins for r in rows]
        mean = [r.mean_ece for r in rows]
        sem = [r.sem_ece for r in rows]
        ax.errorbar(bins, mean, yerr=sem, marker="o", ms=4, lw=1.2, color=color, label=f"n = {n}")
        ax2.plot(
            [np.sqrt(r.n_bins / r.n_samples) for r in rows],
            [r.bias for r in rows],
            "o",
            ms=5,
            color=color,
            label=f"n = {n}",
        )
    true_ece = curve.rows[0].true_ece
    ax.axhline(
        true_ece,
        color="tab:red",
        ls="--",
        lw=1.2,
        label=f"population ECE = {true_ece:.4f}",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of bins B")
    ax.set_ylabel("measured ECE")
    ax.set_title(f"{title}\nspec = {curve.spec_name}, {curve.strategy}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    try:
        slope, intercept, r2 = curve.power_law_fit()
        x = np.linspace(
            min(np.sqrt(r.n_bins / r.n_samples) for r in curve.rows),
            max(np.sqrt(r.n_bins / r.n_samples) for r in curve.rows),
            64,
        )
        ax2.plot(
            x,
            np.exp(intercept) * (x**2) ** slope,
            color="0.3",
            lw=1.2,
            label=f"fit: bias ~ (B/n)^{slope:.3f}, R2 = {r2:.4f}",
        )
    except ValueError:
        pass
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\sqrt{B/n}$")
    ax2.set_ylabel("bias = measured ECE - population ECE")
    ax2.set_title("the bias collapses onto one curve")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    return fig


def plot_decomposition_bars(
    decompositions: dict[str, BrierDecomposition],
    *,
    title: str = "Brier decomposition",
) -> plt.Figure:
    """Grouped bars of the five terms, plus the dropped three-term residual."""
    names = list(decompositions)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
    terms = ["reliability", "resolution", "uncertainty", "within_bin_variance"]
    labels = ["REL", "RES", "UNC", "WBV"]
    width = 0.8 / len(terms)
    x = np.arange(len(names))
    for j, (term, label) in enumerate(zip(terms, labels, strict=True)):
        vals = [getattr(decompositions[n], term) for n in names]
        ax.bar(x + j * width - 0.4 + width / 2, vals, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("score contribution (dimensionless)")
    ax.set_title(f"{title}\nterms of BS = REL - RES + UNC + WBV - 2 WBC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    resid = [decompositions[n].three_term_residual for n in names]
    ident = [abs(decompositions[n].identity_residual) for n in names]
    ax2.bar(x - 0.2, resid, width=0.4, color="tab:orange", label="three-term residual")
    ax2.bar(
        x + 0.2,
        np.maximum(ident, 1e-18),
        width=0.4,
        color="tab:green",
        label="|five-term residual|",
    )
    ax2.set_yscale("symlog", linthresh=1e-17)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("residual (dimensionless, symlog)")
    ax2.set_title("what a three-term report drops,\nand what the five-term identity leaves")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")
    return fig


def plot_sample_size_sweep(
    sweep: SampleSizeSweep, *, title: str = "does recalibration help?"
) -> plt.Figure:
    """Mean held-out Brier change and harm rate against training sample size."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
    methods = [m for m in dict.fromkeys(r.method for r in sweep.rows) if m != "raw"]
    sizes = sorted({r.n_samples for r in sweep.rows})
    lookup = {(r.n_samples, r.method): r for r in sweep.rows}
    for m in methods:
        rows = [lookup[(n, m)] for n in sizes]
        ax.errorbar(
            sizes,
            [r.mean_delta_brier for r in rows],
            yerr=[r.sem_delta_brier for r in rows],
            marker="o",
            ms=4,
            lw=1.2,
            label=m,
        )
        ax2.plot(sizes, [r.harm_rate for r in rows], "o-", ms=4, lw=1.2, label=m)
    ax.axhline(0.0, color="tab:red", ls="--", lw=1.2, label="raw baseline")
    ax.set_xscale("log")
    ax.set_xlabel("total samples (split into train / test)")
    ax.set_ylabel("mean held-out Brier change vs raw")
    ax.set_title(f"{title}\nspec = {sweep.spec_name}, above 0 is worse than doing nothing")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax2.axhline(0.5, color="0.4", ls=":", lw=1.0, label="coin flip")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("total samples")
    ax2.set_ylabel("fraction of replicates made worse")
    ax2.set_title("harm rate")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    return fig
