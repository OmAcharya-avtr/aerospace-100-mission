"""Matplotlib figures for the audit. Agg backend only, never ``show()``.

Every function takes an already-computed result object and a destination path,
so a figure can never disagree with the numbers in ``validation/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .audit import AuditResult, BreakingPointResult  # noqa: E402
from .bounds import split_conformal_coverage_bound  # noqa: E402

METHOD_STYLE: dict[str, tuple[str, str]] = {
    "parametric": ("tab:red", "o"),
    "split": ("tab:blue", "s"),
    "mondrian": ("tab:green", "^"),
    "weighted_declared": ("tab:purple", "D"),
    "weighted_learned": ("tab:orange", "v"),
}
"""Colour and marker per method; the baseline is red so it cannot be mistaken."""

FIGURE_DPI = 130


def _save(figure: plt.Figure, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_coverage_audit(result: AuditResult, path: str | Path) -> Path:
    """Empirical coverage against declared shift severity, one panel per model."""
    models = sorted({row.model for row in result.rows})
    figure, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.6), squeeze=False)
    bound = split_conformal_coverage_bound(result.n_calibration, result.alpha)
    for axis, model in zip(axes[0], models, strict=True):
        for method in result_methods(result):
            rows = result.select(model=model, method=method)
            if not rows:
                continue
            colour, marker = METHOD_STYLE.get(method, ("grey", "x"))
            severity = [r.severity for r in rows]
            coverage = [r.coverage for r in rows]
            low = [r.coverage - r.ci_low for r in rows]
            high = [r.ci_high - r.coverage for r in rows]
            axis.errorbar(
                severity,
                coverage,
                yerr=[low, high],
                color=colour,
                marker=marker,
                capsize=3,
                lw=1.6,
                label=method,
            )
        axis.axhline(1.0 - result.alpha, color="black", ls="--", lw=1.2)
        axis.axhspan(bound.lower, bound.upper, color="black", alpha=0.08)
        axis.text(
            0.02,
            bound.upper + 0.002,
            f"finite-sample band [{bound.lower:.4f}, {bound.upper:.4f}], n={bound.n_calibration}",
            fontsize=7,
            transform=axis.get_yaxis_transform(),
        )
        axis.set_xlabel("declared shift severity [-]")
        axis.set_ylabel(f"empirical coverage [-], nominal {1.0 - result.alpha:.2f}")
        axis.set_title(f"{model} point predictor")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, loc="lower left")
    figure.suptitle(
        f"Coverage audit: {result.replicates} replicates, n_cal={result.n_calibration}, "
        f"n_test={result.n_test}, 95 % replicate-level intervals",
        fontsize=10,
    )
    return _save(figure, path)


def result_methods(result: AuditResult) -> list[str]:
    """Methods present in a result, in :data:`conformalband.audit.METHODS` order."""
    from .audit import METHODS

    present = {row.method for row in result.rows}
    return [m for m in METHODS if m in present]


def plot_interval_width(result: AuditResult, path: str | Path) -> Path:
    """Mean finite interval width against severity, and coverage against width."""
    models = sorted({row.model for row in result.rows})
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for model, linestyle in zip(models, ("-", "--"), strict=False):
        for method in result_methods(result):
            rows = result.select(model=model, method=method)
            if not rows:
                continue
            colour, marker = METHOD_STYLE.get(method, ("grey", "x"))
            axes[0].plot(
                [r.severity for r in rows],
                [r.mean_width for r in rows],
                color=colour,
                marker=marker,
                ls=linestyle,
                lw=1.5,
                label=f"{method} / {model}",
            )
            axes[1].scatter(
                [r.mean_width for r in rows],
                [r.coverage for r in rows],
                color=colour,
                marker=marker,
                s=28,
            )
    axes[0].set_xlabel("declared shift severity [-]")
    axes[0].set_ylabel("mean width of finite intervals [Wh]")
    axes[0].set_title("Interval width is the price of coverage")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncols=2)
    axes[1].axhline(1.0 - result.alpha, color="black", ls="--", lw=1.2)
    axes[1].set_xlabel("mean width of finite intervals [Wh]")
    axes[1].set_ylabel("empirical coverage [-]")
    axes[1].set_title("Coverage against width, all rows")
    axes[1].grid(alpha=0.3)
    return _save(figure, path)


def plot_breaking_point(result: BreakingPointResult, path: str | Path) -> Path:
    """Coverage and effective sample size against the assumed weight fraction."""
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    fractions = np.array([r.fraction for r in result.rows])
    coverage = np.array([r.coverage for r in result.rows])
    low = coverage - np.array([r.ci_low for r in result.rows])
    high = np.array([r.ci_high for r in result.rows]) - coverage
    axis.errorbar(
        fractions, coverage, yerr=[low, high], color="tab:purple", marker="D", capsize=2, lw=1.6
    )
    axis.axhline(result.rows[0].nominal, color="black", ls="--", lw=1.2, label="nominal")
    axis.axvline(1.0, color="tab:green", ls=":", lw=1.4, label="correct weights")
    if result.breaking_fraction is not None:
        axis.axvline(
            result.breaking_fraction,
            color="tab:red",
            ls="-",
            lw=1.6,
            label=f"breaking point f={result.breaking_fraction:.2f}",
        )
    axis.set_xlabel("assumed shift as a fraction of the true shift [-]")
    axis.set_ylabel(f"empirical coverage [-], nominal {result.rows[0].nominal:.2f}")
    axis.grid(alpha=0.3)
    twin = axis.twinx()
    twin.plot(
        fractions,
        [r.ess_fraction for r in result.rows],
        color="tab:grey",
        ls="-.",
        lw=1.3,
        label="ESS fraction",
    )
    twin.set_ylabel("effective sample size / n_cal [-]")
    handles, labels = axis.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    axis.legend(handles + h2, labels + l2, fontsize=8, loc="lower right")
    axis.set_title(
        f"Weighted conformal under misspecified weights: true severity "
        f"{result.true_severity:.1f}, {result.replicates} replicates, {result.model} model"
    )
    return _save(figure, path)


def plot_stratified_coverage(
    tally: dict[str, dict[int, tuple[float, float, float, int]]],
    path: str | Path,
    *,
    title: str = "Conditional coverage by tercile of predicted energy",
) -> Path:
    """Grouped bars of per-stratum coverage and width, split against Mondrian."""
    methods = list(tally)
    bins = sorted({b for per_bin in tally.values() for b in per_bin})
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    width = 0.8 / max(len(methods), 1)
    for index, method in enumerate(methods):
        offset = (index - (len(methods) - 1) / 2) * width
        colour = METHOD_STYLE.get(method, ("grey", "x"))[0]
        axes[0].bar(
            np.array(bins) + offset,
            [tally[method][b][0] for b in bins],
            width=width,
            color=colour,
            label=method,
        )
        axes[1].bar(
            np.array(bins) + offset,
            [tally[method][b][1] for b in bins],
            width=width,
            color=colour,
            label=method,
        )
    nominal = next(iter(next(iter(tally.values())).values()))[2]
    axes[0].axhline(nominal, color="black", ls="--", lw=1.2)
    axes[0].set_ylim(0.7, 1.0)
    axes[0].set_ylabel("empirical coverage in stratum [-]")
    axes[1].set_ylabel("mean interval width in stratum [Wh]")
    for axis in axes:
        axis.set_xticks(bins)
        axis.set_xticklabels([f"tercile {b + 1}" for b in bins])
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize=8)
    figure.suptitle(title, fontsize=10)
    return _save(figure, path)


def plot_coverage_bound(path: str | Path, *, alpha: float = 0.1, max_n: int = 200) -> Path:
    """The finite-sample bound as a staircase in the calibration size."""
    sizes = np.arange(max(9, int(np.ceil(1.0 / alpha)) - 1), max_n + 1)
    exact = np.array([split_conformal_coverage_bound(int(n), alpha).exact for n in sizes])
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.step(sizes, exact, where="post", color="tab:blue", lw=1.6, label="ceil((n+1)(1-a))/(n+1)")
    axis.axhline(1.0 - alpha, color="black", ls="--", lw=1.2, label="1 - alpha")
    axis.plot(
        sizes, 1.0 - alpha + 1.0 / (sizes + 1.0), color="tab:red", ls=":", lw=1.4,
        label="1 - alpha + 1/(n+1)"
    )
    axis.set_xlabel("calibration size n")
    axis.set_ylabel("exact coverage of split conformal [-]")
    axis.set_title(f"Finite-sample coverage is a staircase, alpha = {alpha}")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    return _save(figure, path)


def plot_model_comparison(
    severities: np.ndarray,
    rmse: dict[str, np.ndarray],
    path: str | Path,
    *,
    ylabel: str = "RMSE on the shifted test sample [Wh]",
) -> Path:
    """Point-prediction error of each model against severity."""
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for name, values in rmse.items():
        axis.plot(severities, values, marker="o", lw=1.6, label=name)
    axis.set_xlabel("declared shift severity [-]")
    axis.set_ylabel(ylabel)
    axis.set_title("Analytic baseline against the learned model under shift")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=9)
    return _save(figure, path)
