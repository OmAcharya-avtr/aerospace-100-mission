"""Expected calibration error, and the bias of the estimator itself.

The expected calibration error of a forecaster is the population quantity

    ECE = E_f |E[o | f] - f|,

and the usual estimator replaces the conditional expectation by a bin average:

    ECEhat(B, n) = sum_k (n_k / n) |fbar_k - obar_k|.

``ECEhat`` is biased upward, for a reason that needs no statistics: ``obar_k``
is a mean of ``n_k`` Bernoulli draws, so it misses ``fbar_k`` by about
``sqrt(fbar_k (1 - fbar_k) / n_k)`` even when the forecaster is perfect, and
the absolute value turns that noise into a positive contribution that cannot
cancel. The bias grows with the bin count and falls with the sample size,
roughly as ``sqrt(B / n)``. On a *perfectly calibrated* forecaster the
population ECE is zero, so every digit the estimator reports is bias.

This module therefore reports three things rather than one:

``expected_calibration_error``
    the usual estimator, so results are comparable with other libraries;

``null_ece_distribution``
    the distribution of the same estimator under the null hypothesis that the
    forecaster is perfectly calibrated, obtained by a parametric bootstrap
    that keeps the forecasts fixed and redraws ``o_i ~ Bernoulli(f_i)``. Under
    that resampling the forecaster is calibrated by construction, so the
    distribution is exactly the estimator's own noise at this sample size,
    bin count and forecast distribution;

``debiased_ece``
    the raw estimate minus the mean of that null distribution, with a
    bootstrap p-value for "this is only binning bias".

**The subtraction is not a free lunch, and the measurement says how it fails.**
The null is built from the forecasts themselves, so it describes a *calibrated*
forecaster with the same forecast distribution. On a forecaster that really is
calibrated that is the right null and the correction works: in
``validation/validate_ece_bias.py`` it improved the error in 9 of 9 calibrated
configurations, by a median factor of about 10. On a *miscalibrated*
forecaster it is the wrong null, and subtracting it removes most of the real
miscalibration along with the bias: the same run made the error worse in 9 of
9 miscalibrated configurations, by up to 58x. It can also push the estimate
below zero, which happened in 100 of 360 individual estimates there, and it is
not clipped, because clipping would hide exactly that.

**So read ``p_value`` before quoting ``debiased``.** A large p-value means the
calibrated null is tenable and the debiased value is the number to report. A
small p-value means it is not, and the raw ECE together with the null interval
``[null_q05, null_q95]`` is what to report: the gap between the raw estimate
and that interval is the part of the ECE that binning bias cannot explain.

References
----------
Naeini, M. P., Cooper, G. F. and Hauskrecht, M. (2015). "Obtaining well
calibrated probabilities using Bayesian binning." *AAAI 2015*, 2901-2907.

Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). "On calibration of
modern neural networks." *ICML 2017*, PMLR 70, 1321-1330. arXiv:1706.04599.

Nixon, J., Dusenberry, M. W., Zhang, L., Jerfel, G. and Tran, D. (2019).
"Measuring calibration in deep learning." *CVPR 2019 Workshops*.
arXiv:1904.01685.

Kumar, A., Liang, P. and Ma, T. (2019). "Verified uncertainty calibration."
*NeurIPS 2019*, 3787-3798. arXiv:1909.10155. (states and analyses the
upward bias of the plug-in estimator)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from .binning import assign_bins
from .scores import check_forecasts

__all__ = [
    "DebiasedECE",
    "ECEBiasCurve",
    "ECEBiasRow",
    "calibration_gaps",
    "debiased_ece",
    "ece_bias_curve",
    "expected_calibration_error",
    "maximum_calibration_error",
    "null_ece_distribution",
]


def calibration_gaps(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    *,
    n_bins: int = 10,
    strategy: str = "equal_width",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(weights, gaps)`` over occupied bins.

    ``weights`` are ``n_k / n`` and sum to 1; ``gaps`` are
    ``fbar_k - obar_k``, signed, dimensionless. Empty bins are dropped, which
    is why the arrays can be shorter than ``n_bins``.
    """
    f, o = check_forecasts(forecasts, outcomes)
    labels, _ = assign_bins(f, n_bins=n_bins, strategy=strategy)
    return _gaps_from_labels(f, o, labels, int(n_bins))


def _gaps_from_labels(
    f: np.ndarray, o: np.ndarray, labels: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-occupied-bin weights and signed gaps, by bin-count accumulation."""
    counts = np.bincount(labels, minlength=n_bins).astype(np.int64)
    sum_f = np.bincount(labels, weights=f, minlength=n_bins)
    sum_o = np.bincount(labels, weights=o, minlength=n_bins)
    occupied = counts > 0
    nk = counts[occupied].astype(float)
    weights = nk / f.size
    gaps = sum_f[occupied] / nk - sum_o[occupied] / nk
    return weights, gaps


def expected_calibration_error(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    *,
    n_bins: int = 10,
    strategy: str = "equal_width",
) -> float:
    """Plug-in ECE, dimensionless, in [0, 1].

    This is the estimator everyone quotes, including ``netcal.metrics.ECE``.
    It is reported here so numbers are comparable, not because it is unbiased;
    see :func:`debiased_ece`.
    """
    weights, gaps = calibration_gaps(
        forecasts, outcomes, n_bins=n_bins, strategy=strategy
    )
    return float(np.sum(weights * np.abs(gaps)))


def maximum_calibration_error(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    *,
    n_bins: int = 10,
    strategy: str = "equal_width",
) -> float:
    """Largest absolute bin gap, dimensionless.

    Taken over occupied bins only and therefore dominated by the emptiest one;
    a bin holding a single sample contributes a gap of 0 or 1 and nothing in
    between.
    """
    _, gaps = calibration_gaps(forecasts, outcomes, n_bins=n_bins, strategy=strategy)
    return float(np.max(np.abs(gaps)))


def null_ece_distribution(
    forecasts: ArrayLike,
    *,
    n_bins: int = 10,
    strategy: str = "equal_mass",
    n_replicates: int = 400,
    seed: int = 0,
) -> np.ndarray:
    """ECE values under the null that the forecaster is perfectly calibrated.

    Keeps ``forecasts`` fixed and draws ``o_i ~ Bernoulli(f_i)`` once per
    replicate, so the forecaster is calibrated by construction and the only
    thing the returned values contain is the estimator's own noise at this
    forecast distribution, sample size, bin count and strategy.

    Returns
    -------
    ndarray of shape (n_replicates,)
        One ECE per replicate, dimensionless.
    """
    f = np.asarray(forecasts, dtype=float)
    f, _ = check_forecasts(f, np.zeros_like(f))
    if n_replicates < 1:
        raise ValueError(f"n_replicates must be at least 1, got {n_replicates!r}")
    rng = np.random.default_rng(int(seed))
    # The forecasts are fixed, so the bin assignment is fixed too: compute it
    # once instead of once per replicate.
    labels, _ = assign_bins(f, n_bins=n_bins, strategy=strategy)
    out = np.empty(int(n_replicates), dtype=float)
    for r in range(int(n_replicates)):
        o = (rng.random(f.size) < f).astype(float)
        weights, gaps = _gaps_from_labels(f, o, labels, int(n_bins))
        out[r] = float(np.sum(weights * np.abs(gaps)))
    return out


@dataclass(frozen=True)
class DebiasedECE:
    """Raw ECE, its null distribution, and the bias-corrected value.

    Attributes
    ----------
    raw
        The plug-in estimate.
    null_mean, null_std
        Mean and standard deviation of the estimator under the calibrated null.
        ``null_mean`` is the estimated binning bias at this configuration.
    null_q05, null_q95
        5th and 95th percentiles of the null distribution.
    debiased
        ``raw - null_mean``. Can be negative; it is not clipped, because
        clipping would hide the one case where the correction overshoots.
    p_value
        ``(1 + #{null >= raw}) / (1 + n_replicates)``, the parametric-bootstrap
        p-value for "the forecaster is perfectly calibrated". Large values mean
        the measured ECE is indistinguishable from binning bias.
    """

    raw: float
    null_mean: float
    null_std: float
    null_q05: float
    null_q95: float
    debiased: float
    p_value: float
    n_samples: int
    n_bins: int
    strategy: str
    n_replicates: int
    null_values: np.ndarray = field(repr=False)

    def report(self) -> str:
        """Multi-line summary; every quantity is dimensionless."""
        return "\n".join(
            [
                f"ECE audit ({self.strategy}, n = {self.n_samples}, "
                f"bins = {self.n_bins}, replicates = {self.n_replicates})",
                f"  raw ECE                   : {self.raw:.9f}",
                f"  null mean (binning bias)  : {self.null_mean:.9f}",
                f"  null std                  : {self.null_std:.9f}",
                f"  null 5-95 pct             : [{self.null_q05:.9f}, {self.null_q95:.9f}]",
                f"  debiased ECE              : {self.debiased:+.9f}",
                f"  p-value (calibrated null) : {self.p_value:.4f}",
            ]
        )


def debiased_ece(
    forecasts: ArrayLike,
    outcomes: ArrayLike,
    *,
    n_bins: int = 10,
    strategy: str = "equal_mass",
    n_replicates: int = 400,
    seed: int = 0,
) -> DebiasedECE:
    """Plug-in ECE with its binning bias estimated and subtracted.

    The bias is estimated by :func:`null_ece_distribution`, which costs
    ``n_replicates`` extra ECE evaluations. The subtraction assumes the
    estimator's noise is the same under the fitted forecast distribution as
    under the true one; that assumption is exact when the forecaster is in
    fact calibrated and approximate otherwise, which is why ``p_value`` is
    returned alongside and why the overshoot is measured rather than claimed
    absent.
    """
    f, o = check_forecasts(forecasts, outcomes)
    raw = expected_calibration_error(f, o, n_bins=n_bins, strategy=strategy)
    null = null_ece_distribution(
        f, n_bins=n_bins, strategy=strategy, n_replicates=n_replicates, seed=seed
    )
    n_ge = int(np.sum(null >= raw))
    return DebiasedECE(
        raw=float(raw),
        null_mean=float(null.mean()),
        null_std=float(null.std(ddof=1)) if null.size > 1 else 0.0,
        null_q05=float(np.quantile(null, 0.05)),
        null_q95=float(np.quantile(null, 0.95)),
        debiased=float(raw - null.mean()),
        p_value=float((1 + n_ge) / (1 + null.size)),
        n_samples=int(f.size),
        n_bins=int(n_bins),
        strategy=str(strategy),
        n_replicates=int(null.size),
        null_values=null,
    )


@dataclass(frozen=True)
class ECEBiasRow:
    """One cell of the bias curve: a (bin count, sample size) pair."""

    n_bins: int
    n_samples: int
    strategy: str
    true_ece: float
    mean_ece: float
    std_ece: float
    sem_ece: float
    bias: float
    mean_debiased: float
    debiased_bias: float
    n_replicates: int


@dataclass(frozen=True)
class ECEBiasCurve:
    """The measured bias of the ECE estimator over a (bins, samples) grid."""

    spec_name: str
    rows: tuple[ECEBiasRow, ...]
    strategy: str

    def table(self) -> str:
        """Fixed-width table, one line per grid cell."""
        head = (
            f"{'bins':>5} {'n':>7} {'true_ECE':>10} {'mean_ECE':>10} {'sem':>9} "
            f"{'bias':>10} {'bias/sqrt(B/n)':>15} {'debiased':>10} {'deb_bias':>10}"
        )
        lines = [head, "-" * len(head)]
        for r in self.rows:
            scale = np.sqrt(r.n_bins / r.n_samples)
            lines.append(
                f"{r.n_bins:>5d} {r.n_samples:>7d} {r.true_ece:>10.6f} "
                f"{r.mean_ece:>10.6f} {r.sem_ece:>9.6f} {r.bias:>10.6f} "
                f"{r.bias / scale:>15.6f} {r.mean_debiased:>10.6f} "
                f"{r.debiased_bias:>+10.6f}"
            )
        return "\n".join(lines)

    def power_law_fit(self) -> tuple[float, float, float]:
        """Least-squares fit of ``log bias = c + m log(B / n)``.

        Returns ``(slope, intercept, r_squared)``. The theoretical slope for
        the leading Bernoulli-noise term is 0.5. Cells with a non-positive
        measured bias are dropped, and the fit needs at least three cells.
        """
        x, y = [], []
        for r in self.rows:
            if r.bias > 0.0:
                x.append(np.log(r.n_bins / r.n_samples))
                y.append(np.log(r.bias))
        if len(x) < 3:
            raise ValueError(
                f"need at least 3 cells with positive bias to fit, got {len(x)}"
            )
        xa, ya = np.asarray(x), np.asarray(y)
        slope, intercept = np.polyfit(xa, ya, 1)
        pred = slope * xa + intercept
        ss_res = float(np.sum((ya - pred) ** 2))
        ss_tot = float(np.sum((ya - ya.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return float(slope), float(intercept), float(r2)


def ece_bias_curve(
    spec,
    *,
    n_bins_grid,
    n_samples_grid,
    strategy: str = "equal_width",
    n_replicates: int = 200,
    seed: int = 0,
    debias_replicates: int = 0,
) -> ECEBiasCurve:
    """Measure the ECE estimator's bias against the population value.

    For each ``(B, n)`` cell, draw ``n_replicates`` independent samples of size
    ``n`` from ``spec``, compute the plug-in ECE with ``B`` bins, and compare
    the mean against the population ECE from
    :func:`calibaudit.synthetic.analytic_truth`. When ``debias_replicates`` is
    positive, also compute the debiased estimate on every replicate, which
    costs ``n_replicates * debias_replicates`` extra ECE evaluations per cell
    and is the dominant cost of this function.

    Parameters
    ----------
    spec
        A :class:`calibaudit.synthetic.ForecastSpec`.
    n_bins_grid, n_samples_grid
        Iterables of bin counts and sample sizes.
    """
    from .synthetic import analytic_truth, sample_forecast

    truth = analytic_truth(spec)
    bins_list = [int(b) for b in n_bins_grid]
    rows: list[ECEBiasRow] = []
    stream = 0
    for n_samples in n_samples_grid:
        # One draw per replicate, evaluated at every bin count, so the bin
        # counts are compared on identical data and the sampling cost does not
        # multiply by the length of the bin grid.
        vals = np.empty((len(bins_list), int(n_replicates)), dtype=float)
        deb = np.full((len(bins_list), int(n_replicates)), np.nan, dtype=float)
        for r in range(int(n_replicates)):
            stream += 1
            s = sample_forecast(spec, int(n_samples), seed=int(seed) * 100003 + stream)
            for j, n_bins in enumerate(bins_list):
                vals[j, r] = expected_calibration_error(
                    s.forecasts, s.outcomes, n_bins=n_bins, strategy=strategy
                )
                if debias_replicates > 0:
                    deb[j, r] = debiased_ece(
                        s.forecasts,
                        s.outcomes,
                        n_bins=n_bins,
                        strategy=strategy,
                        n_replicates=int(debias_replicates),
                        seed=int(seed) * 7919 + stream,
                    ).debiased
        for j, n_bins in enumerate(bins_list):
            cell = vals[j]
            mean_deb = (
                float(np.nanmean(deb[j])) if debias_replicates > 0 else float("nan")
            )
            rows.append(
                ECEBiasRow(
                    n_bins=int(n_bins),
                    n_samples=int(n_samples),
                    strategy=strategy,
                    true_ece=float(truth.ece),
                    mean_ece=float(cell.mean()),
                    std_ece=float(cell.std(ddof=1)) if cell.size > 1 else 0.0,
                    sem_ece=float(cell.std(ddof=1) / np.sqrt(cell.size))
                    if cell.size > 1
                    else 0.0,
                    bias=float(cell.mean() - truth.ece),
                    mean_debiased=mean_deb,
                    debiased_bias=float(mean_deb - truth.ece)
                    if debias_replicates > 0
                    else float("nan"),
                    n_replicates=int(n_replicates),
                )
            )
    return ECEBiasCurve(spec_name=spec.name, rows=tuple(rows), strategy=strategy)
