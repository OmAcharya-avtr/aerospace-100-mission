"""Uncertainty budget for timing measurements.

Every latency this package reports carries a combined standard uncertainty.
Three contributions are accounted, following the *Guide to the Expression of
Uncertainty in Measurement* (JCGM 100:2008, the GUM, which is ISO/IEC Guide
98-3):

**Type A --- repeatability.** The standard uncertainty of the mean of ``n``
repeats is ``s / sqrt(n)`` where ``s`` is the experimental standard deviation
of the repeats (GUM §4.2.3). For a quantile rather than a mean there is no
closed form that survives a heavy right tail, so a quantile's uncertainty is
taken from a nonparametric bootstrap (Efron & Tibshirani 1993, *An
Introduction to the Bootstrap*, Chapman & Hall, §6 and §13) --- the standard
deviation of the quantile over resamples.

**Type B --- clock quantisation.** A timer of resolution ``d`` quantises each
of the two reads bounding an interval. Treating each read's error as uniform
on an interval of full width ``d`` gives standard uncertainty ``d /
sqrt(12)`` (GUM §4.3.7, rectangular distribution of half-width ``a`` has
standard uncertainty ``a / sqrt(3)``; with ``a = d/2`` this is ``d /
sqrt(12)``). Two independent reads combine in quadrature, so the interval's
clock uncertainty is ``d * sqrt(2/12) = d / sqrt(6)``.

**Type B --- timer call overhead.** The two ``perf_counter`` calls themselves
take time, which biases a short measurement upward. :func:`timer_overhead_s`
measures that overhead empirically; it is reported as a *bias*, not folded
into the standard uncertainty, because a known bias should be corrected or
stated rather than inflated into noise.

Combination is by root-sum-square of independent contributions (GUM §5.1.2
with unit sensitivity coefficients, since latency is measured directly rather
than computed from other quantities).

What is **not** covered
-----------------------
The dominant error on a shared host is not any of the above: it is the
scheduler. A preempted repeat produces a latency with no relation to the
code under test. This shows up as a heavy right tail, which is why the tail
quantiles are reported separately and why an uncertainty figure here must not
be read as a bound on the spread of the distribution --- it is the
uncertainty of the *statistic*, not the width of the distribution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

__all__ = [
    "UncertaintyBudget",
    "bootstrap_quantile_uncertainty",
    "clock_quantisation_uncertainty_s",
    "combine_standard_uncertainties",
    "mean_uncertainty",
    "timer_overhead_s",
    "uncertainty_budget",
]


def clock_quantisation_uncertainty_s(resolution_s: float, reads: int = 2) -> float:
    """Standard uncertainty from timer quantisation [s].

    ``resolution_s / sqrt(12)`` per read (GUM §4.3.7, rectangular
    distribution of full width ``resolution_s``), ``reads`` independent reads
    combined in quadrature.

    Parameters
    ----------
    resolution_s
        Timer resolution [s], non-negative.
    reads
        Number of clock reads bounding the interval, >= 1. Two for a
        start/stop pair.
    """
    if resolution_s < 0:
        raise ValueError(f"resolution_s must be non-negative, got {resolution_s}")
    if reads < 1:
        raise ValueError(f"reads must be >= 1, got {reads}")
    return float(resolution_s / np.sqrt(12.0) * np.sqrt(reads))


def mean_uncertainty(samples: np.ndarray) -> float:
    """Type A standard uncertainty of the mean [same unit as ``samples``].

    ``s / sqrt(n)`` with ``s`` the sample standard deviation using ``n - 1``
    in the denominator (GUM §4.2.2-4.2.3). Returns ``nan`` for ``n < 2``.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def bootstrap_quantile_uncertainty(
    samples: np.ndarray,
    quantile: float,
    n_resamples: int = 400,
    seed: int = 0,
) -> float:
    """Bootstrap standard uncertainty of a sample quantile.

    Nonparametric bootstrap (Efron & Tibshirani 1993 §6): resample with
    replacement ``n_resamples`` times and take the standard deviation of the
    quantile across resamples.

    Parameters
    ----------
    samples
        Observations, length >= 2.
    quantile
        In (0, 1).
    n_resamples
        >= 20. 400 is used throughout this package; it costs about 2 ms for
        n = 400 samples and is stated in every report alongside the figure.
    seed
        Seed for the resampling generator, so the uncertainty of a pinned
        regression output is itself reproducible.

    Returns
    -------
    float
        Standard uncertainty in the unit of ``samples``; ``nan`` for n < 2.
    """
    arr = np.asarray(samples, dtype=float)
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must lie in (0, 1), got {quantile}")
    if n_resamples < 20:
        raise ValueError(f"n_resamples must be >= 20, got {n_resamples}")
    if arr.size < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    draws = np.quantile(arr[idx], quantile, axis=1)
    return float(np.std(draws, ddof=1))


def combine_standard_uncertainties(*components: float) -> float:
    """Root-sum-square of independent standard uncertainties (GUM §5.1.2)."""
    arr = np.asarray([c for c in components if c is not None], dtype=float)
    if arr.size == 0:
        return 0.0
    if np.any(arr < 0):
        raise ValueError("standard uncertainties must be non-negative")
    return float(np.sqrt(np.sum(arr**2)))


def timer_overhead_s(n: int = 2000) -> float:
    """Median cost of one back-to-back ``perf_counter`` pair [s].

    Measured, not assumed. Reported as a bias on short intervals rather than
    folded into the standard uncertainty.

    Parameters
    ----------
    n
        Number of pairs to time, >= 10. 2000 pairs cost well under 10 ms.
    """
    if n < 10:
        raise ValueError(f"n must be >= 10, got {n}")
    deltas = np.empty(n, dtype=float)
    counter = time.perf_counter
    for i in range(n):
        t0 = counter()
        t1 = counter()
        deltas[i] = t1 - t0
    return float(np.median(deltas))


@dataclass(frozen=True)
class UncertaintyBudget:
    """The full uncertainty budget for one timing statistic.

    Attributes
    ----------
    statistic
        Which statistic this budget applies to, e.g. ``"mean"`` or ``"p99"``.
    value
        The statistic's value [s].
    type_a_s
        Repeatability contribution [s]: standard error of the mean, or the
        bootstrap standard deviation for a quantile.
    clock_s
        Clock-quantisation contribution [s].
    combined_s
        Root-sum-square of the above [s].
    timer_bias_s
        Measured timer-call overhead [s], a bias on the value rather than an
        uncertainty.
    n_samples
        Repeat count behind the statistic.
    method
        Verbatim measurement-method string for the report.
    """

    statistic: str
    value: float
    type_a_s: float
    clock_s: float
    combined_s: float
    timer_bias_s: float
    n_samples: int
    method: str

    def summary_lines(self) -> list[str]:
        """Human-readable budget, every number with its unit."""
        return [
            f"statistic               : {self.statistic}",
            f"value                   : {self.value * 1e6:.3f} us",
            f"u_A (repeatability)     : {self.type_a_s * 1e6:.3f} us",
            f"u_B (clock quantisation): {self.clock_s * 1e9:.3f} ns",
            f"u_c (combined, RSS)     : {self.combined_s * 1e6:.3f} us",
            f"timer-call bias         : {self.timer_bias_s * 1e9:.1f} ns (stated, not corrected)",
            f"n                       : {self.n_samples}",
            f"method                  : {self.method}",
        ]


def uncertainty_budget(
    samples: np.ndarray,
    statistic: str,
    resolution_s: float,
    *,
    quantile: float | None = None,
    n_resamples: int = 400,
    seed: int = 0,
    timer_bias_s: float = 0.0,
    method: str = "unspecified",
) -> UncertaintyBudget:
    """Build the uncertainty budget for one statistic of a sample.

    Parameters
    ----------
    samples
        Timing observations [s].
    statistic
        ``"mean"``, or a quantile label such as ``"p99"``; when ``quantile``
        is given the statistic is that quantile.
    resolution_s
        Timer resolution [s].
    quantile
        If given, the statistic is this quantile and its Type A uncertainty is
        bootstrapped. If ``None``, the statistic is the mean.
    n_resamples, seed
        Bootstrap settings; recorded in ``method``.
    timer_bias_s
        Measured timer-call overhead [s].
    method
        Measurement-method string, extended with the repeat count.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        raise ValueError("samples must be non-empty")
    if quantile is None:
        value = float(np.mean(arr))
        type_a = mean_uncertainty(arr)
        detail = f"{method}; n={arr.size}; u_A = s/sqrt(n) (GUM 4.2.3)"
    else:
        value = float(np.quantile(arr, quantile))
        type_a = bootstrap_quantile_uncertainty(arr, quantile, n_resamples, seed)
        detail = (
            f"{method}; n={arr.size}; u_A = bootstrap sd of p{quantile * 100:g} over "
            f"{n_resamples} resamples, seed={seed} (Efron & Tibshirani 1993)"
        )
    clock = clock_quantisation_uncertainty_s(resolution_s)
    nan_safe_a = 0.0 if np.isnan(type_a) else type_a
    return UncertaintyBudget(
        statistic=statistic,
        value=value,
        type_a_s=type_a,
        clock_s=clock,
        combined_s=combine_standard_uncertainties(nan_safe_a, clock),
        timer_bias_s=timer_bias_s,
        n_samples=int(arr.size),
        method=detail,
    )
