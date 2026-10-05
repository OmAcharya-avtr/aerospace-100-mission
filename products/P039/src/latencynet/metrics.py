"""Scoring: interval coverage with its binomial standard error, and accuracy.

Coverage
--------
Whether a true value falls inside a prediction interval is a Bernoulli trial
per pipeline, so over ``n`` held-out pipelines the measured coverage ``c_hat``
has standard error

    SE(c_hat) = sqrt(c_hat (1 - c_hat) / n)

and a measured coverage is only meaningful quoted with it. With ``n = 100``
and a nominal 0.90 the standard error is 0.030, so a measured 0.87 and a
measured 0.93 are both consistent with the nominal value and neither is
evidence of mis-calibration. Standard binomial result; see e.g. Brown, Cai &
DasGupta (2001), *Statistical Science* 16(2): 101-133, which also notes that
the Wald interval under-covers near 0 and 1 -- the Wilson interval is
reported alongside for that reason.

Accuracy
--------
Errors are reported in log space, where they are relative errors:
``mean |ln q_hat - ln q|`` is, to first order, the mean absolute *fractional*
error of the prediction. Median absolute error is reported too because a
single badly mis-predicted pipeline should not be allowed to decide which
model wins.

Interval width is reported in log space for the same reason: a log-space width
of 0.2 is a factor of ``exp(0.2) = 1.22`` from end to end, whatever the
absolute latency of the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .predictors import IntervalPrediction


@dataclass(frozen=True)
class CoverageResult:
    """Measured interval coverage against its nominal level.

    Attributes
    ----------
    nominal:
        Nominal coverage of the interval, dimensionless.
    measured:
        Fraction of held-out pipelines whose true value fell inside.
    standard_error:
        ``sqrt(c (1 - c) / n)``, the binomial standard error of ``measured``.
    wilson_lower, wilson_upper:
        Wilson score interval on the true coverage at the same level as the
        standard error (one sigma, 68.27 %).
    n:
        Number of held-out pipelines.
    n_covered:
        Number inside the interval.
    mean_log_width:
        Mean interval width in log space, dimensionless.
    median_log_width:
        Median interval width in log space, dimensionless.
    """

    nominal: float
    measured: float
    standard_error: float
    wilson_lower: float
    wilson_upper: float
    n: int
    n_covered: int
    mean_log_width: float
    median_log_width: float

    @property
    def deviation_sigma(self) -> float:
        """``(measured - nominal) / standard_error``; ``nan`` if the SE is zero."""
        if self.standard_error == 0.0:
            return float("nan")
        return (self.measured - self.nominal) / self.standard_error


def wilson_interval(n_success: int, n: int, z: float = 1.0) -> tuple[float, float]:
    """Wilson score interval on a binomial proportion, dimensionless.

    ``z = 1.0`` gives the one-sigma (68.27 %) interval, matching the standard
    error reported next to it. Source: Wilson (1927), *JASA* 22(158): 209-212.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be an integer >= 1, got {n!r}")
    if not (0 <= int(n_success) <= int(n)):
        raise ValueError(f"n_success must lie in [0, {n}], got {n_success!r}")
    phat = float(n_success) / float(n)
    denom = 1.0 + z**2 / n
    centre = (phat + z**2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z**2 / (4.0 * n**2))
    return max(0.0, centre - half), min(1.0, centre + half)


def interval_coverage(prediction: IntervalPrediction, log_truth: np.ndarray) -> CoverageResult:
    """Measured coverage of a log-space interval against the true log quantiles."""
    truth = np.asarray(log_truth, dtype=float).ravel()
    if truth.shape != prediction.log_point.shape:
        raise ValueError(
            f"log_truth has shape {truth.shape} but the prediction has "
            f"{prediction.log_point.shape}"
        )
    if truth.size == 0:
        raise ValueError("log_truth must be non-empty")
    inside = (truth >= prediction.log_lower) & (truth <= prediction.log_upper)
    n = int(truth.size)
    n_cov = int(inside.sum())
    c = n_cov / n
    se = math.sqrt(c * (1.0 - c) / n)
    lo, hi = wilson_interval(n_cov, n, z=1.0)
    widths = prediction.log_width
    return CoverageResult(
        nominal=float(prediction.level),
        measured=c,
        standard_error=se,
        wilson_lower=lo,
        wilson_upper=hi,
        n=n,
        n_covered=n_cov,
        mean_log_width=float(np.mean(widths)),
        median_log_width=float(np.median(widths)),
    )


@dataclass(frozen=True)
class AccuracyResult:
    """Point-prediction accuracy in log space.

    Attributes
    ----------
    mean_abs_log_error:
        Mean of ``|ln q_hat - ln q|``; approximately the mean relative error.
    median_abs_log_error:
        Median of the same, robust to a single bad pipeline.
    rms_log_error:
        Root mean square of the log error.
    bias_log:
        Mean signed log error; negative means systematic under-prediction,
        which for a deadline is the dangerous direction.
    p90_abs_log_error:
        90th percentile of the absolute log error.
    n:
        Number of pipelines scored.
    """

    mean_abs_log_error: float
    median_abs_log_error: float
    rms_log_error: float
    bias_log: float
    p90_abs_log_error: float
    n: int

    @property
    def mean_relative_error(self) -> float:
        """``exp(mean_abs_log_error) - 1``, a relative error, dimensionless."""
        return math.exp(self.mean_abs_log_error) - 1.0


def log_accuracy(log_prediction: np.ndarray, log_truth: np.ndarray) -> AccuracyResult:
    """Score point predictions of ``ln q_p`` against the reference truth."""
    pred = np.asarray(log_prediction, dtype=float).ravel()
    truth = np.asarray(log_truth, dtype=float).ravel()
    if pred.shape != truth.shape:
        raise ValueError(f"shapes differ: {pred.shape} and {truth.shape}")
    if pred.size == 0:
        raise ValueError("predictions must be non-empty")
    err = pred - truth
    return AccuracyResult(
        mean_abs_log_error=float(np.mean(np.abs(err))),
        median_abs_log_error=float(np.median(np.abs(err))),
        rms_log_error=float(math.sqrt(float(np.mean(err**2)))),
        bias_log=float(np.mean(err)),
        p90_abs_log_error=float(np.quantile(np.abs(err), 0.9, method="linear")),
        n=int(pred.size),
    )


def paired_difference_test(
    errors_a: np.ndarray, errors_b: np.ndarray
) -> tuple[float, float, float]:
    """Paired comparison of two models' absolute log errors on the same pipelines.

    Returns ``(mean_difference, t_statistic, two_sided_p_value)`` for
    ``|errors_a| - |errors_b|``. A negative mean difference means model A is
    more accurate. The t-test assumes only that the mean of the paired
    differences is approximately normal, which the central limit theorem gives
    for a hundred pipelines; it does not assume the errors themselves are
    normal.
    """
    a = np.abs(np.asarray(errors_a, dtype=float).ravel())
    b = np.abs(np.asarray(errors_b, dtype=float).ravel())
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} and {b.shape}")
    if a.size < 3:
        raise ValueError("need at least 3 paired observations")
    diff = a - b
    result = stats.ttest_rel(a, b)
    return float(np.mean(diff)), float(result.statistic), float(result.pvalue)
