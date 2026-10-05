"""Baseline 1: the analytic sum-of-stages latency model.

This is the model an engineer writes on the back of an envelope, and it is the
first baseline any learned latency predictor has to beat.

The model
---------
End-to-end latency of a serial pipeline is ``S = sum_i X_i``. Two moment
results are used, and they differ in how much they assume.

Mean, exact under any dependence:

    E[S] = sum_i E[X_i]                                                   (1)

by linearity of expectation. No independence assumption, no approximation.

Variance, exact given the covariances:

    Var[S] = sum_i Var[X_i] + 2 sum_{i<j} Cov(X_i, X_j)                   (2)

which reduces to ``Var[S] = sum_i Var[X_i]`` exactly when the stages are
uncorrelated. Both are standard (e.g. Casella & Berger (2002), *Statistical
Inference*, 2nd ed., Thm. 4.5.6 and Sec. 4.5).

Equations (1) and (2) are the whole of the sum-of-stages model, and they are
*identities*: given stage moments they return the total moments with no error
beyond binary64 rounding. That is what
``validation/validate_analytic_exactness.py`` checks, and it is the
correctness check for the whole harness.

From moments to a tail quantile
-------------------------------
A deadline needs a quantile, and the sum of lognormals has no closed-form
distribution. This module uses the Fenton-Wilkinson approximation: match a
single lognormal to the exact mean and variance of the sum and read its
quantile off,

    sigma_S^2 = ln(1 + Var[S] / E[S]^2)
    mu_S      = ln(E[S]) - sigma_S^2 / 2
    q_p       = exp(mu_S + sigma_S Phi^-1(p))                             (3)

Source: L. F. Fenton (1960), "The Sum of Log-Normal Probability Distributions
in Scatter Transmission Systems", *IRE Transactions on Communications
Systems* 8(1): 57-67.

Equation (3) is an approximation and is labelled as one everywhere in this
package. Its known validity range: it is accurate for moderate total
coefficient of variation and degrades in the far tail and for a small number
of highly skewed stages, where the true sum is heavier-tailed than any
lognormal matched on two moments. ``validation/validate_analytic_exactness.py``
measures that error against a large Monte Carlo reference rather than
asserting it away.

Prediction interval
-------------------
In use, the stage moments are not declared but *estimated* from a short probe
trace of ``n`` passes. The analytic model's native interval propagates that
estimation uncertainty through (1)-(3) by the law of propagation of
uncertainty, JCGM 100:2008 (GUM) Sec. 5.1.2, with

    u(mean_i) = s_i / sqrt(n)
    u(s_i)    = sqrt((m4_i - s_i^4) / (4 n s_i^2))

the second being the delta-method standard error of a sample standard
deviation in terms of the fourth central moment (Cramer (1946),
*Mathematical Methods of Statistics*, Sec. 27.4). Sensitivity coefficients
are taken numerically by central differences on ``ln q_p``, and the interval
is formed in log space so it cannot contain a negative latency.

This interval accounts for probe sampling uncertainty and for nothing else.
It does not account for the Fenton-Wilkinson model error, and it does not
account for stage dependence when the model is run in its independence mode.
Both omissions are real and show up as under-coverage in
``validation/validate_interval_coverage.py``; see :mod:`latencynet.conformal`
for the calibrated alternative.

Units: latencies and standard deviations in seconds, variances in s^2, fourth
central moments in s^4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import stats


@lru_cache(maxsize=256)
def _norm_ppf(p: float) -> float:
    """Standard normal quantile, cached.

    The GUM sensitivity coefficients below are taken by central
    differences, which evaluates the quantile map a few dozen times per
    pipeline at a single fixed ``p``. ``scipy.stats.norm.ppf`` costs tens
    of microseconds per scalar call, so caching it turns a measurable
    fraction of the validation runtime into a dictionary lookup. The
    cached value is exact: ``p`` is the only argument.
    """
    return float(stats.norm.ppf(p))


def sum_of_stages_mean(stage_mean_s: np.ndarray) -> float:
    """Equation (1): end-to-end mean, seconds. Exact under any dependence."""
    arr = np.asarray(stage_mean_s, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("stage_mean_s must be non-empty")
    if np.any(arr <= 0.0) or not np.all(np.isfinite(arr)):
        raise ValueError("stage means must be finite and strictly positive")
    return float(arr.sum())


def sum_of_stages_variance(
    stage_std_s: np.ndarray, stage_cov_s2: np.ndarray | None = None
) -> float:
    """Equation (2): end-to-end variance, s^2.

    With ``stage_cov_s2 is None`` the stages are assumed uncorrelated and the
    result is ``sum(std^2)``. With a covariance matrix supplied the result is
    its total sum, which is exact.
    """
    sd = np.asarray(stage_std_s, dtype=float).ravel()
    if sd.size == 0:
        raise ValueError("stage_std_s must be non-empty")
    if np.any(sd < 0.0) or not np.all(np.isfinite(sd)):
        raise ValueError("stage standard deviations must be finite and non-negative")
    if stage_cov_s2 is None:
        return float(np.sum(sd**2))
    cov = np.asarray(stage_cov_s2, dtype=float)
    if cov.shape != (sd.size, sd.size):
        raise ValueError(f"stage_cov_s2 must have shape {(sd.size, sd.size)}, got {cov.shape}")
    if not np.all(np.isfinite(cov)):
        raise ValueError("stage_cov_s2 must be finite")
    return float(cov.sum())


def fenton_wilkinson_quantile(mean_s: float, variance_s2: float, p: float) -> float:
    """Equation (3): lognormal moment-matched quantile of the sum, seconds.

    A zero variance returns ``mean_s`` exactly, which is the correct
    degenerate limit.
    """
    mean_s = float(mean_s)
    variance_s2 = float(variance_s2)
    if not math.isfinite(mean_s) or mean_s <= 0.0:
        raise ValueError(f"mean_s must be finite and > 0, got {mean_s!r}")
    if not math.isfinite(variance_s2) or variance_s2 < 0.0:
        raise ValueError(f"variance_s2 must be finite and >= 0, got {variance_s2!r}")
    if not (0.0 < float(p) < 1.0):
        raise ValueError(f"p must lie strictly in (0, 1), got {p!r}")
    if variance_s2 == 0.0:
        return mean_s
    sigma_sq = math.log1p(variance_s2 / mean_s**2)
    sigma = math.sqrt(sigma_sq)
    mu = math.log(mean_s) - 0.5 * sigma_sq
    return float(math.exp(mu + sigma * _norm_ppf(float(p))))


@dataclass(frozen=True)
class AnalyticPrediction:
    """Result of one analytic sum-of-stages prediction.

    Attributes
    ----------
    mean_s:
        End-to-end mean from equation (1), seconds.
    std_s:
        End-to-end standard deviation from equation (2), seconds.
    quantile_s:
        Mapping from tail probability to the Fenton-Wilkinson quantile,
        seconds. Approximate; see module docstring.
    assume_independent:
        Whether equation (2) was evaluated without covariance terms.
    """

    mean_s: float
    std_s: float
    quantile_s: dict[float, float]
    assume_independent: bool


class SumOfStagesModel:
    """Baseline 1. Predicts end-to-end latency moments and tail quantiles.

    Parameters
    ----------
    assume_independent:
        ``True`` (the default) is the textbook sum-of-stages model: variances
        add and covariances are ignored. ``False`` uses a supplied stage
        covariance matrix, which makes equation (2) exact. The independence
        mode is the baseline named in the specification; the covariance mode
        is reported alongside it as a diagnostic so that the cost of the
        independence assumption is visible rather than inferred.
    """

    def __init__(self, assume_independent: bool = True) -> None:
        self.assume_independent = bool(assume_independent)
        self.name = "analytic_sum_indep" if assume_independent else "analytic_sum_cov"

    def predict(
        self,
        stage_mean_s: np.ndarray,
        stage_std_s: np.ndarray,
        probabilities: tuple[float, ...] = (0.5, 0.99, 0.999),
        stage_cov_s2: np.ndarray | None = None,
    ) -> AnalyticPrediction:
        """Predict end-to-end moments and quantiles from per-stage moments."""
        if not self.assume_independent and stage_cov_s2 is None:
            raise ValueError("assume_independent=False requires stage_cov_s2")
        mean = sum_of_stages_mean(stage_mean_s)
        var = sum_of_stages_variance(
            stage_std_s, None if self.assume_independent else stage_cov_s2
        )
        var = max(var, 0.0)
        quant = {float(p): fenton_wilkinson_quantile(mean, var, p) for p in probabilities}
        return AnalyticPrediction(
            mean_s=mean,
            std_s=math.sqrt(var),
            quantile_s=quant,
            assume_independent=self.assume_independent,
        )

    def predict_quantile(
        self,
        stage_mean_s: np.ndarray,
        stage_std_s: np.ndarray,
        p: float,
        stage_cov_s2: np.ndarray | None = None,
    ) -> float:
        """Predicted end-to-end quantile at ``p``, seconds."""
        return self.predict(stage_mean_s, stage_std_s, (float(p),), stage_cov_s2).quantile_s[
            float(p)
        ]

    def log_quantile_uncertainty(
        self,
        stage_mean_s: np.ndarray,
        stage_std_s: np.ndarray,
        stage_m4_s4: np.ndarray,
        n_probe: int,
        p: float,
        stage_cov_s2: np.ndarray | None = None,
        rel_step: float = 1.0e-6,
    ) -> float:
        """GUM standard uncertainty of ``ln q_p`` from probe sampling, dimensionless.

        See the module docstring for the input uncertainties and for what this
        term does and does not include.
        """
        means = np.asarray(stage_mean_s, dtype=float).ravel().copy()
        sds = np.asarray(stage_std_s, dtype=float).ravel().copy()
        m4 = np.asarray(stage_m4_s4, dtype=float).ravel()
        if not isinstance(n_probe, (int, np.integer)) or n_probe < 4:
            raise ValueError(f"n_probe must be an integer >= 4, got {n_probe!r}")
        if not (means.size == sds.size == m4.size):
            raise ValueError("stage_mean_s, stage_std_s and stage_m4_s4 must match in length")
        n = float(n_probe)
        u_mean = sds / math.sqrt(n)
        var4 = np.maximum(m4 - sds**4, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            u_sd = np.where(sds > 0.0, np.sqrt(var4 / (4.0 * n * np.maximum(sds**2, 1e-300))), 0.0)

        def log_q(m_vec: np.ndarray, s_vec: np.ndarray) -> float:
            return math.log(self.predict_quantile(m_vec, s_vec, p, stage_cov_s2))

        total = 0.0
        for i in range(means.size):
            for vec, unc in ((means, u_mean), (sds, u_sd)):
                if unc[i] == 0.0:
                    continue
                step = max(abs(vec[i]) * rel_step, 1e-300)
                original = vec[i]
                vec[i] = original + step
                hi = log_q(means, sds)
                vec[i] = original - step
                lo = log_q(means, sds)
                vec[i] = original
                sens = (hi - lo) / (2.0 * step)
                total += (sens * unc[i]) ** 2
        return float(math.sqrt(total))

    def predict_quantile_interval(
        self,
        stage_mean_s: np.ndarray,
        stage_std_s: np.ndarray,
        stage_m4_s4: np.ndarray,
        n_probe: int,
        p: float,
        level: float = 0.9,
        stage_cov_s2: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        """Native prediction interval for the end-to-end quantile, seconds.

        Returns ``(lower_s, point_s, upper_s)``. The interval is symmetric in
        ``ln q`` with half-width ``z_(1+level)/2 * u(ln q)``.
        """
        if not (0.0 < float(level) < 1.0):
            raise ValueError(f"level must lie strictly in (0, 1), got {level!r}")
        point = self.predict_quantile(stage_mean_s, stage_std_s, p, stage_cov_s2)
        u_log = self.log_quantile_uncertainty(
            stage_mean_s, stage_std_s, stage_m4_s4, n_probe, p, stage_cov_s2
        )
        z = _norm_ppf(0.5 * (1.0 + float(level)))
        half = z * u_log
        return float(point * math.exp(-half)), float(point), float(point * math.exp(half))
