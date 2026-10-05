"""Per-stage features extracted from a short probe trace.

The operational setting this encodes
------------------------------------
Profiling each stage of an embedded pipeline in isolation is cheap: a handful
of passes with per-stage instrumentation. Characterising the *end-to-end tail*
is expensive, because p99.9 needs thousands of passes before the estimate
stops moving (see :mod:`latencynet.tails`). So the task is to predict the
expensive quantity from the cheap one.

A probe trace is an aligned record of per-stage latencies over ``n_probe``
passes: shape ``(n_probe, K)``, seconds. Aligned matters -- it is what makes
stage *dependence* observable at all. From marginals alone the correlation
between stages is not identifiable, and no model of any kind could recover it.

Feature vector
--------------
Thirteen features, all computed from the probe trace only, never from the
end-to-end reference sample:

===  ==========================  ======================================
idx  name                        meaning
===  ==========================  ======================================
0    ``n_stages``                K
1    ``log_sum_mean``            ln of sum of per-stage sample means
2    ``log_indep_sd``            ln of sqrt(sum of per-stage variances)
3    ``log_dep_sd``              ln of sqrt(total sum of the covariance
                                 matrix), the dependence-aware total sd
4    ``indep_cv``                independence sd / sum of means
5    ``dep_cv``                  dependence-aware sd / sum of means
6    ``dependence_inflation``    dep sd / indep sd, 1.0 under independence
7    ``mean_offdiag_corr``       mean off-diagonal Pearson correlation
8    ``max_offdiag_corr``        largest off-diagonal correlation
9    ``mean_stage_cv``           mean per-stage coefficient of variation
10   ``max_stage_cv``            largest per-stage coefficient of variation
11   ``mean_stage_skew``         mean per-stage sample skewness
12   ``log_sum_stage_p99``       ln of sum of per-stage probe p99s, the
                                 "every stage is simultaneously late" bound
===  ==========================  ======================================

Features 3, 5, 6, 7 and 8 are the only ones that carry dependence information.
They are available to the linear-regression baseline and to the learned model.
They are deliberately *not* available to the analytic sum-of-stages baseline,
because that baseline is defined by the independence assumption -- that is the
comparison the specification asks for. A covariance-aware analytic variant is
reported separately as a diagnostic.

All features are computed from ``n_probe`` samples and are therefore noisy;
feature 7 in particular has standard error of order ``1/sqrt(n_probe)``, so at
``n_probe = 256`` a true correlation of 0.0 is observed as roughly
``0.00 +/- 0.06``. That noise is part of the problem, not a defect.

Units: features 1, 2, 3 and 12 are logs of seconds (dimensionless after the
log, with an additive offset that is constant across the dataset); the rest
are dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tails import quantile

FEATURE_NAMES: tuple[str, ...] = (
    "n_stages",
    "log_sum_mean",
    "log_indep_sd",
    "log_dep_sd",
    "indep_cv",
    "dep_cv",
    "dependence_inflation",
    "mean_offdiag_corr",
    "max_offdiag_corr",
    "mean_stage_cv",
    "max_stage_cv",
    "mean_stage_skew",
    "log_sum_stage_p99",
)

N_FEATURES = len(FEATURE_NAMES)

#: Indices of the features that carry stage-dependence information.
DEPENDENCE_FEATURE_INDICES: tuple[int, ...] = (3, 5, 6, 7, 8)

_FLOOR = 1.0e-300


@dataclass(frozen=True)
class ProbeSummary:
    """Per-stage moment estimates from a probe trace, plus the feature vector.

    Attributes
    ----------
    n_probe:
        Number of passes in the probe trace.
    stage_mean_s, stage_std_s:
        Per-stage sample mean and sample standard deviation (ddof=1), seconds.
    stage_m4_s4:
        Per-stage fourth central moment estimate, s^4, used by the analytic
        model's uncertainty propagation.
    stage_cov_s2:
        Per-stage sample covariance matrix (ddof=1), s^2, flattened row-major.
    stage_p99_s:
        Per-stage probe p99, seconds.
    features:
        The thirteen features in :data:`FEATURE_NAMES` order.
    """

    n_probe: int
    stage_mean_s: tuple[float, ...]
    stage_std_s: tuple[float, ...]
    stage_m4_s4: tuple[float, ...]
    stage_cov_s2: tuple[float, ...]
    stage_p99_s: tuple[float, ...]
    features: tuple[float, ...]

    @property
    def n_stages(self) -> int:
        """Number of stages in the probe trace."""
        return len(self.stage_mean_s)

    def covariance_matrix(self) -> np.ndarray:
        """Sample covariance matrix as a ``(K, K)`` array, s^2."""
        k = self.n_stages
        return np.asarray(self.stage_cov_s2, dtype=float).reshape(k, k)


def summarise_probe_trace(trace_s: np.ndarray) -> ProbeSummary:
    """Summarise an aligned per-stage probe trace.

    Parameters
    ----------
    trace_s:
        Shape ``(n_probe, K)``, seconds, strictly positive and finite.
        ``n_probe`` must be at least 8 so that the fourth moment and the
        correlation estimates are defined.
    """
    arr = np.asarray(trace_s, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"trace_s must be 2-D (n_probe, K), got shape {arr.shape}")
    n, k = arr.shape
    if n < 8:
        raise ValueError(f"n_probe must be >= 8 to estimate the needed moments, got {n}")
    if k < 1:
        raise ValueError("trace_s must have at least one stage column")
    if not np.all(np.isfinite(arr)):
        raise ValueError("trace_s must be finite")
    if np.any(arr <= 0.0):
        raise ValueError("trace_s latencies must be strictly positive")

    mean = arr.mean(axis=0)
    centred = arr - mean
    var = centred.var(axis=0, ddof=1)
    sd = np.sqrt(var)
    m4 = (centred**4).mean(axis=0)
    cov = np.cov(arr, rowvar=False, ddof=1)
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    p99 = np.array([quantile(arr[:, j], 0.99, method="linear") for j in range(k)])

    sum_mean = float(mean.sum())
    indep_var = float(var.sum())
    dep_var = float(cov.sum())
    indep_sd = float(np.sqrt(max(indep_var, 0.0)))
    dep_sd = float(np.sqrt(max(dep_var, 0.0)))

    with np.errstate(divide="ignore", invalid="ignore"):
        stage_cv = np.where(mean > 0.0, sd / np.maximum(mean, _FLOOR), 0.0)
        m3 = (centred**3).mean(axis=0)
        skew = np.where(sd > 0.0, m3 / np.maximum(sd**3, _FLOOR), 0.0)

    if k >= 2:
        denom = np.outer(sd, sd)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 0.0, cov / np.maximum(denom, _FLOOR), 0.0)
        off = corr[~np.eye(k, dtype=bool)]
        mean_off = float(off.mean())
        max_off = float(off.max())
    else:
        mean_off = 0.0
        max_off = 0.0

    features = (
        float(k),
        float(np.log(max(sum_mean, _FLOOR))),
        float(np.log(max(indep_sd, _FLOOR))),
        float(np.log(max(dep_sd, _FLOOR))),
        float(indep_sd / max(sum_mean, _FLOOR)),
        float(dep_sd / max(sum_mean, _FLOOR)),
        float(dep_sd / max(indep_sd, _FLOOR)),
        mean_off,
        max_off,
        float(stage_cv.mean()),
        float(stage_cv.max()),
        float(skew.mean()),
        float(np.log(max(float(p99.sum()), _FLOOR))),
    )
    if len(features) != N_FEATURES:
        raise AssertionError("feature vector length does not match FEATURE_NAMES")

    return ProbeSummary(
        n_probe=int(n),
        stage_mean_s=tuple(float(v) for v in mean),
        stage_std_s=tuple(float(v) for v in sd),
        stage_m4_s4=tuple(float(v) for v in m4),
        stage_cov_s2=tuple(float(v) for v in cov.ravel()),
        stage_p99_s=tuple(float(v) for v in p99),
        features=features,
    )
