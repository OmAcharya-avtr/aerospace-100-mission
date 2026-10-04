"""Probabilistic forecast verification: proper scores, decomposition, calibration.

Every quantity here is a standard forecast-verification statistic with a named
source.  Accuracy is deliberately a secondary output: a link-availability
probability is used to decide whether to schedule a contact, and a
mis-calibrated probability corrupts that decision even when the thresholded
classification is right.

Brier score
-----------
    BS = (1 / N) sum_i (p_i - o_i)^2                                      (1)

for probabilities ``p_i`` in [0, 1] and binary outcomes ``o_i`` (Brier 1950,
"Verification of Forecasts Expressed in Terms of Probability", Monthly Weather
Review 78(1), 1-3).  Lower is better; it is a strictly proper score, so it
cannot be improved by misreporting a believed probability.

Murphy decomposition
--------------------
Partitioning the forecasts into ``K`` bins with ``n_k`` members, mean forecast
``pbar_k`` and observed frequency ``obar_k``, and with overall base rate
``obar`` (Murphy 1973, "A New Vector Partition of the Probability Score",
Journal of Applied Meteorology 12(4), 595-600)::

    BS = REL - RES + UNC
    REL = (1 / N) sum_k n_k (pbar_k - obar_k)^2        (reliability, lower better)
    RES = (1 / N) sum_k n_k (obar_k - obar)^2          (resolution, higher better)
    UNC = obar (1 - obar)                              (uncertainty, data property)

REL is the calibration term: it is zero for a perfectly calibrated forecast.

The three-term identity is EXACT only when every forecast inside a bin has the
same value -- that is, when the bins are the distinct forecast values.  With
equal-width bins over continuous forecasts there are two further components,
a within-bin forecast variance and a within-bin forecast-outcome covariance,
derived and named by Stephenson, Coelho & Jolliffe 2008, "Two Extra
Components in the Brier Score Decomposition", Weather and Forecasting 23(4),
752-757.  :func:`brier_decomposition` therefore reports
``identity_residual = BS - (REL - RES + UNC)``, which is exactly the sum of
those two extra components, and does not pretend it is floating-point noise.
Pass ``by_distinct_value=True`` to bin by distinct forecast value, where the
residual collapses to rounding error and the identity is exact.

Binning caveat, stated because it is often hidden: REL and RES depend on the
binning.  With few bins a mis-calibrated forecast can appear reliable; with
many bins REL is biased upward by sampling noise within bins.  All reported
values here state the bin count alongside.

Expected calibration error
--------------------------
    ECE = (1 / N) sum_k n_k | pbar_k - obar_k |                           (2)

the weighted mean absolute calibration gap (Naeini, Cooper & Hauskrecht 2015,
"Obtaining well calibrated probabilities using Bayesian binning", AAAI 2015;
the usage in modern model evaluation follows Guo, Pleiss, Sun & Weinberger
2017, "On calibration of modern neural networks", ICML 2017).  ECE is NOT a
proper score -- a constant forecast at the base rate has ECE near zero and no
skill at all -- so it is always reported next to the Brier score and the
resolution term, never alone.

Reliability diagram
-------------------
:func:`reliability_curve` returns the binned ``(pbar_k, obar_k, n_k)`` triples
plus Wilson score intervals on ``obar_k`` (Wilson 1927, "Probable inference,
the law of succession, and statistical inference", JASA 22, 209-212), so a bin
with three samples is visibly uninformative instead of looking like a
calibration failure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "brier_score",
    "log_loss_safe",
    "BrierDecomposition",
    "brier_decomposition",
    "expected_calibration_error",
    "ReliabilityCurve",
    "reliability_curve",
    "wilson_interval",
    "bootstrap_ci",
]


def _check_prob_obs(p: np.ndarray, o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float).ravel()
    o = np.asarray(o, dtype=float).ravel()
    if p.shape != o.shape:
        raise ValueError(f"p and o must have the same shape, got {p.shape} and {o.shape}")
    if p.size == 0:
        raise ValueError("p and o must be non-empty")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.all(np.isin(o, (0.0, 1.0))):
        raise ValueError("outcomes must be binary 0/1")
    return p, o


def brier_score(p: np.ndarray, o: np.ndarray) -> float:
    """Brier score, Eq. (1).  Lower is better; range [0, 1]."""
    p, o = _check_prob_obs(p, o)
    return float(np.mean((p - o) ** 2))


def log_loss_safe(p: np.ndarray, o: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood [nats], probabilities clipped to ``eps``.

    Reported alongside the Brier score because the two proper scores penalise
    confident errors differently: the log score is unbounded, so one confident
    mistake dominates it.
    """
    p, o = _check_prob_obs(p, o)
    if not 0.0 < eps < 0.5:
        raise ValueError(f"eps must be in (0, 0.5), got {eps}")
    q = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(o * np.log(q) + (1.0 - o) * np.log(1.0 - q)))


@dataclass(frozen=True)
class BrierDecomposition:
    """Murphy (1973) three-term partition of the Brier score.

    Attributes
    ----------
    brier : the score itself, Eq. (1).
    reliability : calibration term, lower is better.
    resolution : discrimination term, higher is better.
    uncertainty : base-rate variance, a property of the data.
    n_bins : number of bins used (equal-width, or the number of distinct
        forecast values when ``by_distinct_value``).
    n_occupied_bins : bins containing at least one forecast.
    identity_residual : ``brier - (reliability - resolution + uncertainty)``.
        This is the sum of the two extra components of Stephenson, Coelho &
        Jolliffe 2008 -- within-bin forecast variance minus twice the
        within-bin forecast-outcome covariance -- and is NOT floating-point
        noise unless the bins are the distinct forecast values.
    by_distinct_value : whether the bins were the distinct forecast values.
    """

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    n_bins: int
    n_occupied_bins: int
    identity_residual: float
    by_distinct_value: bool = False


def brier_decomposition(p: np.ndarray, o: np.ndarray, n_bins: int = 10,
                        by_distinct_value: bool = False) -> BrierDecomposition:
    """Murphy decomposition of the Brier score.

    With ``by_distinct_value=False`` (default) the forecasts are binned into
    ``n_bins >= 2`` equal-width bins and ``identity_residual`` carries the two
    extra Stephenson-Coelho-Jolliffe components.  With
    ``by_distinct_value=True`` the bins ARE the distinct forecast values, the
    three-term identity is exact, and ``n_bins`` is ignored.  See the module
    docstring.
    """
    p, o = _check_prob_obs(p, o)
    if by_distinct_value:
        values, idx = np.unique(p, return_inverse=True)
        n_bins = int(values.size)
    else:
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    n = p.size
    obar = float(o.mean())
    rel = 0.0
    res = 0.0
    occupied = 0
    for k in range(n_bins):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            continue
        occupied += 1
        pbar_k = float(p[m].mean())
        obar_k = float(o[m].mean())
        rel += nk * (pbar_k - obar_k) ** 2
        res += nk * (obar_k - obar) ** 2
    rel /= n
    res /= n
    unc = obar * (1.0 - obar)
    bs = brier_score(p, o)
    return BrierDecomposition(brier=bs, reliability=rel, resolution=res,
                              uncertainty=unc, n_bins=n_bins,
                              n_occupied_bins=occupied,
                              identity_residual=bs - (rel - res + unc),
                              by_distinct_value=by_distinct_value)


def expected_calibration_error(p: np.ndarray, o: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, Eq. (2).  Not a proper score -- see module docs."""
    p, o = _check_prob_obs(p, o)
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for k in range(n_bins):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            continue
        total += nk * abs(float(p[m].mean()) - float(o[m].mean()))
    return float(total / p.size)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054,
                    ) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (Wilson 1927).

    ``z`` defaults to the two-sided 95 % normal quantile.  Returns
    ``(lower, upper)``; a zero-trial bin returns ``(0.0, 1.0)``.

    The analytic interval always contains the observed proportion, but at
    ``phat`` of exactly 0 or 1 the endpoint is recovered only to rounding
    error, so the returned bounds are additionally clamped to bracket
    ``phat``.  Callers that subtract the bounds from ``phat`` to draw error
    bars therefore never see a negative half-width.
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(
            f"need 0 <= successes <= trials, got successes={successes}, trials={trials}")
    if trials == 0:
        return 0.0, 1.0
    n = float(trials)
    phat = successes / n
    denom = 1.0 + z ** 2 / n
    centre = (phat + z ** 2 / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1.0 - phat) / n + z ** 2 / (4.0 * n ** 2))
    lower = min(max(0.0, centre - half), phat)
    upper = max(min(1.0, centre + half), phat)
    return float(lower), float(upper)


@dataclass(frozen=True)
class ReliabilityCurve:
    """Binned reliability-diagram data.

    Attributes
    ----------
    bin_centre : nominal bin centre [-], shape ``(K,)``.
    mean_forecast : ``pbar_k``, NaN for an empty bin.
    observed_frequency : ``obar_k``, NaN for an empty bin.
    count : ``n_k``.
    ci_low, ci_high : Wilson 95 % interval on ``obar_k``, NaN for empty bins.
    """

    bin_centre: np.ndarray
    mean_forecast: np.ndarray
    observed_frequency: np.ndarray
    count: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray


def reliability_curve(p: np.ndarray, o: np.ndarray, n_bins: int = 10,
                      ) -> ReliabilityCurve:
    """Reliability-diagram data with Wilson intervals (see module docstring)."""
    p, o = _check_prob_obs(p, o)
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    mean_f = np.full(n_bins, np.nan)
    obs = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=int)
    lo = np.full(n_bins, np.nan)
    hi = np.full(n_bins, np.nan)
    for k in range(n_bins):
        m = idx == k
        nk = int(m.sum())
        cnt[k] = nk
        if nk == 0:
            continue
        mean_f[k] = float(p[m].mean())
        obs[k] = float(o[m].mean())
        lo[k], hi[k] = wilson_interval(int(o[m].sum()), nk)
    return ReliabilityCurve(bin_centre=centres, mean_forecast=mean_f,
                            observed_frequency=obs, count=cnt,
                            ci_low=lo, ci_high=hi)


def bootstrap_ci(p: np.ndarray, o: np.ndarray, statistic=brier_score,
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
                 ) -> tuple[float, float, float]:
    """Percentile bootstrap interval for a verification statistic.

    Resamples forecast/outcome PAIRS with replacement (Efron & Tibshirani
    1993, "An Introduction to the Bootstrap", Chapman & Hall, Ch. 6).  Returns
    ``(point_estimate, lower, upper)`` at confidence ``1 - alpha``.

    The pairs are assumed exchangeable.  They are not, strictly, when several
    rows come from the same link: the interval is therefore optimistic by an
    amount set by the within-link correlation.  The grouped split used in
    :mod:`constellink.availability` keeps a link entirely on one side of the
    split, which removes the leakage but not the within-test correlation.
    """
    p, o = _check_prob_obs(p, o)
    if n_boot < 10:
        raise ValueError(f"n_boot must be >= 10, got {n_boot}")
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    rng = np.random.default_rng(seed)
    n = p.size
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # A resample can be single-class; the statistics here all tolerate it.
        vals[b] = statistic(p[idx], o[idx])
    return (float(statistic(p, o)),
            float(np.quantile(vals, alpha / 2.0)),
            float(np.quantile(vals, 1.0 - alpha / 2.0)))
