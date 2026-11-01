"""Bootstrap and split-conformal bounds: tighter, under exchangeability.

This module is **baseline 2**. Everything in it is tighter than
:mod:`transfergap.bounds` and everything in it rests on an assumption that
:mod:`transfergap.bounds` does not need. The assumption is stated here and
**tested** in :mod:`transfergap.exchangeability`; it is never assumed silently,
because a tighter interval that does not hold is worse than a loose one that
does.

The load-bearing assumption
---------------------------
Split conformal prediction needs the calibration scores and the new point's
score to be **exchangeable**: their joint law is invariant under permutation.
That is strictly weaker than i.i.d. and strictly stronger than nothing. A
flight-test campaign flown in order of increasing Mach number is not
exchangeable in the index, and the bound then has no guarantee at all --
not a weakened one, none. The nonparametric bootstrap needs i.i.d. sampling
for its usual asymptotic justification (Efron and Tibshirani 1993, chapter 6),
and in a finite sample of tens it is an approximation whose error is not
bounded by anything computed here.

Equations, with sources, units and validity ranges
--------------------------------------------------
**(12) The split-conformal quantile.** Vovk, V., Gammerman, A. and Shafer, G.
(2005), *Algorithmic Learning in a Random World*, Springer; and the modern
statement in Angelopoulos, A. N. and Bates, S. (2023), "Conformal prediction:
a gentle introduction", *Foundations and Trends in Machine Learning* 16(4),
494-591, section 1. Given ``n`` exchangeable calibration scores ``s_1..s_n``
and a miscoverage level ``alpha``, let

    k = ceil( (n + 1) (1 - alpha) ),    qhat = s_(k)                      (12)

the ``k``-th smallest score. Then for an exchangeable new score ``s_{n+1}``,

    1 - alpha  <=  P( s_{n+1} <= qhat )  <=  1 - alpha + 1/(n+1).         (13)

The upper half of (13) needs the scores to be almost surely distinct and is
given in Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. and Wasserman, L.
(2018), "Distribution-free predictive inference for regression", *Journal of
the American Statistical Association* 113(523), 1094-1111, Theorem 2.2.
Equation (13) is a two-sided bracket on the achieved coverage and is used in
``validation/validate_independent_bound.py`` as a bound derived from
mathematics this implementation does not use: the implementation takes an
order statistic by integer index and never computes a coverage probability.

``k > n`` happens exactly when ``n < ceil(1/alpha) - 1``; the sample cannot
support the statement and :func:`split_conformal_quantile` raises rather than
returning the maximum and calling it a bound.

**(14) The percentile bootstrap.** Efron, B. and Tibshirani, R. J. (1993), *An
Introduction to the Bootstrap*, Chapman and Hall, chapters 12-13. Draw ``B``
resamples with replacement, compute the statistic on each, and take the
``1 - alpha`` empirical quantile of the ``B`` values as a one-sided upper
confidence limit. **(15) The basic (reverse-percentile) bootstrap** reflects
instead: ``2 thetahat - q_alpha``. Validity: asymptotic, i.i.d. sampling,
and a statistic that is a smooth functional of the empirical distribution.
At ``n`` of tens neither is distribution-free and neither carries a
finite-sample guarantee. Their measured coverage is in
``validation/validate_bounds_output.txt``.

Units: every limit returned here is in the unit of the measured quantity.
"""

from __future__ import annotations

import math

import numpy as np

from .bounds import Bound
from .discrepancy import validate_confidence, validate_gaps

__all__ = [
    "bootstrap_mean_upper",
    "conformal_order_index",
    "conformal_required_n",
    "split_conformal_quantile",
    "split_conformal_upper",
]


def conformal_order_index(n: int, alpha: float) -> int:
    """``k = ceil((n+1)(1-alpha))`` of equation (12), a 1-based order index."""
    count = int(n)
    if count < 1:
        raise ValueError(f"n must be at least 1; got {n}")
    level = validate_confidence(1.0 - float(alpha), "1 - alpha")
    return int(math.ceil((count + 1) * level))


def conformal_required_n(alpha: float) -> int:
    """Smallest ``n`` for which equation (12) has ``k <= n``.

    ``k = ceil((n+1)(1-alpha)) <= n`` first holds at ``n = ceil(1/alpha) - 1``.
    At ``alpha = 0.05`` that is 19, which is why a 95 per cent conformal upper
    bound from 15 field points does not exist. Closed form, exact in exact
    arithmetic.

    **A defect found during this build, and why the implementation is not the
    closed form alone.** The first version of this function computed
    ``ceil(1 / (1 - level)) - 1`` with ``level = 1 - alpha``, reconstructing
    ``alpha`` from its own complement. That round trip is not exact in binary
    floating point: at ``alpha = 0.10`` it gives ``1 - (1 - 0.10) =
    0.09999999999999998``, whose reciprocal is ``10.000000000000002``, whose
    ceiling is 11 rather than 10, and the function returned 10 where the
    answer is 9. The same happened at ``alpha = 0.20``, returning 5 for 4.
    Both are pinned with literal inputs in ``tests/test_regressions.py``.

    The reciprocal is now taken from ``alpha`` directly, and the result is then
    reconciled against :func:`conformal_order_index` by a short walk, so that
    the two functions cannot disagree about where the limit starts to exist
    whatever either one's rounding does. The walk is the authority; the closed
    form is its starting point.
    """
    level = validate_confidence(1.0 - float(alpha), "1 - alpha")
    miscoverage = 1.0 - level
    candidate = max(1, int(math.ceil(1.0 / float(alpha))) - 1)
    while candidate > 1 and conformal_order_index(candidate - 1, miscoverage) <= candidate - 1:
        candidate -= 1
    while conformal_order_index(candidate, miscoverage) > candidate:
        candidate += 1
    return candidate


def split_conformal_quantile(scores: np.ndarray, alpha: float = 0.05) -> float:
    """The conformal quantile ``qhat`` of equation (12), in the unit of ``scores``.

    Parameters
    ----------
    scores:
        ``(n,)`` finite nonconformity scores from the calibration split.
    alpha:
        Miscoverage level, strictly inside ``(0, 1)``.

    Raises
    ------
    ValueError
        If ``k > n``, naming the ``n`` that equation (12) needs. Returning
        ``max(scores)`` in that case would be a bound with no guarantee
        wearing the name of one.

    Notes
    -----
    The quantile is taken as an order statistic by integer index, with
    ``numpy.partition``, and never through ``numpy.quantile``: a quantile
    interpolation rule is a library default that can change, and the
    finite-sample guarantee of equation (13) holds for the order statistic and
    not for an interpolated value between two of them.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"scores must be one-dimensional; got shape {values.shape}")
    if values.size == 0:
        raise ValueError("scores is empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores contains a non-finite value (nan or inf)")
    n = values.size
    k = conformal_order_index(n, alpha)
    if k > n:
        raise ValueError(
            f"split conformal at alpha = {float(alpha):.6g} needs n >= "
            f"{conformal_required_n(alpha)} "
            f"calibration scores by equation (12); got n = {n}, which asks for the "
            f"{k}-th smallest of {n}. There is no finite conformal quantile at this "
            f"sample size."
        )
    return float(np.partition(values, k - 1)[k - 1])


def split_conformal_upper(
    calibration_gaps: np.ndarray,
    bound_range: float,
    confidence: float = 0.95,
    unit: str = "",
) -> Bound:
    """Marginal split-conformal upper bound on a future gap, equations (12)-(13).

    With the identity nonconformity score ``s_i = d_i`` this is a prediction
    bound on the discrepancy of one future field measurement, valid under
    exchangeability with no distributional assumption at all. It is directly
    comparable to :func:`transfergap.bounds.wilks_tolerance_upper`: both
    target a future observation. It is **not** comparable to a mean bound.

    The distinction from a tolerance interval is worth stating. Equation (13)
    is a statement about coverage **averaged over the calibration draw**; a
    tolerance interval is a statement that holds for ``1 - delta`` of
    calibration draws. Conformal is marginally valid, Wilks is
    training-conditionally valid, and the second is the stronger statement.
    Both are reported in ``validation/validate_learned_model_output.txt``.
    """
    values = validate_gaps(calibration_gaps, bound_range)
    level = validate_confidence(confidence)
    limit = split_conformal_quantile(values, 1.0 - level)
    centre = float(np.mean(values))
    return Bound(
        method="split_conformal",
        target="quantile",
        estimate=centre,
        upper=limit,
        slack=limit - centre,
        confidence=level,
        proportion=level,
        n=int(values.size),
        unit=unit,
        distribution_free=True,
    )


def bootstrap_mean_upper(
    gaps: np.ndarray,
    bound_range: float,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
    kind: str = "percentile",
    unit: str = "",
) -> Bound:
    """One-sided bootstrap upper confidence limit on the mean gap.

    Parameters
    ----------
    gaps:
        ``(n,)`` discrepancies in ``[0, bound_range]``.
    bound_range:
        The declared ``R``.
    confidence:
        ``1 - alpha``.
    resamples:
        ``B``, the number of resamples. At least 100; the Monte Carlo error of
        the reported limit falls as ``1/sqrt(B)`` and is not the dominant error
        at ``n`` of tens.
    seed:
        Seed for ``numpy.random.default_rng``. The limit is a random variable
        given the data and this makes it reproducible.
    kind:
        ``"percentile"`` for equation (14) or ``"basic"`` for equation (15).
    unit:
        Unit string carried into the result.

    Returns
    -------
    Bound
        ``distribution_free=False``: the bootstrap's justification is
        asymptotic and nothing here bounds its finite-sample error.
    """
    values = validate_gaps(gaps, bound_range)
    level = validate_confidence(confidence)
    if kind not in {"percentile", "basic"}:
        raise ValueError(f"kind must be 'percentile' or 'basic'; got {kind!r}")
    count = int(resamples)
    if count < 100:
        raise ValueError(
            f"resamples must be at least 100; got {resamples}. Below that the "
            f"Monte Carlo error of the reported limit is larger than the difference "
            f"between the methods being compared."
        )
    rng = np.random.default_rng(int(seed))
    n = values.size
    draws = rng.integers(0, n, size=(count, n))
    means = values[draws].mean(axis=1)
    mean = float(np.mean(values))
    if kind == "percentile":
        limit = float(np.quantile(means, level))
    else:
        limit = float(2.0 * mean - np.quantile(means, 1.0 - level))
    return Bound(
        method=f"bootstrap_{kind}",
        target="mean",
        estimate=mean,
        upper=limit,
        slack=limit - mean,
        confidence=level,
        proportion=None,
        n=int(n),
        unit=unit,
        distribution_free=False,
    )
