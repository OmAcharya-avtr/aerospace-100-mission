"""Distribution-free bounds on the sim-to-field gap, and their sample-count inversions.

This module is **baseline 1**. It is implemented and validated before any
learned component exists, and the learned component is benchmarked against it
on the same held-out data. Everything here is conservative and says so.

Two different targets, kept apart
---------------------------------
A bound on the **mean** gap and a bound covering a **future** gap are not the
same object, and reporting them in one column is the most common way to make a
tolerance interval look tighter than a confidence interval when it is simply
answering an easier question. They are separated throughout:

``target = "mean"``
    An upper confidence bound on ``mu = E[d]``, the mean discrepancy over the
    operating-condition population. Hoeffding and empirical Bernstein.

``target = "quantile"``
    An upper tolerance limit that covers at least a declared proportion ``p``
    of the discrepancy population, with confidence ``1 - delta``. Wilks'
    nonparametric limit and the exact normal-theory limit.

Equations, with sources, units and validity ranges
--------------------------------------------------
**(2) Hoeffding's inequality.** Hoeffding, W. (1963), "Probability inequalities
for sums of bounded random variables", *Journal of the American Statistical
Association* 58(301), 13-30, Theorem 2. For independent ``X_i`` with
``a_i <= X_i <= b_i``,

    P(Xbar - E[Xbar] >= t) <= exp(-2 n^2 t^2 / sum_i (b_i - a_i)^2).

With every ``d_i`` in ``[0, R]`` the denominator is ``n R^2``, and setting the
right-hand side to ``delta`` and solving for ``t`` gives the one-sided upper
confidence bound

    mu <= dbar + R sqrt( ln(1/delta) / (2 n) )          with prob >= 1-delta. (3)

Units: ``dbar``, ``R`` and the bound are in the unit of the measured quantity.
Validity: independence, and ``d_i`` in ``[0, R]`` with ``R`` declared in
advance. The slack in (3) does not depend on the data at all, which is both
its weakness and the reason its sample-count inversion is exact.

**(4) The empirical Bernstein bound.** Maurer, A. and Pontil, M. (2009),
"Empirical Bernstein bounds and sample-variance penalization", *Proceedings of
the 22nd Conference on Learning Theory (COLT 2009)*, Theorem 4. For i.i.d.
``X_i`` in ``[0, 1]`` and ``n >= 2``, with probability at least ``1 - delta``,

    E[X] <= Xbar + sqrt( 2 V_n ln(2/delta) / n ) + 7 ln(2/delta) / (3 (n-1)),

``V_n`` being the unbiased sample variance. The inequality is positively
homogeneous in the scale of the variable, so for ``d_i`` in ``[0, R]`` the same
statement reads

    mu <= dbar + sqrt( 2 V_n ln(2/delta) / n ) + 7 R ln(2/delta) / (3 (n-1))  (5)

with ``V_n`` in squared units of the measurement. Validity: i.i.d., bounded in
``[0, R]``, ``n >= 2``. Equation (5) beats (3) whenever the sample variance is
small relative to ``R^2``, which is the usual case for a twin that is roughly
right, and LOSES to (3) at very small ``n`` because of the ``1/(n-1)`` term.
Both are computed and the tighter is reported as such, never silently.

**(6) Wilks' nonparametric one-sided tolerance limit.** Wilks, S. S. (1941),
"Determination of sample sizes for setting tolerance limits", *Annals of
Mathematical Statistics* 12(1), 91-96. For ``n`` i.i.d. draws from any
continuous distribution, the population proportion below the largest order
statistic ``d_(n)`` has a ``Beta(n, 1)`` distribution, so

    P( F(d_(n)) >= p ) = 1 - p^n,

and ``d_(n)`` is an upper tolerance limit for proportion ``p`` at confidence
``1 - delta`` exactly when ``p^n <= delta``, that is when

    n >= ln(delta) / ln(p).                                                 (7)

Equation (7) is the closed-form sample-count requirement and it is exact, not
asymptotic. Validity: continuity of the discrepancy distribution, i.i.d.
draws. The extension to several order statistics and to multivariate regions
is Wald, A. (1943), "An extension of Wilks' method for setting tolerance
limits", *Annals of Mathematical Statistics* 14(1), 45-55; this package uses
only the largest order statistic, where the inversion is closed form.

**(8) The exact normal-theory one-sided tolerance factor.** Krishnamoorthy, K.
and Mathew, T. (2009), *Statistical Tolerance Regions: Theory, Applications,
and Computation*, Wiley, chapter 1. If the ``d_i`` are normal with unknown
mean and variance, the limit ``dbar + k s`` covers proportion ``p`` with
confidence ``1 - delta`` when

    k = t_{n-1, 1-delta}( z_p sqrt(n) ) / sqrt(n),                          (9)

``t_{nu, q}(ncp)`` being the ``q``-quantile of the non-central t distribution
with ``nu`` degrees of freedom and non-centrality ``ncp``, and ``z_p`` the
standard normal ``p``-quantile. Validity: normality. A discrepancy magnitude
is a folded quantity and is NOT normal, so (8) is shipped as the parametric
comparison it is, and its measured coverage on this package's own synthetic
field sets is reported in ``validation/validate_bounds_output.txt`` rather
than asserted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .discrepancy import validate_confidence, validate_gaps

__all__ = [
    "Bound",
    "empirical_bernstein_mean_upper",
    "empirical_bernstein_slack",
    "hoeffding_mean_upper",
    "normal_tolerance_required_n",
    "normal_tolerance_upper",
    "required_n_empirical_bernstein",
    "required_n_hoeffding",
    "required_n_wilks",
    "tolerance_factor_normal",
    "wilks_tolerance_upper",
]


@dataclass(frozen=True)
class Bound:
    """One upper bound on the gap, with everything needed to read it honestly.

    Attributes
    ----------
    method:
        Short name of the bound, e.g. ``"hoeffding"``.
    target:
        ``"mean"`` for a bound on ``E[d]``, ``"quantile"`` for a tolerance
        limit covering a proportion of the population. The two are never
        comparable and this field is why.
    estimate:
        The point estimate the bound is built on, in ``unit``: the sample mean
        for a mean bound, the sample location for a tolerance limit.
    upper:
        The upper bound itself, in ``unit``.
    slack:
        ``upper - estimate``, in ``unit``. The price of the confidence
        statement.
    confidence:
        ``1 - delta``, the confidence the bound is stated at.
    proportion:
        ``p`` for a tolerance limit; ``None`` for a mean bound.
    n:
        The number of paired measurements the bound was computed from.
    unit:
        The unit of ``estimate``, ``upper`` and ``slack``.
    distribution_free:
        ``True`` when the bound holds for every distribution on ``[0, R]``
        satisfying the stated independence assumption, ``False`` when it rests
        on a distributional assumption such as normality or on an asymptotic
        argument.
    """

    method: str
    target: str
    estimate: float
    upper: float
    slack: float
    confidence: float
    proportion: float | None
    n: int
    unit: str
    distribution_free: bool


def hoeffding_mean_upper(
    gaps: np.ndarray, bound_range: float, confidence: float = 0.95, unit: str = ""
) -> Bound:
    """Upper confidence bound on the mean gap by equation (3).

    Parameters
    ----------
    gaps:
        ``(n,)`` discrepancies in ``[0, bound_range]``, in ``unit``.
    bound_range:
        The declared range ``R``, in ``unit``, strictly positive.
    confidence:
        ``1 - delta``, strictly inside ``(0, 1)``.
    unit:
        Unit string carried into the result.

    Returns
    -------
    Bound
        ``target="mean"``, ``distribution_free=True``.

    Notes
    -----
    The slack ``R sqrt(ln(1/delta) / (2n))`` is a function of ``n``, ``R`` and
    ``delta`` only. It does not look at the data, so this bound is identical
    for a twin that is perfect and a twin that is wrong by ``R`` every time.
    That is what makes it trustworthy at ``n`` of tens and what makes it loose.
    """
    values = validate_gaps(gaps, bound_range)
    level = validate_confidence(confidence)
    delta = 1.0 - level
    n = values.size
    slack = float(bound_range) * math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    mean = float(np.mean(values))
    return Bound(
        method="hoeffding",
        target="mean",
        estimate=mean,
        upper=mean + slack,
        slack=slack,
        confidence=level,
        proportion=None,
        n=int(n),
        unit=unit,
        distribution_free=True,
    )


def empirical_bernstein_mean_upper(
    gaps: np.ndarray, bound_range: float, confidence: float = 0.95, unit: str = ""
) -> Bound:
    """Upper confidence bound on the mean gap by equation (5).

    Parameters
    ----------
    gaps:
        ``(n,)`` discrepancies in ``[0, bound_range]``, ``n >= 2``, in ``unit``.
    bound_range:
        The declared range ``R``, in ``unit``.
    confidence:
        ``1 - delta``, strictly inside ``(0, 1)``.
    unit:
        Unit string carried into the result.

    Raises
    ------
    ValueError
        If ``n < 2``: equation (5) has an ``n - 1`` in a denominator and the
        unbiased sample variance is undefined at ``n = 1``.
    """
    values = validate_gaps(gaps, bound_range)
    level = validate_confidence(confidence)
    delta = 1.0 - level
    n = values.size
    if n < 2:
        raise ValueError(
            "the empirical Bernstein bound of equation (5) needs at least 2 "
            f"measurements; got {n}. Use hoeffding_mean_upper at n = 1."
        )
    log_term = math.log(2.0 / delta)
    variance = float(np.var(values, ddof=1))
    slack = math.sqrt(2.0 * variance * log_term / n) + 7.0 * float(bound_range) * log_term / (
        3.0 * (n - 1)
    )
    mean = float(np.mean(values))
    return Bound(
        method="empirical_bernstein",
        target="mean",
        estimate=mean,
        upper=mean + slack,
        slack=slack,
        confidence=level,
        proportion=None,
        n=int(n),
        unit=unit,
        distribution_free=True,
    )


def wilks_tolerance_upper(
    gaps: np.ndarray,
    bound_range: float,
    proportion: float = 0.95,
    confidence: float = 0.95,
    unit: str = "",
) -> Bound:
    """Wilks' nonparametric upper tolerance limit, equations (6) and (7).

    The limit is the largest order statistic ``d_(n)``. It covers at least
    ``proportion`` of the population with confidence at least ``confidence``
    **only when** ``n >= ln(delta)/ln(p)``; below that ``n`` the sample simply
    cannot support the statement and this function raises rather than
    returning a limit that does not mean what it says.

    Raises
    ------
    ValueError
        If ``n`` is below the requirement of equation (7). The message states
        the required ``n``.
    """
    values = validate_gaps(gaps, bound_range)
    level = validate_confidence(confidence)
    p = validate_confidence(proportion, "proportion")
    delta = 1.0 - level
    n = values.size
    needed = required_n_wilks(p, level)
    if n < needed:
        raise ValueError(
            f"Wilks' limit at proportion {p} and confidence {level} needs n >= {needed} "
            f"by equation (7) (ln({delta:.6g})/ln({p:.6g})); got n = {n}. The largest "
            f"of {n} observations does not cover {p} of the population at that "
            f"confidence, and returning it anyway would be the defect this package "
            f"exists to avoid."
        )
    limit = float(np.max(values))
    centre = float(np.mean(values))
    return Bound(
        method="wilks_tolerance",
        target="quantile",
        estimate=centre,
        upper=limit,
        slack=limit - centre,
        confidence=level,
        proportion=p,
        n=int(n),
        unit=unit,
        distribution_free=True,
    )


def tolerance_factor_normal(n: int, proportion: float, confidence: float) -> float:
    """The exact one-sided normal tolerance factor ``k`` of equation (9).

    Parameters
    ----------
    n:
        Sample size, ``n >= 2``.
    proportion:
        ``p``, the proportion of the population to cover, in ``(0, 1)``.
    confidence:
        ``1 - delta``, in ``(0, 1)``.

    Returns
    -------
    float
        Dimensionless ``k``; the limit is ``mean + k * s`` in the unit of the
        measurement.
    """
    if int(n) < 2:
        raise ValueError(f"the normal tolerance factor needs n >= 2; got {n}")
    p = validate_confidence(proportion, "proportion")
    level = validate_confidence(confidence)
    count = int(n)
    noncentrality = stats.norm.ppf(p) * math.sqrt(count)
    return float(stats.nct.ppf(level, df=count - 1, nc=noncentrality) / math.sqrt(count))


def normal_tolerance_upper(
    gaps: np.ndarray,
    bound_range: float,
    proportion: float = 0.95,
    confidence: float = 0.95,
    unit: str = "",
) -> Bound:
    """Exact normal-theory upper tolerance limit, equations (8) and (9).

    This bound is **not** distribution-free and the returned
    :attr:`Bound.distribution_free` says so. A discrepancy magnitude is a
    folded quantity with a hard floor at zero and is not normal; this limit is
    shipped as the parametric comparison and its measured coverage on the
    synthetic field sets of this package is reported, not assumed.
    """
    values = validate_gaps(gaps, bound_range)
    level = validate_confidence(confidence)
    p = validate_confidence(proportion, "proportion")
    n = values.size
    if n < 2:
        raise ValueError(f"the normal tolerance limit needs n >= 2; got {n}")
    mean = float(np.mean(values))
    deviation = float(np.std(values, ddof=1))
    factor = tolerance_factor_normal(n, p, level)
    limit = mean + factor * deviation
    return Bound(
        method="normal_tolerance",
        target="quantile",
        estimate=mean,
        upper=limit,
        slack=limit - mean,
        confidence=level,
        proportion=p,
        n=int(n),
        unit=unit,
        distribution_free=False,
    )


def required_n_hoeffding(bound_range: float, target_slack: float, confidence: float) -> int:
    """Smallest ``n`` whose Hoeffding slack is at most ``target_slack``.

    Inverting equation (3) for ``n`` is exact because the slack does not depend
    on the data:

        n >= R^2 ln(1/delta) / (2 eps^2).                                  (10)

    Parameters
    ----------
    bound_range:
        The declared range ``R``, strictly positive, in the measurement unit.
    target_slack:
        ``eps``, the slack to reach, strictly positive, same unit as ``R``.
    confidence:
        ``1 - delta``.

    Returns
    -------
    int
        ``ceil`` of the right-hand side of (10), at least 1.
    """
    if not np.isfinite(bound_range) or bound_range <= 0.0:
        raise ValueError(f"bound_range must be finite and positive; got {bound_range!r}")
    if not np.isfinite(target_slack) or target_slack <= 0.0:
        raise ValueError(
            f"target_slack must be finite and positive; got {target_slack!r}. A target "
            f"of zero needs an infinite sample."
        )
    level = validate_confidence(confidence)
    delta = 1.0 - level
    exact = float(bound_range) ** 2 * math.log(1.0 / delta) / (2.0 * float(target_slack) ** 2)
    return max(1, int(math.ceil(exact)))


def required_n_empirical_bernstein(
    variance: float,
    bound_range: float,
    target_slack: float,
    confidence: float,
) -> int:
    """Smallest ``n`` whose empirical Bernstein slack is at most ``target_slack``.

    The slack of equation (5) is ``a / sqrt(n) + b / (n - 1)`` with
    ``a = sqrt(2 V ln(2/delta))`` and ``b = 7 R ln(2/delta) / 3``. Replacing
    ``n - 1`` by ``n`` makes it a quadratic in ``u = 1 / sqrt(n)``,

        b u^2 + a u - eps = 0,   u = (-a + sqrt(a^2 + 4 b eps)) / (2 b),   (11)

    whose root gives ``n = 1/u^2`` in closed form. The ``n - 1`` neglected in
    (11) makes that value very slightly optimistic, so this function then walks
    ``n`` upward against the TRUE slack expression until it is satisfied. The
    correction is at most a handful of samples and
    ``validation/validate_inversion_output.txt`` reports the measured size of
    it; the walk, not the closed form, is what the returned number satisfies.

    Parameters
    ----------
    variance:
        A plug-in for ``V_n``, in squared measurement units. Non-negative.
        This is the one place in this module where a bound rests on an
        estimate: ``V_n`` is random, so the returned ``n`` is a prediction and
        is verified operationally rather than being asserted.
    bound_range:
        The declared ``R``.
    target_slack:
        ``eps``, strictly positive.
    confidence:
        ``1 - delta``.
    """
    if not np.isfinite(variance) or variance < 0.0:
        raise ValueError(f"variance must be finite and non-negative; got {variance!r}")
    if not np.isfinite(bound_range) or bound_range <= 0.0:
        raise ValueError(f"bound_range must be finite and positive; got {bound_range!r}")
    if not np.isfinite(target_slack) or target_slack <= 0.0:
        raise ValueError(f"target_slack must be finite and positive; got {target_slack!r}")
    level = validate_confidence(confidence)
    log_term = math.log(2.0 / (1.0 - level))
    a = math.sqrt(2.0 * float(variance) * log_term)
    b = 7.0 * float(bound_range) * log_term / 3.0
    eps = float(target_slack)
    u = (-a + math.sqrt(a * a + 4.0 * b * eps)) / (2.0 * b)
    n = max(2, int(math.ceil(1.0 / (u * u))))
    while empirical_bernstein_slack(n, variance, bound_range, level) > eps:
        n += 1
        if n > 10**9:  # pragma: no cover - unreachable for a positive eps
            raise ValueError("the empirical Bernstein slack did not reach the target")
    return n


def empirical_bernstein_slack(
    n: int, variance: float, bound_range: float, confidence: float
) -> float:
    """The slack term of equation (5) at a given ``n`` and plug-in variance.

    Units: same as ``bound_range``. Exposed so that the inversion of
    :func:`required_n_empirical_bernstein` can be checked against the forward
    expression by anything that wants to, including the test suite.
    """
    count = int(n)
    if count < 2:
        raise ValueError(f"equation (5) needs n >= 2; got {n}")
    level = validate_confidence(confidence)
    log_term = math.log(2.0 / (1.0 - level))
    return math.sqrt(2.0 * float(variance) * log_term / count) + 7.0 * float(
        bound_range
    ) * log_term / (3.0 * (count - 1))


def required_n_wilks(proportion: float, confidence: float) -> int:
    """Smallest ``n`` for Wilks' limit, equation (7): ``ceil(ln(delta)/ln(p))``.

    Exact and closed form. At ``p = 0.95`` and ``1 - delta = 0.95`` this is
    ``ceil(ln(0.05)/ln(0.95)) = ceil(58.404) = 59``, the classical result.
    """
    p = validate_confidence(proportion, "proportion")
    level = validate_confidence(confidence)
    delta = 1.0 - level
    return max(1, int(math.ceil(math.log(delta) / math.log(p))))


def normal_tolerance_required_n(
    proportion: float, confidence: float, factor_target: float, max_n: int = 100000
) -> int:
    """Smallest ``n`` whose normal tolerance factor ``k`` is at most ``factor_target``.

    Equation (9) has no closed-form inverse, so this is a monotone search. It
    is here so that the normal-theory limit can be put in the same
    sample-count table as the distribution-free ones, with the fact that its
    inversion is a SEARCH and not a closed form visible in the table.
    """
    if not np.isfinite(factor_target) or factor_target <= 0.0:
        raise ValueError(f"factor_target must be finite and positive; got {factor_target!r}")
    p = validate_confidence(proportion, "proportion")
    level = validate_confidence(confidence)
    low, high = 2, 2
    while tolerance_factor_normal(high, p, level) > factor_target:
        low = high
        high *= 2
        if high > max_n:
            raise ValueError(
                f"no n <= {max_n} reaches a tolerance factor of {factor_target}; the "
                f"factor is bounded below by the normal quantile z_p = "
                f"{stats.norm.ppf(p):.6g}"
            )
    while low < high:
        middle = (low + high) // 2
        if tolerance_factor_normal(max(2, middle), p, level) > factor_target:
            low = middle + 1
        else:
            high = middle
    return max(2, int(low))
