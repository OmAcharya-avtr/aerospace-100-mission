"""Binomial confidence intervals for a rare-event verification campaign.

A verification campaign that runs ``n`` independent simulations and records
``k`` requirement violations observes a binomial count, and the quantity the
campaign has to defend is an interval for the violation probability ``p`` of the
*model that was simulated*.  This module implements the two intervals that are
actually defensible for that purpose and treats the ``k = 0`` case explicitly,
because a campaign aimed at a rare violation normally ends with ``k = 0``.

Notation and assumptions
------------------------
``k`` violations out of ``n`` independent and identically distributed runs,
each a Bernoulli trial with the same unknown probability ``p`` (dimensionless,
valid range ``0 <= p <= 1``).  Independence and identical distribution are
assumptions about the campaign, not properties of the arithmetic here: a
campaign that reuses a seed, that correlates runs through a shared stochastic
input, or that changes the model mid-campaign violates them and nothing in this
module detects that.

Methods
-------
Clopper-Pearson ("exact"): the interval obtained by inverting the binomial
tail probabilities.  For the equal-tailed two-sided interval at confidence
``1 - alpha``,

    lower = BetaInv(alpha / 2;     k,     n - k + 1),    lower = 0 if k = 0
    upper = BetaInv(1 - alpha / 2; k + 1, n - k),        upper = 1 if k = n

The Beta-quantile form is the standard identity between the binomial CDF and
the incomplete beta function.  Source: Clopper, C. J. and Pearson, E. S.
(1934), "The use of confidence or fiducial limits illustrated in the case of
the binomial", Biometrika 26(4):404-413.  Coverage is guaranteed to be at least
``1 - alpha`` for every ``p``, and is strictly greater than ``1 - alpha`` for
almost every ``p`` because the binomial count is discrete; the interval is
therefore conservative, which is the property that makes it usable as evidence.

Wilson score interval: the interval obtained by inverting the score test,

    centre     = (k + z^2 / 2) / (n + z^2)
    halfwidth  = z / (n + z^2) * sqrt(k (n - k) / n + z^2 / 4)

with ``z`` the standard normal quantile for the chosen side.  Source: Wilson,
E. B. (1927), "Probable inference, the law of succession, and statistical
inference", Journal of the American Statistical Association 22(158):209-212.
Wilson coverage oscillates about the nominal level and dips below it for some
``p``; see :func:`exact_coverage`, which computes the dip exactly rather than
by simulation.  Recommended over the Wald interval in Brown, L. D., Cai, T. T.
and DasGupta, A. (2001), "Interval estimation for a binomial proportion",
Statistical Science 16(2):101-133, and in Agresti, A. and Coull, B. A. (1998),
"Approximate is better than 'exact' for interval estimation of binomial
proportions", The American Statistician 52(2):119-126.

Zero-failure case
-----------------
At ``k = 0`` both intervals have a closed form:

    Clopper-Pearson one-sided upper  : 1 - alpha ** (1 / n)
    Clopper-Pearson two-sided upper  : 1 - (alpha / 2) ** (1 / n)
    Wilson two-sided upper           : z^2 / (n + z^2)

and both lower limits are exactly 0.  The one-sided Clopper-Pearson form is the
one a campaign should quote, and ``3 / n`` is its well-known approximation (the
"rule of three", Hanley, J. A. and Lippman-Hand, A. (1983), "If nothing goes
wrong, is everything all right? Interpreting zero numerators", JAMA
249(13):1743-1745); :func:`rule_of_three_upper` reports it and
validation/validate_intervals.py measures the approximation error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

SIDES = ("two-sided", "upper", "lower")
METHODS = ("clopper-pearson", "wilson")

__all__ = [
    "METHODS",
    "SIDES",
    "ProportionInterval",
    "clopper_pearson",
    "exact_coverage",
    "proportion_interval",
    "rule_of_three_upper",
    "wilson",
    "zero_failure_upper",
]


@dataclass(frozen=True)
class ProportionInterval:
    """A confidence interval for a binomial proportion (all values dimensionless).

    Attributes
    ----------
    k, n
        Violations observed and runs executed.
    lower, upper
        Interval limits, in probability units on ``[0, 1]``.
    confidence
        Nominal confidence level, e.g. ``0.95``.
    method
        ``"clopper-pearson"`` or ``"wilson"``.
    side
        ``"two-sided"``, ``"upper"`` (lower limit forced to 0) or ``"lower"``
        (upper limit forced to 1).
    point
        The maximum-likelihood point estimate ``k / n``.
    """

    k: int
    n: int
    lower: float
    upper: float
    confidence: float
    method: str
    side: str
    point: float

    @property
    def width(self) -> float:
        """Interval width ``upper - lower`` (dimensionless)."""
        return self.upper - self.lower

    @property
    def zero_failure(self) -> bool:
        """True when the campaign observed no violation."""
        return self.k == 0

    def contains(self, p: float) -> bool:
        """Whether probability ``p`` lies in the closed interval."""
        return self.lower <= p <= self.upper

    def describe(self) -> str:
        """One-line human-readable summary. Contains no I/O."""
        return (
            f"{self.method} {self.side} {self.confidence:.4g}: "
            f"k={self.k} n={self.n} p_hat={self.point:.6e} "
            f"[{self.lower:.6e}, {self.upper:.6e}]"
        )


def _validate_counts(k: int, n: int) -> tuple[int, int]:
    if isinstance(k, bool) or isinstance(n, bool):
        raise TypeError("k and n must be integers, not bool")
    if not isinstance(k, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise TypeError(f"k and n must be integers, got k={type(k).__name__}, n={type(n).__name__}")
    k = int(k)
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be a positive number of runs, got n={n}")
    if k < 0:
        raise ValueError(f"k must be a non-negative violation count, got k={k}")
    if k > n:
        raise ValueError(f"k must not exceed n; got k={k}, n={n}")
    return k, n


def _validate_confidence(confidence: float) -> float:
    confidence = float(confidence)
    if not math.isfinite(confidence) or not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must lie strictly in (0, 1), got {confidence}")
    return confidence


def _validate_side(side: str) -> str:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return side


def _tail_alpha(confidence: float, side: str) -> float:
    """Per-tail error probability for the requested side."""
    alpha = 1.0 - confidence
    return alpha / 2.0 if side == "two-sided" else alpha


def clopper_pearson(
    k: int, n: int, confidence: float = 0.95, side: str = "two-sided"
) -> ProportionInterval:
    """Clopper-Pearson interval for ``k`` violations in ``n`` runs.

    Parameters
    ----------
    k, n
        Violation count and run count (dimensionless integers, ``0 <= k <= n``).
    confidence
        Nominal confidence level in ``(0, 1)``.
    side
        ``"two-sided"``, ``"upper"`` or ``"lower"``.

    Returns
    -------
    ProportionInterval

    Notes
    -----
    Exact in the sense of guaranteed coverage, not in the sense of being short.
    The ``k = 0`` and ``k = n`` boundaries are set to 0 and 1 analytically
    rather than obtained from a Beta quantile with a zero shape parameter.
    """
    k, n = _validate_counts(k, n)
    confidence = _validate_confidence(confidence)
    side = _validate_side(side)
    tail = _tail_alpha(confidence, side)

    if side == "lower":
        upper = 1.0
    elif k == n:
        upper = 1.0
    else:
        upper = float(stats.beta.ppf(1.0 - tail, k + 1, n - k))

    if side == "upper":
        lower = 0.0
    elif k == 0:
        lower = 0.0
    else:
        lower = float(stats.beta.ppf(tail, k, n - k + 1))

    return ProportionInterval(
        k=k,
        n=n,
        lower=lower,
        upper=upper,
        confidence=confidence,
        method="clopper-pearson",
        side=side,
        point=k / n,
    )


def wilson(
    k: int, n: int, confidence: float = 0.95, side: str = "two-sided"
) -> ProportionInterval:
    """Wilson score interval for ``k`` violations in ``n`` runs.

    Parameters
    ----------
    k, n
        Violation count and run count (dimensionless integers, ``0 <= k <= n``).
    confidence
        Nominal confidence level in ``(0, 1)``.
    side
        ``"two-sided"``, ``"upper"`` or ``"lower"``.

    Returns
    -------
    ProportionInterval

    Notes
    -----
    No continuity correction is applied.  At ``k = 0`` the formula already
    returns ``lower = 0`` exactly and ``upper = z^2 / (n + z^2)``; the code
    clips to ``[0, 1]`` only to remove floating-point overshoot.
    """
    k, n = _validate_counts(k, n)
    confidence = _validate_confidence(confidence)
    side = _validate_side(side)
    tail = _tail_alpha(confidence, side)
    z = float(stats.norm.isf(tail))

    z2 = z * z
    denom = n + z2
    centre = (k + z2 / 2.0) / denom
    halfwidth = (z / denom) * math.sqrt(k * (n - k) / n + z2 / 4.0)

    lower = 0.0 if side == "upper" else min(max(centre - halfwidth, 0.0), 1.0)
    upper = 1.0 if side == "lower" else min(max(centre + halfwidth, 0.0), 1.0)

    return ProportionInterval(
        k=k,
        n=n,
        lower=lower,
        upper=upper,
        confidence=confidence,
        method="wilson",
        side=side,
        point=k / n,
    )


def proportion_interval(
    k: int,
    n: int,
    confidence: float = 0.95,
    method: str = "clopper-pearson",
    side: str = "two-sided",
) -> ProportionInterval:
    """Dispatch to :func:`clopper_pearson` or :func:`wilson` by name."""
    if method == "clopper-pearson":
        return clopper_pearson(k, n, confidence=confidence, side=side)
    if method == "wilson":
        return wilson(k, n, confidence=confidence, side=side)
    raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def zero_failure_upper(
    n: int, confidence: float = 0.95, method: str = "clopper-pearson", side: str = "upper"
) -> float:
    """Closed-form upper confidence limit when no violation was observed.

    Parameters
    ----------
    n
        Runs executed (dimensionless integer, ``n >= 1``).
    confidence
        Nominal confidence level in ``(0, 1)``.
    method
        ``"clopper-pearson"`` or ``"wilson"``.
    side
        ``"upper"`` for the one-sided limit (the one a campaign should quote) or
        ``"two-sided"`` for the upper limit of the equal-tailed interval.

    Returns
    -------
    float
        Upper confidence limit on the violation probability, dimensionless.

    Notes
    -----
    For Clopper-Pearson this is ``1 - tail ** (1 / n)`` with ``tail`` the
    per-tail error probability, which is the ``k = 0`` specialisation of the
    Beta-quantile formula because ``BetaInv(q; 1, n) = 1 - (1 - q) ** (1 / n)``.
    The closed form and the general code path are checked against each other in
    validation/validate_intervals.py.
    """
    _validate_counts(0, n)
    confidence = _validate_confidence(confidence)
    if side not in ("upper", "two-sided"):
        raise ValueError(
            "side must be 'upper' or 'two-sided' for a zero-failure bound, got "
            f"{side!r}"
        )
    tail = _tail_alpha(confidence, side)
    if method == "clopper-pearson":
        return 1.0 - tail ** (1.0 / n)
    if method == "wilson":
        z2 = float(stats.norm.isf(tail)) ** 2
        return z2 / (n + z2)
    raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def rule_of_three_upper(n: int) -> float:
    """The ``3 / n`` approximation to the 95 % one-sided zero-failure bound.

    Provided so that the approximation error can be measured rather than
    assumed; see validation/validate_intervals.py.  Dimensionless.
    """
    _validate_counts(0, n)
    return 3.0 / n


def exact_coverage(
    n: int,
    p: float,
    confidence: float = 0.95,
    method: str = "clopper-pearson",
    side: str = "two-sided",
) -> float:
    """Exact coverage probability of an interval method at a given true ``p``.

    Computed by summing the binomial probability mass over every ``k`` whose
    interval contains ``p``; this is an exact evaluation, not a simulation, so
    it carries no Monte-Carlo error.

    Parameters
    ----------
    n
        Runs per campaign (dimensionless integer, ``n >= 1``).
    p
        True violation probability in ``[0, 1]``.
    confidence, method, side
        Passed to :func:`proportion_interval`.

    Returns
    -------
    float
        Coverage probability in ``[0, 1]``.  Values below ``confidence``
        indicate the method under-covers at this ``p``.
    """
    _validate_counts(0, n)
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must lie in [0, 1], got {p}")
    ks = np.arange(n + 1)
    pmf = stats.binom.pmf(ks, n, p)
    covered = np.array(
        [
            proportion_interval(
                int(k), n, confidence=confidence, method=method, side=side
            ).contains(p)
            for k in ks
        ]
    )
    return float(pmf[covered].sum())
