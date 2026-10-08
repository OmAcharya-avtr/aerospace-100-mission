"""Sample-size planning for a rare-event verification campaign.

The question this module answers: given a target violation probability and a
confidence requirement, how many independent runs does the campaign need.
There are two distinct questions hidden in that sentence and the module keeps
them separate, because conflating them is the usual planning error.

1. *Demonstration*: "if the campaign sees no violation, I want to be able to
   state ``p <= p_target`` at confidence ``1 - alpha``."  The answer is exact
   and closed-form.  The one-sided Clopper-Pearson upper limit after ``k = 0``
   violations is ``1 - alpha ** (1 / n)``; requiring that limit to be at most
   ``p_target`` gives

       n >= ln(alpha) / ln(1 - p_target)

   so ``n = ceil(ln(alpha) / ln(1 - p_target))``.  Source: the ``k = 0``
   specialisation of Clopper, C. J. and Pearson, E. S. (1934), Biometrika
   26(4):404-413.  Dimensionless; valid for ``0 < p_target < 1`` and
   ``0 < alpha < 1``.

2. *Estimation*: "I want an interval whose width is at most a stated fraction
   of ``p``."  This needs an assumed ``p`` and has no closed form once the
   discreteness of ``k`` is respected, so :func:`samples_for_relative_width`
   searches.  The Wald starting guess

       n0 = 4 z^2 (1 - p) / (p * relative_width ** 2)

   comes from equating the Wald half-width ``z sqrt(p(1-p)/n)`` to
   ``relative_width * p / 2``; it is only a starting point and the returned
   value is checked against the chosen exact or score interval.

Both questions assume ``n`` independent and identically distributed runs of the
*same model*.  A plan computed here says nothing about how well that model
represents any vehicle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .intervals import METHODS, proportion_interval, zero_failure_upper

__all__ = [
    "CampaignPlan",
    "detection_probability",
    "expected_violations",
    "plan_campaign",
    "probability_of_zero_violations",
    "samples_for_relative_width",
    "samples_for_zero_failure_bound",
]


def _validate_probability(p: float, name: str) -> float:
    p = float(p)
    if not math.isfinite(p) or not (0.0 < p < 1.0):
        raise ValueError(f"{name} must lie strictly in (0, 1), got {p}")
    return p


def samples_for_zero_failure_bound(
    p_target: float, confidence: float = 0.95, method: str = "clopper-pearson"
) -> int:
    """Runs needed so that observing zero violations demonstrates ``p <= p_target``.

    Parameters
    ----------
    p_target
        Target violation probability to be bounded, in ``(0, 1)``, dimensionless.
    confidence
        One-sided confidence level in ``(0, 1)``.
    method
        ``"clopper-pearson"`` (exact, closed form) or ``"wilson"`` (score,
        solved from ``z^2 / (n + z^2) <= p_target``).

    Returns
    -------
    int
        Smallest ``n`` whose one-sided zero-failure upper limit is at most
        ``p_target``.

    Notes
    -----
    Clopper-Pearson: ``n = ceil(ln(alpha) / ln(1 - p_target))``.  The result is
    then verified against :func:`rareverify.intervals.zero_failure_upper` and
    incremented if floating-point rounding leaves it one short, so the returned
    value always satisfies the requirement.
    """
    p_target = _validate_probability(p_target, "p_target")
    confidence = _validate_probability(confidence, "confidence")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    alpha = 1.0 - confidence

    if method == "clopper-pearson":
        n = max(1, math.ceil(math.log(alpha) / math.log1p(-p_target)))
    else:
        from scipy import stats

        z2 = float(stats.norm.isf(alpha)) ** 2
        n = max(1, math.ceil(z2 * (1.0 - p_target) / p_target))

    while zero_failure_upper(n, confidence=confidence, method=method, side="upper") > p_target:
        n += 1
    while n > 1 and zero_failure_upper(
        n - 1, confidence=confidence, method=method, side="upper"
    ) <= p_target:
        n -= 1
    return n


def samples_for_relative_width(
    p: float,
    relative_width: float = 0.5,
    confidence: float = 0.95,
    method: str = "clopper-pearson",
    max_samples: int = 2_000_000_000,
    back_scan: int = 512,
) -> int:
    """Runs needed for an interval no wider than ``relative_width * p``.

    Parameters
    ----------
    p
        Assumed violation probability, in ``(0, 1)``, dimensionless.  The plan
        is only as good as this assumption.
    relative_width
        Target interval width as a fraction of ``p``; ``0.5`` means the
        two-sided interval must be no wider than half of ``p``.  Must be > 0.
    confidence
        Nominal two-sided confidence level in ``(0, 1)``.
    method
        ``"clopper-pearson"`` or ``"wilson"``.
    max_samples
        Refuse to plan beyond this many runs; raises ``ValueError`` instead of
        returning a number nobody can execute.
    back_scan
        How many steps to scan downward from the bracketed solution looking for
        a smaller feasible ``n``.

    Returns
    -------
    int
        Number of runs.

    Notes
    -----
    The feasibility predicate uses ``k = round(n * p)`` violations, which makes
    it mildly non-monotone in ``n``: incrementing ``n`` can move ``k`` and
    widen the interval.  The search therefore brackets geometrically, bisects,
    and then scans downward up to ``back_scan`` steps.  The returned ``n``
    always satisfies the predicate, but minimality is only guaranteed within
    the scan window.  validation/validate_planner.py measures how often the
    predicate is non-monotone.
    """
    p = _validate_probability(p, "p")
    confidence = _validate_probability(confidence, "confidence")
    relative_width = float(relative_width)
    if not math.isfinite(relative_width) or relative_width <= 0.0:
        raise ValueError(f"relative_width must be positive, got {relative_width}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if max_samples < 1:
        raise ValueError(f"max_samples must be at least 1, got {max_samples}")
    if back_scan < 0:
        raise ValueError(f"back_scan must be non-negative, got {back_scan}")

    target = relative_width * p

    def feasible(n: int) -> bool:
        k = int(round(n * p))
        interval = proportion_interval(
            min(k, n), n, confidence=confidence, method=method, side="two-sided"
        )
        return interval.width <= target

    from scipy import stats

    z = float(stats.norm.isf((1.0 - confidence) / 2.0))
    guess = 4.0 * z * z * (1.0 - p) / (p * relative_width * relative_width)
    hi = min(max(2, int(guess * 2)), max_samples)
    lo = max(1, min(int(guess / 8), hi))

    while not feasible(hi):
        if hi >= max_samples:
            raise ValueError(
                f"relative_width={relative_width} at p={p} needs more than "
                f"max_samples={max_samples} runs"
            )
        hi = min(hi * 2, max_samples)
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    n = lo
    if n > max_samples:
        raise ValueError(
            f"relative_width={relative_width} at p={p} needs {n} runs, more than "
            f"max_samples={max_samples}"
        )
    for candidate in range(max(1, n - back_scan), n):
        if feasible(candidate):
            n = candidate
            break
    return n


def expected_violations(n: int, p: float) -> float:
    """Expected violation count ``n * p`` (dimensionless)."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    p = _validate_probability(p, "p")
    return n * p


def probability_of_zero_violations(n: int, p: float) -> float:
    """Probability ``(1 - p) ** n`` that a campaign of ``n`` runs sees nothing."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    p = _validate_probability(p, "p")
    return math.exp(n * math.log1p(-p))


def detection_probability(n: int, p: float) -> float:
    """Probability ``1 - (1 - p) ** n`` of at least one violation in ``n`` runs."""
    return -math.expm1(n * math.log1p(-_validate_probability(p, "p"))) if n >= 1 else 0.0


@dataclass(frozen=True)
class CampaignPlan:
    """A planned verification campaign.

    All probabilities are dimensionless and refer to the simulated model.

    Attributes
    ----------
    p_target
        The violation probability the campaign is required to bound.
    confidence
        Confidence level the bound is quoted at.
    method
        Interval method the plan was computed for.
    n_demonstration
        Runs needed so that zero observed violations demonstrates
        ``p <= p_target``.
    n_estimation
        Runs needed for a two-sided interval of width
        ``relative_width * p_target``, assuming ``p = p_target``.
    relative_width
        The relative width ``n_estimation`` was computed for.
    expected_violations_demonstration
        ``n_demonstration * p_target``: the number of violations a campaign of
        the demonstration size would see if ``p`` really equals ``p_target``.
    probability_zero_violations_demonstration
        ``(1 - p_target) ** n_demonstration``: the probability that such a
        campaign ends with the zero-failure case, which is ``alpha`` by
        construction.
    zero_failure_bound
        The upper limit the campaign would actually quote after zero violations
        at ``n_demonstration`` runs.
    """

    p_target: float
    confidence: float
    method: str
    n_demonstration: int
    n_estimation: int
    relative_width: float
    expected_violations_demonstration: float
    probability_zero_violations_demonstration: float
    zero_failure_bound: float

    def describe(self) -> str:
        """Multi-line human-readable summary. Contains no I/O."""
        return "\n".join(
            [
                f"target violation probability : {self.p_target:.6e}",
                f"confidence                   : {self.confidence:.6g}",
                f"interval method              : {self.method}",
                f"runs to demonstrate (k=0)    : {self.n_demonstration}",
                f"  upper bound if k=0         : {self.zero_failure_bound:.6e}",
                f"  E[violations] at p_target  : "
                f"{self.expected_violations_demonstration:.6f}",
                f"  P(k=0) at p_target         : "
                f"{self.probability_zero_violations_demonstration:.6f}",
                f"runs to estimate to {self.relative_width:.3g}*p : {self.n_estimation}",
            ]
        )


def plan_campaign(
    p_target: float,
    confidence: float = 0.95,
    relative_width: float = 0.5,
    method: str = "clopper-pearson",
    max_samples: int = 2_000_000_000,
) -> CampaignPlan:
    """Build a :class:`CampaignPlan` for a target violation probability.

    Parameters
    ----------
    p_target
        Violation probability to bound, in ``(0, 1)``, dimensionless.
    confidence
        Confidence level in ``(0, 1)``.
    relative_width
        Target interval width as a fraction of ``p_target`` for the estimation
        sample size.
    method
        ``"clopper-pearson"`` or ``"wilson"``.
    max_samples
        Cap for the estimation search.

    Returns
    -------
    CampaignPlan
    """
    n_demo = samples_for_zero_failure_bound(p_target, confidence=confidence, method=method)
    n_est = samples_for_relative_width(
        p_target,
        relative_width=relative_width,
        confidence=confidence,
        method=method,
        max_samples=max_samples,
    )
    return CampaignPlan(
        p_target=p_target,
        confidence=confidence,
        method=method,
        n_demonstration=n_demo,
        n_estimation=n_est,
        relative_width=relative_width,
        expected_violations_demonstration=expected_violations(n_demo, p_target),
        probability_zero_violations_demonstration=probability_of_zero_violations(
            n_demo, p_target
        ),
        zero_failure_bound=zero_failure_upper(
            n_demo, confidence=confidence, method=method, side="upper"
        ),
    )
