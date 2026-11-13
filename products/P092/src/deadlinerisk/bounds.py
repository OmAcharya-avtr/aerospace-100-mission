"""Bounds derived from mathematics this product's estimators do not use.

"Two implementations agree" and "the answer lies inside a bound derived from
different mathematics" are different checks, and the second catches what the
first cannot. Every bound here is computed from quantities the estimators in
:mod:`deadlinerisk.baselines` already produce, but by a route none of them
takes:

- :func:`markov_upper_bound` -- Markov's inequality. ``P(X > D) <= E[X]/D`` for
  a non-negative ``X``. Uses the first moment and nothing else: no normality, no
  variance, no extreme-value limit. It upper-bounds the **executed-job** miss
  rate, which is a counted quantity, so the check is a counted number against a
  moment bound.
- :func:`cantelli_upper_bound` -- the one-sided Chebyshev (Cantelli) inequality,
  ``P(X - mu >= t) <= s^2 / (s^2 + t^2)`` for ``t > 0``. Uses the first two
  moments and no distributional form. The mean-plus-margin rule's Gaussian tail
  must lie at or below it for any mean and variance, which is a statement about
  the rule itself that holds independently of any data.
- :func:`bonferroni_pair_bracket` -- the union bound and the monotonicity of
  probability, applied to consecutive release pairs:
  ``max(p1, p2) <= P(miss in either) <= p1 + p2``. Nothing in the miss
  accounting uses either inequality, and the bracket is exact in exact
  arithmetic, so a violation is a defect and not a tolerance question.

Each bound ships with a slack function. **A slack follows the CONDITIONING of
the bound expression**, not an operation count and not an absolute constant: a
fixed floor makes a narrow bracket vacuous, which is how an earlier product of
this mission admitted the very expression its check existed to catch.

References
----------
Markov's and Chebyshev's inequalities, and the one-sided Cantelli form: Feller
1968, "An Introduction to Probability Theory and Its Applications", Volume 1,
3rd edition, chapter 9. The union bound as Bonferroni's first inequality: same
volume, chapter 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = float(np.finfo(np.float64).eps)


def markov_upper_bound(mean_s: float, deadline_s: float) -> float:
    """``E[X]/D``: Markov's upper bound on ``P(X > D)`` for non-negative ``X``.

    Dimensionless, clipped above at 1. ``mean_s`` and ``deadline_s`` are
    seconds.

    Raises
    ------
    ValueError
        On a negative mean or a non-positive deadline.
    """
    if not np.isfinite(mean_s) or mean_s < 0.0:
        raise ValueError(f"mean_s must be finite and >= 0, got {mean_s!r}")
    if not np.isfinite(deadline_s) or deadline_s <= 0.0:
        raise ValueError(f"deadline_s must be finite and > 0, got {deadline_s!r}")
    return float(min(1.0, mean_s / deadline_s))


def markov_slack(mean_s: float, deadline_s: float) -> float:
    """Slack for a :func:`markov_upper_bound` comparison.

    The bound is a quotient of two positive quantities, so its relative
    condition number is 1 in each argument and the only amplification is the
    division itself. The slack is therefore a few eps of the bound's own
    magnitude, and vanishes as the bound does rather than being floored.
    """
    return 4.0 * EPS * markov_upper_bound(mean_s, deadline_s)


def cantelli_upper_bound(mean_s: float, sd_s: float, deadline_s: float) -> float:
    """``s^2/(s^2 + t^2)`` with ``t = D - mean``: Cantelli's one-sided bound.

    Valid only for ``t > 0``, i.e. a deadline above the mean response. For
    ``t <= 0`` the inequality is vacuous and 1.0 is returned, which is the
    correct trivial bound.

    Raises
    ------
    ValueError
        On a negative standard deviation or a non-positive deadline.
    """
    if not np.isfinite(sd_s) or sd_s < 0.0:
        raise ValueError(f"sd_s must be finite and >= 0, got {sd_s!r}")
    if not np.isfinite(deadline_s) or deadline_s <= 0.0:
        raise ValueError(f"deadline_s must be finite and > 0, got {deadline_s!r}")
    t = deadline_s - mean_s
    if t <= 0.0:
        return 1.0
    if sd_s == 0.0:
        return 0.0
    return float(sd_s**2 / (sd_s**2 + t**2))


def cantelli_slack(mean_s: float, sd_s: float, deadline_s: float) -> float:
    """Slack for a :func:`cantelli_upper_bound` comparison.

    The ill-conditioned step is ``t = D - mean``, whose relative condition
    number is ``(|D| + |mean|) / |D - mean|``. The bound depends on ``t``
    through ``t^2`` in a denominator, so the relative perturbation of the bound
    is at most twice that of ``t``, and the slack is that relative figure times
    the bound's own magnitude. It therefore tightens as the bound shrinks and
    loosens only when the deadline sits close to the mean, which is exactly
    where the subtraction loses digits.
    """
    bound = cantelli_upper_bound(mean_s, sd_s, deadline_s)
    t = deadline_s - mean_s
    if t <= 0.0:
        # The bound is the trivial 1.0 and carries one rounding at most.
        return 4.0 * EPS
    condition = (abs(deadline_s) + abs(mean_s)) / abs(t)
    # No floor. A slack floored at an absolute constant makes a narrow
    # comparison vacuous, which is how an earlier product of this mission
    # admitted the very expression its check existed to catch. This slack is
    # proportional to the bound, so it vanishes with it.
    return 8.0 * EPS * condition * bound


@dataclass(frozen=True)
class PairBracket:
    """A Bonferroni bracket on the pair-miss rate of a release sequence.

    Attributes
    ----------
    p_first, p_second
        Miss rate over positions ``0..n-2`` and ``1..n-1``, dimensionless.
    observed
        Measured fraction of overlapping consecutive pairs containing at least
        one miss, dimensionless.
    lower
        ``max(p_first, p_second)`` -- monotonicity of probability.
    upper
        ``min(1, p_first + p_second)`` -- the union bound.
    slack
        Tolerance for the comparison, from the conditioning of the two bound
        expressions.
    inside
        Whether ``lower - slack <= observed <= upper + slack``.
    """

    n_pairs: int
    p_first: float
    p_second: float
    observed: float
    lower: float
    upper: float
    slack: float
    inside: bool


def bonferroni_pair_bracket(indicator: np.ndarray) -> PairBracket:
    """Bracket the consecutive-pair miss rate from the two marginal rates.

    For a 0/1 sequence, the empirical mean of ``max(a_i, a_{i+1})`` over the
    ``n-1`` overlapping pairs satisfies ``max(p1, p2) <= q <= p1 + p2``, where
    ``p1`` and ``p2`` are the empirical means of the left and right members.
    Both inequalities are exact in exact arithmetic: the lower one because the
    maximum of two indicators is at least either, the upper one because their
    union's measure is at most the sum of theirs.

    Raises
    ------
    ValueError
        On a sequence shorter than 2, or on a value that is not 0 or 1.
    """
    series = np.asarray(indicator)
    if series.ndim != 1:
        raise ValueError(f"indicator must be one-dimensional, got shape {series.shape}")
    if series.size < 2:
        raise ValueError(f"indicator must have at least 2 entries, got {series.size}")
    values = series.astype(np.int64)
    if not np.all(np.isin(values, (0, 1))):
        raise ValueError("indicator must contain only 0 and 1")
    left = values[:-1]
    right = values[1:]
    n_pairs = int(left.size)
    p_first = float(left.sum()) / n_pairs
    p_second = float(right.sum()) / n_pairs
    observed = float(np.maximum(left, right).sum()) / n_pairs
    lower = max(p_first, p_second)
    upper = min(1.0, p_first + p_second)
    # Conditioning: every quantity is a sum of 0/1 over the same denominator, so
    # each is exact to one rounding of the division. The upper bound adds two
    # such numbers, which is well conditioned because both are non-negative, so
    # the relative perturbation of the sum is bounded by one eps. The slack is
    # a few eps of the largest magnitude entering the comparison, and so is zero
    # when the sequence has no misses at all.
    slack = 4.0 * EPS * max(upper, observed, lower)
    return PairBracket(
        n_pairs=n_pairs,
        p_first=p_first,
        p_second=p_second,
        observed=observed,
        lower=float(lower),
        upper=float(upper),
        slack=float(slack),
        inside=bool(lower - slack <= observed <= upper + slack),
    )
