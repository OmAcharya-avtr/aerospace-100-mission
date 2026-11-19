"""Bounds on total wasted time derived from mathematics the accumulator does not use.

"Two implementations agree" and "the answer lies inside a bound derived from
different mathematics" are not the same check, and the second catches what the
first cannot. An earlier product of this mission had 777 tests, property tests
included, that could not catch catastrophic cancellation in a closed form,
because every test compared the series against expressions sharing the same
cancellation; a check against an independently derived bound caught it
immediately.

WHAT THE ACCUMULATOR DOES
-------------------------

:func:`flakerank.baselines.accumulated_wasted_time` computes, for each test,
the sum of the durations of that test's failed attempts, by boolean-masking the
attempt table on ``attempt_failed``, grouping by ``attempt_test`` and summing
with ``numpy.add.at``. The total is the sum over sessions of the same masked
durations. Every operation is a grouped floating-point addition of the selected
durations, in the order the attempt table happens to be in.

WHAT THESE BOUNDS USE INSTEAD
-----------------------------

Neither bound here groups anything, sums any selected subset, or looks at which
attempt belongs to which test. Each uses two scalars computed over the **whole**
duration sample plus one integer count:

**1. The order-statistic bracket.** Let ``F`` be the number of failed attempts
and let ``t_min`` and ``t_max`` be the smallest and largest attempt duration in
the history, over **all** attempts, failed or not. Since each of the ``F``
wasted durations is one of the sampled durations,

    F * t_min  <=  W  <=  F * t_max                                 (1)

This is a counting argument plus two extreme order statistics. It uses no
grouping, no subset sum and no per-test information; ``t_min`` and ``t_max``
are computed by ``min`` and ``max`` over the full array, which are selections
and not additions, so they cannot share an accumulation error with the
accumulator. The bracket is loose -- it is as wide as the duration spread, and
for this product's history the spread is four orders of magnitude -- and that
is the point: a loose bound derived differently catches a structural defect
(the wrong mask, a double count, a sign) that a tight bound sharing the
implementation's arithmetic cannot.

**2. The Cauchy-Schwarz upper bound.** By the Cauchy-Schwarz inequality applied
to the vector of wasted durations against the all-ones vector of length ``F``,

    W = sum_{failed} t  <=  sqrt(F * sum_{failed} t**2)
                        <=  sqrt(F * sum_{all} t**2)                (2)

The second step is monotonicity of a sum of non-negative terms. So an upper
bound on ``W`` follows from ``F`` and the **second moment of the whole duration
sample**, a statistic the accumulator never forms. (2) is tighter than (1)
whenever the durations are spread out, since it is sensitive to the shape of
the distribution and not only to its maximum.

*Source:* the Cauchy-Schwarz inequality; Hardy, G. H., Littlewood, J. E. &
Polya, G. (1952), *Inequalities*, 2nd edition, Cambridge University Press,
develops it and the related power-mean inequalities. The specialisation used
here, ``sum x_i <= sqrt(n * sum x_i**2)``, is the case of Cauchy-Schwarz
against the all-ones vector.

THE SLACK FOLLOWS THE CONDITIONING, NOT A CONSTANT
--------------------------------------------------

An absolute slack constant is the wrong shape and has already made a bound
check vacuous once in this mission: a slack floored at 1.8e-15 admitted a
bracket of width 4.5e-23. The slack here is the standard forward error bound
for recursive floating-point summation (Wilkinson's bound; see Higham, N. J.,
2002, *Accuracy and Stability of Numerical Algorithms*, 2nd edition, SIAM,
chapter 4, for the ``n * eps * sum|x_i|`` form):

    slack = SAFETY * F * eps * sum_{all} |t|                        (3)

with ``eps`` the double-precision unit roundoff ``2**-53`` and ``SAFETY = 4``
covering the grouped rather than strictly recursive order of the accumulation.
So the slack scales with the number of terms and with the magnitude of what is
being summed -- the conditioning of the quantity the bound brackets -- and goes
to zero with either. For this product's history, durations are stored as exact
integer milliseconds, so the input contributes no representation error at all
and (3) bounds the accumulation alone.

A note on how tight this is in practice: ``F * eps * sum|t|`` for 1000 failed
attempts and a 100000-second duration total is about 1e-8 seconds, which is
twelve orders of magnitude below the bracket width. The slack is therefore
never what makes the check pass; it exists so the check is not falsified by a
rounding difference, and its size is reported so a reader can see that it is
not doing the work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .history import RunHistory

__all__ = ["BoundBracket", "SAFETY", "UNIT_ROUNDOFF", "wasted_time_bracket"]

#: Double-precision unit roundoff, ``2**-53``.
UNIT_ROUNDOFF = 2.0**-53

#: Multiplier on the summation error bound, covering the grouped rather than
#: strictly recursive accumulation order of ``numpy.add.at``.
SAFETY = 4.0


@dataclass(frozen=True)
class BoundBracket:
    """An independently derived bracket on total wasted time, in seconds.

    Attributes:
        accumulated_seconds: the value being bracketed, from the accumulator.
        order_statistic_low: ``F * t_min`` from equation (1).
        order_statistic_high: ``F * t_max`` from equation (1).
        cauchy_schwarz_high: ``sqrt(F * sum t**2)`` from equation (2).
        failed_attempts: ``F``.
        min_attempt_seconds: ``t_min`` over all attempts.
        max_attempt_seconds: ``t_max`` over all attempts.
        second_moment_seconds_squared: ``sum t**2`` over all attempts, in s^2.
        absolute_total_seconds: ``sum |t|`` over all attempts.
        slack_seconds: equation (3).
        brackets: whether the accumulated value lies inside the tightest
            available bracket, allowing the slack.
        tightest_high_seconds: the smaller of the two upper bounds.
        margin_low_seconds: ``accumulated - (low - slack)``; positive passes.
        margin_high_seconds: ``(tightest_high + slack) - accumulated``;
            positive passes.
    """

    accumulated_seconds: float
    order_statistic_low: float
    order_statistic_high: float
    cauchy_schwarz_high: float
    failed_attempts: int
    min_attempt_seconds: float
    max_attempt_seconds: float
    second_moment_seconds_squared: float
    absolute_total_seconds: float
    slack_seconds: float
    brackets: bool
    tightest_high_seconds: float
    margin_low_seconds: float
    margin_high_seconds: float

    @property
    def relative_width(self) -> float:
        """Bracket width divided by the accumulated value, or ``inf`` at zero."""
        if self.accumulated_seconds == 0.0:
            return float("inf")
        return (
            self.tightest_high_seconds - self.order_statistic_low
        ) / self.accumulated_seconds


def wasted_time_bracket(
    history: RunHistory,
    accumulated_seconds: float,
) -> BoundBracket:
    """Bracket an accumulated wasted-time total with equations (1), (2) and (3).

    Args:
        history: the recorded history or window the total was accumulated over.
        accumulated_seconds: the total to be bracketed, in seconds, as produced
            by :func:`flakerank.baselines.accumulated_wasted_time`. It is
            passed in rather than recomputed, so that this function shares no
            code path with the thing it checks.

    Returns:
        A :class:`BoundBracket`.

    Raises:
        TypeError: if ``accumulated_seconds`` is not a real number.
        ValueError: for a non-finite or negative accumulated total, or an empty
            history.
    """
    if isinstance(accumulated_seconds, bool) or not isinstance(
        accumulated_seconds, (int, float, np.floating, np.integer)
    ):
        raise TypeError(
            "accumulated_seconds must be a real number of seconds, got "
            f"{type(accumulated_seconds).__name__}"
        )
    total = float(accumulated_seconds)
    if not np.isfinite(total) or total < 0.0:
        raise ValueError(
            f"accumulated_seconds must be finite and >= 0, got {total!r}; wasted time "
            "is a sum of non-negative durations"
        )
    if history.n_attempts == 0:
        raise ValueError("cannot bracket a total over a history with no attempts")

    durations = history.duration_seconds
    failed = int(np.count_nonzero(history.attempt_failed))
    minimum = float(durations.min())
    maximum = float(durations.max())
    second_moment = float(np.square(durations).sum())
    absolute_total = float(np.abs(durations).sum())

    low = failed * minimum
    high_order = failed * maximum
    high_cauchy = float(np.sqrt(failed * second_moment))
    tightest_high = min(high_order, high_cauchy)
    slack = SAFETY * failed * UNIT_ROUNDOFF * absolute_total

    margin_low = total - (low - slack)
    margin_high = (tightest_high + slack) - total
    return BoundBracket(
        accumulated_seconds=total,
        order_statistic_low=low,
        order_statistic_high=high_order,
        cauchy_schwarz_high=high_cauchy,
        failed_attempts=failed,
        min_attempt_seconds=minimum,
        max_attempt_seconds=maximum,
        second_moment_seconds_squared=second_moment,
        absolute_total_seconds=absolute_total,
        slack_seconds=slack,
        brackets=bool(margin_low >= 0.0 and margin_high >= 0.0),
        tightest_high_seconds=tightest_high,
        margin_low_seconds=margin_low,
        margin_high_seconds=margin_high,
    )
