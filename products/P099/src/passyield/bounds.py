"""Brackets derived from mathematics the shortfall estimators do not use.

WHY THIS MODULE EXISTS SEPARATELY
---------------------------------
"Two implementations agree" and "the answer lies inside a bound derived from
different mathematics" are different checks, and only the second catches an
error the two implementations share. This product's analytic baseline is a
mass-preserving pushforward followed by a compound-Poisson convolution on a
uniform volume grid, and its cross-check is a Monte Carlo of the same
distributions evaluated through the same cumulative rate integral. A sign
error in that integral, a Mbit-versus-bit unit slip, a missing grid-step
factor in the mass normalisation or a mis-stated geometric term would move
**both** of them together and every agreement test would pass.

The brackets here use none of that machinery. They use interval lengths and
the extreme values of the integrand, and nothing else: no quadrature of the
rate profile, no pushforward, no convolution, no random numbers.

THE TWO BRACKETS
----------------
1. :func:`shortfall_bracket`. The realised yield is a non-negative integral of
   a rate bounded above by the link budget's own ceiling
   :math:`R_{max}=B\\eta_{max}`, over an interval no longer than the window:

   .. math:: 0 \\le Y_{real} \\le R_{max} T
      \\quad\\Longrightarrow\\quad
      Y_{nom} - R_{max} T \\le S \\le Y_{nom}

   Every quantile of every shortfall distribution this product produces must
   lie inside that, because the inequality holds path by path. It is loose,
   and looseness is the price of independence; it is still tight enough to
   catch a unit error, a sign error or an unnormalised mass, which is what it
   is for.

2. :func:`capacity_bracket`. The mean-value inequality for integrals applied
   to the window capacity:

   .. math:: R_{min} T \\le Y_{cap} \\le R_{max} T

   with :math:`R_{min}` and :math:`R_{max}` the smallest and largest rate
   actually attained on the window. This brackets the composite-Simpson
   capacity integral without using Simpson's rule, so a wrong quadrature
   weight vector -- which would leave the decomposition self-consistent --
   falls outside it.

THE SLACK FOLLOWS THE CONDITIONING, NOT A CONSTANT
--------------------------------------------------
A bracket on a floating-point quantity needs a slack, and a slack that is an
absolute constant makes a tight bracket vacuous and a wide bracket useless. A
previous product of this mission had both wrong forms falsified on the same
structure by a slack floored at an absolute constant, which admitted the very
expression the check existed to catch.

So :func:`bracket_slack_mbit` returns

.. math:: \\sigma = c\\,\\epsilon \\max(|a_1|, |a_2|, \\dots)

over the **operands** of the bound expression, not over its result. For a
difference of two numbers of magnitude :math:`M` the absolute rounding error
is of order :math:`\\epsilon M` whatever the difference comes to, so the slack
grows with the operands and the bracket becomes correspondingly harder to
satisfy, in relative terms, exactly when it is tight. ``c`` is the operation
count of the bound expression, which is small and stated.

UNITS
-----
Volumes Mbit, rates Mbit/s, times s.
"""

from __future__ import annotations

import numpy as np

#: Double-precision unit roundoff.
MACHINE_EPSILON = float(np.finfo(float).eps)

#: Operation-count coefficient of :func:`bracket_slack_mbit`. The widest bound
#: expression here is a product and a difference, so 8 leaves a factor of four
#: over the worst case and is stated rather than tuned.
SLACK_OPERATION_COUNT = 8.0


def bracket_slack_mbit(*operands: float) -> float:
    """Slack for a bracket, Mbit, scaled by the magnitude of its operands.

    See the module docstring for why the scale is the operands and not the
    result.
    """
    magnitude = max((abs(float(value)) for value in operands), default=0.0)
    return SLACK_OPERATION_COUNT * MACHINE_EPSILON * magnitude


def shortfall_bracket(
    nominal_bits: float, window_duration_s: float, max_rate_mbit_s: float
) -> tuple[float, float]:
    """Bracket on the shortfall, Mbit, from durations and the rate ceiling.

    Parameters
    ----------
    nominal_bits:
        The booked volume ``Y_nom``, Mbit.
    window_duration_s:
        Geometric mask-to-mask window length ``T``, seconds.
    max_rate_mbit_s:
        The link budget's rate ceiling ``R_max = B * eta_max``, Mbit/s. This
        is a declared constant of the waveform, not a measurement of the pass.

    Returns
    -------
    tuple of float
        ``(lower, upper)`` in Mbit, inclusive, already widened by
        :func:`bracket_slack_mbit`.
    """
    if window_duration_s < 0.0:
        raise ValueError(f"window_duration_s must be non-negative, got {window_duration_s!r}")
    if max_rate_mbit_s <= 0.0:
        raise ValueError(f"max_rate_mbit_s must be positive, got {max_rate_mbit_s!r}")
    ceiling = max_rate_mbit_s * window_duration_s
    slack = bracket_slack_mbit(nominal_bits, ceiling)
    return nominal_bits - ceiling - slack, nominal_bits + slack


def capacity_bracket(
    window_duration_s: float, min_rate_mbit_s: float, max_rate_mbit_s: float
) -> tuple[float, float]:
    """Bracket on the window capacity, Mbit, by the mean-value inequality."""
    if window_duration_s < 0.0:
        raise ValueError(f"window_duration_s must be non-negative, got {window_duration_s!r}")
    if min_rate_mbit_s < 0.0:
        raise ValueError(f"min_rate_mbit_s must be non-negative, got {min_rate_mbit_s!r}")
    if max_rate_mbit_s < min_rate_mbit_s:
        raise ValueError(
            f"max_rate_mbit_s {max_rate_mbit_s!r} must not be below min_rate_mbit_s "
            f"{min_rate_mbit_s!r}"
        )
    low = min_rate_mbit_s * window_duration_s
    high = max_rate_mbit_s * window_duration_s
    slack = bracket_slack_mbit(low, high)
    return low - slack, high + slack


def inside(value: float, bracket: tuple[float, float]) -> bool:
    """Whether ``value`` lies inside an inclusive bracket."""
    low, high = bracket
    return bool(low <= value <= high)
