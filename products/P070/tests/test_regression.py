"""Counterexamples found once, pinned with literal inputs.

``.hypothesis/`` is gitignored, so a falsifying example a property-based run
turns up is lost on a fresh clone unless it is written down here. Every test
in this file carries the exact inputs that produced the behaviour and says
what the behaviour is.
"""

from __future__ import annotations

import math

import pytest

from conftest import make_supervisor, make_task
from watchdogspec import (
    DEFAULT_TOL_MS,
    Margin,
    comparison_band_ms,
    simulate_detection,
    verdict,
    wilson_interval,
    worst_case_detection_latency_ms,
)


def test_summation_order_flips_a_strict_deadline_comparison():
    """Two algebraically identical sums differ by two units in the last place.

    Found by a search over the declared-parameter space on 2026-10-16. The
    inputs are k = 2, P = 3.7 ms, J = 0.1 ms, Q = 2.3 ms, Jq = 0, O = 0.

        (k+1)*(P+J) + Q  = 3 * 3.8 + 2.3            = 13.7
        (k+1)*P + (k+1)*J + Q
                         = 11.100000000000001
                         + 0.30000000000000004
                         + 2.3                      = 13.700000000000003

    At a deadline equal to the second value the strict comparison
    ``latency <= deadline`` answers met for one order and missed for the
    other, while the two latencies agree to within 3.6e-15 ms. This is why
    nothing in this package returns a two-valued deadline verdict.
    """
    k, period, jitter, poll = 2, 3.7, 0.1, 2.3
    direct = worst_case_detection_latency_ms(period, jitter, k, poll)
    reassociated = (k + 1) * period + (k + 1) * jitter + poll

    assert direct != reassociated
    assert abs(direct - reassociated) == pytest.approx(3.55e-15, rel=0.2)
    assert abs(direct - reassociated) <= 2.0 * math.ulp(direct)

    deadline = reassociated
    # The two-valued "is there slack left" boolean disagrees with itself.
    assert ((deadline - direct) > 0.0) is True
    assert ((deadline - reassociated) > 0.0) is False
    # So does the verdict with the tolerance band closed to zero width.
    assert verdict(deadline - direct, tol_ms=0.0) is Margin.MET
    assert verdict(deadline - reassociated, tol_ms=0.0) is Margin.MARGINAL

    # The three-valued verdict at the default tolerance does not.
    assert verdict(deadline - direct, tol_ms=DEFAULT_TOL_MS) is Margin.MARGINAL
    assert verdict(deadline - reassociated, tol_ms=DEFAULT_TOL_MS) is Margin.MARGINAL


def test_poll_jitter_above_the_poll_period_makes_the_bound_loose():
    """Equation (3) is an upper bound everywhere and tight only for Jq <= Q.

    Found by Hypothesis on 2026-10-16 while checking that the adversarial
    simulation attains the closed form. Literal falsifying inputs:
    P = 1.0 ms, J = 0.0 ms, k = 0, Q = 1.0 ms, Jq = 2.0 ms, O = 0.0 ms.

    Equation (3) gives (0+1)*1 + 1 + 2 + 0 = 4.0 ms. The simulation observes
    3.0 ms, because with Jq > Q the poll released one period later executes
    EARLIER than the delayed next poll: the supervisor sees the expiry at
    2Q = 2 ms rather than Q + Jq = 3 ms after it. The bound still holds; it is
    simply no longer attainable, and the model's tightness claim is therefore
    restricted to Jq <= Q.
    """
    task = make_task(period=1.0, jitter=0.0, k=0)
    supervisor = make_supervisor(poll=1.0, jitter=2.0, overhead=0.0)
    closed = worst_case_detection_latency_ms(1.0, 0.0, 0, 1.0, 2.0, 0.0)
    assert closed == pytest.approx(4.0, abs=1e-12)

    result = simulate_detection(task, supervisor, seed=0, trials=1, adversarial=True)
    assert result.max_latency_from_receipt_ms == pytest.approx(3.0, abs=1e-9)
    assert result.max_latency_from_receipt_ms < closed
    # The bound is never violated, which is the property that matters.
    assert result.violations == 0

    stochastic = simulate_detection(task, supervisor, seed=1, trials=500)
    assert stochastic.violations == 0
    assert stochastic.max_latency_from_receipt_ms <= closed + 1e-9


def test_absolute_tolerance_alone_is_too_narrow_at_large_latencies():
    """An absolute comparison band of 1e-9 ms falls below one ulp above 4.5e6 ms.

    Found by Hypothesis on 2026-10-16, running the test suite at the declared
    DEPENDENCY FLOOR (hypothesis 6.100.0 explores a different part of the
    space from 6.170.0 and only the floor run turned this up). Literal
    falsifying inputs:

        P = 100000.0 ms, J = 86413.50000000362 ms, k = 44, Q = 1.0 ms

    The two summation orders of equation (3) give

        (k+1)(P+J) + Q          = 8388608.500000162 ms
        (k+1)P + (k+1)J + Q     = 8388608.500000164 ms

    which differ by 1.862645149230957e-09 ms, one ulp at that magnitude and
    nearly twice the 1e-9 ms absolute tolerance. With the absolute tolerance
    alone the three-valued verdict therefore flips from MET to MARGINAL, which
    is the behaviour the band exists to prevent. Giving the band the scale of
    the quantities being compared, as watchdogspec.latency.comparison_band_ms
    does, restores stability.
    """
    period, jitter, k, poll = 100000.0, 86413.50000000362, 44, 1.0
    direct = worst_case_detection_latency_ms(period, jitter, k, poll)
    reassociated = (k + 1) * period + (k + 1) * jitter + poll

    assert direct == pytest.approx(8388608.500000162, abs=1e-7)
    assert reassociated == pytest.approx(8388608.500000164, abs=1e-7)
    difference = reassociated - direct
    assert difference == pytest.approx(1.862645149230957e-09, rel=1e-9)
    assert difference > DEFAULT_TOL_MS
    assert difference == pytest.approx(math.ulp(direct), rel=1e-12)

    deadline = reassociated
    # Absolute tolerance alone: the verdict is not stable.
    assert verdict(deadline - direct, DEFAULT_TOL_MS) is Margin.MET
    assert verdict(deadline - reassociated, DEFAULT_TOL_MS) is Margin.MARGINAL

    # Scale-aware band: it is.
    band = comparison_band_ms(DEFAULT_TOL_MS, deadline)
    assert band > difference
    assert verdict(deadline - direct, DEFAULT_TOL_MS, deadline) is Margin.MARGINAL
    assert verdict(deadline - reassociated, DEFAULT_TOL_MS, deadline) is Margin.MARGINAL


def test_wilson_lower_bound_at_zero_successes_is_not_exactly_zero():
    """The Wilson lower bound is 0 in exact arithmetic and 2.17e-19 in doubles.

    Found while writing tests/test_edge_cases.py. At 0 successes in 1000
    trials the centre and the half-width are equal in exact arithmetic and
    cancel; as two separately rounded doubles they do not.
    """
    low, high = wilson_interval(0, 1000)
    assert low != 0.0
    assert low == pytest.approx(2.168404344971009e-19, rel=1e-9)
    assert 0.0 < low < 1e-18 < high
