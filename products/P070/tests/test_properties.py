"""Property-based tests.

Every identity here is one the arithmetic should satisfy for any declared
numbers inside the documented validity range. Comparisons carry a tolerance:
these are floating-point sums, not integers.

Any falsifying example Hypothesis turns up is pinned as a literal regression
test in ``tests/test_regression.py``, because ``.hypothesis/`` is gitignored
and a counterexample found once is otherwise lost on a fresh clone.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from conftest import make_ladder, make_supervisor, make_task
from watchdogspec import (
    DEFAULT_TOL_MS,
    Margin,
    false_trip_probability,
    max_tolerance_for_deadline,
    simulate_detection,
    task_timing,
    verdict,
    worst_case_detection_latency_ms,
)

periods = st.floats(min_value=0.001, max_value=1e5, allow_nan=False, allow_infinity=False)
jitters = st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False)
tolerances = st.integers(min_value=0, max_value=50)
polls = st.floats(min_value=0.001, max_value=1e4, allow_nan=False, allow_infinity=False)
small = st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False)
deadlines = st.floats(min_value=0.001, max_value=1e7, allow_nan=False, allow_infinity=False)
sigmas = st.floats(min_value=1e-6, max_value=1e4, allow_nan=False, allow_infinity=False)

SLOW = settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@given(periods, jitters, tolerances, polls, small, small)
def test_wcdl_equals_its_reassociated_sum_to_within_rounding(p, j, k, q, jq, o):
    direct = worst_case_detection_latency_ms(p, j, k, q, jq, o)
    reassociated = (k + 1) * p + (k + 1) * j + q + jq + o
    assert direct == pytest.approx(reassociated, rel=1e-12, abs=1e-12)


@given(periods, jitters, tolerances, polls, small, small)
def test_wcdl_is_at_least_one_beat_window_plus_the_poll_period(p, j, k, q, jq, o):
    value = worst_case_detection_latency_ms(p, j, k, q, jq, o)
    assert value >= (p + j) + q - 1e-9


@given(periods, jitters, tolerances, polls, small, small)
def test_wcdl_does_not_decrease_with_the_missed_beat_tolerance(p, j, k, q, jq, o):
    lower = worst_case_detection_latency_ms(p, j, k, q, jq, o)
    higher = worst_case_detection_latency_ms(p, j, k + 1, q, jq, o)
    assert higher >= lower - 1e-9


@given(periods, jitters, tolerances, polls, small, small, small)
def test_wcdl_does_not_decrease_with_the_declaration_overhead(p, j, k, q, jq, o, extra):
    lower = worst_case_detection_latency_ms(p, j, k, q, jq, o)
    higher = worst_case_detection_latency_ms(p, j, k, q, jq, o + extra)
    assert higher >= lower - 1e-9


@given(periods, jitters, tolerances, polls, small, small)
def test_three_valued_verdict_is_stable_under_reassociation(p, j, k, q, jq, o):
    # The two summation orders differ by a couple of units in the last place,
    # which is enough to flip a strict met/missed comparison at a deadline
    # equal to one of them. The three-valued verdict puts both inside the
    # marginal band instead. The pinned counterexample for the strict
    # comparison is test_regression.py::test_summation_order_flips_a_strict
    # _deadline_comparison.
    #
    # The band must be given the SCALE of the quantities compared. With the
    # absolute tolerance alone this property is false above about 4.5e6 ms,
    # where one ulp exceeds 1e-9 ms; that counterexample is pinned in
    # test_regression.py::test_absolute_tolerance_alone_is_too_narrow_at_large
    # _latencies.
    direct = worst_case_detection_latency_ms(p, j, k, q, jq, o)
    reassociated = (k + 1) * p + (k + 1) * j + q + jq + o
    deadline = reassociated
    assume(math.isfinite(deadline))
    first = verdict(deadline - direct, DEFAULT_TOL_MS, deadline)
    second = verdict(deadline - reassociated, DEFAULT_TOL_MS, deadline)
    assert first is second


@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), st.floats(
    min_value=0.0, max_value=1e3, allow_nan=False))
def test_verdict_partitions_the_real_line(slack, tol):
    result = verdict(slack, tol_ms=tol)
    if slack > tol:
        assert result is Margin.MET
    elif slack < -tol:
        assert result is Margin.MISSED
    else:
        assert result is Margin.MARGINAL


@given(periods, jitters, sigmas, tolerances)
def test_false_trip_probability_is_a_probability_at_most_one_half(p, j, sigma, k):
    value = false_trip_probability(p, j, sigma, k)
    assert 0.0 <= value <= 0.5


@given(periods, jitters, sigmas, st.integers(min_value=0, max_value=20))
def test_false_trip_probability_does_not_increase_with_the_tolerance(p, j, sigma, k):
    assert false_trip_probability(p, j, sigma, k + 1) <= false_trip_probability(
        p, j, sigma, k
    ) + 1e-18


@given(periods, jitters, sigmas, tolerances, st.floats(min_value=0.0, max_value=1e4,
                                                       allow_nan=False))
def test_false_trip_probability_does_not_increase_with_the_jitter_bound(p, j, sigma, k, extra):
    assert false_trip_probability(p, j + extra, sigma, k) <= false_trip_probability(
        p, j, sigma, k
    ) + 1e-18


@given(
    periods,
    jitters,
    tolerances,
    deadlines,
    st.lists(st.floats(min_value=0.0, max_value=1e4, allow_nan=False), min_size=1, max_size=6),
)
@SLOW
def test_deepest_rung_never_exceeds_the_ladder_length(p, j, k, d, latencies):
    timing = task_timing(
        make_task(period=p, jitter=j, k=k, deadline=d),
        make_supervisor(),
        make_ladder(*latencies),
    )
    assert 0 <= timing.deepest_rung_reachable <= len(latencies)


@given(
    periods,
    jitters,
    tolerances,
    deadlines,
    st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
    st.lists(st.floats(min_value=0.0, max_value=1e4, allow_nan=False), min_size=1, max_size=6),
)
@SLOW
def test_deepest_rung_does_not_decrease_with_the_deadline(p, j, k, d, extra, latencies):
    supervisor = make_supervisor()
    ladder = make_ladder(*latencies)
    tight = task_timing(make_task(period=p, jitter=j, k=k, deadline=d), supervisor, ladder)
    loose = task_timing(
        make_task(period=p, jitter=j, k=k, deadline=d + extra), supervisor, ladder
    )
    assert loose.deepest_rung_reachable >= tight.deepest_rung_reachable


@given(
    periods,
    jitters,
    tolerances,
    deadlines,
    st.lists(st.floats(min_value=0.0, max_value=1e4, allow_nan=False), min_size=1, max_size=6),
)
@SLOW
def test_rung_finish_times_are_non_decreasing_and_start_at_detection(p, j, k, d, latencies):
    timing = task_timing(
        make_task(period=p, jitter=j, k=k, deadline=d),
        make_supervisor(),
        make_ladder(*latencies),
    )
    assert timing.rungs[0].start_ms == pytest.approx(timing.detection_latency_ms, rel=1e-12)
    finishes = [rung.finish_ms for rung in timing.rungs]
    for earlier, later in zip(finishes[:-1], finishes[1:], strict=True):
        assert later >= earlier - 1e-9
    assert timing.full_ladder_latency_ms == pytest.approx(finishes[-1], rel=1e-12)


@given(
    st.floats(min_value=0.1, max_value=1e3, allow_nan=False),
    st.floats(min_value=0.0, max_value=1e3, allow_nan=False),
    st.floats(min_value=1.0, max_value=1e6, allow_nan=False),
    st.floats(min_value=0.1, max_value=1e2, allow_nan=False),
    st.floats(min_value=0.0, max_value=1e2, allow_nan=False),
    st.floats(min_value=0.0, max_value=1e2, allow_nan=False),
    st.floats(min_value=0.0, max_value=1e4, allow_nan=False),
)
@SLOW
def test_max_tolerance_is_the_largest_k_that_fits(p, j, d, q, jq, o, ladder):
    k = max_tolerance_for_deadline(p, j, d, q, jq, o, ladder)
    window = p + j
    if k >= 0:
        assert (k + 1) * window + q + jq + o + ladder <= d + 1e-6
    assert (k + 2) * window + q + jq + o + ladder > d - 1e-6


@given(
    st.floats(min_value=0.5, max_value=500.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=200.0, allow_nan=False),
    st.integers(min_value=0, max_value=6),
    st.floats(min_value=0.5, max_value=200.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=50.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=20.0, allow_nan=False),
    st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_simulated_detection_never_exceeds_the_closed_form(p, j, k, q, jq, o, seed):
    task = make_task(period=p, jitter=j, k=k)
    supervisor = make_supervisor(poll=q, jitter=jq, overhead=o)
    result = simulate_detection(task, supervisor, seed=seed, trials=20)
    assert result.violations == 0
    assert result.max_latency_from_receipt_ms <= result.bound_ms + 1e-9


@given(
    st.floats(min_value=0.5, max_value=500.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=200.0, allow_nan=False),
    st.integers(min_value=0, max_value=6),
    st.floats(min_value=0.5, max_value=200.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=50.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=20.0, allow_nan=False),
)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_adversarial_simulation_attains_the_closed_form(p, j, k, q, jq, o):
    # Equation (3) is tight only while the poll release jitter does not
    # exceed the poll period. Beyond that a LATER poll can execute EARLIER
    # than the delayed next one, so the observation delay is capped at 2Q
    # rather than Q + Jq and the closed form is a strict over-estimate. This
    # condition was found by Hypothesis, not assumed; the counterexample is
    # pinned in test_regression.py::test_poll_jitter_above_the_poll_period
    # _makes_the_bound_loose.
    assume(jq <= q)
    task = make_task(period=p, jitter=j, k=k)
    supervisor = make_supervisor(poll=q, jitter=jq, overhead=o)
    closed = worst_case_detection_latency_ms(p, j, k, q, jq, o)
    result = simulate_detection(task, supervisor, seed=0, trials=1, adversarial=True)
    assert result.max_latency_from_receipt_ms == pytest.approx(closed, rel=1e-9, abs=1e-9)
