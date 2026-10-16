"""Known-answer tests for the closed-form arithmetic.

Every expected number below is hand-calculated and the calculation is written
out in the comment above the assertion. Comparisons use an absolute tolerance
because the quantities are floating-point sums; see the package build guide on
why nothing here asserts bit-exact equality.
"""

from __future__ import annotations

import pytest

from conftest import make_ladder, make_supervisor, make_task
from watchdogspec import (
    DEFAULT_TOL_MS,
    Margin,
    beat_window_ms,
    max_tolerance_for_deadline,
    task_timing,
    verdict,
    worst_case_detection_latency_ms,
)

TOL = 1e-9


def test_beat_window_known_answer():
    # W = P + J = 20 + 3 = 23 ms.
    assert beat_window_ms(20.0, 3.0) == pytest.approx(23.0, abs=TOL)


def test_wcdl_known_answer_worked_example_gnc():
    # TASK-GNC of examples/specs/smallsat_fc.yaml.
    # W = P + J = 20 + 3 = 23 ms.
    # WCDL = (k+1) W + Q + Jq + O = 3 * 23 + 5 + 1 + 0.5 = 69 + 6.5 = 75.5 ms.
    value = worst_case_detection_latency_ms(20.0, 3.0, 2, 5.0, 1.0, 0.5)
    assert value == pytest.approx(75.5, abs=TOL)


def test_wcdl_known_answer_zero_tolerance_is_one_window_plus_observation():
    # k = 0: WCDL = 1 * (100 + 10) + 10 + 2 + 1 = 110 + 13 = 123 ms.
    value = worst_case_detection_latency_ms(100.0, 10.0, 0, 10.0, 2.0, 1.0)
    assert value == pytest.approx(123.0, abs=TOL)


def test_wcdl_known_answer_no_jitter_no_overhead():
    # The degenerate case: WCDL = (k+1) P + Q = 4 * 50 + 10 = 210 ms.
    value = worst_case_detection_latency_ms(50.0, 0.0, 3, 10.0)
    assert value == pytest.approx(210.0, abs=TOL)


def test_wcdl_is_the_sum_of_its_five_documented_terms():
    # Independent re-association of equation (3). The two orders agree to
    # within rounding, not exactly; see tests/test_regression.py for the
    # pinned case where they differ by two units in the last place.
    k, period, jitter, poll, poll_jitter, overhead = 4, 7.5, 1.25, 3.0, 0.5, 0.25
    direct = worst_case_detection_latency_ms(
        period, jitter, k, poll, poll_jitter, overhead
    )
    reassociated = (k + 1) * period + (k + 1) * jitter + poll + poll_jitter + overhead
    assert direct == pytest.approx(reassociated, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("period", "jitter", "k", "poll", "poll_jitter", "overhead", "expected"),
    [
        # 1 * (10 + 0) + 1 + 0 + 0 = 11 ms
        (10.0, 0.0, 0, 1.0, 0.0, 0.0, 11.0),
        # 2 * (10 + 1) + 1 + 0 + 0 = 23 ms
        (10.0, 1.0, 1, 1.0, 0.0, 0.0, 23.0),
        # 3 * (10 + 1) + 1 + 2 + 3 = 33 + 6 = 39 ms
        (10.0, 1.0, 2, 1.0, 2.0, 3.0, 39.0),
        # 11 * (1 + 0.5) + 0.25 = 16.5 + 0.25 = 16.75 ms
        (1.0, 0.5, 10, 0.25, 0.0, 0.0, 16.75),
    ],
)
def test_wcdl_table_of_hand_calculations(
    period, jitter, k, poll, poll_jitter, overhead, expected
):
    value = worst_case_detection_latency_ms(period, jitter, k, poll, poll_jitter, overhead)
    assert value == pytest.approx(expected, abs=TOL)


def test_escalation_arithmetic_known_answer():
    # Hand calculation for T-WS003 of the planted fixture.
    #   W    = 20 + 2 = 22 ms
    #   WCDL = 2 * 22 + (10 + 2 + 1) = 44 + 13 = 57 ms
    #   rung finish times: 57 + 2 = 59, + 20 = 79, + 50 = 129, + 100 = 229 ms
    #   D = 100 ms, so rungs 1 and 2 fit and rung 3 does not: deepest = 2
    #   final-rung slack = 100 - 229 = -129 ms
    task = make_task(period=20.0, jitter=2.0, k=1, deadline=100.0)
    timing = task_timing(task, make_supervisor(), make_ladder(2.0, 20.0, 50.0, 100.0))
    assert timing.detection_latency_ms == pytest.approx(57.0, abs=TOL)
    finishes = [rung.finish_ms for rung in timing.rungs]
    for value, expected in zip(finishes, (59.0, 79.0, 129.0, 229.0), strict=True):
        assert value == pytest.approx(expected, abs=TOL)
    assert timing.deepest_rung_reachable == 2
    assert timing.ladder_slack_ms == pytest.approx(-129.0, abs=TOL)
    assert timing.full_ladder_latency_ms == pytest.approx(229.0, abs=TOL)


def test_escalation_cumulative_cost_known_answer():
    # make_ladder assigns cost = rung index starting at 0, so the cumulative
    # costs over four rungs are 0, 1, 3, 6.
    timing = task_timing(
        make_task(deadline=10_000.0), make_supervisor(), make_ladder(2.0, 20.0, 50.0, 100.0)
    )
    costs = [rung.cumulative_cost for rung in timing.rungs]
    for value, expected in zip(costs, (0.0, 1.0, 3.0, 6.0), strict=True):
        assert value == pytest.approx(expected, abs=TOL)
    assert timing.total_cost == pytest.approx(6.0, abs=TOL)


def test_tolerated_silence_slack_known_answer():
    # T-WS002 of the planted fixture: k = 4, W = 55, k*W = 220 ms against
    # D = 200 ms, so the tolerance slack is -20 ms.
    task = make_task(period=50.0, jitter=5.0, k=4, deadline=200.0)
    timing = task_timing(task, make_supervisor(), make_ladder(1.0))
    assert timing.tolerated_silence_ms == pytest.approx(220.0, abs=TOL)
    assert timing.tolerance_slack_ms == pytest.approx(-20.0, abs=TOL)
    assert timing.tolerance_margin is Margin.MISSED


def test_deepest_rung_is_zero_when_detection_alone_misses():
    # W = 110, WCDL = 2 * 110 + 13 = 233 ms against D = 200 ms.
    task = make_task(period=100.0, jitter=10.0, k=1, deadline=200.0)
    timing = task_timing(task, make_supervisor(), make_ladder(2.0, 20.0))
    assert timing.detection_margin is Margin.MISSED
    assert timing.deepest_rung_reachable == 0


def test_deepest_rung_is_every_rung_when_the_whole_ladder_fits():
    timing = task_timing(
        make_task(deadline=10_000.0), make_supervisor(), make_ladder(2.0, 20.0, 50.0)
    )
    assert timing.deepest_rung_reachable == 3
    assert timing.ladder_margin is Margin.MET


def test_verdict_is_three_valued_and_marginal_at_the_threshold():
    assert verdict(1.0) is Margin.MET
    assert verdict(-1.0) is Margin.MISSED
    assert verdict(0.0) is Margin.MARGINAL
    assert verdict(DEFAULT_TOL_MS / 2) is Margin.MARGINAL
    assert verdict(-DEFAULT_TOL_MS / 2) is Margin.MARGINAL
    assert verdict(DEFAULT_TOL_MS * 2) is Margin.MET


def test_verdict_rejects_a_negative_tolerance():
    with pytest.raises(ValueError, match="tol_ms"):
        verdict(1.0, tol_ms=-1e-9)


def test_max_tolerance_known_answer():
    # Budget = D - Q - Jq - O - ladder = 500 - 10 - 2 - 1 - 172 = 315 ms.
    # W = 22 ms, 315 / 22 = 14.318..., floor 14, minus 1 gives k = 13.
    # Check: 14 * 22 + 13 + 172 = 308 + 185 = 493 <= 500, and k = 14 gives
    # 15 * 22 + 185 = 515 > 500.
    value = max_tolerance_for_deadline(20.0, 2.0, 500.0, 10.0, 2.0, 1.0, 172.0)
    assert value == 13
    assert 14 * 22.0 + 13.0 + 172.0 <= 500.0
    assert 15 * 22.0 + 13.0 + 172.0 > 500.0


def test_max_tolerance_is_minus_one_when_nothing_fits():
    # D = 20 ms cannot absorb even one window of 110 ms plus 13 ms of poll.
    assert max_tolerance_for_deadline(100.0, 10.0, 20.0, 10.0, 2.0, 1.0, 0.0) == -1


def test_spec_timing_skips_a_task_whose_ladder_is_undeclared(planted_spec):
    from watchdogspec import spec_timing

    timings = spec_timing(planted_spec)
    assert "T-WS012" not in timings
    assert "T-CLEAN" in timings
