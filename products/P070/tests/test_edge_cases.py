"""Input validation and boundary behaviour of the public arithmetic."""

from __future__ import annotations

import math

import pytest

from conftest import make_ladder, make_supervisor, make_task
from watchdogspec import (
    beat_window_ms,
    false_trip_probability,
    false_trip_rate,
    normal_upper_tail,
    task_timing,
    wilson_interval,
    worst_case_detection_latency_ms,
)


@pytest.mark.parametrize("bad", [0.0, -1.0, -1e-9, float("nan"), float("inf")])
def test_beat_window_rejects_a_non_positive_or_non_finite_period(bad):
    with pytest.raises(ValueError, match="period_ms"):
        beat_window_ms(bad, 0.0)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_beat_window_rejects_a_bad_jitter(bad):
    with pytest.raises(ValueError, match="jitter_ms"):
        beat_window_ms(10.0, bad)


def test_beat_window_rejects_a_string():
    with pytest.raises(TypeError, match="period_ms"):
        beat_window_ms("10", 0.0)  # type: ignore[arg-type]


def test_wcdl_rejects_a_non_integer_tolerance():
    with pytest.raises(TypeError, match="missed_beats_tolerated"):
        worst_case_detection_latency_ms(10.0, 0.0, 1.5, 1.0)  # type: ignore[arg-type]


def test_wcdl_rejects_a_boolean_tolerance():
    # bool is a subclass of int; True must not silently mean one missed beat.
    with pytest.raises(TypeError, match="missed_beats_tolerated"):
        worst_case_detection_latency_ms(10.0, 0.0, True, 1.0)  # type: ignore[arg-type]


def test_wcdl_rejects_a_negative_tolerance():
    with pytest.raises(ValueError, match="missed_beats_tolerated"):
        worst_case_detection_latency_ms(10.0, 0.0, -1, 1.0)


def test_wcdl_rejects_a_non_positive_poll_period():
    with pytest.raises(ValueError, match="poll_period_ms"):
        worst_case_detection_latency_ms(10.0, 0.0, 0, 0.0)


def test_zero_jitter_and_zero_overhead_are_accepted():
    assert worst_case_detection_latency_ms(10.0, 0.0, 0, 1.0, 0.0, 0.0) == pytest.approx(
        11.0, abs=1e-12
    )


def test_task_timing_rejects_a_non_positive_deadline():
    task = make_task(deadline=1.0)
    broken = type(task)(
        id=task.id,
        heartbeat_period_ms=task.heartbeat_period_ms,
        missed_beats_tolerated=task.missed_beats_tolerated,
        liveness_deadline_ms=0.0,
        escalation=task.escalation,
    )
    with pytest.raises(ValueError, match="liveness_deadline_ms"):
        task_timing(broken, make_supervisor(), make_ladder(1.0))


def test_false_trip_probability_is_exactly_zero_for_zero_sigma():
    assert false_trip_probability(20.0, 2.0, 0.0, 1) == 0.0


def test_false_trip_rate_for_zero_sigma_reports_exact_and_infinite_mtbft():
    rate = false_trip_rate(make_task(sigma=0.0))
    assert rate.exact is True
    assert rate.per_gap == 0.0
    assert rate.per_hour == 0.0
    assert rate.mtbft_hours == math.inf


def test_false_trip_probability_is_one_half_when_the_threshold_is_zero():
    # k = 0 and J = 0: the supervisor declares as soon as a beat is one
    # instant late, which a symmetric jitter does half the time.
    assert false_trip_probability(20.0, 0.0, 1.0, 0) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_false_trip_probability_rejects_a_bad_sigma(bad):
    with pytest.raises(ValueError, match="jitter_sigma_ms"):
        false_trip_probability(20.0, 2.0, bad, 0)


def test_false_trip_probability_rejects_a_non_positive_period():
    with pytest.raises(ValueError, match="period_ms"):
        false_trip_probability(0.0, 2.0, 1.0, 0)


def test_normal_upper_tail_at_zero_is_one_half():
    assert normal_upper_tail(0.0) == pytest.approx(0.5, abs=1e-15)


def test_normal_upper_tail_handles_infinities():
    assert normal_upper_tail(float("inf")) == 0.0
    assert normal_upper_tail(float("-inf")) == 1.0


def test_normal_upper_tail_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        normal_upper_tail(float("nan"))


def test_normal_upper_tail_matches_published_values():
    # Abramowitz and Stegun 1964, table 26.1: the one-sided tail at 1.96 is
    # 0.0249979 to seven places, and at 3 standard deviations 0.00134990.
    assert normal_upper_tail(1.959963984540054) == pytest.approx(0.025, abs=5e-9)
    assert normal_upper_tail(3.0) == pytest.approx(0.0013498980, abs=1e-9)


def test_wilson_interval_contains_the_point_estimate():
    low, high = wilson_interval(50, 1000)
    assert low < 0.05 < high
    assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_at_zero_successes_starts_at_zero():
    # The lower bound is zero in exact arithmetic. It evaluates to
    # 2.168404344971009e-19 here, because the centre and the half-width are
    # two separately rounded quantities that cancel only to within rounding.
    # This is the reason nothing in this suite compares a floating-point
    # pipeline for bit-exact equality.
    low, high = wilson_interval(0, 1000)
    assert low == pytest.approx(0.0, abs=1e-15)
    assert 0.0 <= low < high < 0.01


@pytest.mark.parametrize(("successes", "trials"), [(-1, 10), (11, 10), (0, 0)])
def test_wilson_interval_rejects_impossible_counts(successes, trials):
    with pytest.raises(ValueError):
        wilson_interval(successes, trials)
