"""Known answers for the false-trip arithmetic, equations (7) to (9)."""

from __future__ import annotations

import math

import pytest

from conftest import make_task
from watchdogspec import false_trip_probability, false_trip_rate, spec_false_trip_rates
from watchdogspec.falsetrip import MS_PER_HOUR


def test_ms_per_hour_is_the_literal_conversion():
    assert MS_PER_HOUR == 3600.0 * 1000.0


def test_false_trip_probability_known_answer_k_zero():
    # k = 0, so the threshold is J = 2 ms. With sigma = 1 ms,
    # p = 0.5 * erfc(2 / (2 * 1)) = 0.5 * erfc(1) = 0.5 * 0.157299207...
    #   = 0.0786496035...
    # erfc(1) = 0.1572992070502851 (Abramowitz and Stegun 1964, table 7.1).
    value = false_trip_probability(20.0, 2.0, 1.0, 0)
    assert value == pytest.approx(0.07864960352514, rel=1e-12)
    assert value == pytest.approx(0.5 * math.erfc(1.0), rel=1e-15)


def test_false_trip_probability_known_answer_k_one():
    # k = 1: threshold = 1 * P + 2 * J = 20 + 4 = 24 ms. With sigma = 4 ms,
    # p = 0.5 * erfc(24 / 8) = 0.5 * erfc(3) = 0.5 * 2.20905e-05
    #   = 1.104524...e-05.
    value = false_trip_probability(20.0, 2.0, 4.0, 1)
    assert value == pytest.approx(0.5 * math.erfc(3.0), rel=1e-15)
    assert value == pytest.approx(1.1045241e-05, rel=1e-6)


def test_tail_argument_in_sigma_units_known_answer():
    # z = threshold / (sigma sqrt 2) = 24 / (4 * 1.41421356) = 4.2426407.
    rate = false_trip_rate(make_task(period=20.0, jitter=2.0, k=1, sigma=4.0))
    assert rate.threshold_ms == pytest.approx(24.0, abs=1e-12)
    assert rate.z == pytest.approx(24.0 / (4.0 * math.sqrt(2.0)), rel=1e-14)
    assert rate.z == pytest.approx(4.2426406871, rel=1e-9)


def test_false_trip_per_hour_known_answer():
    # p = 0.5 erfc(1) = 0.0786496035; at P = 1000 ms there are 3600 beats per
    # hour, so the rate is 0.0786496035 * 3600 = 283.1385727 per hour and the
    # mean time between false trips is 1 / 283.1385727 = 0.003531897 hours.
    rate = false_trip_rate(make_task(period=1000.0, jitter=2.0, k=0, sigma=1.0))
    assert rate.per_gap == pytest.approx(0.5 * math.erfc(1.0), rel=1e-15)
    assert rate.per_hour == pytest.approx(283.1385727, rel=1e-9)
    assert rate.mtbft_hours == pytest.approx(1.0 / 283.1385727, rel=1e-9)
    assert rate.exact is False


def test_rate_decreases_as_the_tolerance_rises():
    # Each extra tolerated beat adds P + J to the threshold, which moves the
    # Gaussian tail argument out by (P + J) / (2 sigma).
    rates = [
        false_trip_probability(20.0, 2.0, 5.0, k) for k in range(5)
    ]
    for earlier, later in zip(rates[:-1], rates[1:], strict=True):
        assert later <= earlier + 1e-18


def test_rate_decreases_as_the_declared_jitter_bound_widens():
    rates = [false_trip_probability(20.0, bound, 5.0, 0) for bound in (0.0, 1.0, 2.0, 8.0)]
    for earlier, later in zip(rates[:-1], rates[1:], strict=True):
        assert later <= earlier + 1e-18


def test_rate_increases_with_sigma():
    rates = [false_trip_probability(20.0, 4.0, sigma, 0) for sigma in (0.5, 1.0, 2.0, 8.0)]
    for earlier, later in zip(rates[:-1], rates[1:], strict=True):
        assert later >= earlier - 1e-18


def test_probability_never_exceeds_one_half():
    # The threshold is non-negative, so the tail argument is non-negative and
    # erfc of a non-negative argument is at most 1.
    for k in range(4):
        for jitter in (0.0, 0.5, 3.0):
            assert 0.0 <= false_trip_probability(10.0, jitter, 2.0, k) <= 0.5


def test_underflow_is_reported_as_zero_with_infinite_mtbft():
    # threshold / (2 sigma) = 49 / 1.8 = 27.2, and erfc(27.2) underflows the
    # double range, so the rate is reported as 0 and the mean time between
    # false trips as infinite. That means "below what this arithmetic can
    # represent", not "impossible".
    rate = false_trip_rate(make_task(period=20.0, jitter=3.0, k=2, sigma=0.9))
    assert rate.per_gap == 0.0
    assert rate.mtbft_hours == math.inf
    assert rate.exact is False


def test_spec_rates_cover_every_task(worked_spec):
    rates = spec_false_trip_rates(worked_spec)
    assert set(rates) == {task.id for task in worked_spec.tasks}


def test_worked_example_telemetry_rate_known_answer(worked_spec):
    # TASK-TLM: k = 0, J = 250 ms, sigma = 90 ms, P = 1000 ms.
    # p = 0.5 erfc(250 / 180) = 0.5 erfc(1.388889) = 0.024754...
    # per hour = p * 3600 = 89.1 trips per hour.
    rate = spec_false_trip_rates(worked_spec)["TASK-TLM"]
    assert rate.per_gap == pytest.approx(0.5 * math.erfc(250.0 / 180.0), rel=1e-14)
    assert rate.per_gap == pytest.approx(0.0247543, rel=1e-5)
    assert rate.per_hour == pytest.approx(89.1155, rel=1e-5)
