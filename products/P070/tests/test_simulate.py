"""The discrete-event simulation, and its agreement with the closed form.

Nothing here asserts bit-exact equality. The adversarial trial is compared to
the closed form with an absolute tolerance of 1e-6 ms, which is nine orders of
magnitude below the smallest latency in any shipped specification and ten
orders above the rounding of the sums involved.
"""

from __future__ import annotations

import math

import pytest

from conftest import make_supervisor, make_task
from watchdogspec import (
    simulate_detection,
    simulate_false_trips,
    wilson_interval,
    worst_case_detection_latency_ms,
)
from watchdogspec.falsetrip import false_trip_probability

AGREEMENT_TOL_MS = 1e-6


def test_adversarial_trial_attains_the_closed_form():
    task = make_task(period=20.0, jitter=3.0, k=2)
    supervisor = make_supervisor(poll=5.0, jitter=1.0, overhead=0.5)
    closed = worst_case_detection_latency_ms(20.0, 3.0, 2, 5.0, 1.0, 0.5)
    result = simulate_detection(task, supervisor, seed=0, trials=1, adversarial=True)
    assert result.max_latency_from_receipt_ms == pytest.approx(closed, abs=AGREEMENT_TOL_MS)
    assert result.violations == 0


@pytest.mark.parametrize("k", [0, 1, 3, 7])
@pytest.mark.parametrize(("period", "jitter"), [(20.0, 3.0), (5.0, 0.0), (1000.0, 250.0)])
def test_adversarial_trial_attains_the_closed_form_over_a_grid(k, period, jitter):
    task = make_task(period=period, jitter=jitter, k=k)
    supervisor = make_supervisor(poll=5.0, jitter=1.0, overhead=0.5)
    closed = worst_case_detection_latency_ms(period, jitter, k, 5.0, 1.0, 0.5)
    result = simulate_detection(task, supervisor, seed=k, trials=1, adversarial=True)
    assert result.max_latency_from_receipt_ms == pytest.approx(closed, abs=AGREEMENT_TOL_MS)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_stochastic_trials_never_exceed_the_closed_form(seed):
    task = make_task(period=20.0, jitter=3.0, k=2)
    supervisor = make_supervisor(poll=5.0, jitter=1.0, overhead=0.5)
    result = simulate_detection(task, supervisor, seed=seed, trials=400)
    assert result.violations == 0
    assert result.max_latency_from_receipt_ms <= result.bound_ms + AGREEMENT_TOL_MS
    assert result.worst_overshoot_ms <= AGREEMENT_TOL_MS


def test_stochastic_trials_do_not_reach_the_bound():
    # The worst case needs the fault to arrive at the receipt instant and the
    # poll phase to be adversarial; both are measure-zero events, so a random
    # run falls short. The gap is reported rather than tuned away.
    task = make_task(period=20.0, jitter=3.0, k=2)
    supervisor = make_supervisor(poll=5.0, jitter=1.0, overhead=0.5)
    result = simulate_detection(task, supervisor, seed=11, trials=2000)
    shortfall = result.bound_ms - result.max_latency_from_receipt_ms
    assert shortfall > 0.0
    assert shortfall < 5.0


def test_simulation_is_a_pure_function_of_its_seed():
    task = make_task()
    supervisor = make_supervisor()
    first = simulate_detection(task, supervisor, seed=5, trials=50)
    second = simulate_detection(task, supervisor, seed=5, trials=50)
    assert first == second


def test_different_seeds_give_different_samples():
    task = make_task()
    supervisor = make_supervisor()
    first = simulate_detection(task, supervisor, seed=5, trials=50)
    second = simulate_detection(task, supervisor, seed=6, trials=50)
    assert first.max_latency_from_receipt_ms != second.max_latency_from_receipt_ms


def test_latency_from_onset_is_never_above_latency_from_receipt():
    task = make_task()
    supervisor = make_supervisor()
    result = simulate_detection(task, supervisor, seed=3, trials=200)
    assert result.max_latency_from_onset_ms <= result.max_latency_from_receipt_ms + 1e-12


def test_gaussian_jitter_model_runs():
    task = make_task(sigma=0.5)
    result = simulate_detection(
        task, make_supervisor(), seed=1, trials=50, jitter_model="gaussian"
    )
    assert result.trials == 50
    assert math.isfinite(result.mean_latency_from_receipt_ms)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"jitter_model": "poisson"}, "jitter_model"),
        ({"trials": 0}, "trials"),
        ({"beats_before_fault": 0}, "beats_before_fault"),
    ],
)
def test_simulate_detection_rejects_bad_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        simulate_detection(make_task(), make_supervisor(), seed=0, **kwargs)


def test_false_trip_simulation_agrees_with_the_closed_form():
    # k = 0, J = 2.4 ms, sigma = 1.0 ms, so the closed form is
    # p = 0.5 erfc(2.4 / 2) = 0.5 erfc(1.2) = 0.0448430...
    task = make_task(period=20.0, jitter=2.4, k=0, sigma=1.0)
    closed = false_trip_probability(20.0, 2.4, 1.0, 0)
    assert closed == pytest.approx(0.5 * math.erfc(1.2), rel=1e-14)
    result = simulate_false_trips(task, seed=7, gaps=200_000)
    low, high = wilson_interval(result.false_trips, result.gaps)
    assert low <= closed <= high, f"closed form {closed} outside Wilson interval {low}..{high}"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_false_trip_simulation_agrees_over_several_seeds(seed):
    task = make_task(period=20.0, jitter=1.4, k=0, sigma=1.0)
    closed = false_trip_probability(20.0, 1.4, 1.0, 0)
    result = simulate_false_trips(task, seed=seed, gaps=100_000)
    low, high = wilson_interval(result.false_trips, result.gaps)
    assert low <= closed <= high


def test_false_trip_simulation_rejects_zero_sigma():
    with pytest.raises(ValueError, match="jitter_sigma_ms"):
        simulate_false_trips(make_task(sigma=0.0), seed=0, gaps=10)


def test_false_trip_simulation_rejects_zero_gaps():
    with pytest.raises(ValueError, match="gaps"):
        simulate_false_trips(make_task(sigma=1.0), seed=0, gaps=0)


def test_false_trip_rate_per_hour_matches_the_count():
    task = make_task(period=20.0, jitter=1.4, k=0, sigma=1.0)
    result = simulate_false_trips(task, seed=4, gaps=50_000)
    # 50 000 gaps at 20 ms each is 1 000 000 ms, which is 0.27777... hours.
    assert result.simulated_hours == pytest.approx(50_000 * 20.0 / 3_600_000.0, rel=1e-12)
    assert result.empirical_per_hour == pytest.approx(
        result.false_trips / result.simulated_hours, rel=1e-12
    )
