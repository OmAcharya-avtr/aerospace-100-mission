"""The event-level simulator must reproduce every rate equation it generalises."""

from __future__ import annotations

import numpy as np
import pytest

from photoncount.afterpulse import observed_from_primary
from photoncount.deadtime import nonparalyzable_observed, paralyzable_observed
from photoncount.simulate import DetectorSpec, RunResult, simulate_run, simulate_windows

TAU = 1e-7


def _pooled_rate(n, duration, spec, seed, repeats=20):
    rng = np.random.default_rng(seed)
    total = 0
    for _ in range(repeats):
        total += simulate_run(n, duration, spec, rng).registered
    return total / (duration * repeats)


def test_ideal_counter_reproduces_the_incident_rate():
    spec = DetectorSpec()
    rate = _pooled_rate(1e5, 1e-2, spec, 0)
    # 20 windows x 1000 expected counts = 2e4 counts, relative SE 0.7 %.
    assert rate == pytest.approx(1e5, rel=0.03)


@pytest.mark.parametrize("x", [0.05, 0.3, 1.0, 2.0])
def test_nonparalyzable_rate_equation_recovered(x):
    spec = DetectorSpec(dead_time_s=TAU, model="nonparalyzable")
    n = x / TAU
    predicted = float(nonparalyzable_observed(n, TAU)[0])
    measured = _pooled_rate(n, 2e-3, spec, 11)
    # 20 x 2 ms at these rates is >= 2e4 counts; 3 % tolerance is ~4 SE.
    assert measured == pytest.approx(predicted, rel=0.03)


@pytest.mark.parametrize("x", [0.05, 0.3, 1.0, 2.0])
def test_paralyzable_rate_equation_recovered(x):
    spec = DetectorSpec(dead_time_s=TAU, model="paralyzable")
    n = x / TAU
    predicted = float(paralyzable_observed(n, TAU)[0])
    measured = _pooled_rate(n, 2e-3, spec, 23)
    assert measured == pytest.approx(predicted, rel=0.04)


def test_paralyzable_observed_rate_falls_past_the_maximum():
    """The simulator must reproduce the non-monotonicity, not just the formula."""
    spec = DetectorSpec(dead_time_s=TAU, model="paralyzable")
    below = _pooled_rate(0.5 / TAU, 2e-3, spec, 1)
    at = _pooled_rate(1.0 / TAU, 2e-3, spec, 1)
    above = _pooled_rate(3.0 / TAU, 2e-4, spec, 1)
    assert at > below
    assert above < below


def test_afterpulsing_rate_equation_recovered_without_dead_time():
    spec = DetectorSpec(dead_time_s=0.0, afterpulse_probability=0.2, afterpulse_mean_delay_s=1e-9)
    measured = _pooled_rate(1e5, 1e-2, spec, 5)
    predicted = float(observed_from_primary(1e5, 0.2, "cascading")[0])
    assert measured == pytest.approx(predicted, rel=0.02)


def test_first_order_afterpulsing_is_weaker_than_cascading():
    rng = np.random.default_rng(3)
    casc = DetectorSpec(
        dead_time_s=0.0, afterpulse_probability=0.3, afterpulse_mean_delay_s=1e-9, cascading=True
    )
    first = DetectorSpec(
        dead_time_s=0.0, afterpulse_probability=0.3, afterpulse_mean_delay_s=1e-9, cascading=False
    )
    a = simulate_run(1e5, 1e-1, casc, rng).registered
    b = simulate_run(1e5, 1e-1, first, rng).registered
    assert a > b


def test_dead_time_suppresses_afterpulses():
    """A dead time long against the release delay must remove nearly all afterpulses."""
    rng = np.random.default_rng(9)
    long_dead = DetectorSpec(
        dead_time_s=1e-6,
        model="nonparalyzable",
        afterpulse_probability=0.3,
        afterpulse_mean_delay_s=1e-9,
    )
    run = simulate_run(1e4, 1e-1, long_dead, rng)
    assert run.afterpulse_fraction < 0.01
    short_dead = DetectorSpec(
        dead_time_s=1e-10,
        model="nonparalyzable",
        afterpulse_probability=0.3,
        afterpulse_mean_delay_s=1e-6,
    )
    run2 = simulate_run(1e4, 1e-1, short_dead, rng)
    assert run2.afterpulse_fraction > 0.2


def test_counts_are_conserved():
    rng = np.random.default_rng(4)
    run = simulate_run(
        2e5,
        1e-2,
        DetectorSpec(
            dead_time_s=TAU,
            model="paralyzable",
            afterpulse_probability=0.1,
            afterpulse_mean_delay_s=3e-7,
        ),
        rng,
    )
    assert run.registered == run.registered_primary + run.registered_afterpulse
    assert run.primary_events == run.registered_primary + run.lost_primary


def test_timestamps_are_sorted_and_respect_the_dead_time():
    rng = np.random.default_rng(6)
    spec = DetectorSpec(dead_time_s=TAU, model="nonparalyzable")
    run = simulate_run(1e6, 1e-3, spec, rng, keep_timestamps=True)
    assert run.timestamps.size == run.registered
    gaps = np.diff(run.timestamps)
    assert np.all(gaps >= TAU - 1e-15)
    assert np.all(np.diff(run.timestamps) > 0)


def test_determinism_for_a_seed():
    spec = DetectorSpec(dead_time_s=TAU, model="paralyzable", afterpulse_probability=0.1)
    a = simulate_run(1e5, 1e-3, spec, np.random.default_rng(77))
    b = simulate_run(1e5, 1e-3, spec, np.random.default_rng(77))
    assert (a.registered, a.lost_primary, a.registered_afterpulse) == (
        b.registered,
        b.lost_primary,
        b.registered_afterpulse,
    )


def test_fano_factor_below_one_with_dead_time_and_above_one_with_afterpulsing():
    rng = np.random.default_rng(31)
    dead = simulate_windows(
        2e6, 1e-3, 200, DetectorSpec(dead_time_s=TAU, model="nonparalyzable"), rng
    )
    assert dead["fano_factor"] < 0.9
    ap = simulate_windows(
        2e5,
        1e-3,
        200,
        DetectorSpec(dead_time_s=0.0, afterpulse_probability=0.25, afterpulse_mean_delay_s=1e-9),
        rng,
    )
    assert ap["fano_factor"] > 1.2


def test_zero_rate_gives_zero_counts():
    run = simulate_run(0.0, 1e-3, DetectorSpec(dead_time_s=TAU), np.random.default_rng(0))
    assert run.registered == 0
    assert run.observed_rate_hz == 0.0
    assert run.afterpulse_fraction == 0.0


def test_run_result_rate_property():
    r = RunResult(incident_rate_hz=1.0, duration_s=2.0, registered=10)
    assert r.observed_rate_hz == 5.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dead_time_s": -1e-9},
        {"model": "extending"},
        {"afterpulse_probability": 1.0},
        {"afterpulse_probability": -0.1},
        {"afterpulse_mean_delay_s": 0.0},
    ],
)
def test_invalid_spec_raises(kwargs):
    with pytest.raises(ValueError):
        DetectorSpec(**kwargs)


@pytest.mark.parametrize(("n", "t"), [(-1.0, 1e-3), (1e5, 0.0), (1e5, -1.0)])
def test_invalid_run_arguments_raise(n, t):
    with pytest.raises(ValueError):
        simulate_run(n, t, DetectorSpec(), np.random.default_rng(0))


def test_oversized_run_is_refused():
    with pytest.raises(ValueError, match="beyond the intended scale"):
        simulate_run(1e9, 1.0, DetectorSpec(), np.random.default_rng(0))


def test_simulate_windows_needs_at_least_two_windows():
    with pytest.raises(ValueError):
        simulate_windows(1e5, 1e-3, 1, DetectorSpec(), np.random.default_rng(0))
