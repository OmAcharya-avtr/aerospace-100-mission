"""The assurance accounting: run algebra, dwell statistics, pairing checks."""

from __future__ import annotations

import numpy as np
import pytest

from simplexguard import (
    account,
    disturbance_sequence,
    dwell_statistics,
    mode_runs,
    simulate_baseline,
    simulate_guarded,
    simulate_unguarded,
    square_wave_reference,
)


def test_mode_runs_known_answer():
    # [0,0,1,1,1,0] splits into (0, start 0, len 2), (1, 2, 3), (0, 5, 1).
    assert mode_runs(np.array([0, 0, 1, 1, 1, 0])) == [(0, 0, 2), (1, 2, 3), (0, 5, 1)]
    assert mode_runs(np.array([], dtype=int)) == []
    assert mode_runs(np.array([1])) == [(1, 0, 1)]


def test_mode_runs_cover_the_whole_sequence(rng):
    modes = rng.integers(0, 2, size=200)
    runs = mode_runs(modes)
    assert sum(length for _, _, length in runs) == modes.size
    rebuilt = np.concatenate([np.full(length, value) for value, _, length in runs])
    assert np.array_equal(rebuilt, modes)


def test_dwell_statistics_known_answer():
    # lengths [1, 1, 2, 4, 7]: n = 5, min 1, median 2, mean 3.0, max 7,
    # fraction of length one = 2/5 = 0.4, total 15 steps.
    stats = dwell_statistics([1, 1, 2, 4, 7], dt=0.05)
    assert stats.n_intervals == 5
    assert stats.minimum == 1
    assert stats.median == pytest.approx(2.0)
    assert stats.mean == pytest.approx(3.0)
    assert stats.maximum == 7
    assert stats.fraction_of_length_one == pytest.approx(0.4)
    assert stats.total_steps == 15
    assert stats.mean_seconds == pytest.approx(0.15)


def test_dwell_statistics_histogram_known_answer():
    stats = dwell_statistics([1, 1, 2, 12])
    hist = stats.histogram([1, 1, 2, 12], max_bin=3)
    assert hist == {"1": 2, "2": 1, "3": 0, ">3": 1}


def test_dwell_statistics_empty_and_invalid():
    empty = dwell_statistics([])
    assert empty.n_intervals == 0
    assert empty.mean == 0.0
    with pytest.raises(ValueError, match="at least 1 step"):
        dwell_statistics([0, 2])


def _three_runs(plant, controllers, guard, seed=51, steps=600, mode="uniform"):
    rng = np.random.default_rng(seed)
    w = disturbance_sequence(plant, steps, rng, mode)
    ref = square_wave_reference(0.18, 80, plant.n_states)
    return (
        simulate_guarded(plant, guard, controllers[1], steps, ref, w),
        simulate_unguarded(plant, controllers[1], steps, ref, w),
        simulate_baseline(plant, controllers[0], steps, ref, w),
    )


def test_report_is_internally_consistent(plant, controllers, guard, invariant_set):
    g, u, b = _three_runs(plant, controllers, guard)
    report = account(g, u, b, invariant_set)
    assert report.n_steps == g.n_steps
    assert report.switches_per_1000_steps == pytest.approx(
        1000.0 * report.n_switches / report.n_steps
    )
    assert report.switches_per_second == pytest.approx(
        report.n_switches / (report.n_steps * plant.dt)
    )
    assert report.baseline_dwell.total_steps + report.performance_dwell.total_steps == (
        report.n_steps
    )
    assert report.baseline_fraction == pytest.approx(
        report.baseline_dwell.total_steps / report.n_steps
    )
    # The number of authority changes is one less than the number of runs.
    assert report.n_switches == (
        report.baseline_dwell.n_intervals + report.performance_dwell.n_intervals - 1
    )


def test_conservatism_cost_sits_between_the_two_extremes(
    plant, controllers, guard, invariant_set
):
    g, u, b = _three_runs(plant, controllers, guard)
    report = account(g, u, b, invariant_set)
    assert report.unguarded_cost < report.guarded_cost < report.baseline_cost
    assert report.conservatism_cost_ratio > 1.0
    assert report.conservatism_cost_absolute > 0.0
    assert 0.0 < report.performance_gap_recovered < 1.0


def test_guard_eliminates_the_violations_the_unguarded_run_makes(
    plant, controllers, guard, invariant_set
):
    g, u, b = _three_runs(plant, controllers, guard)
    report = account(g, u, b, invariant_set)
    assert report.guarded_violations == 0
    assert report.guarded_invariant_exits == 0
    assert report.unguarded_violations > 0
    assert report.unguarded_invariant_exits >= report.unguarded_violations
    assert report.guarded_worst_residual < 0.0
    assert report.unguarded_worst_residual > 0.0


def test_margin_is_negative_exactly_at_the_firing_steps(
    plant, controllers, guard, invariant_set
):
    g, u, b = _three_runs(plant, controllers, guard)
    report = account(g, u, b, invariant_set)
    assert report.margin_at_switch_median < 0.0
    assert report.margin_median > 0.0


def test_minimum_dwell_reduces_the_switch_rate(
    plant, controllers, invariant_set
):
    from simplexguard import SimplexGuard

    rates = []
    for dwell in (1, 4, 10):
        g = SimplexGuard(plant, controllers[0], invariant_set, min_baseline_dwell=dwell)
        guarded, unguarded, base = _three_runs(plant, controllers, g)
        report = account(guarded, unguarded, base, invariant_set)
        rates.append(report.switches_per_1000_steps)
        # Holding the baseline longer is always safe, which is the point.
        assert report.guarded_violations == 0
        assert report.guarded_invariant_exits == 0
        assert report.baseline_dwell.minimum >= dwell or report.baseline_dwell.n_intervals == 0
    assert rates[0] > rates[1] > rates[2]


def test_account_refuses_unpaired_episodes(plant, controllers, guard, invariant_set):
    g, u, b = _three_runs(plant, controllers, guard, seed=51)
    _, u2, _ = _three_runs(plant, controllers, guard, seed=99)
    with pytest.raises(ValueError, match="different disturbance sequences"):
        account(g, u2, b, invariant_set)
    with pytest.raises(ValueError, match="baseline-only episode"):
        account(g, u, _three_runs(plant, controllers, guard, seed=99)[2], invariant_set)


def test_account_refuses_different_lengths(plant, controllers, guard, invariant_set):
    g, u, b = _three_runs(plant, controllers, guard, steps=200)
    g2, _, _ = _three_runs(plant, controllers, guard, steps=100)
    with pytest.raises(ValueError, match="same length"):
        account(g2, u, b, invariant_set)


def test_describe_contains_all_six_headings(plant, controllers, guard, invariant_set):
    g, u, b = _three_runs(plant, controllers, guard, steps=300)
    text = account(g, u, b, invariant_set).describe()
    for heading in (
        "1  switch rate",
        "2  dwell-time distribution",
        "3  authority share",
        "4  conservatism cost",
        "5  constraint violations",
        "6  invariant-set exits",
    ):
        assert heading in text


def test_vertex_sampler_also_gives_zero_violations(
    plant, controllers, guard, invariant_set
):
    g, u, b = _three_runs(plant, controllers, guard, mode="vertex")
    report = account(g, u, b, invariant_set)
    assert report.guarded_violations == 0
    assert report.guarded_invariant_exits == 0
