"""Performance tests.

Thresholds are deliberately loose (roughly 5-10x below the rates measured on
the 2-core reference machine, recorded in validation/VALIDATION.md) so the
suite is a guard against algorithmic regressions rather than a machine
benchmark.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from trackbench.dynamics import JitterPSD, synthesize_jitter
from trackbench.reacq import AlwaysLocalPolicy, ReacqConfig, evaluate_policy, train_q_learning
from trackbench.scan import GaussianUncertainty, coverage_fraction, spiral_scan
from trackbench.sim import DEFAULT_SCENARIO, Scenario, run_episode, sim_steps_per_second


def test_closed_loop_throughput_above_floor():
    """Reference machine: ~3.5e4 steps/s; require > 5e3 steps/s."""
    perf = sim_steps_per_second(DEFAULT_SCENARIO, duration=0.5)
    assert perf["steps_per_second"] > 5_000


def test_closed_loop_runs_faster_than_realtime():
    perf = sim_steps_per_second(DEFAULT_SCENARIO, duration=0.5)
    assert perf["realtime_factor"] > 1.0


def test_full_episode_under_two_seconds():
    t0 = time.perf_counter()
    run_episode(DEFAULT_SCENARIO, seed=1, keep_series=False)
    assert time.perf_counter() - t0 < 2.0


def test_spiral_generation_is_fast():
    u = GaussianUncertainty(1e-3)
    t0 = time.perf_counter()
    p = spiral_scan(u, 1e-5)
    dt = time.perf_counter() - t0
    assert p.n_points > 10_000
    assert dt < 5.0


def test_coverage_query_is_vectorised():
    u = GaussianUncertainty(3e-4)
    p = spiral_scan(u, 2e-5)
    t0 = time.perf_counter()
    coverage_fraction(p, u, n_samples=50_000, rng=np.random.default_rng(0))
    assert time.perf_counter() - t0 < 10.0


def test_jitter_synthesis_scales_to_a_million_samples():
    t0 = time.perf_counter()
    x = synthesize_jitter(JitterPSD(1e-12, 3.0), 2**20, 5000.0, np.random.default_rng(0))
    assert x.size == 2**20
    assert time.perf_counter() - t0 < 5.0


def test_q_learning_training_within_compute_budget():
    """20 000 episodes must train in well under the 3-minute mission budget."""
    t0 = time.perf_counter()
    train_q_learning(ReacqConfig(), episodes=20_000, seed=12345)
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0


def test_monte_carlo_evaluation_within_budget():
    t0 = time.perf_counter()
    evaluate_policy(AlwaysLocalPolicy(), ReacqConfig(), n_episodes=2_000, seed=999)
    assert time.perf_counter() - t0 < 30.0


def test_short_scenario_episode_scales_linearly_in_duration():
    short = Scenario(track_duration=0.5, spike_time=0.25)
    long = Scenario(track_duration=2.0, spike_time=1.0)
    t0 = time.perf_counter()
    run_episode(short, seed=1, keep_series=False)
    t_short = time.perf_counter() - t0
    t0 = time.perf_counter()
    run_episode(long, seed=1, keep_series=False)
    t_long = time.perf_counter() - t0
    # 4x the simulated time must not cost more than 12x the wall time
    assert t_long < 12.0 * max(t_short, 1e-3)


def test_step_metrics_do_not_allocate_quadratically():
    """A 10x longer run must not cost more than 30x the wall time."""
    from trackbench.control import PIDController, pid_gains_from_bandwidth, step_response
    from trackbench.dynamics import GimbalAxis

    kp, ki, kd = pid_gains_from_bandwidth(0.05, 31.4, 0.707)

    def run(duration: float) -> float:
        ax = GimbalAxis(0.05, 0.02, 2.0, 1.0)
        t0 = time.perf_counter()
        step_response(ax, PIDController(kp, ki, kd, 2.0), 1e-4, 1e-3, duration)
        return time.perf_counter() - t0

    assert run(2.0) < 30.0 * max(run(0.2), 1e-4)


@pytest.mark.parametrize("n", [1000, 10_000])
def test_scan_time_property_is_constant_time(n):
    u = GaussianUncertainty(3e-4)
    p = spiral_scan(u, 2e-5)
    t0 = time.perf_counter()
    for _ in range(n):
        _ = p.scan_time
    assert time.perf_counter() - t0 < 1.0
