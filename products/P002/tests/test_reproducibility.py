"""Reproducibility tests: identical seeds must give identical results."""

from __future__ import annotations

import numpy as np
import pytest

from trackforge.dynamics import JitterPSD, synthesize_jitter
from trackforge.reacq import (
    AlwaysFullPolicy,
    ReacqConfig,
    ReacqEnv,
    evaluate_policy,
    train_q_learning,
)
from trackforge.scan import GaussianUncertainty, simulate_acquisition, spiral_scan
from trackforge.sim import DEFAULT_SCENARIO, Scenario, run_episode, run_monte_carlo


def test_scan_pattern_is_deterministic():
    u = GaussianUncertainty(3e-4)
    a = spiral_scan(u, 2e-5)
    b = spiral_scan(u, 2e-5)
    assert np.array_equal(a.points, b.points)


def test_acquisition_simulation_is_seed_reproducible():
    u = GaussianUncertainty(3e-4)
    p = spiral_scan(u, 2e-5)
    t = np.array([2e-4, 1e-4])
    a = simulate_acquisition(p, t, p_dwell=0.3, rng=np.random.default_rng(17))
    b = simulate_acquisition(p, t, p_dwell=0.3, rng=np.random.default_rng(17))
    assert a == b


def test_target_sampling_is_seed_reproducible():
    u = GaussianUncertainty(3e-4)
    a = u.sample(100, np.random.default_rng(5))
    b = u.sample(100, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_jitter_synthesis_is_bitwise_reproducible():
    psd = JitterPSD(1e-12, 3.0, 2.0)
    a = synthesize_jitter(psd, 4096, 5000.0, np.random.default_rng(31))
    b = synthesize_jitter(psd, 4096, 5000.0, np.random.default_rng(31))
    assert np.array_equal(a, b)


def test_episode_is_bitwise_reproducible():
    a = run_episode(DEFAULT_SCENARIO, seed=99)
    b = run_episode(DEFAULT_SCENARIO, seed=99)
    assert np.array_equal(a.los_error, b.los_error)
    assert np.array_equal(a.torque, b.torque)
    assert a.summary() == b.summary()


def test_episode_differs_between_seeds():
    a = run_episode(DEFAULT_SCENARIO, seed=1, keep_series=False)
    b = run_episode(DEFAULT_SCENARIO, seed=2, keep_series=False)
    assert a.summary() != b.summary()


def test_scenario_seed_used_when_no_override():
    a = run_episode(Scenario(seed=1234), keep_series=False)
    b = run_episode(Scenario(seed=1234), seed=1234, keep_series=False)
    assert a.summary() == b.summary()


def test_monte_carlo_is_reproducible():
    sc = Scenario(track_duration=0.6, spike_time=0.3)
    a = run_monte_carlo(sc, n_episodes=4, base_seed=50)
    b = run_monte_carlo(sc, n_episodes=4, base_seed=50)
    assert a == b


def test_reacq_environment_is_seed_reproducible():
    env_a, env_b = ReacqEnv(), ReacqEnv()
    env_a.reset(seed=1001)
    env_b.reset(seed=1001)
    ra, rb = [], []
    while not env_a.done:
        ra.append(env_a.step(0)[1])
    while not env_b.done:
        rb.append(env_b.step(0)[1])
    assert ra == rb


def test_q_learning_is_bitwise_reproducible():
    a = train_q_learning(ReacqConfig(), episodes=1000, seed=2024)
    b = train_q_learning(ReacqConfig(), episodes=1000, seed=2024)
    assert np.array_equal(a.q, b.q)
    assert a.metadata == b.metadata


def test_policy_evaluation_is_reproducible():
    cfg = ReacqConfig()
    pol = train_q_learning(cfg, episodes=800, seed=5)
    a = evaluate_policy(pol, cfg, n_episodes=120, seed=606)
    b = evaluate_policy(pol, cfg, n_episodes=120, seed=606)
    assert a == b


def test_common_random_numbers_across_policies():
    """Different policies choosing the same first action must see the same draw."""
    cfg = ReacqConfig()
    e1, e2 = ReacqEnv(cfg), ReacqEnv(cfg)
    e1.reset(seed=808)
    e2.reset(seed=808)
    r1 = e1.step(AlwaysFullPolicy().act(0))[1]
    r2 = e2.step(1)[1]
    assert r1 == pytest.approx(r2)
