"""Unit tests for trackforge.reacq (environment, baselines, Q-learning)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trackforge.reacq import (
    ACTIONS,
    AlwaysFullPolicy,
    AlwaysLocalPolicy,
    QLearningPolicy,
    ReacqConfig,
    ReacqEnv,
    compare_policies,
    evaluate_policy,
    train_q_learning,
)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [
    {"sigma0": 0.0}, {"drift_rate": -1.0}, {"cone_radius": 0.0},
    {"coverage_rate": 0.0}, {"p_detect": 0.0}, {"p_detect": 1.5},
    {"kappa": -1.0}, {"max_time": 0.0},
])
def test_config_validation(kw):
    with pytest.raises(ValueError):
        ReacqConfig(**kw)


def test_config_defaults_are_physical():
    c = ReacqConfig()
    assert c.sigma0 > 0 and c.cone_radius > c.sigma0
    assert 0 < c.p_detect <= 1


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
def test_env_state_space_size_matches_bins():
    env = ReacqEnv()
    assert env.N_STATES == int(np.prod(env.SHAPE))
    assert env.N_STATES == 5 * 4 * 4 * 4


def test_env_reset_returns_valid_state():
    env = ReacqEnv()
    s = env.reset(seed=1)
    assert 0 <= s < env.N_STATES
    assert env.t == 0.0 and env.n_attempts == 0 and not env.done


def test_env_reset_is_seeded():
    a = ReacqEnv()
    b = ReacqEnv()
    a.reset(seed=42)
    b.reset(seed=42)
    assert np.array_equal(a.delta0, b.delta0)
    assert np.array_equal(a.p_lk, b.p_lk)


def test_env_different_seeds_give_different_episodes():
    a, b = ReacqEnv(), ReacqEnv()
    a.reset(seed=1)
    b.reset(seed=2)
    assert not np.array_equal(a.delta0, b.delta0)


def test_env_sigma_growth_matches_equation_12():
    env = ReacqEnv(ReacqConfig(sigma0=5e-5, drift_rate=1e-4))
    env.reset(seed=0)
    got = env.observation()[2]
    assert got == pytest.approx(5e-5)
    env.t = 2.0
    want = math.sqrt(5e-5**2 + (1e-4 * 2.0) ** 2)
    assert env.observation()[2] == pytest.approx(want)


def test_env_action_plan_durations_follow_area_model():
    """Disc of radius R costs pi R^2 / coverage_rate; FULL uses cone_radius."""
    cfg = ReacqConfig(coverage_rate=1e-6, cone_radius=1e-3)
    env = ReacqEnv(cfg)
    env.reset(seed=0)
    dur_full, centre, r_in, r_out = env.action_plan(1)
    assert r_in == 0.0 and r_out == pytest.approx(cfg.cone_radius)
    assert centre == pytest.approx(np.zeros(2))
    assert dur_full == pytest.approx(math.pi * cfg.cone_radius**2 / cfg.coverage_rate)


def test_env_local_radius_scales_with_sigma():
    cfg = ReacqConfig(k_local=3.0)
    env = ReacqEnv(cfg)
    env.reset(seed=0)
    _, _, _, r0 = env.action_plan(0)
    env.t = 5.0
    _, _, _, r1 = env.action_plan(0)
    assert r1 > r0
    assert r0 == pytest.approx(3.0 * cfg.sigma0)


def test_env_ring_starts_at_searched_radius():
    env = ReacqEnv()
    env.reset(seed=0)
    env.step(0)  # LOCAL first, sets r_searched
    _, _, r_in, r_out = env.action_plan(2)
    assert r_in == pytest.approx(env.r_searched)
    assert r_out > r_in


def test_env_rejects_invalid_action():
    env = ReacqEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.action_plan(7)


def test_env_step_after_done_raises():
    env = ReacqEnv(ReacqConfig(p_detect=1.0))
    env.reset(seed=0)
    while not env.done:
        env.step(1)
    with pytest.raises(RuntimeError):
        env.step(1)


def test_env_reward_is_negative_duration():
    env = ReacqEnv()
    env.reset(seed=3)
    dur, _, _, _ = env.action_plan(1)
    _, r, done, info = env.step(1)
    assert info["duration_s"] == pytest.approx(dur)
    if done and not info.get("timeout"):
        assert r == pytest.approx(-dur)


def test_env_time_advances_monotonically():
    env = ReacqEnv()
    env.reset(seed=5)
    times = []
    while not env.done:
        env.step(0)
        times.append(env.t)
    assert all(b > a for a, b in zip(times, times[1:]))


def test_env_timeout_is_terminal_with_penalty():
    env = ReacqEnv(ReacqConfig(max_time=0.5, p_detect=1e-9))
    env.reset(seed=0)
    _, r, done, info = env.step(1)
    assert done and info.get("timeout") and r < -0.5


def test_env_target_position_drifts_linearly():
    env = ReacqEnv()
    env.reset(seed=8)
    p0 = env.target_position(0.0)
    p1 = env.target_position(1.0)
    p2 = env.target_position(2.0)
    assert (p2 - p1) == pytest.approx(p1 - p0)


def test_env_encode_is_within_range_over_many_states():
    env = ReacqEnv()
    for s in range(50):
        st = env.reset(seed=s)
        assert 0 <= st < env.N_STATES
        while not env.done:
            st, _, _, _ = env.step(int(s % 3))
            assert 0 <= st < env.N_STATES


# --------------------------------------------------------------------------
# baselines (classical, implemented FIRST)
# --------------------------------------------------------------------------
def test_baselines_choose_fixed_actions():
    assert AlwaysFullPolicy().act(0) == ACTIONS.index("FULL")
    assert AlwaysLocalPolicy().act(123) == ACTIONS.index("LOCAL")


def test_baseline_confidence_is_one():
    assert AlwaysFullPolicy().confidence(0) == 1.0


def test_baseline_evaluation_produces_finite_metrics():
    cfg = ReacqConfig()
    res = evaluate_policy(AlwaysFullPolicy(), cfg, n_episodes=200, seed=5)
    assert res["policy"] == "baseline-always-full"
    assert 0.0 < res["mean_time_s"] <= cfg.max_time
    assert res["ci_low_s"] < res["mean_time_s"] < res["ci_high_s"]
    assert 0.0 <= res["success_rate"] <= 1.0
    assert res["action_mix"]["FULL"] == res["n_episodes"] * res["mean_attempts"]


def test_local_baseline_uses_more_attempts_than_full():
    cfg = ReacqConfig()
    full = evaluate_policy(AlwaysFullPolicy(), cfg, n_episodes=300, seed=11)
    loc = evaluate_policy(AlwaysLocalPolicy(), cfg, n_episodes=300, seed=11)
    assert loc["mean_attempts"] > full["mean_attempts"]


def test_evaluation_uses_common_random_numbers():
    """Both baselines must see identical episodes for the same seed."""
    cfg = ReacqConfig(p_detect=1.0)
    e1, e2 = ReacqEnv(cfg), ReacqEnv(cfg)
    e1.reset(seed=77)
    e2.reset(seed=77)
    assert np.array_equal(e1.target_position(1.0), e2.target_position(1.0))


# --------------------------------------------------------------------------
# Q-learning
# --------------------------------------------------------------------------
def test_training_shapes_and_metadata():
    pol = train_q_learning(ReacqConfig(), episodes=300, seed=1)
    assert pol.q.shape == (ReacqEnv.N_STATES, len(ACTIONS))
    assert pol.visits.sum() > 0
    assert pol.metadata["episodes"] == 300 and pol.metadata["seed"] == 1


def test_training_is_reproducible_bitwise():
    a = train_q_learning(ReacqConfig(), episodes=400, seed=7)
    b = train_q_learning(ReacqConfig(), episodes=400, seed=7)
    assert np.array_equal(a.q, b.q)
    assert np.array_equal(a.visits, b.visits)


def test_training_seeds_differ():
    a = train_q_learning(ReacqConfig(), episodes=400, seed=7)
    b = train_q_learning(ReacqConfig(), episodes=400, seed=8)
    assert not np.array_equal(a.q, b.q)


def test_learned_q_values_are_non_positive():
    """Rewards are negative durations, so optimal values must be <= 0."""
    pol = train_q_learning(ReacqConfig(), episodes=800, seed=3)
    visited = pol.visits > 0
    assert np.all(pol.q[visited] <= 1e-9)


@pytest.mark.parametrize("kw", [{"episodes": 0}, {"gamma": 0.0}, {"gamma": 1.5}])
def test_training_input_validation(kw):
    base = {"episodes": 10, "gamma": 0.99}
    base.update(kw)
    with pytest.raises(ValueError):
        train_q_learning(ReacqConfig(), **base)


def test_policy_confidence_bounds_and_fallback():
    q = np.zeros((ReacqEnv.N_STATES, len(ACTIONS)))
    v = np.zeros_like(q, dtype=np.int64)
    pol = QLearningPolicy(q=q, visits=v)
    assert pol.act(0) == pol.fallback_action  # unvisited -> baseline action
    assert pol.confidence(0) == 0.0
    q[5] = [-1.0, -10.0, -10.0]
    v[5] = [100, 100, 100]
    a, c = pol.act_with_confidence(5)
    assert a == 0
    assert 0.0 < c <= 1.0


def test_policy_confidence_low_when_actions_are_tied():
    q = np.zeros((ReacqEnv.N_STATES, len(ACTIONS)))
    v = np.full_like(q, 100, dtype=np.int64)
    q[3] = [-5.0, -5.0, -5.0]
    pol = QLearningPolicy(q=q, visits=v)
    assert pol.confidence(3) == pytest.approx(0.0)


def test_policy_confidence_rises_with_support():
    q = np.zeros((ReacqEnv.N_STATES, len(ACTIONS)))
    v = np.zeros_like(q, dtype=np.int64)
    q[2] = [-1.0, -2.0, -2.0]
    v[2] = [3, 3, 3]
    low = QLearningPolicy(q=q, visits=v).confidence(2)
    v2 = v.copy()
    v2[2] = [300, 300, 300]
    high = QLearningPolicy(q=q, visits=v2).confidence(2)
    assert high > low


def test_greedy_actions_vector_matches_act():
    pol = train_q_learning(ReacqConfig(), episodes=500, seed=4)
    ga = pol.greedy_actions()
    assert ga.shape == (ReacqEnv.N_STATES,)
    for s in (0, 10, 100, 300):
        assert ga[s] == pol.act(s)


def test_compare_policies_returns_one_row_per_policy():
    cfg = ReacqConfig()
    rows = compare_policies(
        {"full": AlwaysFullPolicy(), "local": AlwaysLocalPolicy()},
        cfg,
        n_episodes=100,
        seed=13,
    )
    assert len(rows) == 2
    assert {r["policy"] for r in rows} == {
        "baseline-always-full", "baseline-always-local"
    }


def test_evaluation_is_deterministic_for_fixed_seed():
    cfg = ReacqConfig()
    a = evaluate_policy(AlwaysLocalPolicy(), cfg, n_episodes=150, seed=21)
    b = evaluate_policy(AlwaysLocalPolicy(), cfg, n_episodes=150, seed=21)
    assert a == b
