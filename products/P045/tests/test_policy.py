"""Policies: environment, seed discipline, baselines, and the learned model."""

from __future__ import annotations

import numpy as np
import pytest

from arqlonghaul.datasets import (
    DEFAULT_ACTIONS,
    SEED_SPLIT,
    FadeHarqEnv,
    SeedSplit,
    effective_memory_rounds,
    long_burst_env,
    nominal_rtt_symbols,
    short_burst_env,
)
from arqlonghaul.policy import (
    CONSECUTIVE_HARD_CAP,
    FEATURE_NAMES,
    QUANTISATION_LEVELS,
    AnalyticFixedPolicy,
    EscalatingPolicy,
    LearnedRedundancyPolicy,
    Policy,
    RandomPolicy,
    analytic_fixed,
    collect_transitions,
    evaluate,
    run_episode,
    tune_escalating,
    tune_fixed,
)


def test_seed_sets_are_disjoint() -> None:
    assert set(SEED_SPLIT.fit).isdisjoint(SEED_SPLIT.tune)
    assert set(SEED_SPLIT.fit).isdisjoint(SEED_SPLIT.report)
    assert set(SEED_SPLIT.tune).isdisjoint(SEED_SPLIT.report)
    assert SEED_SPLIT.summary() == {"n_fit": 200, "n_tune": 200, "n_report": 400}


def test_overlapping_seed_sets_are_rejected() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        SeedSplit(fit=(1, 2), tune=(2, 3), report=(4,))


def test_environment_stationary_quantities() -> None:
    # p_gb = 0.01, p_bg = 0.04  =>  pi_b = 0.2, mean burst 25 rounds.
    env = FadeHarqEnv(p_gb=0.01, p_bg=0.04)
    assert env.pi_bad == pytest.approx(0.2)
    assert env.mean_burst_rounds == pytest.approx(25.0)
    assert env.ber_bad > env.ber_good
    assert env.ber_good < env.ber_mixture < env.ber_bad
    assert env.ber_mixture == pytest.approx(0.8 * env.ber_good + 0.2 * env.ber_bad)


def test_named_environments_share_a_marginal_but_not_a_memory() -> None:
    a, b = long_burst_env(), short_burst_env()
    assert a.ber_mixture == pytest.approx(b.ber_mixture)
    assert a.pi_bad == pytest.approx(b.pi_bad)
    assert effective_memory_rounds(a) > 10.0
    assert effective_memory_rounds(b) < 1.0


def test_environment_validation() -> None:
    with pytest.raises(ValueError, match="k must be"):
        FadeHarqEnv(k=0)
    with pytest.raises(ValueError, match="max_rounds"):
        FadeHarqEnv(max_rounds=0)
    with pytest.raises(ValueError, match="actions"):
        FadeHarqEnv(actions=())
    with pytest.raises(ValueError, match="Bad state is the faded one"):
        FadeHarqEnv(esn0_good_db=-5.0, esn0_bad_db=1.0)
    with pytest.raises(ValueError, match="drop_penalty"):
        FadeHarqEnv(drop_penalty=-1.0)
    with pytest.raises(ValueError, match="n_rounds"):
        long_burst_env().state_sequence(0, np.random.default_rng(0))


def test_nominal_rtt_symbols() -> None:
    assert nominal_rtt_symbols(2e6, 0.25) == pytest.approx(5e5)
    with pytest.raises(ValueError):
        nominal_rtt_symbols(0.0, 1.0)


def test_feature_vector_shape_and_quantisation() -> None:
    env = long_burst_env()
    _, tr = run_episode(env, AnalyticFixedPolicy(2, 2), 40, 1000, record=True)
    feats = tr["features"]
    assert feats.shape[1] == len(FEATURE_NAMES) == 8
    # the two quantised columns land on multiples of 1/QUANTISATION_LEVELS
    for col in (4, 5):
        scaled = feats[:, col] * QUANTISATION_LEVELS
        assert np.allclose(scaled, np.round(scaled))
    assert feats[:, 6].max() <= CONSECUTIVE_HARD_CAP
    # rtt/k is constant and correct
    assert np.allclose(feats[:, 7], env.rtt_symbols / env.k)


def test_run_episode_is_deterministic() -> None:
    env = long_burst_env()
    pol = AnalyticFixedPolicy(2, 1)
    a, _ = run_episode(env, pol, 60, 1234)
    b, _ = run_episode(env, pol, 60, 1234)
    assert a.goodput == b.goodput
    assert a.cost == b.cost
    assert a.rounds == b.rounds


def test_run_episode_accounting() -> None:
    env = long_burst_env()
    stats, _ = run_episode(env, AnalyticFixedPolicy(2, 2), 100, 1001)
    assert stats.frames == 100
    assert 0 <= stats.delivered <= 100
    assert stats.rounds >= stats.frames
    assert stats.rounds <= stats.frames * env.max_rounds
    assert stats.action_counts.sum() == stats.rounds
    assert stats.elapsed_symbols > stats.symbols_sent  # feedback waits are charged
    assert stats.residual_fer == pytest.approx(1.0 - stats.delivered / 100)
    d = stats.as_dict()
    assert d["frames"] == 100


def test_run_episode_validation() -> None:
    with pytest.raises(ValueError, match="n_frames"):
        run_episode(long_burst_env(), AnalyticFixedPolicy(0, 0), 0, 1)


def test_policy_returning_a_bad_action_is_rejected() -> None:
    class Rogue(Policy):
        name = "rogue"

        def act(self, obs: np.ndarray) -> int:
            return 99

    with pytest.raises(ValueError, match="outside"):
        run_episode(long_burst_env(), Rogue(), 5, 1)


def test_escalating_policy_saturates() -> None:
    pol = EscalatingPolicy(base_index=3, n_actions=5)
    obs = np.zeros(len(FEATURE_NAMES))
    for m, expected in ((1, 3), (2, 4), (3, 4), (4, 4)):
        obs[0] = m
        assert pol.act(obs) == expected


def test_analytic_policy_uses_the_first_action_only_in_round_one() -> None:
    pol = AnalyticFixedPolicy(first_index=0, later_index=4)
    obs = np.zeros(len(FEATURE_NAMES))
    obs[0] = 1
    assert pol.act(obs) == 0
    obs[0] = 2
    assert pol.act(obs) == 4


def test_random_policy_covers_the_action_set() -> None:
    pol = RandomPolicy(5, np.random.default_rng(0))
    seen = {pol.act(np.zeros(len(FEATURE_NAMES))) for _ in range(300)}
    assert seen == set(range(5))


def test_analytic_fixed_uses_no_data_and_returns_a_valid_pair() -> None:
    env = long_burst_env()
    pol, detail = analytic_fixed(env)
    assert 0 <= pol.first_index < len(env.actions)
    assert 0 <= pol.later_index < len(env.actions)
    assert detail["mixture_ber"] == pytest.approx(env.ber_mixture)
    assert detail["first_action"] in DEFAULT_ACTIONS
    assert 0.0 < detail["analytic_goodput"] < 1.0


def test_tuning_only_reads_the_seeds_it_is_given() -> None:
    env = long_burst_env()
    seeds = SEED_SPLIT.tune[:6]
    pol, detail = tune_fixed(env, seeds, 30)
    assert detail["tune_cost_per_frame"] > 0.0
    assert 0 <= pol.first_index < len(env.actions)
    pol2, detail2 = tune_escalating(env, seeds, 30)
    assert detail2["base_action"] in DEFAULT_ACTIONS
    assert 0 <= pol2.base_index < len(env.actions)


def test_evaluate_requires_seeds() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        evaluate(long_burst_env(), AnalyticFixedPolicy(0, 0), [], 10)


def test_evaluate_returns_the_documented_keys() -> None:
    out = evaluate(long_burst_env(), AnalyticFixedPolicy(2, 2), SEED_SPLIT.report[:4], 25)
    for key in (
        "goodput",
        "goodput_stderr",
        "residual_fer",
        "mean_rounds",
        "cost_per_frame",
        "cost_stderr",
        "symbols_per_frame",
        "n_seeds",
        "n_frames_total",
    ):
        assert key in out
    assert out["n_seeds"] == 4.0
    assert out["n_frames_total"] == 100.0


def test_learned_policy_fits_acts_and_reports_uncertainty() -> None:
    env = long_burst_env()
    tr = collect_transitions(env, SEED_SPLIT.fit[:8], 60, rng_seed=1)
    assert tr["features"].shape[0] == tr["actions"].shape[0] == tr["cost"].shape[0]
    pol = LearnedRedundancyPolicy(len(env.actions), n_estimators=20, max_depth=6)
    pol.fit(tr["features"], tr["actions"], tr["cost"])
    obs = tr["features"][0]
    mean, sd = pol.predict_with_uncertainty(obs)
    assert mean.shape == sd.shape == (len(env.actions),)
    assert np.all(sd >= 0.0)
    conf = pol.decision_confidence(obs)
    assert set(conf) == {"action", "predicted_cost", "std", "margin", "margin_sigma"}
    assert conf["margin"] >= 0.0
    assert 0 <= int(conf["action"]) < len(env.actions)
    assert pol.act(obs) == int(np.argmin(mean))
    # the action cache grows and does not change the answer
    first = pol.act(obs)
    assert pol.cache_size >= 1
    assert pol.act(obs) == first


def test_learned_policy_refuses_to_act_before_fitting() -> None:
    pol = LearnedRedundancyPolicy(5, n_estimators=5)
    obs = np.zeros(len(FEATURE_NAMES))
    with pytest.raises(RuntimeError, match="fit"):
        pol.act(obs)
    with pytest.raises(RuntimeError, match="fit"):
        pol.predict_with_uncertainty(obs)


def test_learned_policy_input_validation() -> None:
    pol = LearnedRedundancyPolicy(5, n_estimators=5)
    with pytest.raises(ValueError, match="n_actions"):
        LearnedRedundancyPolicy(1)
    good = np.zeros((10, len(FEATURE_NAMES)))
    with pytest.raises(ValueError, match="shape"):
        pol.fit(np.zeros((10, 3)), np.zeros(10, dtype=int), np.zeros(10))
    with pytest.raises(ValueError, match="agree in length"):
        pol.fit(good, np.zeros(9, dtype=int), np.zeros(10))
    with pytest.raises(ValueError, match="action indices"):
        pol.fit(good, np.full(10, 7), np.zeros(10))
    pol.fit(good, np.zeros(10, dtype=int), np.zeros(10))
    with pytest.raises(ValueError, match="features"):
        pol.predict_with_uncertainty(np.zeros(3))


def test_fitting_clears_the_action_cache() -> None:
    env = long_burst_env()
    tr = collect_transitions(env, SEED_SPLIT.fit[:4], 40, rng_seed=1)
    pol = LearnedRedundancyPolicy(len(env.actions), n_estimators=15, max_depth=5)
    pol.fit(tr["features"], tr["actions"], tr["cost"])
    pol.act(tr["features"][0])
    assert pol.cache_size > 0
    pol.fit(tr["features"], tr["actions"], tr["cost"])
    assert pol.cache_size == 0
