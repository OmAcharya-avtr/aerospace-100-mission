"""Tests for the scheduling problem, the classical baseline and the learned scheduler."""

from __future__ import annotations

import numpy as np
import pytest

from momentummgr import (
    FEATURE_NAMES,
    N_FEATURES,
    AlwaysOnScheduler,
    FixedThresholdScheduler,
    NeverScheduler,
    evaluate_policy,
    rollout,
    sample_episode,
    simulate_masks,
    tune_fixed_threshold,
)
from momentummgr.episodes import SAFETY_OVERRIDE_FRACTION
from momentummgr.learned import (
    LearnedScheduler,
    harvest_training_rows,
    search_best_mask,
    train_scheduler,
)


def tiny(seed: int) -> object:
    """A short episode, so the tests stay fast."""
    return sample_episode(seed, n_orbits=2.0, window_s=900.0, substeps=3)


def test_episode_is_deterministic_in_its_seed() -> None:
    a, b = tiny(11), tiny(11)
    assert np.array_equal(a.torque_body_nm, b.torque_body_nm)
    assert np.array_equal(a.b_body_t, b.b_body_t)
    assert a.gain == b.gain and a.envelope_nms == b.envelope_nms


def test_different_seeds_give_different_episodes() -> None:
    assert not np.array_equal(tiny(11).torque_body_nm, tiny(12).torque_body_nm)


def test_feature_names_match_the_feature_count() -> None:
    assert len(FEATURE_NAMES) == N_FEATURES
    ep = tiny(11)
    roll = rollout(ep, lambda _k, _f: (False, 1.0))
    assert roll.features.shape == (ep.n_windows, N_FEATURES)
    assert np.all(np.isfinite(roll.features))


def test_rollout_and_batch_simulation_agree_exactly() -> None:
    ep = tiny(13)
    roll = rollout(ep, FixedThresholdScheduler(0.5, 0.3).decider())
    batch = simulate_masks(ep, roll.actions[None, :])[0]
    assert batch.dipole_cost_am2s == pytest.approx(roll.metrics.dipole_cost_am2s, rel=1e-12)
    assert batch.near_saturation_fraction == pytest.approx(
        roll.metrics.near_saturation_fraction, abs=1e-15
    )
    assert batch.max_h_fraction == pytest.approx(roll.metrics.max_h_fraction, rel=1e-12)


def test_history_shapes_and_optional_recording() -> None:
    ep = tiny(14)
    full = rollout(ep, NeverScheduler().decider(), record_history=True)
    n = ep.n_windows * ep.substeps
    assert full.h_history_nms.shape == (n + 1, 3)
    assert full.dipole_history_am2.shape == (n, 3)
    assert full.time_s.shape == (n + 1,)
    lean = rollout(ep, NeverScheduler().decider(), record_history=False)
    assert lean.h_history_nms.shape == (0, 3)
    assert lean.metrics.cost == pytest.approx(full.metrics.cost, rel=1e-15)


def test_safety_override_fires_and_is_recorded() -> None:
    ep = tiny(15)
    roll = rollout(ep, NeverScheduler().decider())
    fired = roll.features[:, 0] >= SAFETY_OVERRIDE_FRACTION
    assert np.all(roll.actions[fired])
    assert np.all(roll.confidences[fired] == 1.0)


def test_threshold_hysteresis_keeps_dumping_between_the_two_levels() -> None:
    sched = FixedThresholdScheduler(on_fraction=0.6, off_fraction=0.3)
    decide = sched.decider()
    feats = np.zeros(N_FEATURES)
    feats[0] = 0.45
    assert decide(0, feats)[0] is False        # below the on threshold, stays off
    feats[0] = 0.65
    assert decide(1, feats)[0] is True         # crosses on
    feats[0] = 0.45
    assert decide(2, feats)[0] is True         # between the levels, stays on
    feats[0] = 0.2
    assert decide(3, feats)[0] is False        # drops below off


def test_always_on_costs_more_duty_than_never() -> None:
    ep = tiny(16)
    on = rollout(ep, AlwaysOnScheduler().decider()).metrics
    off = rollout(ep, NeverScheduler().decider()).metrics
    assert on.duty_fraction > off.duty_fraction
    assert on.near_saturation_fraction <= off.near_saturation_fraction


def test_tuning_returns_a_scheduler_from_the_grid() -> None:
    eps = [tiny(s) for s in (20, 21, 22)]
    best, cost, grid = tune_fixed_threshold(eps, on_grid=[0.4, 0.7], off_ratio_grid=[0.5])
    assert len(grid) == 2
    assert best.on_fraction in (0.4, 0.7)
    assert cost == pytest.approx(min(r[2] for r in grid), rel=1e-15)


def test_search_beats_or_matches_every_seeded_threshold_schedule() -> None:
    ep = tiny(23)
    res = search_best_mask(ep, n_random=24, max_rounds=2)
    assert res.metrics.cost <= res.seed_cost + 1e-12
    for on in (0.3, 0.5, 0.7, 0.9):
        rule = rollout(ep, FixedThresholdScheduler(on, on * 0.4).decider()).metrics
        assert res.metrics.cost <= rule.cost + 1e-12


def test_harvest_drops_the_override_windows() -> None:
    ep = tiny(24)
    mask = np.zeros(ep.n_windows, dtype=bool)
    x, y = harvest_training_rows(ep, mask)
    assert x.shape[1] == N_FEATURES
    assert x.shape[0] == y.shape[0]
    assert np.all(x[:, 0] < 0.95)


def test_end_to_end_training_and_confidence_output() -> None:
    fit = [tiny(s) for s in range(30, 36)]
    tune = [tiny(s) for s in range(60, 63)]
    baseline, _, _ = tune_fixed_threshold(fit + tune, on_grid=[0.5, 0.7], off_ratio_grid=[0.5])
    sched, diag, x, y = train_scheduler(
        fit, tune, fallback=baseline, n_estimators=30,
        threshold_grid=[0.2, 0.5], confidence_grid=[0.0, 0.6],
    )
    assert x.shape[1] == N_FEATURES and x.shape[0] == y.shape[0] > 0
    assert set(np.unique(y)).issubset({0, 1})
    assert diag["decision_threshold"] in (0.2, 0.5)
    assert diag["min_confidence"] in (0.0, 0.6)
    proba = sched.predict_proba(x[:20])
    assert proba.shape == (20,)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    metrics = evaluate_policy(sched, [tiny(90), tiny(91)])
    assert len(metrics) == 2
    roll = rollout(tiny(90), sched.decider())
    assert np.all((roll.confidences >= 0.0) & (roll.confidences <= 1.0))


def test_predict_proba_rejects_the_wrong_feature_count() -> None:
    fit = [tiny(s) for s in range(30, 33)]
    tune = [tiny(60)]
    sched, _, _, _ = train_scheduler(
        fit, tune, n_estimators=10, threshold_grid=[0.5], confidence_grid=[0.0]
    )
    with pytest.raises(ValueError, match=f"must have {N_FEATURES} columns"):
        sched.predict_proba(np.zeros((2, 3)))


def test_deferral_uses_the_fallback_when_confidence_is_low() -> None:
    fit = [tiny(s) for s in range(30, 33)]
    tune = [tiny(60)]
    sched, _, _, _ = train_scheduler(
        fit, tune, n_estimators=10, threshold_grid=[0.5], confidence_grid=[0.0]
    )
    always_defer = LearnedScheduler(
        model=sched.model, decision_threshold=0.5, min_confidence=1.01,
        fallback=FixedThresholdScheduler(0.5, 0.3),
    )
    ep = tiny(92)
    deferred = rollout(ep, always_defer.decider()).actions
    classical = rollout(ep, FixedThresholdScheduler(0.5, 0.3).decider()).actions
    assert np.array_equal(deferred, classical)


def test_train_scheduler_requires_both_splits() -> None:
    with pytest.raises(ValueError, match="must both be non-empty"):
        train_scheduler([], [tiny(60)])


def test_simulate_masks_shape_validation() -> None:
    ep = tiny(25)
    with pytest.raises(ValueError, match=r"masks must have shape"):
        simulate_masks(ep, np.zeros((2, ep.n_windows + 1), dtype=bool))


def test_infeasible_episodes_are_rejected_by_the_sampler() -> None:
    ep = sample_episode(5000, n_orbits=2.0, window_s=900.0, substeps=3)
    always = np.ones((1, ep.n_windows), dtype=bool)
    assert simulate_masks(ep, always)[0].violated is False
