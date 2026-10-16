"""Policies: posterior update, Bayesian-greedy rule, the learned policy, episodes."""

from __future__ import annotations

import numpy as np
import pytest

from acqsearch import (
    AcquisitionEnvironment,
    BayesianGreedyPolicy,
    FixedPatternPolicy,
    LearnedScanPolicy,
    acquisition_rate,
    build_features,
    collect_training_rows,
    greedy_scores,
    initial_posterior,
    mean_censored_time,
    median_time,
    run_episode,
    sample_target_cells,
    shortlist_indices,
    update_posterior,
)
from acqsearch.policy import FEATURE_NAMES, episode_time_standard_error


@pytest.fixture(scope="module")
def environment(grid, geometry, detector, uncertainty):
    """Reference environment with a deadline of three nominal passes."""
    base = AcquisitionEnvironment.build(
        grid, geometry, detector, uncertainty, 1.4e-3, 0.5, deadline_s=1.0
    )
    return AcquisitionEnvironment(
        grid=base.grid,
        geometry=base.geometry,
        cell_detection=base.cell_detection,
        dwell_s=base.dwell_s,
        false_alarm_probability=base.false_alarm_probability,
        verification_time_s=base.verification_time_s,
        deadline_s=3.0 * base.nominal_period_s,
        nominal_period_s=base.nominal_period_s,
        neighbours=base.neighbours,
    )


def test_initial_posterior_is_the_prior_renormalised(grid):
    posterior = initial_posterior(grid)
    assert posterior.sum() == pytest.approx(1.0, abs=1e-14)
    assert posterior == pytest.approx(grid.prior_mass / grid.prior_mass.sum(), rel=1e-14)


def test_posterior_update_matches_the_hand_calculation():
    # Two cells, P = (0.6, 0.4), miss at cell 0 with q = 0.5.
    # Unnormalised (0.3, 0.4), total 0.7, so (3/7, 4/7).
    posterior = update_posterior(np.array([0.6, 0.4]), 0, 0.5)
    assert posterior == pytest.approx(np.array([3.0 / 7.0, 4.0 / 7.0]), rel=1e-14)


def test_posterior_update_leaves_other_cells_in_the_same_ratio():
    before = np.array([0.5, 0.3, 0.2])
    after = update_posterior(before, 0, 0.4)
    assert after[1] / after[2] == pytest.approx(before[1] / before[2], rel=1e-14)


def test_posterior_update_with_certain_detection_empties_the_cell():
    after = update_posterior(np.array([0.5, 0.5]), 0, 1.0)
    assert after[0] == pytest.approx(0.0, abs=1e-15)
    assert after[1] == pytest.approx(1.0, rel=1e-15)


def test_posterior_update_with_zero_detection_changes_nothing():
    before = np.array([0.25, 0.75])
    assert update_posterior(before, 1, 0.0) == pytest.approx(before, rel=1e-15)


def test_posterior_update_rejects_a_collapsed_posterior():
    with pytest.raises(ValueError, match="collapsed"):
        update_posterior(np.array([1.0]), 0, 1.0)


def test_posterior_stays_normalised_over_many_updates(environment):
    posterior = initial_posterior(environment.grid)
    rng = np.random.default_rng(0)
    for _ in range(200):
        cell = int(rng.integers(environment.grid.n_cells))
        posterior = update_posterior(posterior, cell, float(environment.cell_detection[cell]))
        assert posterior.sum() == pytest.approx(1.0, abs=1e-12)


def test_greedy_picks_the_highest_score(environment):
    posterior = initial_posterior(environment.grid)
    current = np.zeros(2)
    scores = greedy_scores(environment, posterior, current)
    policy = BayesianGreedyPolicy()
    chosen = policy.select(
        environment, posterior, current, np.zeros(environment.grid.n_cells, dtype=np.int64), 0, 0.0
    )
    assert chosen == int(np.argmax(scores))


def test_greedy_first_dwell_is_the_centre_cell(environment):
    posterior = initial_posterior(environment.grid)
    chosen = BayesianGreedyPolicy().select(
        environment,
        posterior,
        np.zeros(2),
        np.zeros(environment.grid.n_cells, dtype=np.int64),
        0,
        0.0,
    )
    assert environment.grid.radii_rad[chosen] == pytest.approx(0.0, abs=1e-18)


def test_greedy_moves_off_the_centre_after_a_miss(environment):
    posterior = initial_posterior(environment.grid)
    centre = int(np.argmin(environment.grid.radii_rad))
    posterior = update_posterior(posterior, centre, float(environment.cell_detection[centre]))
    chosen = BayesianGreedyPolicy().select(
        environment,
        posterior,
        environment.grid.centres_rad[centre],
        np.zeros(environment.grid.n_cells, dtype=np.int64),
        1,
        0.0,
    )
    assert chosen != centre


def test_epsilon_greedy_explores(environment):
    policy = BayesianGreedyPolicy(epsilon=1.0, rng=np.random.default_rng(1))
    posterior = initial_posterior(environment.grid)
    picks = {
        policy.select(
            environment,
            posterior,
            np.zeros(2),
            np.zeros(environment.grid.n_cells, dtype=np.int64),
            0,
            0.0,
        )
        for _ in range(50)
    }
    assert len(picks) > 5


def test_shortlist_is_ordered_by_greedy_score(environment):
    posterior = initial_posterior(environment.grid)
    current = np.zeros(2)
    candidates = shortlist_indices(environment, posterior, current, 12)
    scores = greedy_scores(environment, posterior, current)[candidates]
    assert candidates.size == 12
    assert np.all(np.diff(scores) <= 1e-18)
    assert candidates[0] == int(np.argmax(greedy_scores(environment, posterior, current)))


def test_shortlist_is_clipped_to_the_grid_size(environment):
    posterior = initial_posterior(environment.grid)
    candidates = shortlist_indices(environment, posterior, np.zeros(2), 10_000)
    assert candidates.size == environment.grid.n_cells


def test_shortlist_rejects_a_non_positive_size(environment):
    with pytest.raises(ValueError, match="size"):
        shortlist_indices(environment, initial_posterior(environment.grid), np.zeros(2), 0)


def test_feature_matrix_has_the_declared_columns(environment):
    features = build_features(
        environment,
        initial_posterior(environment.grid),
        np.zeros(2),
        np.zeros(environment.grid.n_cells, dtype=np.int64),
        0,
        0.0,
    )
    assert features.shape == (environment.grid.n_cells, len(FEATURE_NAMES))
    assert np.all(np.isfinite(features))


def test_fixed_pattern_cycles_its_order(environment, spiral_order):
    policy = FixedPatternPolicy(spiral_order)
    posterior = initial_posterior(environment.grid)
    visits = np.zeros(environment.grid.n_cells, dtype=np.int64)
    for step in (0, 1, 2, spiral_order.size, spiral_order.size + 1):
        chosen = policy.select(environment, posterior, np.zeros(2), visits, step, 0.0)
        assert chosen == int(spiral_order[step % spiral_order.size])


def test_episode_terminates_and_reports_a_positive_time(environment, spiral_order):
    result = run_episode(
        environment, FixedPatternPolicy(spiral_order), 5, np.random.default_rng(0)
    )
    assert result.dwells >= 1
    assert result.time_s > 0.0
    assert result.time_s <= environment.deadline_s + 1e-12


def test_episode_with_certain_detection_finds_the_target_on_its_first_visit(
    environment, spiral_order
):
    certain = AcquisitionEnvironment(
        grid=environment.grid,
        geometry=environment.geometry,
        cell_detection=np.ones(environment.grid.n_cells),
        dwell_s=environment.dwell_s,
        false_alarm_probability=0.0,
        verification_time_s=0.0,
        deadline_s=environment.deadline_s,
        nominal_period_s=environment.nominal_period_s,
        neighbours=environment.neighbours,
    )
    target = int(spiral_order[7])
    result = run_episode(certain, FixedPatternPolicy(spiral_order), target, np.random.default_rng(0))
    assert result.acquired
    assert result.dwells == 8


def test_episode_is_seeded(environment, spiral_order):
    a = run_episode(environment, FixedPatternPolicy(spiral_order), 11, np.random.default_rng(4))
    b = run_episode(environment, FixedPatternPolicy(spiral_order), 11, np.random.default_rng(4))
    assert a == b


def test_episode_censors_at_the_deadline(environment, spiral_order):
    tight = AcquisitionEnvironment(
        grid=environment.grid,
        geometry=environment.geometry,
        cell_detection=environment.cell_detection,
        dwell_s=environment.dwell_s,
        false_alarm_probability=environment.false_alarm_probability,
        verification_time_s=environment.verification_time_s,
        deadline_s=1.0e-3,
        nominal_period_s=environment.nominal_period_s,
        neighbours=environment.neighbours,
    )
    result = run_episode(tight, FixedPatternPolicy(spiral_order), 50, np.random.default_rng(0))
    assert not result.acquired
    assert result.time_s == pytest.approx(1.0e-3, rel=1e-15)


def test_sample_target_cells_follows_the_prior(environment):
    draws = sample_target_cells(environment.grid, 40000, np.random.default_rng(2))
    counts = np.bincount(draws, minlength=environment.grid.n_cells) / 40000.0
    expected = initial_posterior(environment.grid)
    assert np.max(np.abs(counts - expected)) < 0.01


def test_summary_statistics_agree_with_their_definitions(environment, spiral_order):
    rng = np.random.default_rng(3)
    results = [
        run_episode(environment, FixedPatternPolicy(spiral_order), int(c), rng)
        for c in sample_target_cells(environment.grid, 60, np.random.default_rng(6))
    ]
    times = np.array([r.time_s for r in results])
    assert mean_censored_time(results) == pytest.approx(float(times.mean()), rel=1e-14)
    assert median_time(results) == pytest.approx(float(np.median(times)), rel=1e-14)
    assert 0.0 <= acquisition_rate(results) <= 1.0
    assert episode_time_standard_error(results) > 0.0


def test_summary_statistics_reject_empty_input():
    for function in (mean_censored_time, median_time, acquisition_rate):
        with pytest.raises(ValueError, match="non-empty"):
            function([])


@pytest.fixture(scope="module")
def trained(environment):
    """A learned policy fitted on a small set of behaviour episodes."""
    behaviour = BayesianGreedyPolicy(
        epsilon=0.35, rng=np.random.default_rng(5), explore_shortlist=12
    )
    targets = sample_target_cells(environment.grid, 40, np.random.default_rng(7))
    features, target = collect_training_rows(
        environment, behaviour, targets, np.random.default_rng(9)
    )
    return LearnedScanPolicy(n_estimators=40, max_depth=3).fit(features, target), features, target


def test_training_rows_have_the_declared_shape(trained):
    _policy, features, target = trained
    assert features.shape[1] == len(FEATURE_NAMES)
    assert target.shape == (features.shape[0],)
    assert np.all(target >= 0.0)


def test_learned_policy_reports_fitted_state(trained):
    policy, _features, _target = trained
    assert policy.is_fitted
    assert not LearnedScanPolicy().is_fitted


def test_learned_policy_exposes_an_interval_and_a_confidence(trained, environment):
    policy, _features, _target = trained
    decision = policy.decide(
        environment,
        initial_posterior(environment.grid),
        np.zeros(2),
        np.zeros(environment.grid.n_cells, dtype=np.int64),
        0,
        0.0,
    )
    low, high = decision.interval_s
    assert low <= high
    assert 0.0 < decision.confidence <= 1.0
    assert decision.predicted_remaining_s > 0.0
    assert 0 <= decision.cell < environment.grid.n_cells


def test_learned_policy_chooses_from_its_shortlist(trained, environment):
    policy, _features, _target = trained
    posterior = initial_posterior(environment.grid)
    candidates = {
        int(c) for c in shortlist_indices(environment, posterior, np.zeros(2), policy.shortlist)
    }
    chosen = policy.select(
        environment,
        posterior,
        np.zeros(2),
        np.zeros(environment.grid.n_cells, dtype=np.int64),
        0,
        0.0,
    )
    assert chosen in candidates


def test_learned_policy_select_and_decide_agree(trained, environment):
    policy, _features, _target = trained
    posterior = initial_posterior(environment.grid)
    visits = np.zeros(environment.grid.n_cells, dtype=np.int64)
    assert policy.select(environment, posterior, np.zeros(2), visits, 0, 0.0) == policy.decide(
        environment, posterior, np.zeros(2), visits, 0, 0.0
    ).cell


def test_learned_policy_runs_a_complete_episode(trained, environment):
    policy, _features, _target = trained
    result = run_episode(environment, policy, 20, np.random.default_rng(1))
    assert result.dwells >= 1
    assert result.time_s > 0.0


def test_unfitted_learned_policy_refuses_to_predict(environment):
    policy = LearnedScanPolicy()
    with pytest.raises(RuntimeError, match="fit must be called"):
        policy.select(
            environment,
            initial_posterior(environment.grid),
            np.zeros(2),
            np.zeros(environment.grid.n_cells, dtype=np.int64),
            0,
            0.0,
        )
    with pytest.raises(RuntimeError, match="fit must be called"):
        policy.predict_with_uncertainty(np.zeros((1, len(FEATURE_NAMES))))


def test_learned_policy_rejects_malformed_training_data():
    policy = LearnedScanPolicy()
    with pytest.raises(ValueError, match="features must have shape"):
        policy.fit(np.zeros((4, 3)), np.zeros(4))
    with pytest.raises(ValueError, match="one entry per feature row"):
        policy.fit(np.zeros((4, len(FEATURE_NAMES))), np.zeros(3))
    with pytest.raises(ValueError, match="at least two training rows"):
        policy.fit(np.zeros((1, len(FEATURE_NAMES))), np.zeros(1))
