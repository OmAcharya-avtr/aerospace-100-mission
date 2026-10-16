"""The head-to-head benchmark, and the result it is not allowed to lose quietly.

The reduced configuration here (120 scenarios, one behaviour episode per
training scenario, three target draws per held-out scenario) runs in about
nine seconds and reproduces the direction of the full run in
``validation/validate_policies_output.txt``: the learned policy is slower than
both the Bayesian-greedy baseline and the fixed spiral pattern. The full run
uses 300 scenarios and eight target draws. If a change reverses the ordering,
these tests fail and whoever made the change has to say so.
"""

from __future__ import annotations

import numpy as np
import pytest

from acqsearch import (
    LearnedScanPolicy,
    build_training_set,
    evaluate_policies,
    generate_scenarios,
    paired_difference,
    score_policy,
    split_scenarios,
    summarise,
    winner,
)
from acqsearch.policy import EpisodeResult


@pytest.fixture(scope="module")
def benchmark():
    """Reduced but real benchmark: training, fitting and held-out evaluation."""
    scenarios = generate_scenarios(120, 20261016)
    train, test = split_scenarios(scenarios, 0.65)
    features, target = build_training_set(train, episodes_per_scenario=1)
    learned = LearnedScanPolicy(n_estimators=80, max_depth=4).fit(features, target)
    results = evaluate_policies(test, learned, targets_per_scenario=3)
    return results, summarise(results), features


def test_every_policy_ran_the_same_number_of_episodes(benchmark):
    results, _scores, _features = benchmark
    counts = {len(value) for value in results.values()}
    assert counts == {126}


def test_the_bayesian_greedy_baseline_wins(benchmark):
    _results, scores, _features = benchmark
    assert winner(scores) == "bayesian_greedy"


def test_the_learned_policy_is_slower_than_both_references(benchmark):
    _results, scores, _features = benchmark
    assert scores["learned"].mean_time_s > scores["bayesian_greedy"].mean_time_s
    assert scores["learned"].mean_time_s > scores["fixed_spiral"].mean_time_s
    assert scores["learned"].median_time_s > scores["bayesian_greedy"].median_time_s


def test_the_gap_to_the_baseline_is_larger_than_its_own_standard_error(benchmark):
    results, _scores, _features = benchmark
    difference = paired_difference(
        "learned", results["learned"], "bayesian_greedy", results["bayesian_greedy"]
    )
    assert difference.mean_difference_s > 0.0
    assert difference.z_score > 1.5


def test_the_pinned_numbers_still_hold(benchmark):
    # Measured in the build session at numpy 2.5.3 / scikit-learn 1.9.1:
    # bayesian_greedy 2.705723 s, fixed_spiral 2.730536 s, learned 3.721117 s.
    # The tolerance is 10 percent, not machine precision: these are means over
    # 126 stochastic episodes passed through a fitted model, and asserting
    # them to the last bit would be asserting the floating-point history of
    # three libraries. A 10 percent move is a real change and fails here.
    _results, scores, _features = benchmark
    assert scores["bayesian_greedy"].mean_time_s == pytest.approx(2.705723, rel=0.10)
    assert scores["fixed_spiral"].mean_time_s == pytest.approx(2.730536, rel=0.10)
    assert scores["learned"].mean_time_s == pytest.approx(3.721117, rel=0.10)


def test_training_rows_come_only_from_the_training_split():
    scenarios = generate_scenarios(40, 20261016)
    train, test = split_scenarios(scenarios, 0.65)
    assert not {s.scenario_id for s in train} & {s.scenario_id for s in test}
    features, target = build_training_set(train, episodes_per_scenario=1)
    assert features.shape[0] == target.shape[0]
    assert features.shape[0] > 0


def test_training_set_respects_the_row_cap():
    scenarios = generate_scenarios(30, 1)
    features, target = build_training_set(scenarios, episodes_per_scenario=1, max_rows=200)
    assert features.shape[0] == 200
    assert target.shape == (200,)


def test_evaluation_is_reproducible():
    scenarios = generate_scenarios(12, 7)
    first = evaluate_policies(scenarios, None, targets_per_scenario=2)
    second = evaluate_policies(scenarios, None, targets_per_scenario=2)
    assert first == second


def test_all_patterns_can_be_evaluated_together():
    scenarios = generate_scenarios(8, 3)
    results = evaluate_policies(scenarios, None, targets_per_scenario=1, include_all_patterns=True)
    assert set(results) == {
        "fixed_spiral",
        "fixed_raster",
        "fixed_lissajous",
        "fixed_random_restart",
        "bayesian_greedy",
    }


def test_the_spiral_beats_the_other_fixed_patterns():
    scenarios = generate_scenarios(40, 20261016)
    results = evaluate_policies(scenarios, None, targets_per_scenario=2, include_all_patterns=True)
    scores = summarise(results)
    spiral = scores["fixed_spiral"].mean_time_s
    for name in ("fixed_raster", "fixed_lissajous", "fixed_random_restart"):
        assert spiral < scores[name].mean_time_s


def test_evaluate_rejects_an_unknown_pattern():
    with pytest.raises(ValueError, match="unknown pattern"):
        evaluate_policies(generate_scenarios(2, 1), None, targets_per_scenario=1, pattern="zigzag")


def test_paired_difference_requires_equal_lengths():
    a = [EpisodeResult(1.0, True, 1)]
    b = [EpisodeResult(1.0, True, 1), EpisodeResult(2.0, True, 2)]
    with pytest.raises(ValueError, match="same number of episodes"):
        paired_difference("a", a, "b", b)


def test_paired_difference_of_a_policy_with_itself_is_zero():
    results = [EpisodeResult(float(t), True, 1) for t in (1.0, 2.0, 3.5)]
    difference = paired_difference("a", results, "a", results)
    assert difference.mean_difference_s == pytest.approx(0.0, abs=1e-15)
    assert difference.standard_error_s == pytest.approx(0.0, abs=1e-15)
    assert np.isnan(difference.z_score)


def test_score_policy_reports_the_episode_count():
    results = [EpisodeResult(1.0, True, 3), EpisodeResult(2.0, False, 9)]
    score = score_policy("x", results)
    assert score.n_episodes == 2
    assert score.mean_time_s == pytest.approx(1.5, rel=1e-15)
    assert score.acquisition_rate == pytest.approx(0.5, rel=1e-15)
    assert score.mean_dwells == pytest.approx(6.0, rel=1e-15)


def test_winner_rejects_an_empty_score_set():
    with pytest.raises(ValueError, match="non-empty"):
        winner({})
