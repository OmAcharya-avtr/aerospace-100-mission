"""The learned switch predictor and its metrics, including the losses.

The dataset here is deliberately small (8 episodes of 300 steps) so the module
runs inside the test budget. The full benchmark lives in
``validation/validate_predictor.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from simplexguard import (
    SimplexGuard,
    build_dataset,
    exact_predictor_scores,
    expected_calibration_error,
    fit_switch_predictor,
    guard_condition_scores,
    lead_times,
    measure_decision_cost,
    score_binary,
)
from simplexguard.predictor import FEATURE_NAMES


@pytest.fixture(scope="module")
def small_dataset_lead0(plant, controllers, invariant_set):
    guard = SimplexGuard(plant, controllers[0], invariant_set)
    return build_dataset(plant, guard, controllers[1], 8, 300, seed=5101, lead=0)


@pytest.fixture(scope="module")
def small_dataset_lead5(plant, controllers, invariant_set):
    guard = SimplexGuard(plant, controllers[0], invariant_set)
    return build_dataset(plant, guard, controllers[1], 8, 300, seed=5101, lead=5)


def test_dataset_shape_and_feature_names(small_dataset_lead0):
    data = small_dataset_lead0
    assert data.features.shape == (8 * 300, 5)
    assert data.feature_names == FEATURE_NAMES
    assert data.states.shape == (8 * 300, 2)
    assert 0.0 < data.base_rate < 0.5
    assert 0.0 < data.saturation_rate < 1.0


def test_lead_zero_label_is_fires_now(small_dataset_lead0):
    assert np.array_equal(small_dataset_lead0.labels, small_dataset_lead0.fires_now)


def test_lead_five_label_has_a_higher_base_rate(small_dataset_lead0, small_dataset_lead5):
    assert small_dataset_lead5.base_rate > small_dataset_lead0.base_rate
    assert len(small_dataset_lead5) == 8 * (300 - 5)


def test_dataset_is_deterministic(plant, controllers, invariant_set):
    guard = SimplexGuard(plant, controllers[0], invariant_set)
    a = build_dataset(plant, guard, controllers[1], 3, 200, seed=7, lead=2)
    b = build_dataset(plant, guard, controllers[1], 3, 200, seed=7, lead=2)
    assert np.array_equal(a.features, b.features)
    assert np.array_equal(a.labels, b.labels)


def test_features_are_the_documented_raw_quantities(small_dataset_lead0):
    data = small_dataset_lead0
    assert np.allclose(data.features[:, 0], data.states[:, 0])
    assert np.allclose(data.features[:, 1], data.states[:, 1])
    assert np.allclose(data.features[:, 2], data.references[:, 0])
    assert np.allclose(data.features[:, 3], data.states[:, 0] - data.references[:, 0])


def test_split_by_episode_is_disjoint(small_dataset_lead0):
    train = small_dataset_lead0.select_episodes(np.arange(0, 5))
    test = small_dataset_lead0.select_episodes(np.arange(5, 8))
    assert len(train) + len(test) == len(small_dataset_lead0)
    assert set(train.episode_ids).isdisjoint(set(test.episode_ids))


def test_exact_one_step_condition_is_perfect_on_the_lead_zero_target(
    small_dataset_lead0, plant, controllers, invariant_set
):
    # This is the honest negative the README leads with: on the switching
    # condition's own criterion, the exact computation is right by construction.
    guard = SimplexGuard(plant, controllers[0], invariant_set)
    scores = guard_condition_scores(small_dataset_lead0, guard, controllers[1])
    assert scores.precision == pytest.approx(1.0)
    assert scores.recall == pytest.approx(1.0)
    assert scores.false_positive == 0
    assert scores.false_negative == 0


def test_learned_model_loses_to_the_exact_condition_on_its_own_criterion(
    small_dataset_lead0, plant, controllers, invariant_set
):
    guard = SimplexGuard(plant, controllers[0], invariant_set)
    data = small_dataset_lead0
    train = data.select_episodes(np.arange(0, 5))
    calib = data.select_episodes(np.arange(5, 6))
    test = data.select_episodes(np.arange(6, 8))
    model = fit_switch_predictor(train, calib, "forest", n_estimators=40)
    prob = model.predict_proba(test.features)[:, 1]
    learned = score_binary("forest", test.labels, prob >= 0.5, probability=prob)
    exact = guard_condition_scores(test, guard, controllers[1])
    # The exact condition cannot be beaten on this target; the learned model may
    # tie at best, and here it does not.
    assert learned.f1 <= exact.f1
    assert learned.false_positive + learned.false_negative > 0


def test_learned_model_is_slower_per_decision_than_the_exact_condition(
    small_dataset_lead0, plant, controllers, invariant_set
):
    guard = SimplexGuard(plant, controllers[0], invariant_set)
    data = small_dataset_lead0
    model = fit_switch_predictor(
        data.select_episodes(np.arange(0, 5)),
        data.select_episodes(np.arange(5, 6)),
        "forest",
        n_estimators=40,
    )
    row = data.features[0:1]
    learned_us = measure_decision_cost(lambda: model.predict_proba(row), n_calls=60)
    x0, r0 = data.states[0], data.references[0]
    performance = controllers[1]
    exact_us = measure_decision_cost(
        lambda: guard.condition_margin(x0, performance(x0, r0))[0] < 0.0, n_calls=400
    )
    assert learned_us > exact_us


def test_logistic_model_is_a_weak_lower_bound(small_dataset_lead0):
    # A single hyperplane cannot represent the intersection of 50 facets, so the
    # logistic model is included as a measured lower bound on what a learned
    # model can do here, not as a contender. It must lose to the forest.
    data = small_dataset_lead0
    train = data.select_episodes(np.arange(0, 5))
    calib = data.select_episodes(np.arange(5, 6))
    test = data.select_episodes(np.arange(6, 8))
    logistic = fit_switch_predictor(train, calib, "logistic")
    forest = fit_switch_predictor(train, calib, "forest", n_estimators=40)
    p_log = logistic.predict_proba(test.features)[:, 1]
    p_for = forest.predict_proba(test.features)[:, 1]
    s_log = score_binary("logistic", test.labels, p_log >= 0.5, probability=p_log)
    s_for = score_binary("forest", test.labels, p_for >= 0.5, probability=p_for)
    assert s_log.f1 < s_for.f1
    assert s_log.brier > s_for.brier
    assert s_log.roc_auc < s_for.roc_auc


def test_score_binary_known_answer():
    # y_true = [1,1,0,0,1], y_pred = [1,0,0,1,1]
    # TP = 2 (rows 0, 4), FP = 1 (row 3), TN = 1 (row 2), FN = 1 (row 1)
    # precision = 2/3 = 0.666667, recall = 2/3, F1 = 2/3, accuracy = 3/5 = 0.6
    scores = score_binary(
        "hand", np.array([1, 1, 0, 0, 1]), np.array([1, 0, 0, 1, 1])
    )
    assert scores.true_positive == 2
    assert scores.false_positive == 1
    assert scores.true_negative == 1
    assert scores.false_negative == 1
    assert scores.precision == pytest.approx(2 / 3)
    assert scores.recall == pytest.approx(2 / 3)
    assert scores.f1 == pytest.approx(2 / 3)
    assert scores.accuracy == pytest.approx(0.6)
    assert scores.base_rate == pytest.approx(0.6)


def test_expected_calibration_error_known_answer():
    # Two bins used. Probabilities 0.05 (x4) with 0 positives -> |0 - 0.05| = 0.05
    # weighted 4/5; probability 0.95 (x1) with 1 positive -> |1 - 0.95| = 0.05
    # weighted 1/5. ECE = 0.8 * 0.05 + 0.2 * 0.05 = 0.05.
    y = np.array([0, 0, 0, 0, 1])
    p = np.array([0.05, 0.05, 0.05, 0.05, 0.95])
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.05)


def test_expected_calibration_error_is_zero_for_a_perfect_forecaster():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.0, 1.0, 0.0, 1.0])
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0)


def test_lead_times_known_answer():
    # fires_now  = [0,0,0,1,1,0,0,1]   events start at index 3 and index 7
    # predictions= [0,1,1,1,0,0,1,1]
    # event at 3: steps 2, 1, 0 before it -> pred[2]=1, pred[1]=1, pred[0]=0
    #             so the contiguous run of positives before it is 2.
    # event at 7: pred[6]=1, pred[5]=0 -> lead 1.
    # mean = 1.5, median = 1.5, max = 2, fraction with zero lead = 0.
    fires = np.array([0, 0, 0, 1, 1, 0, 0, 1], dtype=bool)
    pred = np.array([0, 1, 1, 1, 0, 0, 1, 1], dtype=bool)
    out = lead_times(pred, fires)
    assert out["n_events"] == 2.0
    assert out["mean_lead"] == pytest.approx(1.5)
    assert out["median_lead"] == pytest.approx(1.5)
    assert out["max_lead"] == 2.0
    assert out["fraction_zero_lead"] == pytest.approx(0.0)


def test_lead_times_with_no_events_returns_nan():
    out = lead_times(np.zeros(10, dtype=bool), np.zeros(10, dtype=bool))
    assert out["n_events"] == 0.0
    assert np.isnan(out["mean_lead"])


def test_exact_lead_predictor_has_high_recall_but_poor_precision(
    small_dataset_lead5, plant, controllers, invariant_set
):
    guard = SimplexGuard(plant, controllers[0], invariant_set)
    scores = exact_predictor_scores(small_dataset_lead5, guard, controllers[1], 5)
    assert scores.recall > 0.7
    assert scores.precision < scores.recall
    assert np.isnan(scores.brier)  # no probability output, by construction


def test_predictor_input_validation(small_dataset_lead0, plant, controllers, invariant_set):
    guard = SimplexGuard(plant, controllers[0], invariant_set)
    with pytest.raises(ValueError, match="n_episodes"):
        build_dataset(plant, guard, controllers[1], 0, 100, 1, 0)
    with pytest.raises(ValueError, match="n_steps must be at least 2"):
        build_dataset(plant, guard, controllers[1], 1, 1, 1, 0)
    with pytest.raises(ValueError, match="lead must be non-negative"):
        build_dataset(plant, guard, controllers[1], 1, 100, 1, -1)
    data = small_dataset_lead0
    with pytest.raises(ValueError, match="unknown kind"):
        fit_switch_predictor(data, data, "svm")
    with pytest.raises(ValueError, match="n_calls"):
        measure_decision_cost(lambda: None, n_calls=0)
    with pytest.raises(ValueError, match="max_lookback"):
        lead_times(np.zeros(3, dtype=bool), np.zeros(3, dtype=bool), max_lookback=0)
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error(np.zeros(3), np.zeros(3), n_bins=0)
    with pytest.raises(ValueError, match="empty split"):
        score_binary("x", np.zeros(0), np.zeros(0))
    with pytest.raises(ValueError, match="y_true has shape"):
        score_binary("x", np.zeros(3), np.zeros(4))


def test_fit_refuses_a_single_class_split(small_dataset_lead0):
    data = small_dataset_lead0
    all_negative = data.select_episodes(np.arange(0, 2))
    object.__setattr__(all_negative, "labels", np.zeros(len(all_negative), dtype=bool))
    with pytest.raises(ValueError, match="only one class"):
        fit_switch_predictor(all_negative, all_negative, "forest", n_estimators=10)
