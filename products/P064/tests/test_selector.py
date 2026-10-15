"""Tests for the feature extraction and the learned selector."""

from __future__ import annotations

import numpy as np
import pytest

from regressselect.features import (
    DISTANCE_CAP,
    FEATURE_NAMES,
    build_design_matrix,
    feature_row,
)
from regressselect.history import History, Revision
from regressselect.selector import LearnedSelector, SelectorConfig


def test_feature_names_and_row_length_agree(graph, prioritiser):
    row = feature_row(
        "tests/test_quat.py::test_identity_multiply",
        ["sampleproj.quat"],
        {"sampleproj.quat": 7},
        graph,
        prioritiser,
    )
    assert len(row) == len(FEATURE_NAMES) == 11


def test_reachable_and_distance_features_hand_calculated(graph, prioritiser):
    names = list(FEATURE_NAMES)
    row = feature_row(
        "tests/test_quat.py::test_identity_multiply",
        ["sampleproj.vecops"],
        {"sampleproj.vecops": 9},
        graph,
        prioritiser,
    )
    assert row[names.index("reachable")] == pytest.approx(1.0)
    # test_quat -> quat -> vecops is two hops
    assert row[names.index("distance_hops")] == pytest.approx(2.0)
    assert row[names.index("inverse_distance")] == pytest.approx(1.0 / 3.0)
    assert row[names.index("log1p_nearest_churn")] == pytest.approx(float(np.log1p(9.0)))


def test_unreachable_uses_the_cap_and_clears_the_flag(graph, prioritiser):
    names = list(FEATURE_NAMES)
    row = feature_row(
        "tests/test_orbit.py::test_period_matches_vis_viva",
        ["sampleproj.quat"],
        {"sampleproj.quat": 40},
        graph,
        prioritiser,
    )
    assert row[names.index("reachable")] == pytest.approx(0.0)
    assert row[names.index("distance_hops")] == pytest.approx(float(DISTANCE_CAP))
    assert row[names.index("log1p_nearest_churn")] == pytest.approx(0.0)


def test_name_overlap_known_answers(graph, prioritiser):
    names = list(FEATURE_NAMES)
    same = feature_row(
        "tests/test_quat.py::test_identity_multiply",
        ["sampleproj.quat"],
        {"sampleproj.quat": 1},
        graph,
        prioritiser,
    )
    # token sets {"quat"} and {"quat"}: intersection 1, union 1
    assert same[names.index("name_overlap")] == pytest.approx(1.0)
    other = feature_row(
        "tests/test_quat.py::test_identity_multiply",
        ["sampleproj.orbit"],
        {"sampleproj.orbit": 1},
        graph,
        prioritiser,
    )
    assert other[names.index("name_overlap")] == pytest.approx(0.0)


def test_unknown_test_module_is_rejected(graph, prioritiser):
    with pytest.raises(ValueError, match="not in the import graph"):
        feature_row("tests/test_nope.py::test_x", ["sampleproj.quat"], {}, graph, prioritiser)


def test_design_matrix_shape_and_labels(split, graph, prioritiser):
    train, _ = split
    x, y, index = build_design_matrix(train, graph, prioritiser)
    assert x.shape == (len(train) * len(train.test_ids), len(FEATURE_NAMES))
    assert y.shape[0] == x.shape[0]
    assert len(index) == x.shape[0]
    expected_positives = sum(len(r.failed_tests) for r in train.revisions)
    assert int(y.sum()) == expected_positives


def test_design_matrix_has_no_nan_or_inf(split, graph, prioritiser):
    train, _ = split
    x, _, _ = build_design_matrix(train, graph, prioritiser)
    assert np.isfinite(x).all()


def test_selector_probabilities_are_in_the_unit_interval(selector, split):
    _, test = split
    revision = next(r for r in test.revisions if r.is_regression)
    probs = selector.probabilities(revision)
    assert probs.shape == (24,)
    assert float(probs.min()) >= 0.0
    assert float(probs.max()) <= 1.0


def test_selector_rank_is_a_permutation_and_is_sorted(selector, split):
    _, test = split
    revision = test.revisions[0]
    ranked = selector.rank(revision)
    assert sorted(rt.node_id for rt in ranked) == sorted(selector.node_ids)
    scores = [rt.score for rt in ranked]
    assert all(a >= b - 1e-12 for a, b in zip(scores, scores[1:], strict=False))


def test_selector_rank_is_deterministic(split, graph):
    train, test = split
    a = LearnedSelector().fit(train, graph).rank(test.revisions[0])
    b = LearnedSelector().fit(train, graph).rank(test.revisions[0])
    assert [rt.node_id for rt in a] == [rt.node_id for rt in b]
    for ra, rb in zip(a, b, strict=True):
        assert ra.score == pytest.approx(rb.score, abs=1e-12)


def test_feature_importances_sum_to_one(selector):
    importances = selector.feature_importances()
    assert set(importances) == set(FEATURE_NAMES)
    assert sum(importances.values()) == pytest.approx(1.0, abs=1e-9)


def test_brier_score_is_a_fraction(selector, split):
    train, test = split
    assert 0.0 <= selector.brier_score(train) <= 1.0
    assert 0.0 <= selector.brier_score(test) <= 1.0


def test_unfitted_selector_is_an_error(split):
    _, test = split
    with pytest.raises(ValueError, match="not fitted"):
        LearnedSelector().rank(test.revisions[0])


def test_history_with_no_failure_cannot_train_a_classifier(graph):
    revs = tuple(
        Revision(f"r{i}", ("sampleproj.quat",), {"sampleproj.quat": 3}) for i in range(5)
    )
    green = History(("tests/test_quat.py::test_identity_multiply",), revs)
    with pytest.raises(ValueError, match="no failing test"):
        LearnedSelector().fit(green, graph)


def test_empty_history_cannot_train(graph):
    with pytest.raises(ValueError, match="empty history"):
        LearnedSelector().fit(History(("t",), ()), graph)


def test_selector_config_validation():
    with pytest.raises(ValueError, match="n_estimators"):
        SelectorConfig(n_estimators=0)
    with pytest.raises(ValueError, match="learning_rate"):
        SelectorConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="max_depth"):
        SelectorConfig(max_depth=0)
    with pytest.raises(ValueError, match="subsample"):
        SelectorConfig(subsample=1.5)
    with pytest.raises(ValueError, match="calibration_folds"):
        SelectorConfig(calibration_folds=1)


def test_calibrated_selector_refuses_to_report_importances(split, graph):
    train, _ = split
    small = SelectorConfig(n_estimators=20, calibrate=True, calibration_folds=2)
    sel = LearnedSelector(config=small).fit(train, graph)
    with pytest.raises(ValueError, match="unavailable for a calibrated selector"):
        sel.feature_importances()
    probs = sel.probabilities(train.revisions[0])
    assert float(probs.min()) >= 0.0
    assert float(probs.max()) <= 1.0
