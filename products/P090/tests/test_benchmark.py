"""Scoring: per-class rates, recall at a fixed false-positive rate, and the area.

The known-answer tests here use hand-built score arrays with the arithmetic
written out in the comments, because a scoring bug is invisible: it produces
plausible numbers that are wrong, and every published figure rests on it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tlecheck.benchmark import (
    accuracy,
    area_under_curve,
    operating_characteristic,
    per_class,
    point_operating_point,
    prepare,
    recall_at_fpr,
)
from tlecheck.consequence import BENIGN, SILENTLY_WRONG, WILL_FAIL_LOUDLY


def test_per_class_known_answer():
    # truth:     B B B S S
    # predicted: B B S S S
    # benign: TP 2, FP 0, FN 1 -> precision 1.0, recall 2/3
    # silent: TP 2, FP 1, FN 0 -> precision 2/3, recall 1.0
    truth = np.array([BENIGN, BENIGN, BENIGN, SILENTLY_WRONG, SILENTLY_WRONG], dtype=object)
    predicted = np.array(
        [BENIGN, BENIGN, SILENTLY_WRONG, SILENTLY_WRONG, SILENTLY_WRONG], dtype=object
    )
    scores = per_class(truth, predicted)
    assert scores[BENIGN].precision == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert scores[BENIGN].recall == pytest.approx(2.0 / 3.0, rel=0.0, abs=1.0e-15)
    assert scores[SILENTLY_WRONG].precision == pytest.approx(2.0 / 3.0, rel=0.0, abs=1.0e-15)
    assert scores[SILENTLY_WRONG].recall == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert scores[BENIGN].support == 3
    assert scores[SILENTLY_WRONG].support == 2


def test_per_class_absent_class_is_nan_not_zero():
    truth = np.array([BENIGN, BENIGN], dtype=object)
    predicted = np.array([BENIGN, BENIGN], dtype=object)
    scores = per_class(truth, predicted)
    assert math.isnan(scores[WILL_FAIL_LOUDLY].precision)
    assert math.isnan(scores[WILL_FAIL_LOUDLY].recall)
    assert scores[WILL_FAIL_LOUDLY].support == 0


def test_per_class_counts_are_consistent():
    truth = np.array([BENIGN, SILENTLY_WRONG, WILL_FAIL_LOUDLY] * 4, dtype=object)
    predicted = np.array([SILENTLY_WRONG, SILENTLY_WRONG, BENIGN] * 4, dtype=object)
    scores = per_class(truth, predicted)
    for score in scores.values():
        assert score.true_positive + score.false_negative == score.support


def test_per_class_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        per_class(np.array([BENIGN]), np.array([BENIGN, BENIGN]))


def test_accuracy_known_answer():
    truth = np.array([BENIGN, BENIGN, SILENTLY_WRONG, SILENTLY_WRONG], dtype=object)
    predicted = np.array([BENIGN, SILENTLY_WRONG, SILENTLY_WRONG, BENIGN], dtype=object)
    assert accuracy(truth, predicted) == pytest.approx(0.5, rel=0.0, abs=0.0)


def test_accuracy_of_empty_is_nan():
    assert math.isnan(accuracy(np.array([], dtype=object), np.array([], dtype=object)))


def test_operating_characteristic_perfect_separation():
    # Scores 1,1,0,0 with positives first: at threshold 1 recall is 1 and the
    # false-positive rate is 0.
    scores = np.array([1.0, 1.0, 0.0, 0.0])
    positive = np.array([True, True, False, False])
    points = operating_characteristic(scores, positive)
    best = max(points, key=lambda point: point.recall - point.false_positive_rate)
    assert best.recall == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert best.false_positive_rate == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_operating_characteristic_includes_both_corners():
    scores = np.array([1.0, 0.5, 0.0])
    positive = np.array([True, False, False])
    points = operating_characteristic(scores, positive)
    assert any(point.flagged == 0 for point in points)
    assert any(point.flagged == 3 for point in points)


def test_operating_characteristic_needs_both_classes():
    with pytest.raises(ValueError, match="needs both classes"):
        operating_characteristic(np.array([1.0, 2.0]), np.array([True, True]))


def test_operating_characteristic_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        operating_characteristic(np.array([1.0]), np.array([True, False]))


def test_operating_characteristic_handles_infinite_scores():
    scores = np.array([np.inf, 1.0, 0.0, 0.0])
    positive = np.array([True, True, False, False])
    points = operating_characteristic(scores, positive)
    assert any(math.isinf(point.threshold) for point in points)
    assert max(point.recall for point in points) == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_area_under_curve_perfect_is_one():
    scores = np.array([1.0, 0.9, 0.1, 0.0])
    positive = np.array([True, True, False, False])
    assert area_under_curve(scores, positive) == pytest.approx(1.0, rel=0.0, abs=1.0e-12)


def test_area_under_curve_constant_score_is_one_half():
    # A score that cannot separate the classes traces the chance diagonal.
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    positive = np.array([True, True, False, False])
    assert area_under_curve(scores, positive) == pytest.approx(0.5, rel=0.0, abs=1.0e-12)


def test_area_under_curve_inverted_is_zero():
    scores = np.array([0.0, 0.1, 0.9, 1.0])
    positive = np.array([True, True, False, False])
    assert area_under_curve(scores, positive) == pytest.approx(0.0, rel=0.0, abs=1.0e-12)


def test_recall_at_fpr_known_answer():
    # Four negatives at 0.1, 0.2, 0.3, 0.4 and four positives at
    # 0.35, 0.45, 0.55, 0.65. A target false-positive rate of 0.25 allows one
    # of the four negatives to be flagged. The lowest threshold that flags only
    # one negative is 0.35: it flags the single negative at 0.4 and all four
    # positives, so the answer is a recall of 1.0 at a measured rate of 0.25.
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.35, 0.45, 0.55, 0.65])
    positive = np.array([False, False, False, False, True, True, True, True])
    point = recall_at_fpr(scores, positive, 0.25)
    assert point.recall == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert point.false_positive_rate == pytest.approx(0.25, rel=0.0, abs=0.0)
    assert point.threshold == pytest.approx(0.35, rel=0.0, abs=0.0)


def test_recall_at_fpr_is_limited_by_the_target():
    # The same negatives, but one positive now scores 0.05, below every
    # negative. No threshold reaches it without flagging all four negatives, so
    # at a target rate of 0.25 the recall is capped at three of four, 0.75.
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.05, 0.45, 0.55, 0.65])
    positive = np.array([False, False, False, False, True, True, True, True])
    point = recall_at_fpr(scores, positive, 0.25)
    assert point.recall == pytest.approx(0.75, rel=0.0, abs=0.0)
    assert point.false_positive_rate <= 0.25


def test_recall_at_fpr_zero_target():
    scores = np.array([0.1, 0.2, 0.9, 1.0])
    positive = np.array([False, False, True, True])
    point = recall_at_fpr(scores, positive, 0.0)
    assert point.recall == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert point.false_positive_rate == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_recall_at_fpr_reports_the_achieved_rate_not_the_target():
    scores = np.array([0.1, 0.2, 0.9, 1.0])
    positive = np.array([False, False, True, True])
    point = recall_at_fpr(scores, positive, 0.5)
    assert point.false_positive_rate < 0.5


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_recall_at_fpr_validates_the_target(bad):
    with pytest.raises(ValueError, match="target_fpr"):
        recall_at_fpr(np.array([1.0, 0.0]), np.array([True, False]), bad)


def test_point_operating_point_known_answer():
    truth_positive = np.array([True, True, False, False])
    predicted = np.array(
        [SILENTLY_WRONG, BENIGN, SILENTLY_WRONG, BENIGN], dtype=object
    )
    point = point_operating_point(predicted, truth_positive)
    assert point.recall == pytest.approx(0.5, rel=0.0, abs=0.0)
    assert point.false_positive_rate == pytest.approx(0.5, rel=0.0, abs=0.0)
    assert point.flagged == 2
    assert math.isnan(point.threshold)


def test_prepare_rejects_empty_records():
    with pytest.raises(ValueError, match="non-empty"):
        prepare([])


def test_prepare_labels_are_measured_classes(prepared_test):
    assert set(prepared_test.measured_classes) <= {
        BENIGN,
        SILENTLY_WRONG,
        WILL_FAIL_LOUDLY,
    }


def test_prepare_errors_are_nan_exactly_for_loud_records(prepared_test):
    loud = prepared_test.measured_classes == WILL_FAIL_LOUDLY
    assert np.all(np.isnan(prepared_test.measured_max_error_km[loud]))
    assert np.all(np.isfinite(prepared_test.measured_max_error_km[~loud]))


def test_prepare_rule_scores_are_zero_for_loud_predictions(prepared_test):
    loud = prepared_test.rule_classes == WILL_FAIL_LOUDLY
    assert np.all(prepared_test.rule_scores[loud] == 0.0)


def test_prepare_has_all_three_matrices(prepared_test):
    assert set(prepared_test.matrices) == {
        "elements_and_history",
        "elements_only",
        "with_rule_score",
    }


def test_measured_classes_include_every_class(prepared_test):
    assert set(prepared_test.measured_classes) == {
        BENIGN,
        SILENTLY_WRONG,
        WILL_FAIL_LOUDLY,
    }


def test_benign_records_have_small_measured_error(prepared_test):
    benign = prepared_test.measured_classes == BENIGN
    assert np.all(prepared_test.measured_max_error_km[benign] <= 1.0)


def test_silent_records_exceed_the_threshold(prepared_test):
    silent = prepared_test.measured_classes == SILENTLY_WRONG
    assert np.all(prepared_test.measured_max_error_km[silent] > 1.0)
