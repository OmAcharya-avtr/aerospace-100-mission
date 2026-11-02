"""Scoring: the exact interval, the confusion counts and the operating points."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hilverdict.scoring import (
    Confusion,
    clopper_pearson,
    confusion,
    operating_points,
    score_decision,
    threshold_at_recall,
)


def test_clopper_pearson_on_no_trials_is_the_whole_interval():
    assert clopper_pearson(0, 0) == (0.0, 1.0)


def test_clopper_pearson_at_a_perfect_score_has_an_upper_limit_of_one():
    lower, upper = clopper_pearson(40, 40)
    assert upper == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert 0.0 < lower < 1.0


def test_clopper_pearson_lower_limit_at_a_perfect_score_is_the_closed_form():
    lower, _upper = clopper_pearson(20, 20, 0.95)
    assert lower == pytest.approx(0.025 ** (1.0 / 20.0), rel=1e-9, abs=0.0)


def test_clopper_pearson_at_zero_successes_has_a_lower_limit_of_zero():
    lower, upper = clopper_pearson(0, 30)
    assert lower == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert 0.0 < upper < 1.0


def test_clopper_pearson_brackets_the_point_estimate():
    for successes, trials in ((1, 10), (5, 10), (9, 10), (37, 48)):
        lower, upper = clopper_pearson(successes, trials)
        assert lower <= successes / trials <= upper


def test_clopper_pearson_narrows_with_more_trials():
    narrow = clopper_pearson(500, 1000)
    wide = clopper_pearson(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_clopper_pearson_widens_with_confidence():
    tight = clopper_pearson(5, 10, 0.5)
    loose = clopper_pearson(5, 10, 0.999)
    assert (loose[1] - loose[0]) > (tight[1] - tight[0])


def test_clopper_pearson_rejects_negative_trials():
    with pytest.raises(ValueError, match="trials must be non-negative"):
        clopper_pearson(0, -1)


def test_clopper_pearson_rejects_more_successes_than_trials():
    with pytest.raises(ValueError, match="successes must lie"):
        clopper_pearson(11, 10)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.2])
def test_clopper_pearson_rejects_an_out_of_range_confidence(confidence):
    with pytest.raises(ValueError, match="confidence must lie"):
        clopper_pearson(5, 10, confidence)


def test_confusion_counts_by_hand():
    flagged = np.array([True, True, False, False])
    truth = np.array([True, False, True, False])
    counts = confusion(flagged, truth)
    assert (counts.true_positive, counts.false_positive,
            counts.false_negative, counts.true_negative) == (1, 1, 1, 1)


def test_confusion_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes must match"):
        confusion(np.zeros(3, dtype=bool), np.zeros(4, dtype=bool))


def test_confusion_rates_by_hand():
    counts = Confusion(true_positive=8, false_positive=2, false_negative=2,
                       true_negative=88)
    assert counts.total == 100
    assert counts.positives == 10
    assert counts.recall == pytest.approx(0.8, rel=1e-15, abs=0.0)
    assert counts.precision == pytest.approx(0.8, rel=1e-15, abs=0.0)
    assert counts.flag_rate == pytest.approx(0.1, rel=1e-15, abs=0.0)
    assert counts.specificity == pytest.approx(88.0 / 90.0, rel=1e-15, abs=0.0)
    assert counts.dismissal_rate == pytest.approx(0.9, rel=1e-15, abs=0.0)


def test_confusion_recall_is_nan_with_no_positives():
    counts = Confusion(true_positive=0, false_positive=1, false_negative=0,
                       true_negative=9)
    assert math.isnan(counts.recall)


def test_confusion_precision_is_nan_with_no_flags():
    counts = Confusion(true_positive=0, false_positive=0, false_negative=2,
                       true_negative=8)
    assert math.isnan(counts.precision)


def test_confusion_rates_are_nan_on_an_empty_sample():
    counts = Confusion(0, 0, 0, 0)
    assert math.isnan(counts.flag_rate)
    assert math.isnan(counts.dismissal_rate)


def test_confusion_specificity_is_nan_with_no_negatives():
    counts = Confusion(true_positive=3, false_positive=0, false_negative=1,
                       true_negative=0)
    assert math.isnan(counts.specificity)


def test_flagging_everything_gives_recall_one():
    truth = np.array([True, False, True, False, False])
    counts = confusion(np.ones(5, dtype=bool), truth)
    assert counts.recall == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert counts.flag_rate == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_dismissing_everything_gives_recall_zero():
    truth = np.array([True, False, True, False, False])
    counts = confusion(np.zeros(5, dtype=bool), truth)
    assert counts.recall == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert counts.dismissal_rate == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_score_decision_marks_soundness():
    truth = np.array([True, False])
    sound = score_decision("a", np.ones(2, dtype=bool), truth, sound=True)
    estimate = score_decision("b", np.ones(2, dtype=bool), truth)
    assert sound.sound is True
    assert estimate.sound is False


def test_score_decision_missed_is_the_false_negative_count():
    truth = np.array([True, True, False])
    score = score_decision("a", np.array([True, False, False]), truth)
    assert score.missed_consequential == 1


def test_score_decision_attaches_the_interval():
    truth = np.array([True] * 10)
    score = score_decision("a", np.ones(10, dtype=bool), truth)
    assert score.recall_interval == clopper_pearson(10, 10)


def test_operating_points_cover_the_all_dismissed_corner():
    scores = np.array([0.1, 0.4, 0.9])
    truth = np.array([False, True, True])
    thresholds, recall, flag_rate = operating_points(scores, truth)
    assert float(flag_rate[-1]) == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert float(recall[-1]) == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_operating_points_are_monotone():
    generator = np.random.default_rng(0)
    scores = generator.random(50)
    truth = scores > 0.6
    _thresholds, recall, flag_rate = operating_points(scores, truth)
    assert all(
        second <= first for first, second in zip(recall, recall[1:], strict=False)
    )
    assert all(
        second <= first for first, second in zip(flag_rate, flag_rate[1:], strict=False)
    )


def test_operating_points_at_the_lowest_threshold_flag_everything():
    scores = np.array([0.2, 0.5, 0.8])
    truth = np.array([False, True, True])
    _thresholds, recall, flag_rate = operating_points(scores, truth)
    assert float(flag_rate[0]) == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert float(recall[0]) == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_operating_points_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes must match"):
        operating_points(np.zeros(3), np.zeros(4, dtype=bool))


def test_operating_points_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="no misses to score"):
        operating_points(np.zeros(0), np.zeros(0, dtype=bool))


def test_threshold_at_recall_finds_the_cheapest_feasible_point():
    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    truth = np.array([False, False, True, True, True])
    threshold, recall, flag_rate = threshold_at_recall(scores, truth, 1.0)
    assert recall == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert flag_rate == pytest.approx(0.6, rel=1e-15, abs=0.0)
    assert threshold == pytest.approx(0.7, rel=1e-15, abs=0.0)


def test_threshold_at_recall_accepts_a_partial_target():
    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    truth = np.array([False, False, True, True, True])
    _threshold, recall, flag_rate = threshold_at_recall(scores, truth, 0.6)
    assert recall >= 0.6
    assert flag_rate <= 0.6


@pytest.mark.parametrize("target", [0.0, -0.1, 1.5])
def test_threshold_at_recall_rejects_an_out_of_range_target(target):
    with pytest.raises(ValueError, match="target_recall must lie"):
        threshold_at_recall(np.zeros(3), np.zeros(3, dtype=bool), target)


def test_threshold_at_recall_raises_when_no_positive_exists():
    with pytest.raises(ValueError, match="no threshold reaches"):
        threshold_at_recall(np.array([0.1, 0.2]), np.array([False, False]), 1.0)
