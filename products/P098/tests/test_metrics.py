"""Error directions and the scoring metrics, with hand arithmetic."""

from __future__ import annotations

import math

import pytest

from evidencestale.metrics import brier_score, error_directions, roc_auc


def test_confusion_known_answer() -> None:
    # predicted [1,1,0,0,1] against truth [1,0,0,1,1]:
    #   position 0: 1,1 -> TP
    #   position 1: 1,0 -> FP
    #   position 2: 0,0 -> TN
    #   position 3: 0,1 -> FN
    #   position 4: 1,1 -> TP
    # TP 2, FP 1, TN 1, FN 1
    errors = error_directions([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert (errors.true_positive, errors.false_positive) == (2, 1)
    assert (errors.true_negative, errors.false_negative) == (1, 1)
    assert errors.actually_stale == 3
    assert errors.actually_current == 2
    # missed rate = 1/3, alarm rate = 1/2, agreement = (2+1)/5
    assert errors.missed_staleness_rate == pytest.approx(1 / 3, rel=1e-15, abs=0.0)
    assert errors.false_alarm_rate == pytest.approx(0.5, rel=0.0, abs=0.0)
    assert errors.agreement == pytest.approx(0.6, rel=1e-15, abs=0.0)


def test_rates_are_nan_when_a_class_is_absent() -> None:
    errors = error_directions([0, 0], [0, 0])
    assert math.isnan(errors.missed_staleness_rate)
    assert errors.false_alarm_rate == 0.0
    empty = error_directions([], [])
    assert math.isnan(empty.agreement)


def test_misaligned_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="aligned item by item"):
        error_directions([1, 0], [1])


def test_non_binary_values_rejected_with_a_thresholding_hint() -> None:
    with pytest.raises(ValueError, match="Threshold a"):
        error_directions([0.7, 0], [1, 0])


def test_auc_known_answer_perfect_separation() -> None:
    # Two positives scored above both negatives: all 4 pairs won, 4/4 = 1.0
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(
        1.0, rel=0.0, abs=0.0
    )


def test_auc_known_answer_with_ties_mid_ranked() -> None:
    # positives [0.5, 0.5], negatives [0.5, 0.1]: pairs
    #   (0.5, 0.5) tie -> 0.5   (0.5, 0.1) win -> 1
    #   (0.5, 0.5) tie -> 0.5   (0.5, 0.1) win -> 1
    # total 3.0 over 4 pairs = 0.75
    assert roc_auc([0.5, 0.5, 0.5, 0.1], [1, 1, 0, 0]) == pytest.approx(
        0.75, rel=0.0, abs=0.0
    )


def test_auc_of_a_binary_predictor_is_the_mean_of_sensitivity_and_specificity() -> None:
    # decisions [1,0,1,0,0] truth [1,1,0,0,0]
    # sensitivity = 1/2, specificity = 2/3, mean = 7/12 = 0.5833...
    assert roc_auc([1.0, 0.0, 1.0, 0.0, 0.0], [1, 1, 0, 0, 0]) == pytest.approx(
        7 / 12, rel=1e-15, abs=0.0
    )


def test_auc_is_nan_when_one_class_is_empty() -> None:
    assert math.isnan(roc_auc([0.1, 0.2], [0, 0]))
    assert math.isnan(roc_auc([0.1, 0.2], [1, 1]))


def test_brier_known_answer() -> None:
    # ((0.8-1)^2 + (0.3-0)^2) / 2 = (0.04 + 0.09)/2 = 0.065
    assert brier_score([0.8, 0.3], [1, 0]) == pytest.approx(0.065, rel=1e-12, abs=0.0)


def test_brier_rejects_a_probability_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="not emitting a probability"):
        brier_score([1.2], [1])


def test_brier_rejects_an_empty_forecast() -> None:
    with pytest.raises(ValueError, match="at least one forecast"):
        brier_score([], [])
