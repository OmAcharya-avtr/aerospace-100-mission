"""Calibration measurement, with the ECE and Brier arithmetic done by hand."""

from __future__ import annotations

import numpy as np
import pytest

from safemodetree import brier_score, expected_calibration_error, reliability_curve


def test_ece_hand_calculation():
    # Hand arithmetic with 5 equal-width bins over [0, 1]:
    #   confidences 0.9, 0.9 land in [0.8, 1.0]: acc 1/2, conf 0.9, gap 0.4,
    #     weight 2/4 -> 0.200
    #   confidence  0.6 lands in [0.6, 0.8): acc 1.0, conf 0.6, gap 0.4,
    #     weight 1/4 -> 0.100
    #   confidence  0.2 lands in [0.2, 0.4): acc 0.0, conf 0.2, gap 0.2,
    #     weight 1/4 -> 0.050
    #   ECE = 0.200 + 0.100 + 0.050 = 0.350, MCE = 0.400
    confidence = np.array([0.9, 0.9, 0.6, 0.2])
    correct = np.array([True, False, True, False])
    result = reliability_curve(confidence, correct, n_bins=5)
    assert result.ece == pytest.approx(0.35, abs=1e-12)
    assert result.mce == pytest.approx(0.40, abs=1e-12)
    assert result.accuracy == pytest.approx(0.5, abs=1e-12)
    assert result.mean_confidence == pytest.approx(0.65, abs=1e-12)
    assert result.overconfidence == pytest.approx(0.15, abs=1e-12)
    assert expected_calibration_error(confidence, correct, n_bins=5) == pytest.approx(
        0.35, abs=1e-12
    )


def test_bin_occupancy_sums_to_the_sample_count():
    confidence = np.linspace(0.0, 1.0, 101)
    correct = confidence > 0.5
    result = reliability_curve(confidence, correct, n_bins=10)
    assert int(result.bin_counts.sum()) == 101
    assert result.bin_edges.shape == (11,)


def test_confidence_of_exactly_one_falls_in_the_last_bin():
    result = reliability_curve(np.array([1.0]), np.array([True]), n_bins=4)
    assert int(result.bin_counts[-1]) == 1
    assert result.ece == pytest.approx(0.0, abs=1e-12)


def test_perfect_calibration_gives_zero_ece():
    # 100 predictions at confidence 0.5 of which exactly 50 are correct.
    confidence = np.full(100, 0.5)
    correct = np.array([True] * 50 + [False] * 50)
    assert expected_calibration_error(confidence, correct, n_bins=10) == pytest.approx(
        0.0, abs=1e-12
    )


def test_single_bin_ece_is_the_global_gap():
    confidence = np.array([0.8, 0.8, 0.8, 0.8])
    correct = np.array([True, True, True, False])
    assert expected_calibration_error(confidence, correct, n_bins=1) == pytest.approx(
        0.05, abs=1e-12
    )


def test_reliability_table_includes_the_ece_line():
    result = reliability_curve(np.array([0.9, 0.1]), np.array([True, False]), n_bins=10)
    text = result.table()
    assert "ECE" in text
    assert "MCE" in text


def test_brier_hand_calculation():
    # Hand arithmetic, two classes:
    #   row 1, true 'a': (0.7-1)^2 + (0.3-0)^2 = 0.09 + 0.09 = 0.18
    #   row 2, true 'b': (0.2-0)^2 + (0.8-1)^2 = 0.04 + 0.04 = 0.08
    #   mean = 0.13
    proba = np.array([[0.7, 0.3], [0.2, 0.8]])
    assert brier_score(proba, np.array(["a", "b"]), ("a", "b")) == pytest.approx(
        0.13, abs=1e-12
    )


def test_brier_is_zero_for_a_confident_correct_prediction():
    proba = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert brier_score(proba, np.array(["a", "b"]), ("a", "b")) == pytest.approx(
        0.0, abs=1e-12
    )


def test_brier_is_two_for_a_confident_wrong_prediction():
    proba = np.array([[1.0, 0.0]])
    assert brier_score(proba, np.array(["b"]), ("a", "b")) == pytest.approx(2.0, abs=1e-12)


def test_brier_rejects_wrong_shapes():
    with pytest.raises(ValueError, match="must be 2-D"):
        brier_score(np.array([0.5, 0.5]), np.array(["a"]), ("a", "b"))
    with pytest.raises(ValueError, match="columns and"):
        brier_score(np.array([[0.5, 0.5]]), np.array(["a"]), ("a", "b", "c"))
    with pytest.raises(ValueError, match="different lengths"):
        brier_score(np.array([[0.5, 0.5]]), np.array(["a", "b"]), ("a", "b"))


def test_non_finite_confidence_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        reliability_curve(np.array([np.nan]), np.array([True]))
