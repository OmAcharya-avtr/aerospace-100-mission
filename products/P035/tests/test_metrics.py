"""Tests for confusion matrices, window-level ROC and detection-delay statistics."""

from __future__ import annotations

import numpy as np
import pytest

from telemetryool.metrics import confusion_matrix, delay_stats, window_roc


def test_confusion_matrix_hand_values() -> None:
    """3 anomalous windows of which 2 alarmed, 5 nominal of which 1 alarmed:

    TP = 2, FN = 1, FP = 1, TN = 4
    tpr = 2/3 = 0.666667, fpr = 1/5 = 0.2, precision = 2/3, specificity = 4/5,
    f1 = 2 * (2/3) * (2/3) / (4/3) = 2/3, accuracy = (2 + 4) / 8 = 0.75
    """
    cm = confusion_matrix(
        np.array([True, True, False]), np.array([True, False, False, False, False])
    )
    assert (cm.true_positive, cm.false_negative, cm.false_positive, cm.true_negative) == (
        2, 1, 1, 4
    )
    assert cm.n_positive == 3 and cm.n_negative == 5 and cm.total == 8
    assert cm.tpr == pytest.approx(2 / 3, abs=1e-15)
    assert cm.fpr == pytest.approx(0.2, abs=1e-15)
    assert cm.precision == pytest.approx(2 / 3, abs=1e-15)
    assert cm.specificity == pytest.approx(0.8, abs=1e-15)
    assert cm.f1 == pytest.approx(2 / 3, abs=1e-15)
    assert cm.accuracy == pytest.approx(0.75, abs=1e-15)


def test_confusion_matrix_table_contains_all_four_counts() -> None:
    cm = confusion_matrix(np.array([True, False]), np.array([True, True, False]))
    text = cm.table()
    assert "predicted alarm" in text and "predicted nominal" in text
    numbers = [int(tok) for tok in text.split() if tok.isdigit()]
    for count in (1, 1, 2, 1):
        assert count in numbers
    assert "tpr=" in cm.summary() and "accuracy=" in cm.summary()


def test_confusion_matrix_degenerate_cases_return_nan_not_zero() -> None:
    no_positives = confusion_matrix(np.array([], dtype=bool), np.array([True, False]))
    assert np.isnan(no_positives.tpr)
    nothing_flagged = confusion_matrix(np.array([False]), np.array([False]))
    assert np.isnan(nothing_flagged.precision)
    assert np.isnan(nothing_flagged.f1)


def test_confusion_matrix_shape_validation() -> None:
    with pytest.raises(ValueError, match="must be 1-D"):
        confusion_matrix(np.zeros((2, 2), dtype=bool), np.array([True]))


def test_roc_perfectly_separable_has_auc_one() -> None:
    nominal = np.zeros((50, 10))
    anomalous = np.ones((50, 10))
    roc = window_roc(nominal, anomalous, 1)
    assert roc.auc == pytest.approx(1.0, abs=1e-12)
    assert roc.tpr_at_fpr(0.0) == pytest.approx(1.0, abs=1e-12)


def test_roc_identical_distributions_has_auc_about_half() -> None:
    rng = np.random.default_rng(0)
    nominal = rng.standard_normal((4000, 20))
    anomalous = rng.standard_normal((4000, 20))
    roc = window_roc(nominal, anomalous, 1)
    assert 0.47 < roc.auc < 0.53


def test_roc_starts_at_origin_and_is_monotone_in_threshold() -> None:
    rng = np.random.default_rng(1)
    roc = window_roc(rng.standard_normal((500, 20)), rng.standard_normal((500, 20)) + 1.0, 2)
    assert roc.thresholds[0] == np.inf and roc.thresholds[-1] == -np.inf
    assert roc.fpr[0] == 0.0 and roc.tpr[0] == 0.0
    assert roc.fpr[-1] == 1.0 and roc.tpr[-1] == 1.0
    assert np.all(np.diff(roc.thresholds) <= 0)
    assert np.all(np.diff(roc.fpr) >= -1e-15)
    assert np.all(np.diff(roc.tpr) >= -1e-15)


def test_tpr_at_fpr_returns_nan_when_unreachable() -> None:
    roc = window_roc(np.ones((10, 5)), np.ones((10, 5)), 1)
    assert np.isnan(roc.tpr_at_fpr(-1.0))


def test_delay_stats_hand_values() -> None:
    """Three windows of length 6, threshold 0.5, persistence 1, onset 2.

    row 0: 0 0 1 0 0 0 -> first exceedance at index 2, delay 0
    row 1: 0 0 0 0 1 0 -> index 4, delay 2
    row 2: 1 0 0 0 0 0 -> index 0, before onset -> counted as early, not detected
    So n_detected = 2, n_early = 1, Pd = 2/3, delays = [0, 2], mean 1.0, median 1.0.
    """
    scores = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    stats = delay_stats(scores, 0.5, 1, onset=2)
    assert stats.n_detected == 2
    assert stats.n_early == 1
    assert stats.delays.tolist() == [0, 2]
    assert stats.detection_probability == pytest.approx(2 / 3, abs=1e-15)
    assert stats.mean_delay == pytest.approx(1.0, abs=1e-15)
    assert stats.median_delay == pytest.approx(1.0, abs=1e-15)
    assert stats.onset == 2 and stats.n_windows == 3
    assert "Pd=" in stats.summary()


def test_delay_stats_no_detection_gives_nan_delays() -> None:
    stats = delay_stats(np.zeros((5, 10)), 1.0, 1, onset=0)
    assert stats.n_detected == 0
    assert np.isnan(stats.mean_delay)
    assert np.isnan(stats.median_delay)
    assert stats.delays.size == 0


def test_delay_cannot_be_shorter_than_the_debounce_count() -> None:
    """With persistence = 3 and a step that starts at onset, the earliest
    possible alarm index is onset + 2, so the minimum delay is 2."""
    scores = np.zeros((20, 30))
    scores[:, 10:] = 5.0
    stats = delay_stats(scores, 1.0, 3, onset=10)
    assert stats.delays.min() == 2
    assert stats.detection_probability == 1.0


def test_delay_stats_validation() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        delay_stats(np.zeros(5), 1.0, 1, 0)
    with pytest.raises(ValueError, match="outside a window"):
        delay_stats(np.zeros((2, 5)), 1.0, 1, 5)
