"""Calibration metric tests, with hand-calculated known answers."""

from __future__ import annotations

import numpy as np
import pytest

from linkoutage.calibration import (
    CalibrationReport,
    brier_decomposition,
    brier_score,
    evaluate_forecast,
    expected_calibration_error,
    reliability_curve,
    wilson_interval,
)


def test_brier_score_hand_calculation():
    # y = [1, 0], p = [0.8, 0.3] -> ((0.8-1)^2 + (0.3-0)^2)/2 = (0.04+0.09)/2
    assert brier_score([1, 0], [0.8, 0.3]) == pytest.approx(0.065)


def test_brier_score_of_a_perfect_forecaster_is_zero():
    assert brier_score([1, 0, 1], [1.0, 0.0, 1.0]) == 0.0


def test_brier_score_of_the_worst_forecaster_is_one():
    assert brier_score([1, 0], [0.0, 1.0]) == pytest.approx(1.0)


def test_constant_forecast_at_the_base_rate_equals_uncertainty():
    rng = np.random.default_rng(40)
    y = (rng.random(20_000) < 0.03).astype(int)
    base = float(y.mean())
    bs, rel, res, unc, residual = brier_decomposition(y, np.full(y.size, base))
    assert bs == pytest.approx(unc, rel=1e-12)
    assert rel == pytest.approx(0.0, abs=1e-12)
    assert res == pytest.approx(0.0, abs=1e-12)
    assert residual == pytest.approx(0.0, abs=1e-12)


def test_decomposition_residual_is_small_for_a_real_forecaster():
    rng = np.random.default_rng(41)
    p = rng.random(20_000) * 0.2
    y = (rng.random(20_000) < p).astype(int)
    bs, rel, res, unc, residual = brier_decomposition(y, p, n_bins=20)
    assert abs(residual) < 0.05 * bs


def test_resolution_is_positive_for_an_informative_forecaster():
    rng = np.random.default_rng(42)
    p = rng.random(20_000) * 0.5
    y = (rng.random(20_000) < p).astype(int)
    _, _, res, _, _ = brier_decomposition(y, p)
    assert res > 0.0


def test_uncertainty_term_is_base_rate_times_complement():
    y = np.array([1] * 30 + [0] * 70)
    _, _, _, unc, _ = brier_decomposition(y, np.full(100, 0.3))
    assert unc == pytest.approx(0.3 * 0.7)


def test_ece_of_a_perfectly_calibrated_constant_forecast_is_tiny():
    y = np.array([1] * 100 + [0] * 900)
    ece, mce = expected_calibration_error(y, np.full(1000, 0.1))
    assert ece == pytest.approx(0.0, abs=1e-12)
    assert mce == pytest.approx(0.0, abs=1e-12)


def test_ece_hand_calculation_with_uniform_bins():
    # Two groups: 10 rows at p = 0.1 with 5 positives (observed 0.5, gap 0.4)
    #             10 rows at p = 0.9 with 9 positives (observed 0.9, gap 0.0)
    # ECE = (10*0.4 + 10*0.0) / 20 = 0.2;  MCE = 0.4
    p = np.array([0.1] * 10 + [0.9] * 10)
    y = np.array([1] * 5 + [0] * 5 + [1] * 9 + [0])
    ece, mce = expected_calibration_error(y, p, n_bins=2, strategy="uniform")
    assert ece == pytest.approx(0.2)
    assert mce == pytest.approx(0.4)


def test_reliability_curve_of_a_single_valued_forecast_has_one_bin():
    y = np.array([1] * 10 + [0] * 90)
    curve = reliability_curve(y, np.full(100, 0.1))
    assert curve.count.size == 1
    assert curve.count[0] == 100
    assert curve.observed_frequency[0] == pytest.approx(0.1)


def test_reliability_curve_counts_sum_to_n():
    rng = np.random.default_rng(43)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    curve = reliability_curve(y, p, n_bins=10)
    assert int(curve.count.sum()) == 5000
    assert int(curve.n_positive.sum()) == int(y.sum())


def test_reliability_curve_table_renders():
    rng = np.random.default_rng(44)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    assert "observed" in reliability_curve(y, p).table()


def test_reliability_curve_interval_brackets_the_observation():
    rng = np.random.default_rng(45)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    curve = reliability_curve(y, p, n_bins=10)
    assert np.all(curve.lower <= curve.observed_frequency + 1e-12)
    assert np.all(curve.upper >= curve.observed_frequency - 1e-12)


def test_wilson_interval_zero_successes():
    lo, hi = wilson_interval(0, 10)
    assert lo == pytest.approx(0.0)
    assert 0.0 < hi < 0.35


def test_wilson_interval_all_successes():
    lo, hi = wilson_interval(10, 10)
    assert hi == pytest.approx(1.0)
    assert 0.6 < lo < 1.0


def test_wilson_interval_symmetric_case():
    lo, hi = wilson_interval(50, 100)
    assert (lo + hi) / 2 == pytest.approx(0.5, abs=1e-12)


def test_wilson_interval_zero_n():
    lo, hi = wilson_interval(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_wilson_interval_rejects_bad_counts():
    with pytest.raises(ValueError, match="0 <= k <= n"):
        wilson_interval(11, 10)


def test_evaluate_forecast_skill_against_the_base_rate():
    rng = np.random.default_rng(46)
    y = (rng.random(20_000) < 0.05).astype(int)
    base = float(y.mean())
    report = evaluate_forecast(y, np.full(y.size, base), name="constant", reference_rate=base)
    assert report.brier_skill_score == pytest.approx(0.0, abs=1e-12)
    assert report.resolution == pytest.approx(0.0, abs=1e-12)
    assert report.accuracy_all_negative == pytest.approx(1.0 - base)


def test_evaluate_forecast_positive_skill_for_an_informative_forecast():
    rng = np.random.default_rng(47)
    p = np.clip(rng.random(20_000) * 0.1, 1e-6, 1.0)
    y = (rng.random(20_000) < p).astype(int)
    base = float(y.mean())
    report = evaluate_forecast(y, p, name="oracle", reference_rate=base)
    assert report.brier_skill_score > 0.0
    assert report.roc_auc > 0.6


def test_evaluate_forecast_accuracy_is_not_informative_at_a_low_base_rate():
    rng = np.random.default_rng(48)
    y = (rng.random(20_000) < 0.02).astype(int)
    report = evaluate_forecast(y, np.full(y.size, 0.02), name="constant")
    # The constant forecaster never crosses 0.5, so it scores exactly the
    # all-negative accuracy: 98 % and no skill.
    assert report.accuracy_at_half == pytest.approx(report.accuracy_all_negative)


def test_evaluate_forecast_auc_is_nan_for_a_single_class():
    report = evaluate_forecast(np.zeros(100, dtype=int), np.full(100, 0.1), name="x")
    assert np.isnan(report.roc_auc)
    assert np.isnan(report.average_precision)


def test_report_line_and_header_align():
    rng = np.random.default_rng(49)
    p = rng.random(1000) * 0.3
    y = (rng.random(1000) < p).astype(int)
    report = evaluate_forecast(y, p, name="x")
    assert len(report.line()) == len(CalibrationReport.header())


def test_probabilities_out_of_range_are_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([1, 0], [1.2, 0.3])


def test_non_binary_labels_are_rejected():
    with pytest.raises(ValueError, match="only 0 and 1"):
        brier_score([2, 0], [0.5, 0.5])


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="elements"):
        brier_score([1, 0], [0.5])


def test_non_finite_probability_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        brier_score([1, 0], [np.nan, 0.5])


def test_too_few_bins_rejected():
    with pytest.raises(ValueError, match="n_bins"):
        reliability_curve([1, 0], [0.5, 0.5], n_bins=1)


def test_bad_strategy_rejected():
    with pytest.raises(ValueError, match="strategy"):
        reliability_curve([1, 0], [0.5, 0.5], strategy="nope")
