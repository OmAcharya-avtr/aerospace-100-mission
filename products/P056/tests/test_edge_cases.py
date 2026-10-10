"""Degenerate inputs that a user will hit: one sample, one class, ties."""

from __future__ import annotations

import numpy as np
import pytest

from calibaudit.binning import assign_bins
from calibaudit.decomposition import binned_decomposition, murphy_decomposition
from calibaudit.ece import debiased_ece, expected_calibration_error
from calibaudit.recalibration import IsotonicCalibration, PlattScaling
from calibaudit.reliability import bootstrap_reliability, reliability_curve
from calibaudit.scores import brier_score, log_score


class TestSingleSample:
    def test_brier_of_one_sample(self):
        assert brier_score([0.3], [1]) == pytest.approx(0.49, abs=1e-15)

    def test_decomposition_of_one_sample_has_zero_uncertainty(self):
        d = murphy_decomposition([0.3], [1])
        assert d.uncertainty == 0.0
        assert d.resolution == 0.0
        assert d.reliability == pytest.approx(0.49, abs=1e-15)
        assert abs(d.identity_residual) < 1e-16

    def test_ece_of_one_sample_is_the_absolute_error(self):
        assert expected_calibration_error([0.3], [1], n_bins=10) == pytest.approx(
            0.7, abs=1e-15
        )

    def test_reliability_curve_of_one_sample_has_one_bin(self):
        c = reliability_curve([0.3], [1], n_bins=10)
        assert c.n_occupied == 1
        assert c.counts.tolist() == [1]


class TestSingleClass:
    def test_all_zero_outcomes_give_zero_uncertainty(self):
        d = binned_decomposition([0.1, 0.2, 0.3], [0, 0, 0], n_bins=3)
        assert d.uncertainty == 0.0
        assert d.resolution == 0.0
        assert abs(d.identity_residual) < 1e-16

    def test_all_one_outcomes_give_zero_uncertainty(self):
        d = binned_decomposition([0.7, 0.8, 0.9], [1, 1, 1], n_bins=3)
        assert d.uncertainty == 0.0

    def test_platt_fits_on_a_single_class(self):
        # Degenerate but must not raise: the likelihood is monotone in the
        # intercept, so L-BFGS-B stops at its bound rather than diverging.
        model = PlattScaling().fit([0.2, 0.4, 0.6], [0, 0, 0])
        assert np.all(np.isfinite(model.predict([0.5])))

    def test_isotonic_fits_on_a_single_class(self):
        model = IsotonicCalibration().fit([0.2, 0.4, 0.6], [1, 1, 1])
        assert model.predict([0.5])[0] == pytest.approx(1.0)
        assert model.n_steps == 1


class TestTiedForecasts:
    def test_all_identical_forecasts_give_one_group(self):
        d = murphy_decomposition([0.4] * 10, [1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        assert d.n_bins_occupied == 1

    def test_equal_mass_bins_collapse_under_heavy_ties(self):
        f = np.array([0.5] * 20)
        labels, _ = assign_bins(f, n_bins=10, strategy="equal_mass")
        assert np.unique(labels).size < 10

    def test_ece_on_tied_forecasts_is_the_overall_gap(self):
        f = [0.5] * 8
        o = [1, 1, 0, 0, 0, 0, 0, 0]
        assert expected_calibration_error(f, o, n_bins=10) == pytest.approx(
            0.25, abs=1e-15
        )

    def test_decomposition_handles_mixed_ties_and_singletons(self):
        f = np.array([0.1, 0.1, 0.1, 0.9])
        o = np.array([0.0, 1.0, 0.0, 1.0])
        d = murphy_decomposition(f, o)
        assert d.n_bins_occupied == 2
        assert abs(d.identity_residual) < 1e-16


class TestExtremeForecasts:
    def test_all_zero_forecasts_with_all_zero_outcomes_score_zero(self):
        assert brier_score([0.0] * 5, [0] * 5) == 0.0
        assert log_score([0.0] * 5, [0] * 5) == pytest.approx(0.0, abs=1e-14)

    def test_all_one_forecasts_with_all_zero_outcomes_score_one(self):
        assert brier_score([1.0] * 5, [0] * 5) == 1.0

    def test_log_score_of_a_confidently_wrong_forecast_is_large_but_finite(self):
        value = log_score([1.0, 1.0], [0, 0])
        assert np.isfinite(value)
        assert value > 30.0

    def test_platt_handles_forecasts_at_the_boundary(self):
        model = PlattScaling().fit([0.0, 0.0, 1.0, 1.0], [0, 1, 1, 1])
        out = model.predict([0.0, 0.5, 1.0])
        assert np.all(np.isfinite(out))
        assert np.all((out >= 0.0) & (out <= 1.0))

    def test_debiased_ece_on_degenerate_forecasts_does_not_raise(self):
        r = debiased_ece([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1], n_bins=4, n_replicates=10)
        assert np.isfinite(r.raw)
        assert np.isfinite(r.debiased)


class TestSmallBootstrap:
    def test_two_bootstrap_replicates_is_accepted(self):
        c = bootstrap_reliability([0.1, 0.9, 0.3, 0.7], [0, 1, 0, 1], n_bins=2,
                                  n_bootstrap=2, seed=1)
        assert c.n_bootstrap == 2
        assert c.lower is not None

    def test_band_on_a_two_sample_bin_is_degenerate_but_finite(self):
        c = bootstrap_reliability([0.1, 0.15], [0, 1], n_bins=2, n_bootstrap=50, seed=2)
        assert np.all(np.isfinite(c.lower))
        assert np.all(np.isfinite(c.upper))
        assert c.lower[0] == pytest.approx(0.0, abs=1e-15)
        assert c.upper[0] == pytest.approx(1.0, abs=1e-15)


class TestManyBins:
    def test_more_bins_than_samples_leaves_most_empty(self):
        d = binned_decomposition([0.1, 0.5, 0.9], [0, 1, 1], n_bins=1000)
        assert d.n_bins_occupied == 3
        assert d.n_bins_requested == 1000
        assert abs(d.identity_residual) < 1e-16

    def test_ece_with_one_bin_per_sample_reaches_its_maximum(self):
        # Every bin holds one sample, so every gap is |f - o|.
        f = np.array([0.2, 0.5, 0.8])
        o = np.array([0.0, 1.0, 1.0])
        expected = float(np.mean(np.abs(f - o)))
        assert expected_calibration_error(f, o, n_bins=1000) == pytest.approx(
            expected, abs=1e-15
        )
