"""Input validation. Every public entry point must reject bad input loudly."""

from __future__ import annotations

import numpy as np
import pytest

from calibaudit.binning import assign_bins, bin_edges
from calibaudit.decomposition import binned_decomposition, murphy_decomposition
from calibaudit.ece import debiased_ece, expected_calibration_error, null_ece_distribution
from calibaudit.recalibration import (
    IsotonicCalibration,
    PlattScaling,
    RawForecast,
    get_method,
    recalibration_audit,
)
from calibaudit.reliability import bootstrap_reliability, reliability_curve
from calibaudit.scores import brier_score, check_forecasts, log_score
from calibaudit.synthetic import ForecastSpec, get_spec, sample_forecast


class TestCheckForecasts:
    def test_two_dimensional_forecasts_rejected(self):
        with pytest.raises(ValueError, match="forecasts must be 1-D"):
            check_forecasts(np.zeros((2, 2)), np.zeros(4))

    def test_two_dimensional_outcomes_rejected(self):
        with pytest.raises(ValueError, match="outcomes must be 1-D"):
            check_forecasts(np.zeros(4), np.zeros((2, 2)))

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            check_forecasts([0.1, 0.2], [0])

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="at least 1 samples"):
            check_forecasts([], [])

    def test_min_samples_enforced(self):
        with pytest.raises(ValueError, match="at least 4 samples"):
            check_forecasts([0.1, 0.2], [0, 1], min_samples=4)

    def test_nan_forecast_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            check_forecasts([0.5, np.nan], [0, 1])

    def test_inf_forecast_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            check_forecasts([0.5, np.inf], [0, 1])

    def test_nan_outcome_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            check_forecasts([0.5, 0.5], [0, np.nan])

    @pytest.mark.parametrize("bad", [-0.5, 1.5, -1e-12, 1.0 + 1e-9])
    def test_forecast_out_of_unit_interval_rejected(self, bad):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            check_forecasts([0.5, bad], [0, 1])

    @pytest.mark.parametrize("bad", [0.5, 2.0, -1.0])
    def test_non_binary_outcome_rejected(self, bad):
        with pytest.raises(ValueError, match="binary 0 or 1"):
            check_forecasts([0.5, 0.5], [0, bad])

    def test_error_message_counts_offenders(self):
        with pytest.raises(ValueError, match="2 of 4 values"):
            check_forecasts([0.5] * 4, [0, 0.3, 1, 0.7])

    def test_boundary_values_accepted(self):
        f, o = check_forecasts([0.0, 1.0], [0, 1])
        assert f.tolist() == [0.0, 1.0]
        assert o.tolist() == [0.0, 1.0]

    def test_integer_input_is_coerced_to_float(self):
        f, o = check_forecasts([0, 1], [0, 1])
        assert f.dtype == np.float64
        assert o.dtype == np.float64


class TestScoreValidation:
    def test_brier_rejects_bad_outcome(self):
        with pytest.raises(ValueError, match="binary"):
            brier_score([0.5], [2])

    def test_log_score_rejects_bad_forecast(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            log_score([1.2], [1])


class TestBinningValidation:
    @pytest.mark.parametrize("bad", [0, -1, -10])
    def test_non_positive_bin_count_rejected(self, bad):
        with pytest.raises(ValueError, match="at least 1"):
            bin_edges([0.5], n_bins=bad)

    @pytest.mark.parametrize("bad", [2.5, "10", None])
    def test_non_integer_bin_count_rejected(self, bad):
        with pytest.raises(TypeError, match="must be an integer"):
            bin_edges([0.5], n_bins=bad)

    def test_bool_bin_count_rejected(self):
        with pytest.raises(TypeError, match="must be an integer"):
            bin_edges([0.5], n_bins=True)

    def test_unknown_strategy_rejected(self):
        with pytest.raises(ValueError, match="unknown binning strategy"):
            bin_edges([0.5], strategy="quantile-ish")

    def test_equal_mass_on_empty_input_rejected(self):
        with pytest.raises(ValueError, match="at least one forecast"):
            bin_edges([], strategy="equal_mass")

    def test_assign_bins_rejects_unknown_strategy(self):
        with pytest.raises(ValueError, match="unknown binning strategy"):
            assign_bins([0.5], strategy="nope")


class TestDecompositionValidation:
    def test_murphy_rejects_bad_input(self):
        with pytest.raises(ValueError, match="binary"):
            murphy_decomposition([0.5], [3])

    def test_binned_rejects_zero_bins(self):
        with pytest.raises(ValueError, match="at least 1"):
            binned_decomposition([0.5], [1], n_bins=0)

    def test_skill_against_climatology_undefined_when_outcomes_constant(self):
        d = binned_decomposition([0.2, 0.3, 0.4], [0, 0, 0], n_bins=3)
        assert d.uncertainty == 0.0
        with pytest.raises(ValueError, match="undefined"):
            _ = d.brier_skill_vs_climatology


class TestECEValidation:
    def test_ece_rejects_bad_strategy(self):
        with pytest.raises(ValueError, match="unknown binning strategy"):
            expected_calibration_error([0.5], [1], strategy="bogus")

    def test_null_distribution_rejects_zero_replicates(self):
        with pytest.raises(ValueError, match="at least 1"):
            null_ece_distribution([0.5, 0.6], n_replicates=0)

    def test_debiased_rejects_bad_forecasts(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            debiased_ece([1.4, 0.2], [1, 0], n_replicates=5)


class TestReliabilityValidation:
    def test_curve_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            reliability_curve([0.1, 0.2], [1])

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_band_level_out_of_range_rejected(self, bad):
        with pytest.raises(ValueError, match=r"level must lie in \(0, 1\)"):
            bootstrap_reliability([0.1, 0.9], [0, 1], level=bad, n_bins=2)

    @pytest.mark.parametrize("bad", [0, 1, -3])
    def test_too_few_bootstrap_replicates_rejected(self, bad):
        with pytest.raises(ValueError, match="n_bootstrap must be at least 2"):
            bootstrap_reliability([0.1, 0.9], [0, 1], n_bootstrap=bad, n_bins=2)

    def test_diagonal_excluded_without_band_raises(self):
        curve = reliability_curve([0.1, 0.9], [0, 1], n_bins=2)
        with pytest.raises(ValueError, match="no bootstrap band"):
            curve.diagonal_excluded()


class TestRecalibrationValidation:
    def test_unknown_method_rejected(self):
        with pytest.raises(KeyError, match="unknown method"):
            get_method("temperature")

    def test_negative_bootstrap_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            PlattScaling(n_bootstrap=-1)

    def test_predict_before_fit_raises(self):
        with pytest.raises(ValueError, match="not fitted"):
            PlattScaling().predict([0.5])

    def test_isotonic_predict_before_fit_raises(self):
        with pytest.raises(ValueError, match="not fitted"):
            IsotonicCalibration().predict([0.5])

    def test_isotonic_n_steps_before_fit_raises(self):
        with pytest.raises(ValueError, match="not fitted"):
            _ = IsotonicCalibration().n_steps

    def test_interval_without_ensemble_raises(self):
        model = PlattScaling(n_bootstrap=0).fit([0.1, 0.9, 0.4, 0.6], [0, 1, 0, 1])
        with pytest.raises(ValueError, match="no bootstrap ensemble"):
            model.predict_with_interval([0.5])

    def test_interval_rejects_bad_level(self):
        model = PlattScaling(n_bootstrap=4, seed=1).fit([0.1, 0.9, 0.4, 0.6], [0, 1, 0, 1])
        with pytest.raises(ValueError, match=r"level must lie in \(0, 1\)"):
            model.predict_with_interval([0.5], level=1.0)

    def test_fit_needs_two_samples(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            RawForecast().fit([0.5], [1])

    def test_audit_requires_raw_baseline(self, overconfident_small):
        s = overconfident_small
        with pytest.raises(ValueError, match="'raw' baseline must be included"):
            recalibration_audit(s.forecasts, s.outcomes, methods=("platt",))

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.4])
    def test_audit_rejects_bad_test_fraction(self, bad, overconfident_small):
        s = overconfident_small
        with pytest.raises(ValueError, match=r"test_fraction must lie in \(0, 1\)"):
            recalibration_audit(s.forecasts, s.outcomes, test_fraction=bad)

    def test_audit_rejects_split_that_leaves_too_few_samples(self):
        f = np.array([0.1, 0.2, 0.8, 0.9])
        o = np.array([0.0, 0.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="both must be at least 2"):
            recalibration_audit(f, o, test_fraction=0.1)

    def test_audit_needs_four_samples(self):
        with pytest.raises(ValueError, match="at least 4 samples"):
            recalibration_audit([0.1, 0.9], [0, 1])

    def test_by_method_unknown_name_raises(self, overconfident_small):
        audit = recalibration_audit(
            overconfident_small.forecasts, overconfident_small.outcomes, n_bootstrap=5
        )
        with pytest.raises(KeyError, match="not in this audit"):
            audit.by_method("temperature")


class TestSyntheticValidation:
    @pytest.mark.parametrize("a,b", [(0.0, 1.0), (1.0, 0.0), (-1.0, 2.0)])
    def test_non_positive_beta_shapes_rejected(self, a, b):
        with pytest.raises(ValueError, match="must be positive"):
            ForecastSpec("bad", a, b, "identity", 0.0, "x")

    def test_unknown_distortion_kind_rejected(self):
        with pytest.raises(ValueError, match="unknown distortion kind"):
            ForecastSpec("bad", 1.0, 1.0, "softmax", 0.0, "x")

    def test_non_positive_temperature_rejected(self):
        with pytest.raises(ValueError, match="temperature must be positive"):
            ForecastSpec("bad", 1.0, 1.0, "temperature", 0.0, "x")

    def test_unknown_spec_name_rejected(self):
        with pytest.raises(KeyError, match="unknown spec"):
            get_spec("not_a_spec")

    def test_zero_samples_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            sample_forecast(get_spec("calibrated"), 0, seed=1)
