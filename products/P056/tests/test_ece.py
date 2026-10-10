"""Expected calibration error, its null distribution, and the bias curve."""

from __future__ import annotations

import numpy as np
import pytest

from calibaudit.ece import (
    calibration_gaps,
    debiased_ece,
    ece_bias_curve,
    expected_calibration_error,
    maximum_calibration_error,
    null_ece_distribution,
)
from calibaudit.synthetic import get_spec, sample_forecast


class TestECEKnownAnswers:
    def test_hand_computed_two_bin_case(self):
        # Equal-width 2 bins. Bin 0 ([0, 0.5)): f = (0.1, 0.3), o = (0, 1)
        #   fbar = 0.2, obar = 0.5, |gap| = 0.3, weight 0.5
        # Bin 1 ([0.5, 1]): f = (0.7, 0.9), o = (1, 1)
        #   fbar = 0.8, obar = 1.0, |gap| = 0.2, weight 0.5
        # ECE = 0.5*0.3 + 0.5*0.2 = 0.25
        f = [0.1, 0.3, 0.7, 0.9]
        o = [0, 1, 1, 1]
        assert expected_calibration_error(f, o, n_bins=2) == pytest.approx(0.25, abs=1e-15)

    def test_hand_computed_mce_is_the_larger_gap(self):
        f = [0.1, 0.3, 0.7, 0.9]
        o = [0, 1, 1, 1]
        assert maximum_calibration_error(f, o, n_bins=2) == pytest.approx(0.3, abs=1e-15)

    def test_one_bin_ece_is_the_overall_gap(self):
        # mean f = 0.5, mean o = 0.75 -> ECE = 0.25
        assert expected_calibration_error(
            [0.2, 0.4, 0.6, 0.8], [0, 1, 1, 1], n_bins=1
        ) == pytest.approx(0.25, abs=1e-15)

    def test_perfect_bin_agreement_gives_zero(self):
        # Two bins, in each the mean forecast equals the observed frequency.
        f = [0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75]
        o = [1, 0, 0, 0, 1, 1, 1, 0]
        assert expected_calibration_error(f, o, n_bins=2) == pytest.approx(0.0, abs=1e-16)

    def test_worst_case_ece_is_one(self):
        assert expected_calibration_error([1.0, 1.0], [0, 0], n_bins=2) == pytest.approx(
            1.0, abs=1e-16
        )

    def test_weights_sum_to_one_and_drop_empty_bins(self):
        weights, gaps = calibration_gaps([0.05, 0.95], [0, 1], n_bins=10)
        assert weights.size == 2
        assert gaps.size == 2
        assert float(weights.sum()) == pytest.approx(1.0, abs=1e-15)

    def test_gaps_are_signed(self):
        # f = 0.9, o = 0 in one bin: gap = 0.9 - 0 = +0.9 (over-forecast).
        _, gaps = calibration_gaps([0.9], [0], n_bins=1)
        assert gaps[0] == pytest.approx(0.9, abs=1e-15)
        _, gaps = calibration_gaps([0.1], [1], n_bins=1)
        assert gaps[0] == pytest.approx(-0.9, abs=1e-15)


class TestBinCountAndStrategyChangeTheAnswer:
    @pytest.mark.parametrize("n_bins", [2, 5, 10, 25, 50])
    def test_ece_grows_with_bin_count_on_a_calibrated_forecast(
        self, n_bins, calibrated_large
    ):
        s = calibrated_large
        value = expected_calibration_error(s.forecasts, s.outcomes, n_bins=n_bins)
        assert value > 0.0

    def test_more_bins_means_more_measured_ece_on_calibrated_data(self, calibrated_large):
        s = calibrated_large
        low = expected_calibration_error(s.forecasts, s.outcomes, n_bins=5)
        high = expected_calibration_error(s.forecasts, s.outcomes, n_bins=50)
        assert high > low

    def test_strategies_disagree(self, calibrated_large):
        s = calibrated_large
        a = expected_calibration_error(s.forecasts, s.outcomes, n_bins=15, strategy="equal_width")
        b = expected_calibration_error(s.forecasts, s.outcomes, n_bins=15, strategy="equal_mass")
        assert a != b

    def test_equal_mass_occupies_every_bin_on_continuous_forecasts(self, rare_large):
        # Equal-width binning on a rare-event forecast leaves most bins empty;
        # equal-mass binning does not. This is the practical reason to use it.
        s = rare_large
        w, _ = calibration_gaps(s.forecasts, s.outcomes, n_bins=20, strategy="equal_width")
        m, _ = calibration_gaps(s.forecasts, s.outcomes, n_bins=20, strategy="equal_mass")
        assert m.size > w.size


class TestNullDistribution:
    def test_null_distribution_has_the_requested_length(self, calibrated_small):
        null = null_ece_distribution(
            calibrated_small.forecasts, n_bins=10, n_replicates=37, seed=1
        )
        assert null.shape == (37,)

    def test_null_values_are_strictly_positive(self, calibrated_small):
        null = null_ece_distribution(
            calibrated_small.forecasts, n_bins=10, n_replicates=50, seed=2
        )
        assert np.all(null > 0.0)

    def test_null_distribution_is_deterministic_in_the_seed(self, calibrated_small):
        a = null_ece_distribution(calibrated_small.forecasts, n_replicates=20, seed=7)
        b = null_ece_distribution(calibrated_small.forecasts, n_replicates=20, seed=7)
        assert np.array_equal(a, b)

    def test_different_seeds_give_different_draws(self, calibrated_small):
        a = null_ece_distribution(calibrated_small.forecasts, n_replicates=20, seed=7)
        b = null_ece_distribution(calibrated_small.forecasts, n_replicates=20, seed=8)
        assert not np.array_equal(a, b)

    def test_null_mean_falls_as_the_sample_grows(self):
        spec = get_spec("calibrated")
        small = sample_forecast(spec, 250, seed=56200)
        large = sample_forecast(spec, 4000, seed=56201)
        m_small = null_ece_distribution(
            small.forecasts, n_bins=10, n_replicates=80, seed=3
        ).mean()
        m_large = null_ece_distribution(
            large.forecasts, n_bins=10, n_replicates=80, seed=3
        ).mean()
        assert m_large < m_small

    def test_null_mean_rises_with_bin_count(self, calibrated_large):
        f = calibrated_large.forecasts
        m5 = null_ece_distribution(f, n_bins=5, n_replicates=60, seed=4).mean()
        m50 = null_ece_distribution(f, n_bins=50, n_replicates=60, seed=4).mean()
        assert m50 > m5


class TestDebiasedECE:
    def test_debiased_is_raw_minus_null_mean(self, calibrated_small):
        s = calibrated_small
        r = debiased_ece(s.forecasts, s.outcomes, n_bins=10, n_replicates=60, seed=5)
        assert r.debiased == pytest.approx(r.raw - r.null_mean, abs=1e-15)

    def test_debiasing_reduces_the_error_on_a_calibrated_forecast(self, calibrated_large):
        # Population ECE is exactly 0, so |debiased| should beat |raw|.
        s = calibrated_large
        r = debiased_ece(s.forecasts, s.outcomes, n_bins=20, n_replicates=120, seed=6)
        assert abs(r.debiased) < abs(r.raw)

    def test_p_value_is_large_on_a_calibrated_forecast(self, calibrated_large):
        s = calibrated_large
        r = debiased_ece(s.forecasts, s.outcomes, n_bins=20, n_replicates=200, seed=7)
        assert r.p_value > 0.05

    def test_p_value_is_small_on_a_badly_miscalibrated_forecast(self):
        s = sample_forecast(get_spec("biased_high"), 4000, seed=56202)
        r = debiased_ece(s.forecasts, s.outcomes, n_bins=15, n_replicates=200, seed=8)
        assert r.p_value < 0.01

    def test_debiased_can_go_negative_and_is_not_clipped(self):
        # Scan seeds until the correction overshoots. That it happens at all is
        # the limitation documented in README.md; clipping would hide it.
        spec = get_spec("calibrated")
        found = False
        for seed in range(12):
            s = sample_forecast(spec, 600, seed=56300 + seed)
            r = debiased_ece(s.forecasts, s.outcomes, n_bins=12, n_replicates=80, seed=seed)
            if r.debiased < 0.0:
                found = True
                break
        assert found, "expected at least one overshoot in 12 seeds"

    def test_quantiles_bracket_the_null_mean(self, calibrated_small):
        s = calibrated_small
        r = debiased_ece(s.forecasts, s.outcomes, n_bins=10, n_replicates=100, seed=9)
        assert r.null_q05 <= r.null_mean <= r.null_q95

    def test_report_mentions_the_bias(self, calibrated_small):
        s = calibrated_small
        text = debiased_ece(s.forecasts, s.outcomes, n_replicates=20, seed=1).report()
        assert "binning bias" in text
        assert "p-value" in text

    def test_recorded_configuration_is_echoed(self, calibrated_small):
        s = calibrated_small
        r = debiased_ece(
            s.forecasts, s.outcomes, n_bins=7, strategy="equal_width", n_replicates=11, seed=1
        )
        assert r.n_bins == 7
        assert r.strategy == "equal_width"
        assert r.n_replicates == 11
        assert r.n_samples == s.n_samples


class TestBiasCurve:
    @pytest.fixture(scope="class")
    @staticmethod
    def curve():
        return ece_bias_curve(
            get_spec("calibrated"),
            n_bins_grid=[5, 10, 20],
            n_samples_grid=[200, 1000],
            n_replicates=25,
            seed=56,
        )

    def test_grid_shape(self, curve):
        assert len(curve.rows) == 6

    def test_every_cell_has_positive_bias_on_a_calibrated_spec(self, curve):
        assert all(r.bias > 0.0 for r in curve.rows)
        assert all(r.true_ece == 0.0 for r in curve.rows)

    def test_bias_rises_with_bins_at_fixed_sample_size(self, curve):
        rows = sorted((r for r in curve.rows if r.n_samples == 1000), key=lambda r: r.n_bins)
        biases = [r.bias for r in rows]
        assert biases == sorted(biases)

    def test_bias_falls_with_sample_size_at_fixed_bin_count(self, curve):
        small = next(r for r in curve.rows if r.n_samples == 200 and r.n_bins == 20)
        large = next(r for r in curve.rows if r.n_samples == 1000 and r.n_bins == 20)
        assert large.bias < small.bias

    def test_power_law_exponent_is_near_one_half(self, curve):
        slope, _, r2 = curve.power_law_fit()
        # Theory: the leading Bernoulli-noise term scales as sqrt(B / n).
        assert 0.4 < slope < 0.6
        assert r2 > 0.95

    def test_table_renders_every_row(self, curve):
        text = curve.table()
        assert len(text.splitlines()) == len(curve.rows) + 2

    def test_power_law_fit_needs_three_positive_cells(self):
        curve = ece_bias_curve(
            get_spec("calibrated"),
            n_bins_grid=[5],
            n_samples_grid=[400],
            n_replicates=5,
            seed=1,
        )
        with pytest.raises(ValueError, match="at least 3 cells"):
            curve.power_law_fit()

    def test_debias_replicates_populate_the_debiased_columns(self):
        curve = ece_bias_curve(
            get_spec("calibrated"),
            n_bins_grid=[10],
            n_samples_grid=[400],
            n_replicates=6,
            seed=2,
            debias_replicates=20,
        )
        row = curve.rows[0]
        assert np.isfinite(row.mean_debiased)
        assert abs(row.debiased_bias) < row.bias
