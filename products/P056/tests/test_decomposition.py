"""Brier decomposition: identities, hand-computed cases, and reported fields."""

from __future__ import annotations

import numpy as np
import pytest

from calibaudit.decomposition import binned_decomposition, murphy_decomposition
from calibaudit.scores import brier_score


class TestExactIdentity:
    def test_three_term_identity_is_exact_on_discrete_forecasts(self, overconfident_large):
        s = overconfident_large
        d = murphy_decomposition(np.round(s.forecasts, 2), s.outcomes)
        assert abs(d.three_term_residual) < 1e-15
        assert abs(d.identity_residual) < 1e-15

    def test_within_bin_terms_are_exactly_zero_for_the_exact_decomposition(
        self, overconfident_large
    ):
        d = murphy_decomposition(
            np.round(overconfident_large.forecasts, 1), overconfident_large.outcomes
        )
        assert d.within_bin_variance == 0.0
        assert d.within_bin_covariance == 0.0

    @pytest.mark.parametrize("n_bins", [1, 2, 3, 5, 10, 17, 50, 200])
    def test_five_term_identity_is_exact_at_every_bin_count(self, n_bins, overconfident_large):
        s = overconfident_large
        d = binned_decomposition(s.forecasts, s.outcomes, n_bins=n_bins)
        assert abs(d.identity_residual) < 1e-14

    @pytest.mark.parametrize("strategy", ["equal_width", "equal_mass"])
    def test_five_term_identity_holds_for_both_strategies(self, strategy, overconfident_large):
        s = overconfident_large
        d = binned_decomposition(s.forecasts, s.outcomes, n_bins=13, strategy=strategy)
        assert abs(d.identity_residual) < 1e-14

    def test_three_term_residual_is_nonzero_on_continuous_forecasts(self, overconfident_large):
        s = overconfident_large
        d = binned_decomposition(s.forecasts, s.outcomes, n_bins=10)
        # The dropped terms are real and larger than floating-point noise.
        assert abs(d.three_term_residual) > 1e-6

    def test_one_bin_puts_everything_in_the_within_bin_terms(self, overconfident_large):
        s = overconfident_large
        d = binned_decomposition(s.forecasts, s.outcomes, n_bins=1)
        assert d.n_bins_occupied == 1
        assert d.resolution == pytest.approx(0.0, abs=1e-15)
        assert abs(d.identity_residual) < 1e-14

    def test_n_bins_equal_to_n_samples_does_not_make_it_exact(self, overconfident_small):
        s = overconfident_small
        d = binned_decomposition(s.forecasts, s.outcomes, n_bins=s.n_samples)
        # Equal-width bins with as many bins as samples still pool some samples.
        assert d.n_bins_occupied < s.n_samples
        assert abs(d.identity_residual) < 1e-14


class TestHandComputedDecompositions:
    def test_two_value_forecast_by_hand(self):
        # f = 0.2 four times with o = (0,0,0,1); f = 0.8 four times with o = (1,1,1,0).
        # obar = 4/8 = 0.5           UNC = 0.25
        # bin 1: n=4, f=0.2, o=0.25  bin 2: n=4, f=0.8, o=0.75
        # REL = 0.5*(0.2-0.25)^2 + 0.5*(0.8-0.75)^2 = 0.5*0.0025 + 0.5*0.0025 = 0.0025
        # RES = 0.5*(0.25-0.5)^2 + 0.5*(0.75-0.5)^2 = 0.5*0.0625*2        = 0.0625
        # BS  = REL - RES + UNC = 0.0025 - 0.0625 + 0.25 = 0.19
        f = np.array([0.2] * 4 + [0.8] * 4)
        o = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        d = murphy_decomposition(f, o)
        assert d.uncertainty == pytest.approx(0.25, abs=1e-15)
        assert d.reliability == pytest.approx(0.0025, abs=1e-15)
        assert d.resolution == pytest.approx(0.0625, abs=1e-15)
        assert d.brier == pytest.approx(0.19, abs=1e-15)
        assert d.three_term_sum == pytest.approx(0.19, abs=1e-15)

    def test_constant_forecast_at_base_rate_is_perfectly_reliable(self):
        # Constant f = 0.4 with obar = 0.4: single group, f = obar so REL = 0,
        # and a single group has no resolution, so BS = UNC = 0.4*0.6 = 0.24.
        o = np.array([1.0] * 4 + [0.0] * 6)
        d = murphy_decomposition(np.full(10, 0.4), o)
        assert d.reliability == pytest.approx(0.0, abs=1e-16)
        assert d.resolution == pytest.approx(0.0, abs=1e-16)
        assert d.uncertainty == pytest.approx(0.24, abs=1e-15)
        assert d.brier == pytest.approx(0.24, abs=1e-15)

    def test_perfect_sharp_forecast_has_resolution_equal_to_uncertainty(self):
        # f = o exactly: two groups 0 and 1, each perfectly reliable,
        # RES = UNC so BS = 0.
        o = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        d = murphy_decomposition(o, o)
        assert d.reliability == pytest.approx(0.0, abs=1e-16)
        assert d.resolution == pytest.approx(d.uncertainty, abs=1e-16)
        assert d.brier == pytest.approx(0.0, abs=1e-16)

    def test_worst_case_forecast_by_hand(self):
        # f = 1 - o: two groups, each maximally unreliable.
        # obar = 0.5, UNC = 0.25, REL = 0.5*1 + 0.5*1 = 1, RES = 0.25, BS = 1.
        o = np.array([0.0, 0.0, 1.0, 1.0])
        d = murphy_decomposition(1.0 - o, o)
        assert d.reliability == pytest.approx(1.0, abs=1e-15)
        assert d.resolution == pytest.approx(0.25, abs=1e-15)
        assert d.brier == pytest.approx(1.0, abs=1e-15)

    def test_within_bin_terms_by_hand(self):
        # One bin [0, 1) holding f = (0.2, 0.8) and o = (0, 1).
        # fbar = 0.5, obar = 0.5, UNC = 0.25, REL = 0, RES = 0.
        # WBV = ((0.2-0.5)^2 + (0.8-0.5)^2)/2 = 0.09
        # WBC = ((0.2-0.5)*0 + (0.8-0.5)*1)/2 = 0.15
        # BS  = 0 - 0 + 0.25 + 0.09 - 0.30 = 0.04, and directly
        #       ((0.2-0)^2 + (0.8-1)^2)/2 = 0.04. Agreement is the point.
        d = binned_decomposition([0.2, 0.8], [0, 1], n_bins=1)
        assert d.within_bin_variance == pytest.approx(0.09, abs=1e-15)
        assert d.within_bin_covariance == pytest.approx(0.15, abs=1e-15)
        assert d.brier == pytest.approx(0.04, abs=1e-15)
        assert d.five_term_sum == pytest.approx(0.04, abs=1e-15)
        assert d.three_term_sum == pytest.approx(0.25, abs=1e-15)
        assert d.three_term_residual == pytest.approx(-0.21, abs=1e-15)


class TestReportedFields:
    def test_base_rate_matches_outcome_mean(self, calibrated_large):
        s = calibrated_large
        d = binned_decomposition(s.forecasts, s.outcomes, n_bins=10)
        assert d.base_rate == pytest.approx(float(s.outcomes.mean()), abs=0)

    def test_brier_field_matches_the_standalone_function(self, calibrated_large):
        s = calibrated_large
        d = binned_decomposition(s.forecasts, s.outcomes, n_bins=10)
        assert d.brier == brier_score(s.forecasts, s.outcomes)

    def test_bin_arrays_have_consistent_lengths(self, calibrated_large):
        s = calibrated_large
        d = binned_decomposition(s.forecasts, s.outcomes, n_bins=12)
        assert d.bin_counts.size == d.n_bins_occupied
        assert d.bin_mean_forecast.size == d.n_bins_occupied
        assert d.bin_observed_frequency.size == d.n_bins_occupied
        assert int(d.bin_counts.sum()) == s.n_samples

    def test_bin_mean_forecasts_are_sorted(self, calibrated_large):
        s = calibrated_large
        d = binned_decomposition(s.forecasts, s.outcomes, n_bins=12)
        assert np.all(np.diff(d.bin_mean_forecast) > 0)

    def test_exact_decomposition_edges_are_the_distinct_values(self):
        f = np.array([0.3, 0.1, 0.3, 0.7])
        d = murphy_decomposition(f, [0, 1, 1, 0])
        assert d.bin_edges.tolist() == [0.1, 0.3, 0.7]
        assert d.n_bins_occupied == 3
        assert d.strategy == "exact"

    def test_report_contains_every_term(self, calibrated_small):
        s = calibrated_small
        text = binned_decomposition(s.forecasts, s.outcomes, n_bins=8).report()
        for token in ("BS", "REL", "RES", "UNC", "WBV", "WBC", "identity residual"):
            assert token in text

    def test_skill_against_climatology_matches_direct_computation(self, overconfident_large):
        s = overconfident_large
        d = murphy_decomposition(np.round(s.forecasts, 2), s.outcomes)
        direct = 1.0 - d.brier / d.uncertainty
        assert d.brier_skill_vs_climatology == pytest.approx(direct, abs=1e-12)

    def test_nonnegativity_of_the_terms(self, overconfident_large):
        s = overconfident_large
        d = binned_decomposition(s.forecasts, s.outcomes, n_bins=20)
        assert d.reliability >= 0.0
        assert d.resolution >= 0.0
        assert 0.0 <= d.uncertainty <= 0.25
        assert d.within_bin_variance >= 0.0
