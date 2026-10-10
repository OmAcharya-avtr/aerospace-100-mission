"""Reliability curves and the behaviour of their bootstrap bands."""

from __future__ import annotations

import numpy as np
import pytest

from calibaudit.reliability import (
    band_coverage,
    bootstrap_reliability,
    reliability_curve,
)
from calibaudit.synthetic import get_spec, sample_forecast


class TestCurveKnownAnswers:
    def test_hand_computed_two_bin_curve(self):
        # Equal-width 2 bins on f = (0.1, 0.3, 0.7, 0.9), o = (0, 1, 1, 1).
        # bin 0: n = 2, mean f = 0.2, observed 0.5
        # bin 1: n = 2, mean f = 0.8, observed 1.0
        c = reliability_curve([0.1, 0.3, 0.7, 0.9], [0, 1, 1, 1], n_bins=2)
        assert c.counts.tolist() == [2, 2]
        assert c.mean_forecast.tolist() == pytest.approx([0.2, 0.8], abs=1e-15)
        assert c.observed_frequency.tolist() == pytest.approx([0.5, 1.0], abs=1e-15)

    def test_empty_bins_are_dropped(self):
        c = reliability_curve([0.05, 0.95], [0, 1], n_bins=10)
        assert c.n_occupied == 2
        assert c.bin_index.tolist() == [0, 9]

    def test_edges_have_n_bins_plus_one_entries(self):
        c = reliability_curve([0.3, 0.6], [0, 1], n_bins=4)
        assert c.edges.size == 5

    def test_forecast_of_exactly_one_lands_in_the_top_bin(self):
        c = reliability_curve([1.0], [1], n_bins=5)
        assert c.bin_index.tolist() == [4]

    def test_forecast_of_exactly_zero_lands_in_the_bottom_bin(self):
        c = reliability_curve([0.0], [0], n_bins=5)
        assert c.bin_index.tolist() == [0]

    def test_counts_sum_to_the_sample_size(self, overconfident_large):
        s = overconfident_large
        c = reliability_curve(s.forecasts, s.outcomes, n_bins=17)
        assert int(c.counts.sum()) == s.n_samples

    def test_perfect_forecast_lies_on_the_diagonal(self):
        # f = o exactly: bin 0 has f = o = 0, bin 9 has f = o = 1.
        o = np.array([0.0] * 10 + [1.0] * 10)
        c = reliability_curve(o, o, n_bins=10)
        assert np.allclose(c.mean_forecast, c.observed_frequency, atol=0)


class TestBootstrapBands:
    def test_band_brackets_the_point_estimate(self, overconfident_large):
        s = overconfident_large
        c = bootstrap_reliability(
            s.forecasts, s.outcomes, n_bins=10, n_bootstrap=200, level=0.9, seed=1
        )
        assert np.all(c.lower <= c.observed_frequency + 1e-12)
        assert np.all(c.upper >= c.observed_frequency - 1e-12)

    def test_wider_level_gives_wider_band(self, overconfident_large):
        s = overconfident_large
        narrow = bootstrap_reliability(
            s.forecasts, s.outcomes, n_bins=10, n_bootstrap=300, level=0.5, seed=2
        )
        wide = bootstrap_reliability(
            s.forecasts, s.outcomes, n_bins=10, n_bootstrap=300, level=0.99, seed=2
        )
        assert np.all((wide.upper - wide.lower) >= (narrow.upper - narrow.lower) - 1e-12)

    def test_band_narrows_with_sample_size(self):
        spec = get_spec("overconfident")
        small = sample_forecast(spec, 400, seed=56400)
        large = sample_forecast(spec, 6400, seed=56401)
        cs = bootstrap_reliability(
            small.forecasts, small.outcomes, n_bins=8, n_bootstrap=200, seed=3
        )
        cl = bootstrap_reliability(
            large.forecasts, large.outcomes, n_bins=8, n_bootstrap=200, seed=3
        )
        assert float(np.mean(cl.upper - cl.lower)) < float(np.mean(cs.upper - cs.lower))

    def test_bands_are_deterministic_in_the_seed(self, overconfident_small):
        s = overconfident_small
        a = bootstrap_reliability(s.forecasts, s.outcomes, n_bootstrap=50, seed=4)
        b = bootstrap_reliability(s.forecasts, s.outcomes, n_bootstrap=50, seed=4)
        assert np.array_equal(a.lower, b.lower)
        assert np.array_equal(a.upper, b.upper)

    def test_equal_mass_edges_are_fixed_across_replicates(self, overconfident_large):
        # If the edges moved per replicate the band would be meaningless; the
        # observable consequence is that the returned edges match a direct call.
        from calibaudit.binning import bin_edges

        s = overconfident_large
        c = bootstrap_reliability(
            s.forecasts,
            s.outcomes,
            n_bins=10,
            strategy="equal_mass",
            n_bootstrap=50,
            seed=5,
        )
        assert np.array_equal(
            c.edges, bin_edges(s.forecasts, n_bins=10, strategy="equal_mass")
        )

    def test_diagonal_exclusion_flags_the_miscalibrated_bins(self, overconfident_large):
        s = overconfident_large
        c = bootstrap_reliability(
            s.forecasts, s.outcomes, n_bins=10, n_bootstrap=300, level=0.9, seed=6
        )
        # The overconfident spec is miscalibrated everywhere except near 0.5,
        # so a large majority of bins must be flagged at n = 8000.
        assert int(np.sum(c.diagonal_excluded())) >= 7

    def test_calibrated_forecast_flags_few_bins(self, calibrated_large):
        s = calibrated_large
        c = bootstrap_reliability(
            s.forecasts, s.outcomes, n_bins=10, n_bootstrap=300, level=0.9, seed=7
        )
        assert int(np.sum(c.diagonal_excluded())) <= 3

    def test_table_has_band_columns_only_when_a_band_exists(self, overconfident_small):
        s = overconfident_small
        plain = reliability_curve(s.forecasts, s.outcomes, n_bins=5).table()
        banded = bootstrap_reliability(
            s.forecasts, s.outcomes, n_bins=5, n_bootstrap=20, seed=8
        ).table()
        assert "band_lo" not in plain
        assert "band_lo" in banded

    def test_recorded_metadata(self, overconfident_small):
        s = overconfident_small
        c = bootstrap_reliability(
            s.forecasts, s.outcomes, n_bins=6, n_bootstrap=23, level=0.8, seed=9
        )
        assert c.n_bins == 6
        assert c.n_bootstrap == 23
        assert c.level == 0.8
        assert c.n_samples == s.n_samples


class TestBandCoverage:
    @pytest.fixture(scope="class")
    @staticmethod
    def coverage():
        return band_coverage(
            get_spec("calibrated"),
            n_samples=1000,
            n_bins=10,
            level=0.9,
            n_bootstrap=100,
            n_replicates=40,
            seed=56,
        )

    def test_pointwise_coverage_is_in_the_right_region(self, coverage):
        # Nominal 0.9. The bootstrap percentile band on a bin mean of ~100
        # Bernoulli draws undercovers; a measurement far outside this window
        # would mean the band is not doing what it says.
        assert 0.78 <= coverage.pointwise_coverage <= 0.95

    def test_simultaneous_coverage_is_much_worse_than_pointwise(self, coverage):
        # Ten pointwise bands at 0.9 cannot give 0.9 simultaneously. This is
        # the limitation the README states and it is asserted here.
        assert coverage.simultaneous_coverage < coverage.pointwise_coverage - 0.2

    def test_counts_are_consistent(self, coverage):
        assert coverage.pointwise_n == int(coverage.per_bin_n.sum())
        assert coverage.n_replicates == 40

    def test_report_names_both_rates(self, coverage):
        text = coverage.report()
        assert "pointwise" in text
        assert "simultaneous" in text

    def test_per_bin_coverage_is_a_probability_where_defined(self, coverage):
        defined = coverage.per_bin_coverage[np.isfinite(coverage.per_bin_coverage)]
        assert defined.size > 0
        assert np.all((defined >= 0.0) & (defined <= 1.0))
