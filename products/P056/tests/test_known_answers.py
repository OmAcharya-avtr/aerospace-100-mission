"""Known answers against closed-form population values, derived by hand.

The synthetic specs in :mod:`calibaudit.synthetic` have population scores that
reduce to elementary integrals for the calibrated cases. The derivations are
reproduced in the test bodies so a reader can check them without running
anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibaudit.decomposition import binned_decomposition
from calibaudit.ece import expected_calibration_error
from calibaudit.scores import brier_score, log_score
from calibaudit.synthetic import (
    analytic_truth,
    calibration_map,
    distortion,
    get_spec,
    sample_forecast,
)


class TestBetaMoments:
    def test_uncertainty_of_beta_2_2(self):
        # obar = a/(a+b) = 2/4 = 0.5, so UNC = 0.25 exactly.
        t = analytic_truth(get_spec("calibrated"))
        assert t.base_rate == pytest.approx(0.5, abs=0)
        assert t.uncertainty == pytest.approx(0.25, abs=0)

    def test_resolution_of_beta_2_2(self):
        # RES = Var(p) = ab / ((a+b)^2 (a+b+1)) = 4 / (16 * 5) = 0.05 exactly.
        assert analytic_truth(get_spec("calibrated")).resolution == pytest.approx(
            0.05, abs=1e-17
        )

    def test_resolution_of_uniform_latent(self):
        # Beta(1,1) is uniform; Var(p) = 1/12 = 0.0833333...
        assert analytic_truth(get_spec("calibrated_uniform")).resolution == pytest.approx(
            1.0 / 12.0, abs=1e-17
        )

    def test_resolution_of_beta_1_9(self):
        # RES = 1*9 / (100 * 11) = 9/1100 = 0.00818181...
        assert analytic_truth(get_spec("calibrated_rare")).resolution == pytest.approx(
            9.0 / 1100.0, abs=1e-17
        )

    def test_brier_of_calibrated_beta_2_2(self):
        # A calibrated forecaster has REL = 0, so BS = UNC - RES = 0.25 - 0.05 = 0.2.
        assert analytic_truth(get_spec("calibrated")).brier == pytest.approx(0.2, abs=1e-16)

    def test_brier_of_calibrated_uniform(self):
        # BS = 0.25 - 1/12 = 1/6 = 0.1666666...
        assert analytic_truth(get_spec("calibrated_uniform")).brier == pytest.approx(
            1.0 / 6.0, abs=1e-16
        )

    def test_brier_of_calibrated_rare(self):
        # BS = 0.09 - 9/1100 = 0.0818181...
        assert analytic_truth(get_spec("calibrated_rare")).brier == pytest.approx(
            0.09 - 9.0 / 1100.0, abs=1e-16
        )


class TestLogScoreClosedForms:
    def test_log_score_of_calibrated_beta_2_2_is_seven_twelfths(self):
        # LS = E[-(p log p + (1-p) log(1-p))] under Beta(2,2), density 6p(1-p).
        # By the p <-> 1-p symmetry of both density and integrand,
        #   LS = -2 * int_0^1 6 p(1-p) * p log p dp = -12 int_0^1 (p^2 - p^3) log p dp.
        # With int_0^1 p^n log p dp = -1/(n+1)^2:
        #   int (p^2 - p^3) log p dp = -1/9 + 1/16 = -(16 - 9)/144 = -7/144
        #   LS = -12 * (-7/144) = 7/12 = 0.5833333333333333
        t = analytic_truth(get_spec("calibrated"))
        assert t.log_score == pytest.approx(7.0 / 12.0, abs=1e-10)
        assert t.log_score_abserr < 1e-7  # quad's own bound, conservative by ~4 orders

    def test_log_score_of_calibrated_uniform_is_one_half(self):
        # Density 1. LS = -2 int_0^1 p log p dp = -2 * (-1/4) = 1/2.
        t = analytic_truth(get_spec("calibrated_uniform"))
        assert t.log_score == pytest.approx(0.5, abs=1e-10)
        assert t.log_score_abserr < 1e-7  # quad's own bound, conservative by ~4 orders

    def test_log_score_of_calibrated_rare_matches_the_harmonic_form(self):
        # Beta(1,9), density 9(1-p)^8. Using int_0^1 (1-p)^8 p log p dp and
        # int_0^1 (1-p)^9 log(1-p) dp = -1/100, the second term contributes
        # 9 * (-(-1/100)) = 0.09 exactly. The first term has no such short
        # closed form, so the whole value is checked against the quadrature
        # with the quadrature's own error estimate as the tolerance.
        t = analytic_truth(get_spec("calibrated_rare"))
        assert t.log_score == pytest.approx(0.28289682539682537, abs=max(1e-9, t.log_score_abserr))


class TestCalibratedSpecsHaveZeroErrorTerms:
    @pytest.mark.parametrize(
        "name", ["calibrated", "calibrated_uniform", "calibrated_rare"]
    )
    def test_population_reliability_and_ece_are_exactly_zero(self, name):
        t = analytic_truth(get_spec(name))
        assert t.reliability == 0.0
        assert t.ece == 0.0

    @pytest.mark.parametrize(
        "name", ["calibrated", "calibrated_uniform", "calibrated_rare"]
    )
    def test_spec_reports_itself_calibrated(self, name):
        assert get_spec(name).is_calibrated

    @pytest.mark.parametrize("name", ["overconfident", "underconfident", "biased_high"])
    def test_miscalibrated_specs_have_positive_reliability_and_ece(self, name):
        t = analytic_truth(get_spec(name))
        assert t.reliability > 0.0
        assert t.ece > 0.0
        assert not get_spec(name).is_calibrated


class TestDistortionAlgebra:
    @pytest.mark.parametrize("name", ["calibrated", "overconfident", "biased_high"])
    def test_calibration_map_inverts_the_distortion(self, name):
        spec = get_spec(name)
        p = np.linspace(0.01, 0.99, 99)
        assert np.allclose(calibration_map(spec, distortion(spec, p)), p, atol=1e-12)

    def test_temperature_one_is_the_identity(self):
        from calibaudit.synthetic import ForecastSpec

        spec = ForecastSpec("t1", 2.0, 2.0, "temperature", 1.0, "identity by another name")
        p = np.linspace(0.05, 0.95, 19)
        assert np.allclose(distortion(spec, p), p, atol=1e-12)
        assert spec.is_calibrated

    def test_sharpening_moves_forecasts_away_from_one_half(self):
        spec = get_spec("overconfident")
        p = np.array([0.2, 0.3, 0.7, 0.8])
        f = distortion(spec, p)
        assert np.all(np.abs(f - 0.5) > np.abs(p - 0.5))

    def test_shrinking_moves_forecasts_toward_one_half(self):
        spec = get_spec("underconfident")
        p = np.array([0.2, 0.3, 0.7, 0.8])
        f = distortion(spec, p)
        assert np.all(np.abs(f - 0.5) < np.abs(p - 0.5))

    def test_logit_shift_raises_every_forecast(self):
        spec = get_spec("biased_high")
        p = np.linspace(0.05, 0.95, 19)
        assert np.all(distortion(spec, p) > p)


class TestMeasurementConvergesToPopulation:
    # Tolerances are 5 standard errors of the Monte-Carlo estimate at this
    # sample size, not round numbers chosen after seeing the result. For the
    # Brier score of a forecast in [0, 1] the per-sample variance is at most
    # 1/4, so sem <= 0.5 / sqrt(n) = 0.5 / sqrt(200000) = 1.118e-3.
    N = 200_000
    BRIER_TOL = 5 * 0.5 / np.sqrt(200_000)

    @pytest.mark.parametrize(
        "name",
        [
            "calibrated",
            "calibrated_uniform",
            "calibrated_rare",
            "overconfident",
            "underconfident",
            "biased_high",
        ],
    )
    def test_measured_brier_approaches_the_population_value(self, name):
        spec = get_spec(name)
        t = analytic_truth(spec)
        s = sample_forecast(spec, self.N, seed=56100)
        assert brier_score(s.forecasts, s.outcomes) == pytest.approx(
            t.brier, abs=self.BRIER_TOL
        )

    @pytest.mark.parametrize("name", ["calibrated", "overconfident", "biased_high"])
    def test_measured_log_score_approaches_the_population_value(self, name):
        spec = get_spec(name)
        t = analytic_truth(spec)
        s = sample_forecast(spec, self.N, seed=56101)
        assert log_score(s.forecasts, s.outcomes) == pytest.approx(t.log_score, abs=0.02)

    @pytest.mark.parametrize("name", ["calibrated", "overconfident"])
    def test_measured_uncertainty_and_resolution_approach_the_population_values(self, name):
        spec = get_spec(name)
        t = analytic_truth(spec)
        s = sample_forecast(spec, self.N, seed=56102)
        d = binned_decomposition(s.forecasts, s.outcomes, n_bins=50, strategy="equal_mass")
        assert d.uncertainty == pytest.approx(t.uncertainty, abs=2e-3)
        assert d.resolution == pytest.approx(t.resolution, abs=5e-3)

    def test_measured_ece_of_a_calibrated_forecast_is_positive_not_zero(self):
        # The population value is exactly 0; the estimator cannot return it.
        # This is the product's central finding and it is asserted, not hoped.
        spec = get_spec("calibrated")
        s = sample_forecast(spec, 2000, seed=56103)
        ece = expected_calibration_error(s.forecasts, s.outcomes, n_bins=20)
        assert analytic_truth(spec).ece == 0.0
        assert ece > 0.01
