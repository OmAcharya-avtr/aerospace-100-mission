"""Tests for linkswitch.analytic: crossing probability and optimal threshold."""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import norm

from linkswitch.analytic import (
    bivariate_normal_cdf,
    crossing_probability,
    expected_throughput_analytic,
    irradiance_to_z,
    optimal_threshold_analytic,
    optimal_threshold_grid,
    p_rf_available_estimate,
    z_to_irradiance,
)
from linkswitch.optical import OpticalParams
from linkswitch.rf import RFParams


class TestBivariateNormalCdf:
    def test_zero_correlation_factorises(self):
        # rho=0: Phi_2(a,b;0) = Phi(a)*Phi(b) exactly (independence).
        a, b = 0.3, -0.7
        got = bivariate_normal_cdf(a, b, 0.0)
        expected = float(norm.cdf(a) * norm.cdf(b))
        assert got == pytest.approx(expected, abs=1e-6)

    def test_perfect_correlation_reduces_to_min(self):
        # rho=1: Phi_2(a,b;1) = Phi(min(a,b)) exactly.
        a, b = 0.5, 1.2
        got = bivariate_normal_cdf(a, b, 1.0 - 1e-9)
        expected = float(norm.cdf(min(a, b)))
        assert got == pytest.approx(expected, abs=1e-4)

    def test_invalid_rho_rejected(self):
        with pytest.raises(ValueError):
            bivariate_normal_cdf(0.0, 0.0, 1.5)

    def test_symmetric_in_a_b(self):
        assert bivariate_normal_cdf(0.4, -0.2, 0.3) == pytest.approx(
            bivariate_normal_cdf(-0.2, 0.4, 0.3), abs=1e-9
        )


class TestCrossingProbability:
    def test_zero_correlation_known_answer(self):
        # rho=0: independent samples. N(z) = 2*[Phi(z) - Phi(z)^2] = 2 Phi(z)(1-Phi(z)).
        z = 0.5
        got = crossing_probability(z, rho=0.0)
        p = float(norm.cdf(z))
        expected = 2.0 * p * (1.0 - p)
        assert got == pytest.approx(expected, abs=1e-6)

    def test_perfect_correlation_zero_crossings(self):
        # rho -> 1: the process never moves, so crossing probability -> 0.
        got = crossing_probability(0.0, rho=1.0 - 1e-9)
        assert got == pytest.approx(0.0, abs=1e-3)

    def test_symmetric_around_zero_for_symmetric_rho(self):
        # N(z) is symmetric in z (Gaussian symmetry of the standardised process).
        assert crossing_probability(1.3, 0.6) == pytest.approx(
            crossing_probability(-1.3, 0.6), abs=1e-9
        )

    def test_nonnegative(self):
        for z in (-3.0, -1.0, 0.0, 1.0, 3.0):
            for rho in (0.0, 0.3, 0.6, 0.9):
                assert crossing_probability(z, rho) >= 0.0

    def test_peaks_near_zero(self):
        # For a fixed rho, crossing probability should be higher at the
        # distribution's mode (z=0) than deep in either tail.
        rho = 0.7
        assert crossing_probability(0.0, rho) > crossing_probability(3.0, rho)
        assert crossing_probability(0.0, rho) > crossing_probability(-3.0, rho)

    @given(z=st.floats(-4.0, 4.0), rho=st.floats(0.0, 0.98))
    @settings(max_examples=50)
    def test_property_bounded_by_one(self, z, rho):
        assert 0.0 <= crossing_probability(z, rho) <= 1.0


class TestZIrradianceRoundTrip:
    @given(z=st.floats(-5.0, 5.0), mu=st.floats(-1.0, 0.0), sigma=st.floats(0.05, 2.0))
    @settings(max_examples=50)
    def test_round_trip(self, z, mu, sigma):
        tau = z_to_irradiance(z, mu, sigma)
        z_back = irradiance_to_z(tau, mu, sigma)
        assert z_back == pytest.approx(z, abs=1e-8)

    def test_invalid_tau_rejected(self):
        with pytest.raises(ValueError):
            irradiance_to_z(-1.0, 0.0, 1.0)
        with pytest.raises(ValueError):
            irradiance_to_z(0.0, 0.0, 1.0)


class TestPRfAvailableEstimate:
    def test_in_unit_interval(self):
        p = p_rf_available_estimate(RFParams())
        assert 0.0 <= p <= 1.0

    def test_deterministic_given_seed(self):
        rf = RFParams()
        a = p_rf_available_estimate(rf, seed=1)
        b = p_rf_available_estimate(rf, seed=1)
        assert a == b

    def test_more_margin_gives_higher_availability(self):
        low_margin = RFParams(snr_clear_db=10.0, snr_min_db=9.0)
        high_margin = RFParams(snr_clear_db=30.0, snr_min_db=9.0)
        assert p_rf_available_estimate(high_margin) >= p_rf_available_estimate(low_margin)

    def test_invalid_n_mc_rejected(self):
        with pytest.raises(ValueError):
            p_rf_available_estimate(RFParams(), n_mc=0)


class TestOptimalThresholdZeroSwitchCost:
    """Known-answer: in the frictionless (zero switch-cost) limit, the
    optimal fixed threshold equals the physical outage threshold exactly
    (see the module docstring derivation: g_opt saturates at z_phys, and
    increasing z_th past it or decreasing it below it is strictly worse
    whenever R_opt > R_rf * p_rf_avail, which holds for every default
    scenario in this package).
    """

    def test_bounded_optimizer_matches_z_phys(self):
        opt = OpticalParams(sigma_i2=0.3, coherence_steps=4.0, margin_db=6.0, rate_mbps=1000.0)
        rf = RFParams(rate_mbps=150.0)
        result = optimal_threshold_analytic(opt, rf, downtime_steps=0.0)
        assert result.z_th == pytest.approx(result.z_phys, abs=1e-5)

    def test_grid_search_matches_z_phys(self):
        opt = OpticalParams(sigma_i2=0.3, coherence_steps=4.0, margin_db=6.0, rate_mbps=1000.0)
        rf = RFParams(rate_mbps=150.0)
        result = optimal_threshold_grid(opt, rf, downtime_steps=0.0, n_points=8001)
        assert result.z_th == pytest.approx(result.z_phys, abs=5e-3)

    def test_optimizer_and_grid_agree_with_each_other(self):
        opt = OpticalParams(sigma_i2=0.25, coherence_steps=5.0, margin_db=6.0)
        rf = RFParams()
        ana = optimal_threshold_analytic(opt, rf, downtime_steps=0.0)
        grid = optimal_threshold_grid(opt, rf, downtime_steps=0.0, n_points=8001)
        assert ana.z_th == pytest.approx(grid.z_th, abs=5e-3)


class TestOptimalThresholdNonzeroSwitchCost:
    def test_higher_switch_cost_never_increases_objective_at_z_phys_relative_optimum(self):
        # As switch cost grows, the achievable optimum J* is non-increasing
        # (more expensive switching can only hurt, never help, the best
        # achievable throughput for a fixed physical scenario).
        opt = OpticalParams(sigma_i2=0.3, coherence_steps=4.0, margin_db=6.0, rate_mbps=1000.0)
        rf = RFParams(rate_mbps=150.0)
        j_prev = math.inf
        for downtime in (0.0, 0.5, 1.0, 2.0, 5.0):
            result = optimal_threshold_analytic(opt, rf, downtime_steps=downtime)
            assert result.objective <= j_prev + 1e-9
            j_prev = result.objective

    def test_optimizer_and_grid_agree_at_nonzero_cost(self):
        opt = OpticalParams(sigma_i2=0.25, coherence_steps=5.0, margin_db=6.0)
        rf = RFParams()
        ana = optimal_threshold_analytic(opt, rf, downtime_steps=1.0)
        grid = optimal_threshold_grid(opt, rf, downtime_steps=1.0, n_points=8001)
        assert ana.z_th == pytest.approx(grid.z_th, abs=5e-2)
        assert ana.objective == pytest.approx(grid.objective, rel=1e-3)


class TestExpectedThroughputAnalytic:
    def test_known_answer_far_negative_threshold(self):
        # z_th -> -inf: g_opt = 1-Phi(z_phys) (never below), g_rf ~ 0,
        # switch term ~ 0. J -> R_opt * (1 - Phi(z_phys)).
        z_phys = -1.0
        rho = 0.5
        r_opt, r_rf = 800.0, 100.0
        j = expected_throughput_analytic(-10.0, z_phys, rho, r_opt, r_rf, downtime_steps=1.0)
        expected = r_opt * (1.0 - float(norm.cdf(z_phys)))
        assert j == pytest.approx(expected, abs=1e-6)

    def test_known_answer_z_th_equals_z_phys_zero_downtime(self):
        z_phys = -0.5
        rho = 0.6
        r_opt, r_rf = 900.0, 120.0
        j = expected_throughput_analytic(z_phys, z_phys, rho, r_opt, r_rf, downtime_steps=0.0)
        expected = r_opt * (1.0 - norm.cdf(z_phys)) + r_rf * norm.cdf(z_phys)
        assert j == pytest.approx(expected, abs=1e-6)

    def test_higher_downtime_never_increases_j_at_fixed_z(self):
        z_phys = -1.5
        rho = 0.7
        j_low = expected_throughput_analytic(-0.5, z_phys, rho, 700.0, 100.0, downtime_steps=0.5)
        j_high = expected_throughput_analytic(-0.5, z_phys, rho, 700.0, 100.0, downtime_steps=5.0)
        assert j_high <= j_low
