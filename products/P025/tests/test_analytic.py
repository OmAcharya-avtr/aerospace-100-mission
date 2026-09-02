"""Threshold design and detection-delay expressions.

The chi-squared distribution with two degrees of freedom is exponential, which
gives exact closed forms for the hand-checked known-answer tests:
``P(chi2_2 > h) = exp(-h/2)``, so ``h = -2 ln(alpha)``.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fdiscope.analytic import (
    SIEGMUND_RHO,
    chi2_detection_power,
    chi2_false_alarm_rate,
    chi2_threshold,
    cusum_arl0_siegmund,
    cusum_delay_mean_path,
    cusum_delay_siegmund,
    cusum_delay_wald,
    cusum_kl_information,
    cusum_threshold_for_arl0,
    innovation_dc_gain,
    normalised_bias_signature,
    steady_state_gain,
    steady_state_innovation_mean,
)
from fdiscope.plant import PlantConfig, loop_matrices
from fdiscope.simulate import build_filter


@pytest.fixture(scope="module")
def kf():
    return build_filter(loop_matrices(PlantConfig()))


class TestChiSquared:
    def test_known_answer_two_dof(self):
        # chi2 with 2 dof is Exp(1/2): P(X > h) = exp(-h/2), so
        #   alpha = 0.05 -> h = -2 ln 0.05 = 2 * 2.99573227355 = 5.9914645471
        #   alpha = 0.01 -> h = -2 ln 0.01 = 9.2103403720
        assert np.isclose(chi2_threshold(0.05, 2), 5.9914645471079817, rtol=1e-12)
        assert np.isclose(chi2_threshold(0.01, 2), 9.2103403719761836, rtol=1e-12)

    def test_known_answer_survival(self):
        # P(chi2_2 > 9.2103403720) = exp(-4.605170186) = 0.01
        assert np.isclose(chi2_false_alarm_rate(9.2103403719761836, 2), 0.01, rtol=1e-12)

    @settings(max_examples=40, deadline=None)
    @given(
        alpha=st.floats(1e-6, 0.5, allow_nan=False),
        dof=st.integers(1, 400),
    )
    def test_threshold_and_rate_are_inverses(self, alpha, dof):
        assert np.isclose(chi2_false_alarm_rate(chi2_threshold(alpha, dof), dof), alpha, rtol=1e-9)

    @settings(max_examples=30, deadline=None)
    @given(dof=st.integers(1, 200))
    def test_threshold_decreases_with_alpha(self, dof):
        assert chi2_threshold(0.1, dof) < chi2_threshold(0.01, dof)

    def test_detection_power_at_zero_noncentrality_is_alpha(self):
        h = chi2_threshold(0.01, 6)
        assert np.isclose(chi2_detection_power(h, 6, 0.0), 0.01, rtol=1e-9)

    def test_detection_power_increases_with_noncentrality(self):
        h = chi2_threshold(0.001, 4)
        powers = [chi2_detection_power(h, 4, lam) for lam in (0.0, 5.0, 20.0, 100.0)]
        assert powers == sorted(powers)
        assert powers[-1] > 0.99

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, float("nan")])
    def test_rejects_bad_alpha(self, bad):
        with pytest.raises(ValueError, match="alpha"):
            chi2_threshold(bad, 2)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_bad_dof(self, bad):
        with pytest.raises(ValueError, match="dof"):
            chi2_threshold(0.05, bad)

    def test_rejects_negative_noncentrality(self):
        with pytest.raises(ValueError, match="noncentrality"):
            chi2_detection_power(5.0, 2, -1.0)

    def test_rejects_non_positive_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            chi2_false_alarm_rate(0.0, 2)


class TestCusumExpressions:
    def test_kl_information_known_answer(self):
        # K = mu^2 / 2: mu = 2 -> 2, mu = 0.5 -> 0.125
        assert np.isclose(cusum_kl_information(2.0), 2.0)
        assert np.isclose(cusum_kl_information(0.5), 0.125)

    def test_wald_delay_known_answer(self):
        # h / K = 8 / (1^2/2) = 16 samples
        assert np.isclose(cusum_delay_wald(8.0, 1.0), 16.0)

    def test_siegmund_delay_known_answer(self):
        # mu = 1, h = 8: b = 8 + 1.1652 = 9.1652
        #   delay = 2 * (exp(-9.1652) + 9.1652 - 1) / 1
        #         = 2 * (1.04434e-4 + 8.1652) = 16.33061
        assert np.isclose(cusum_delay_siegmund(8.0, 1.0), 16.330608, rtol=1e-5)

    def test_siegmund_crosses_wald_at_mu_equals_one_over_rho(self):
        # Siegmund - Wald = 2 (e^-b + 1.1652 mu - 1) / mu^2, whose sign flips
        # at mu = 1 / 1.1652 = 0.8582 (up to the exponentially small e^-b).
        for h in (2.0, 5.0, 10.0):
            assert cusum_delay_siegmund(h, 0.4) < cusum_delay_wald(h, 0.4)
            assert cusum_delay_siegmund(h, 2.0) > cusum_delay_wald(h, 2.0)
        crossing = 1.0 / SIEGMUND_RHO
        assert np.isclose(
            cusum_delay_siegmund(8.0, crossing), cusum_delay_wald(8.0, crossing), rtol=1e-3
        )

    def test_arl0_known_answer(self):
        # mu = 1, h = 5: b = 5 + 1.1652 = 6.1652
        #   e^6.1652 = 475.89611...
        #   ARL0 = 2 * (475.89611 - 6.1652 - 1) / 1^2
        #        = 2 * 468.73091 = 937.46223
        assert np.isclose(cusum_arl0_siegmund(5.0, 1.0), 937.46223, rtol=1e-6)

    def test_arl0_grows_with_threshold(self):
        values = [cusum_arl0_siegmund(h, 1.0) for h in (2.0, 4.0, 6.0, 8.0)]
        assert values == sorted(values)

    @settings(max_examples=40, deadline=None)
    @given(
        arl0=st.floats(50.0, 1e6, allow_nan=False),
        mu=st.floats(0.2, 4.0, allow_nan=False),
    )
    def test_threshold_for_arl0_is_the_inverse(self, arl0, mu):
        h = cusum_threshold_for_arl0(arl0, mu)
        assert np.isclose(cusum_arl0_siegmund(h, mu), arl0, rtol=1e-8)

    def test_threshold_for_arl0_rejects_impossible_targets(self):
        with pytest.raises(ValueError, match="arl0 must be > 1"):
            cusum_threshold_for_arl0(0.5, 1.0)
        with pytest.raises(ValueError, match="below the smallest achievable"):
            cusum_threshold_for_arl0(1.0000001, 1.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_rejects_bad_mu(self, bad):
        with pytest.raises(ValueError, match="mu"):
            cusum_delay_wald(1.0, bad)

    def test_siegmund_constant_is_twice_the_single_boundary_value(self):
        assert np.isclose(SIEGMUND_RHO, 2.0 * 0.5826, atol=1e-4)


class TestMeanPathDelay:
    def test_known_answer_constant_profile(self):
        # profile p = 3 constant, mu = 3: increment = 3*3 - 4.5 = 4.5 per sample.
        # h = 10 -> g = 4.5, 9.0, 13.5; first crossing at index 2, so the run
        # length (samples inspected) is 3.
        assert cusum_delay_mean_path([3.0] * 10, 3.0, 10.0) == 3.0

    def test_returns_inf_when_the_path_never_crosses(self):
        # profile p = 0: increment = -mu^2/2 < 0, statistic pinned at 0.
        assert cusum_delay_mean_path([0.0] * 50, 1.0, 5.0) == float("inf")

    def test_ramp_profile_crosses_later_than_a_step(self):
        step = cusum_delay_mean_path([2.0] * 200, 2.0, 8.0)
        ramp = cusum_delay_mean_path(np.linspace(0.0, 2.0, 200), 2.0, 8.0)
        assert ramp > step

    def test_rejects_empty_profile(self):
        with pytest.raises(ValueError, match="non-empty"):
            cusum_delay_mean_path([], 1.0, 1.0)

    def test_rejects_non_finite_profile(self):
        with pytest.raises(ValueError, match="finite"):
            cusum_delay_mean_path([1.0, np.nan], 1.0, 1.0)


class TestSteadyStateResponse:
    def test_dc_gain_first_column_is_zero(self, kf):
        # F has a unit eigenvalue in the angle direction and both states are
        # measured, so a constant ANGLE bias leaves no steady-state innovation.
        m = innovation_dc_gain(kf)
        assert np.allclose(m[:, 0], 0.0, atol=1e-12)
        assert not np.allclose(m[:, 1], 0.0)

    def test_dc_gain_reproduces_steady_state_innovation_mean(self, kf):
        bias = np.array([1.3, -0.7])
        assert np.allclose(innovation_dc_gain(kf) @ bias, steady_state_innovation_mean(kf, bias))

    def test_steady_state_gain_matches_its_definition(self, kf):
        from fdiscope.kalman import steady_state_covariance

        p, s = steady_state_covariance(kf)
        assert np.allclose(steady_state_gain(kf), p @ kf.h.T @ np.linalg.inv(s))

    def test_rate_bias_signature_scales_linearly(self, kf):
        sigma = float(np.sqrt(PlantConfig().gyro_var_rad2_s2))
        d1, mu1 = normalised_bias_signature(kf, [0.0, 1.0 * sigma])
        d4, mu4 = normalised_bias_signature(kf, [0.0, 4.0 * sigma])
        assert np.allclose(d1, d4)
        assert np.isclose(mu4 / mu1, 4.0, rtol=1e-9)

    def test_rate_bias_of_n_sigma_gives_mu_of_n(self, kf):
        # A useful coincidence of this configuration, worth pinning: the
        # steady-state normalised residual magnitude equals the bias in gyro
        # sigmas to better than 1e-4.
        sigma = float(np.sqrt(PlantConfig().gyro_var_rad2_s2))
        for n in (1.0, 2.0, 4.0):
            _, mu = normalised_bias_signature(kf, [0.0, n * sigma])
            assert np.isclose(mu, n, rtol=1e-4)

    def test_angle_bias_has_no_signature(self, kf):
        sigma = float(np.sqrt(PlantConfig().attitude_var_rad2))
        with pytest.raises(ValueError, match="steady-state residual mean is zero"):
            normalised_bias_signature(kf, [4.0 * sigma, 0.0])

    def test_rejects_wrong_bias_length(self, kf):
        with pytest.raises(ValueError, match="bias must have 2 elements"):
            steady_state_innovation_mean(kf, [1.0, 2.0, 3.0])

    def test_signature_direction_is_a_unit_vector(self, kf):
        d, _ = normalised_bias_signature(kf, [0.0, 1e-3])
        assert np.isclose(np.linalg.norm(d), 1.0)
