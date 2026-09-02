"""Steering laws: exactness, the SR closed form, GSR dither, rate limiting."""

import numpy as np
import pytest

from cmgsteer.arrays import pyramid_array, roof_array
from cmgsteer.singularity import singular_configuration, singularity_measure
from cmgsteer.steering import (
    METHODS,
    apply_rate_limit,
    gsr_inverse_steer,
    pseudo_inverse_steer,
    robustness_parameter,
    sr_inverse_steer,
    sr_torque_error_closed_form,
    steer,
)

TAU = np.array([0.10, -0.05, 0.20])


class TestPseudoInverse:
    def test_reproduces_the_command_exactly_at_zero(self):
        array = pyramid_array()
        result = pseudo_inverse_steer(array, np.zeros(4), TAU)
        assert result.torque_error_norm < 1e-14
        assert np.allclose(result.achieved_torque, TAU, atol=1e-14)

    def test_reproduces_the_command_over_random_regular_configurations(self):
        array = pyramid_array()
        rng = np.random.default_rng(10)
        worst = 0.0
        for _ in range(300):
            d = rng.uniform(-np.pi, np.pi, 4)
            if singularity_measure(array.jacobian(d)) < 0.05:
                continue
            tau = rng.normal(size=3) * 0.3
            worst = max(worst, pseudo_inverse_steer(array, d, tau).torque_error_norm)
        assert worst < 1e-12

    def test_is_the_minimum_norm_solution(self):
        array = pyramid_array()
        d = np.array([0.3, -0.5, 0.8, 0.2])
        result = pseudo_inverse_steer(array, d, TAU)
        a = array.jacobian(d)
        # any other exact solution differs by a null vector and is longer
        null = np.linalg.svd(a)[2][3]
        for scale in (-0.5, 0.2, 1.0):
            other = result.gimbal_rates + scale * null
            assert np.linalg.norm(other) >= np.linalg.norm(result.gimbal_rates) - 1e-12

    def test_rates_blow_up_near_a_singularity(self):
        array = pyramid_array()
        base = singular_configuration(array, np.array([0.3, 0.2, 0.9]), np.array([1.0, 1, -1, 1]))
        peaks = []
        for offset in (1e-1, 1e-2, 1e-3):
            d = base + offset
            peaks.append(np.max(np.abs(pseudo_inverse_steer(array, d, TAU).gimbal_rates)))
        assert peaks[2] > peaks[1] > peaks[0]

    def test_null_rates_do_not_change_the_torque(self):
        array = pyramid_array()
        d = np.array([0.3, -0.5, 0.8, 0.2])
        a = array.jacobian(d)
        null = np.linalg.svd(a)[2][3]
        plain = pseudo_inverse_steer(array, d, TAU)
        with_null = pseudo_inverse_steer(array, d, TAU, null_rates=0.7 * null)
        assert np.allclose(plain.achieved_torque, with_null.achieved_torque, atol=1e-13)
        assert with_null.method == "pinv+null"

    def test_wrong_null_length_raises(self):
        with pytest.raises(ValueError, match="null_rates must have length 4"):
            pseudo_inverse_steer(pyramid_array(), np.zeros(4), TAU, null_rates=np.zeros(2))


class TestSRInverse:
    def test_matches_the_closed_form_error(self):
        array = pyramid_array()
        rng = np.random.default_rng(11)
        worst = 0.0
        for _ in range(200):
            d = rng.uniform(-np.pi, np.pi, 4)
            tau = rng.normal(size=3) * 0.4
            lam = float(10.0 ** rng.uniform(-8, 0))
            result = sr_inverse_steer(array, d, tau, lam=lam)
            predicted = sr_torque_error_closed_form(array.jacobian(d), tau, lam)
            worst = max(worst, float(np.max(np.abs(result.torque_error - predicted))))
        assert worst < 1e-13

    def test_error_grows_monotonically_with_lam(self):
        array = pyramid_array()
        d = np.array([0.4, -0.2, 0.7, 0.1])
        errors = [
            sr_inverse_steer(array, d, TAU, lam=lam).torque_error_norm
            for lam in (1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0)
        ]
        assert all(b > a for a, b in zip(errors, errors[1:]))

    def test_lam_zero_reduces_to_the_pseudo_inverse(self):
        array = pyramid_array()
        d = np.array([0.4, -0.2, 0.7, 0.1])
        sr = sr_inverse_steer(array, d, TAU, lam=0.0)
        pinv = pseudo_inverse_steer(array, d, TAU)
        assert np.allclose(sr.gimbal_rates, pinv.gimbal_rates, atol=1e-10)

    def test_rates_stay_bounded_at_an_exact_singularity(self):
        array = pyramid_array()
        d = np.full(4, np.pi / 2)
        sr = sr_inverse_steer(array, d, TAU, lam=1e-3)
        assert np.all(np.isfinite(sr.gimbal_rates))
        assert np.max(np.abs(sr.gimbal_rates)) < 1e4

    def test_adaptive_lam_is_small_away_from_singularity(self):
        array = pyramid_array()
        far = sr_inverse_steer(array, np.zeros(4), TAU)
        near = sr_inverse_steer(array, np.full(4, np.pi / 2 - 0.05), TAU)
        assert near.lam > far.lam
        assert far.torque_error_norm < near.torque_error_norm

    def test_negative_lam_raises(self):
        with pytest.raises(ValueError, match="lam must be non-negative"):
            sr_inverse_steer(pyramid_array(), np.zeros(4), TAU, lam=-1.0)


class TestRobustnessParameter:
    def test_equals_lam0_h0_squared_at_zero_measure(self):
        array = pyramid_array(rotor_momentum=2.0)
        assert robustness_parameter(array, 0.0, lam0=0.01) == pytest.approx(0.04)

    def test_decays_exponentially(self):
        array = pyramid_array()
        lam1 = robustness_parameter(array, 0.1, lam0=0.01, mu=10.0)
        lam2 = robustness_parameter(array, 0.2, lam0=0.01, mu=10.0)
        assert lam2 / lam1 == pytest.approx(np.exp(-1.0), rel=1e-12)

    def test_mu_zero_is_constant(self):
        array = pyramid_array()
        assert robustness_parameter(array, 5.0, lam0=0.02, mu=0.0) == pytest.approx(0.02)

    def test_invalid_arguments_raise(self):
        array = pyramid_array()
        with pytest.raises(ValueError, match="measure must be non-negative"):
            robustness_parameter(array, -1.0)
        with pytest.raises(ValueError, match="lam0 must be non-negative"):
            robustness_parameter(array, 1.0, lam0=-1.0)
        with pytest.raises(ValueError, match="mu must be non-negative"):
            robustness_parameter(array, 1.0, mu=-1.0)


class TestGSR:
    def test_reduces_to_sr_when_the_dither_is_off(self):
        array = pyramid_array()
        d = np.array([0.4, -0.2, 0.7, 0.1])
        gsr = gsr_inverse_steer(array, d, TAU, eps0=0.0, lam=1e-3)
        sr = sr_inverse_steer(array, d, TAU, lam=1e-3)
        assert np.allclose(gsr.gimbal_rates, sr.gimbal_rates, atol=1e-14)

    def test_dither_changes_the_solution_over_time(self):
        array = pyramid_array()
        d = np.full(4, np.pi / 2 - 0.02)
        a = gsr_inverse_steer(array, d, TAU, time=0.0, lam=1e-2, eps0=0.05)
        b = gsr_inverse_steer(array, d, TAU, time=1.7, lam=1e-2, eps0=0.05)
        assert not np.allclose(a.gimbal_rates, b.gimbal_rates, atol=1e-9)

    def test_extras_record_the_dither(self):
        result = gsr_inverse_steer(pyramid_array(), np.zeros(4), TAU, time=0.5, eps0=0.02)
        assert set(result.extras) == {"e1", "e2", "e3"}
        assert abs(result.extras["e1"]) <= 0.02 + 1e-15

    def test_invalid_dither_raises(self):
        with pytest.raises(ValueError, match="eps0 must be non-negative"):
            gsr_inverse_steer(pyramid_array(), np.zeros(4), TAU, eps0=-0.1)
        with pytest.raises(ValueError, match="phases must have length 3"):
            gsr_inverse_steer(pyramid_array(), np.zeros(4), TAU, phases=(0.0, 1.0))


class TestRateLimit:
    def test_no_limit_returns_the_input(self):
        rates, n = apply_rate_limit([1.0, -2.0], None)
        assert n == 0
        assert np.allclose(rates, [1.0, -2.0])

    def test_clip_limits_each_component(self):
        rates, n = apply_rate_limit([1.0, -2.0, 0.5], 1.5, mode="clip")
        assert n == 1
        assert np.allclose(rates, [1.0, -1.5, 0.5])

    def test_scale_preserves_direction(self):
        rates, n = apply_rate_limit([1.0, -2.0, 0.5], 1.0, mode="scale")
        assert n == 1
        assert np.allclose(rates, np.array([1.0, -2.0, 0.5]) * 0.5)

    def test_scale_preserves_torque_direction_in_a_steering_call(self):
        array = pyramid_array()
        d = np.array([0.1, 0.4, -0.2, 0.3])
        free = sr_inverse_steer(array, d, TAU * 20.0, lam=1e-8)
        scaled = sr_inverse_steer(
            array, d, TAU * 20.0, lam=1e-8, max_gimbal_rate=0.5, saturation_mode="scale"
        )
        cos = float(
            free.achieved_torque
            @ scaled.achieved_torque
            / (np.linalg.norm(free.achieved_torque) * np.linalg.norm(scaled.achieved_torque))
        )
        assert cos == pytest.approx(1.0, abs=1e-9)

    def test_clip_changes_the_torque_direction(self):
        array = pyramid_array()
        d = np.array([0.1, 0.4, -0.2, 0.3])
        free = sr_inverse_steer(array, d, TAU * 20.0, lam=1e-8)
        clipped = sr_inverse_steer(array, d, TAU * 20.0, lam=1e-8, max_gimbal_rate=0.5)
        cos = float(
            free.achieved_torque
            @ clipped.achieved_torque
            / (np.linalg.norm(free.achieved_torque) * np.linalg.norm(clipped.achieved_torque))
        )
        assert cos < 1.0 - 1e-6
        assert clipped.rate_limited
        assert clipped.n_rate_limited >= 1

    def test_invalid_limit_raises(self):
        with pytest.raises(ValueError, match="max_rate must be positive"):
            apply_rate_limit([1.0], -1.0)
        with pytest.raises(ValueError, match="mode must be"):
            apply_rate_limit([5.0], 1.0, mode="squash")


class TestDispatch:
    @pytest.mark.parametrize("method", METHODS)
    def test_every_method_runs(self, method):
        result = steer(pyramid_array(), np.zeros(4), TAU, method=method)
        assert result.method.startswith(method)
        assert result.commanded_torque.shape == (3,)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown steering method"):
            steer(pyramid_array(), np.zeros(4), TAU, method="magic")

    def test_bad_torque_shape_raises(self):
        with pytest.raises(ValueError, match=r"torque must have shape \(3,\)"):
            steer(pyramid_array(), np.zeros(4), [1.0, 2.0])

    def test_non_finite_torque_raises(self):
        with pytest.raises(ValueError, match="finite"):
            steer(pyramid_array(), np.zeros(4), [1.0, np.inf, 0.0])

    def test_closed_form_rejects_bad_input(self):
        with pytest.raises(ValueError, match=r"shape \(3, n\)"):
            sr_torque_error_closed_form(np.zeros((2, 4)), TAU, 1e-3)
        with pytest.raises(ValueError, match="lam must be non-negative"):
            sr_torque_error_closed_form(np.zeros((3, 4)), TAU, -1.0)


class TestRoofArraySteering:
    def test_steering_works_off_the_degenerate_home_configuration(self):
        array = roof_array()
        d = np.array([0.4, -0.4, 0.4, -0.4])
        result = pseudo_inverse_steer(array, d, TAU)
        assert result.torque_error_norm < 1e-12

    def test_home_configuration_is_singular_and_sr_still_returns_finite_rates(self):
        array = roof_array()
        result = sr_inverse_steer(array, np.zeros(4), TAU, lam=1e-3)
        assert result.measure < 1e-14
        assert np.all(np.isfinite(result.gimbal_rates))
        assert result.torque_error_norm > 1e-6
