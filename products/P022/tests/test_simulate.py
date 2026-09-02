"""Torque profiles and steering-run accounting."""

import numpy as np
import pytest

from cmgsteer.arrays import pyramid_array
from cmgsteer.nullmotion import GradientNullMotion, NoNullMotion
from cmgsteer.simulate import (
    TorqueProfile,
    constant_profile,
    rest_to_rest_profile,
    run_steering,
)


class TestTorqueProfile:
    def test_constant_profile_shape_and_duration(self):
        profile = constant_profile([0.1, 0.0, 0.0], 2.0, 0.05)
        assert profile.n_steps == 40
        assert profile.duration == pytest.approx(2.0)
        assert np.allclose(profile.torques[0], [0.1, 0.0, 0.0])

    def test_rest_to_rest_integrates_to_zero(self):
        profile = rest_to_rest_profile([0.0, 0.0, 1.0], 2.0, 10.0, 0.01)
        assert np.max(np.abs(profile.momentum_change)) < 1e-12

    def test_rest_to_rest_peak_momentum_matches_the_request(self):
        for shape in ("sine", "bang-bang"):
            profile = rest_to_rest_profile([0.0, 1.0, 0.0], 1.7, 8.0, 0.001, shape=shape)
            assert profile.peak_momentum == pytest.approx(1.7, rel=2e-3)

    def test_bang_bang_amplitude_is_the_closed_form(self):
        # tau_max = 2 dh / T
        profile = rest_to_rest_profile([1.0, 0.0, 0.0], 2.0, 8.0, 0.01, shape="bang-bang")
        assert np.max(np.abs(profile.torques)) == pytest.approx(0.5, rel=1e-12)

    def test_sine_amplitude_is_the_closed_form(self):
        # tau_max = pi dh / T
        profile = rest_to_rest_profile([1.0, 0.0, 0.0], 2.0, 8.0, 0.0001, shape="sine")
        assert np.max(np.abs(profile.torques)) == pytest.approx(np.pi * 2.0 / 8.0, rel=1e-6)

    def test_momentum_is_stored_along_the_requested_axis(self):
        axis = np.array([0.0, 0.0, 1.0])
        profile = rest_to_rest_profile(axis, 2.0, 8.0, 0.001)
        running = -np.cumsum(profile.torques, axis=0) * profile.dt
        peak = running[int(np.argmax(np.linalg.norm(running, axis=1)))]
        assert peak @ axis > 0.0

    def test_invalid_profiles_raise(self):
        with pytest.raises(ValueError, match=r"shape \(n_steps, 3\)"):
            TorqueProfile(np.zeros((5, 2)), 0.1)
        with pytest.raises(ValueError, match="dt must be positive"):
            TorqueProfile(np.zeros((5, 3)), 0.0)
        with pytest.raises(ValueError, match="finite"):
            TorqueProfile(np.full((2, 3), np.nan), 0.1)
        with pytest.raises(ValueError, match="shape must be one of"):
            rest_to_rest_profile([1.0, 0, 0], 1.0, 1.0, 0.1, shape="ramp")
        with pytest.raises(ValueError, match="momentum_change must be positive"):
            rest_to_rest_profile([1.0, 0, 0], 0.0, 1.0, 0.1)
        with pytest.raises(ValueError, match="axis must be a non-zero"):
            rest_to_rest_profile([0.0, 0, 0], 1.0, 1.0, 0.1)
        with pytest.raises(ValueError, match=r"axis must have shape \(3,\)"):
            rest_to_rest_profile([0.0, 0], 1.0, 1.0, 0.1)
        with pytest.raises(ValueError, match=r"torque must have shape \(3,\)"):
            constant_profile([0.0, 0], 1.0, 0.1)
        with pytest.raises(ValueError, match="must be positive"):
            constant_profile([0.0, 0, 0], -1.0, 0.1)
        with pytest.raises(ValueError, match="must be positive"):
            rest_to_rest_profile([1.0, 0, 0], 1.0, -1.0, 0.1)


class TestRunSteering:
    def _easy(self):
        return rest_to_rest_profile([0.2, 0.3, 0.93], 1.2, 10.0, 0.02)

    def test_shapes_are_consistent(self):
        array = pyramid_array()
        profile = self._easy()
        history = run_steering(array, np.zeros(4), profile)
        n = profile.n_steps
        assert history.deltas.shape == (n + 1, 4)
        assert history.momentum.shape == (n + 1, 3)
        assert history.measure.shape == (n + 1,)
        assert history.torque_error.shape == (n, 3)
        assert history.gimbal_rates.shape == (n, 4)
        assert history.times.shape == (n + 1,)

    def test_pseudo_inverse_torque_error_is_numerically_zero(self):
        history = run_steering(pyramid_array(), np.zeros(4), self._easy(), method="pinv")
        assert history.max_torque_error < 1e-13
        assert history.rms_torque_error < 1e-13

    def test_momentum_error_is_first_order_in_dt(self):
        array = pyramid_array()
        errors = []
        for dt in (0.08, 0.04, 0.02, 0.01):
            profile = rest_to_rest_profile([0.2, 0.3, 0.93], 1.2, 10.0, dt)
            errors.append(
                run_steering(array, np.zeros(4), profile, method="pinv")
                .accumulated_momentum_error
            )
        ratios = [a / b for a, b in zip(errors, errors[1:])]
        assert all(1.8 < r < 2.2 for r in ratios)

    def test_momentum_returns_to_the_start_for_a_rest_to_rest_profile(self):
        array = pyramid_array()
        history = run_steering(array, np.zeros(4), self._easy(), method="pinv")
        assert np.linalg.norm(history.momentum[-1] - history.momentum[0]) < 5e-3

    def test_history_summary_properties(self):
        history = run_steering(pyramid_array(), np.zeros(4), self._easy(), method="sr")
        assert history.min_measure > 0.0
        assert history.peak_gimbal_rate > 0.0
        assert history.n_rate_limited == 0
        assert history.total_momentum_error_path >= history.accumulated_momentum_error
        assert history.steps_below_measure(1e9) == history.measure.size
        assert history.steps_below_measure(0.0) == 0

    def test_null_policy_is_recorded_and_changes_the_path(self):
        array = pyramid_array()
        profile = self._easy()
        plain = run_steering(array, np.zeros(4), profile, null_policy=NoNullMotion())
        with_null = run_steering(
            array, np.zeros(4), profile, null_policy=GradientNullMotion(gain=1.0, max_rate=0.2)
        )
        assert plain.policy == "none"
        assert with_null.policy == "gradient"
        assert not np.allclose(plain.deltas[-1], with_null.deltas[-1])

    def test_null_motion_does_not_change_the_instantaneous_torque(self):
        array = pyramid_array()
        profile = self._easy()
        plain = run_steering(array, np.zeros(4), profile, method="pinv")
        with_null = run_steering(
            array,
            np.zeros(4),
            profile,
            method="pinv",
            null_policy=GradientNullMotion(gain=1.0, max_rate=0.2),
        )
        assert with_null.max_torque_error < 1e-12
        assert plain.max_torque_error < 1e-12

    def test_rate_limit_is_recorded(self):
        array = pyramid_array()
        profile = rest_to_rest_profile([0.0, 0.0, 1.0], 3.0, 4.0, 0.02)
        history = run_steering(array, np.zeros(4), profile, max_gimbal_rate=0.3)
        assert history.n_rate_limited > 0
        assert history.peak_gimbal_rate <= 0.3 + 1e-12

    def test_gsr_receives_the_running_time(self):
        array = pyramid_array()
        history = run_steering(array, np.zeros(4), self._easy(), method="gsr", eps0=0.05)
        assert history.method == "gsr"
        assert history.times[-1] == pytest.approx(self._easy().duration)

    def test_invalid_arguments_raise(self):
        array = pyramid_array()
        profile = self._easy()
        with pytest.raises(ValueError, match="unknown steering method"):
            run_steering(array, np.zeros(4), profile, method="nope")
        with pytest.raises(ValueError, match="initial_deltas must have length 4"):
            run_steering(array, np.zeros(3), profile)
