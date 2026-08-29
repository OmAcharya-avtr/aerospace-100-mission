"""Unit tests for navbench.ekf — extended Kalman filter and numerical Jacobians."""

from __future__ import annotations

import numpy as np
import pytest

from navbench import (
    CovarianceCollapseError,
    ExtendedKalmanFilter,
    KalmanFilter,
    constant_velocity_2d,
    constant_velocity_cwna,
    numerical_jacobian,
    radar_jacobian,
    radar_measurement,
    simulate_linear_system,
)


class TestNumericalJacobian:
    def test_linear_map_exact(self, rng):
        a = rng.standard_normal((3, 4))
        jac = numerical_jacobian(lambda x: a @ x, rng.standard_normal(4))
        assert np.allclose(jac, a, atol=1e-9)

    def test_scalar_quadratic(self):
        jac = numerical_jacobian(lambda x: np.array([x[0] ** 2]), np.array([3.0]))
        assert jac[0, 0] == pytest.approx(6.0, rel=1e-8)

    def test_matches_analytic_radar_jacobian(self, rng):
        for _ in range(20):
            x = np.array([rng.uniform(500, 5000), 1.0, rng.uniform(500, 5000), -1.0])
            num = numerical_jacobian(radar_measurement, x)
            assert np.allclose(num, radar_jacobian(x), rtol=1e-6, atol=1e-10)

    def test_output_shape(self):
        jac = numerical_jacobian(lambda x: np.array([x[0], x[1], x[0] * x[1]]), np.ones(2))
        assert jac.shape == (3, 2)

    def test_constant_function_gives_zero(self):
        jac = numerical_jacobian(lambda x: np.array([1.0, 2.0]), np.array([5.0, 6.0]))
        assert np.allclose(jac, 0.0)

    def test_nonfinite_input_raises(self):
        with pytest.raises(ValueError, match="finite"):
            numerical_jacobian(lambda x: x, np.array([np.nan, 1.0]))

    def test_step_scales_with_magnitude(self):
        """Large-magnitude states still get a usable Jacobian."""
        jac = numerical_jacobian(lambda x: np.array([x[0] ** 2]), np.array([1e6]))
        assert jac[0, 0] == pytest.approx(2e6, rel=1e-6)


class TestEkfConstruction:
    def test_flags_numerical_jacobians(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        ekf = ExtendedKalmanFilter(lambda x: f @ x, lambda x: h @ x, q, r, x0, p0)
        assert ekf.uses_numerical_jacobian

    def test_analytic_jacobians_flagged_off(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        ekf = ExtendedKalmanFilter(
            lambda x: f @ x, lambda x: h @ x, q, r, x0, p0,
            f_jac=lambda x: f, h_jac=lambda x: h,
        )
        assert not ekf.uses_numerical_jacobian

    def test_non_callable_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        with pytest.raises(TypeError, match="callable"):
            ExtendedKalmanFilter(f, lambda x: h @ x, q, r, x0, p0)

    def test_asymmetric_q_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        bad = q.copy()
        bad[0, 1] += 1.0
        with pytest.raises(ValueError, match="symmetric"):
            ExtendedKalmanFilter(lambda x: f @ x, lambda x: h @ x, bad, r, x0, p0)

    def test_singular_r_raises(self, cv_model):
        f, q, h, _, x0, p0 = cv_model
        with pytest.raises(ValueError, match="positive definite"):
            ExtendedKalmanFilter(lambda x: f @ x, lambda x: h @ x, q, np.zeros((1, 1)), x0, p0)

    def test_empty_x0_raises(self, cv_model):
        f, q, h, r, _, p0 = cv_model
        with pytest.raises(ValueError, match="non-empty"):
            ExtendedKalmanFilter(lambda x: f @ x, lambda x: h @ x, q, r, np.zeros(0), p0)


class TestEkfReducesToKf:
    def test_identical_on_linear_system(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        _, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 120, rng)
        kf = KalmanFilter(f, h, q, r, x0, p0).run(z)
        ekf = ExtendedKalmanFilter(
            lambda x: f @ x, lambda x: h @ x, q, r, x0, p0,
            f_jac=lambda x: f, h_jac=lambda x: h,
        ).run(z)
        assert np.allclose(kf.x_post, ekf.x_post, atol=1e-12)
        assert np.allclose(kf.p_post, ekf.p_post, atol=1e-12)
        assert np.allclose(kf.gain, ekf.gain, atol=1e-12)
        assert np.allclose(kf.nis, ekf.nis, atol=1e-12)

    def test_numerical_jacobians_agree_closely(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        _, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 60, rng)
        kf = KalmanFilter(f, h, q, r, x0, p0).run(z)
        ekf = ExtendedKalmanFilter(lambda x: f @ x, lambda x: h @ x, q, r, x0, p0).run(z)
        scale = float(np.max(np.abs(kf.x_post)))
        assert np.max(np.abs(kf.x_post - ekf.x_post)) / scale < 1e-8


class TestEkfNonlinear:
    def test_tracks_a_radar_target(self, rng):
        from navbench import simulate_radar_scenario

        f, q = constant_velocity_2d(1.0, 0.05)
        r = np.diag([400.0, 1e-4])
        truth, meas = simulate_radar_scenario(
            dt=1.0, n_steps=120, q_psd=0.05, sigma_range=20.0, sigma_bearing=0.01,
            x0=np.array([5000.0, -5.0, 5000.0, 3.0]), rng=rng,
        )
        ekf = ExtendedKalmanFilter(
            lambda x: f @ x, radar_measurement, q, r,
            np.array([truth[0, 0], 0.0, truth[0, 2], 0.0]),
            np.diag([400.0, 100.0, 400.0, 100.0]),
            f_jac=lambda x: f, h_jac=radar_jacobian,
        )
        res = ekf.run(meas)
        err = np.hypot(truth[20:, 0] - res.x_post[20:, 0], truth[20:, 2] - res.x_post[20:, 2])
        assert float(np.sqrt(np.mean(err**2))) < 60.0

    def test_nan_measurement_skipped(self, rng):
        f, q = constant_velocity_cwna(1.0, 0.05)
        h = np.array([[1.0, 0.0]])
        r = np.array([[9.0]])
        _, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 30, rng)
        z[7] = np.nan
        ekf = ExtendedKalmanFilter(
            lambda x: f @ x, lambda x: h @ x, q, r, np.zeros(2), np.diag([100.0, 10.0]),
            f_jac=lambda x: f, h_jac=lambda x: h,
        )
        res = ekf.run(z)
        assert not res.updated[7]
        assert np.isnan(res.nis[7])
        assert res.innovation_cov[7].shape == (1, 1)

    def test_nan_skipped_with_numerical_jacobian(self, rng):
        f, q = constant_velocity_cwna(1.0, 0.05)
        h = np.array([[1.0, 0.0]])
        r = np.array([[9.0]])
        _, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 20, rng)
        z[3] = np.nan
        res = ExtendedKalmanFilter(
            lambda x: f @ x, lambda x: h @ x, q, r, np.zeros(2), np.diag([100.0, 10.0])
        ).run(z)
        assert not res.updated[3]

    def test_control_input_passed_through(self):
        def f_fun(x, u):
            return x + np.asarray(u, dtype=float)

        ekf = ExtendedKalmanFilter(
            f_fun, lambda x: x, np.eye(2) * 1e-6, np.eye(2), np.zeros(2), np.eye(2),
            f_jac=lambda x, u: np.eye(2), h_jac=lambda x: np.eye(2),
        )
        x, _ = ekf.predict(u=[1.0, 2.0])
        assert np.allclose(x, [1.0, 2.0])


class TestEkfValidation:
    def test_wrong_f_fun_output_size_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        ekf = ExtendedKalmanFilter(
            lambda x: np.zeros(3), lambda x: h @ x, q, r, x0, p0,
            f_jac=lambda x: f, h_jac=lambda x: h,
        )
        with pytest.raises(ValueError, match="f_fun must return"):
            ekf.predict()

    def test_wrong_h_fun_output_size_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        ekf = ExtendedKalmanFilter(
            lambda x: f @ x, lambda x: np.zeros(2), q, r, x0, p0,
            f_jac=lambda x: f, h_jac=lambda x: h,
        )
        ekf.predict()
        with pytest.raises(ValueError, match="h_fun must return"):
            ekf.update([1.0])

    def test_wrong_f_jac_shape_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        ekf = ExtendedKalmanFilter(
            lambda x: f @ x, lambda x: h @ x, q, r, x0, p0,
            f_jac=lambda x: np.eye(3), h_jac=lambda x: h,
        )
        with pytest.raises(ValueError, match="f_jac must return"):
            ekf.predict()

    def test_wrong_h_jac_shape_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        ekf = ExtendedKalmanFilter(
            lambda x: f @ x, lambda x: h @ x, q, r, x0, p0,
            f_jac=lambda x: f, h_jac=lambda x: np.zeros((2, 2)),
        )
        ekf.predict()
        with pytest.raises(ValueError, match="h_jac must return"):
            ekf.update([1.0])

    def test_wrong_z_size_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        ekf = ExtendedKalmanFilter(
            lambda x: f @ x, lambda x: h @ x, q, r, x0, p0,
            f_jac=lambda x: f, h_jac=lambda x: h,
        )
        with pytest.raises(ValueError, match="elements"):
            ekf.update([1.0, 2.0])

    def test_nonfinite_z_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        ekf = ExtendedKalmanFilter(
            lambda x: f @ x, lambda x: h @ x, q, r, x0, p0,
            f_jac=lambda x: f, h_jac=lambda x: h,
        )
        with pytest.raises(ValueError, match="finite"):
            ekf.update([np.nan])

    def test_wrong_measurement_width_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        ekf = ExtendedKalmanFilter(
            lambda x: f @ x, lambda x: h @ x, q, r, x0, p0,
            f_jac=lambda x: f, h_jac=lambda x: h,
        )
        with pytest.raises(ValueError, match="shape"):
            ekf.run(np.zeros((5, 3)))

    def test_collapsed_covariance_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        ekf = ExtendedKalmanFilter(
            lambda x: f @ x, lambda x: h @ x, q, r, x0, p0,
            f_jac=lambda x: f, h_jac=lambda x: h,
        )
        ekf.p = np.array([[-1.0, 0.0], [0.0, -1.0]])
        ekf.r = np.array([[1e-30]])
        with pytest.raises(CovarianceCollapseError):
            ekf.update([0.0])
