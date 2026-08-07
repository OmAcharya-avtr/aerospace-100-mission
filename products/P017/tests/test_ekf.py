"""Tests for the extended Kalman filter and the numerical-Jacobian helper."""

from __future__ import annotations

import numpy as np
import pytest

from estimkit import (
    ExtendedKalmanFilter,
    KalmanFilter,
    constant_velocity_cwna,
    numerical_jacobian,
)


def test_numerical_jacobian_known_answer():
    # f(x) = [x0^2, x0 * x1]  =>  J = [[2 x0, 0], [x1, x0]].
    # At x = [2, 3]: J = [[4, 0], [3, 2]].
    def f(x):
        return np.array([x[0] ** 2, x[0] * x[1]])

    jac = numerical_jacobian(f, np.array([2.0, 3.0]))
    assert jac == pytest.approx(np.array([[4.0, 0.0], [3.0, 2.0]]), abs=1e-8)


def test_numerical_jacobian_exact_for_affine_map():
    a = np.array([[1.0, 2.0, -0.5], [0.0, 3.0, 1.0]])
    b = np.array([7.0, -1.0])
    jac = numerical_jacobian(lambda x: a @ x + b, np.array([0.3, -2.0, 5.0]))
    assert jac == pytest.approx(a, abs=1e-9)


def test_numerical_jacobian_non_square_shape():
    jac = numerical_jacobian(lambda x: np.array([x.sum()]), np.zeros(4))
    assert jac.shape == (1, 4)


def test_numerical_jacobian_rejects_non_positive_epsilon():
    with pytest.raises(ValueError, match="epsilon must be positive"):
        numerical_jacobian(lambda x: x, np.zeros(2), epsilon=0.0)


def test_numerical_jacobian_rejects_variable_length_output():
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return np.zeros(2) if calls["n"] < 2 else np.zeros(3)

    with pytest.raises(ValueError, match="constant length"):
        numerical_jacobian(f, np.zeros(2))


def test_ekf_reduces_to_kf_on_linear_system():
    f, q = constant_velocity_cwna(1.0, 0.05)
    h = np.array([[1.0, 0.0]])
    r = np.array([[4.0]])
    rng = np.random.default_rng(5)
    zs = np.cumsum(rng.standard_normal((80, 1)), axis=0)

    kf = KalmanFilter(f, h, q, r)
    ekf = ExtendedKalmanFilter(
        f=lambda x: f @ x,
        h=lambda x: h @ x,
        process_noise=q,
        measurement_noise=r,
        f_jac=lambda x: f,
        h_jac=lambda x: h,
    )
    a = kf.filter(np.zeros(2), np.eye(2) * 50.0, zs)
    b = ekf.filter(np.zeros(2), np.eye(2) * 50.0, zs)
    assert b.x_post == pytest.approx(a.x_post, abs=1e-12)
    assert b.p_post == pytest.approx(a.p_post, abs=1e-12)
    assert b.nis == pytest.approx(a.nis, abs=1e-12)


def test_ekf_numerical_jacobians_track_analytic_ones():
    f, q = constant_velocity_cwna(0.5, 0.1)
    h = np.array([[1.0, 0.0]])
    r = np.array([[2.0]])
    rng = np.random.default_rng(9)
    zs = rng.standard_normal((40, 1))
    common = {"f": lambda x: f @ x, "h": lambda x: h @ x,
              "process_noise": q, "measurement_noise": r}
    analytic = ExtendedKalmanFilter(**common, f_jac=lambda x: f, h_jac=lambda x: h)
    numeric = ExtendedKalmanFilter(**common)
    a = analytic.filter(np.zeros(2), np.eye(2) * 5.0, zs)
    b = numeric.filter(np.zeros(2), np.eye(2) * 5.0, zs)
    # Central differences on an affine map are exact up to round-off.
    assert b.x_post == pytest.approx(a.x_post, abs=1e-8)
    assert b.p_post == pytest.approx(a.p_post, abs=1e-8)


def test_ekf_tracks_a_nonlinear_range_measurement():
    # 2-D constant position, range-only measurement from a sensor at (0, 0):
    #   h(x) = sqrt(x0^2 + x1^2),  dh/dx = [x0, x1] / h
    truth = np.array([30.0, 40.0])  # range 50 m

    def h(x):
        return np.array([np.hypot(x[0], x[1])])

    def h_jac(x):
        rng_ = np.hypot(x[0], x[1])
        return np.array([[x[0] / rng_, x[1] / rng_]])

    ekf = ExtendedKalmanFilter(
        f=lambda x: x,
        h=h,
        process_noise=np.eye(2) * 1e-6,
        measurement_noise=np.array([[0.25]]),
        f_jac=lambda x: np.eye(2),
        h_jac=h_jac,
    )
    rng = np.random.default_rng(17)
    zs = 50.0 + 0.5 * rng.standard_normal((300, 1))
    res = ekf.filter(np.array([28.0, 42.0]), np.eye(2) * 4.0, zs)
    # Range-only from one sensor is unobservable along the tangential
    # direction, so only the range of the estimate is expected to converge.
    est_range = float(np.hypot(*res.x_post[-1]))
    assert est_range == pytest.approx(float(np.hypot(*truth)), abs=0.3)


def test_ekf_h_jac_wrong_shape_raises():
    ekf = ExtendedKalmanFilter(
        f=lambda x: x,
        h=lambda x: np.array([x[0]]),
        process_noise=np.eye(2),
        measurement_noise=np.eye(1),
        f_jac=lambda x: np.eye(2),
        h_jac=lambda x: np.eye(2),
    )
    with pytest.raises(ValueError, match="h_jac must return shape"):
        ekf.update(np.zeros(2), np.eye(2), np.zeros(1))


def test_ekf_f_jac_wrong_shape_raises():
    ekf = ExtendedKalmanFilter(
        f=lambda x: x,
        h=lambda x: np.array([x[0]]),
        process_noise=np.eye(2),
        measurement_noise=np.eye(1),
        f_jac=lambda x: np.ones((3, 3)),
    )
    with pytest.raises(ValueError, match="f_jac must return shape"):
        ekf.predict(np.zeros(2), np.eye(2))


def test_ekf_rejects_non_callable():
    with pytest.raises(TypeError, match="f and h must be callables"):
        ExtendedKalmanFilter(f=None, h=lambda x: x, process_noise=np.eye(1),
                             measurement_noise=np.eye(1))


def test_ekf_rejects_non_callable_jacobian():
    with pytest.raises(TypeError, match="f_jac must be callable"):
        ExtendedKalmanFilter(f=lambda x: x, h=lambda x: x, process_noise=np.eye(1),
                             measurement_noise=np.eye(1), f_jac=np.eye(1))


def test_ekf_rejects_bad_f_output_length():
    ekf = ExtendedKalmanFilter(
        f=lambda x: np.zeros(3),
        h=lambda x: np.array([x[0]]),
        process_noise=np.eye(2),
        measurement_noise=np.eye(1),
        f_jac=lambda x: np.eye(2),
    )
    with pytest.raises(ValueError, match="f must return 2 elements"):
        ekf.predict(np.zeros(2), np.eye(2))


def test_ekf_rejects_empty_measurements():
    ekf = ExtendedKalmanFilter(
        f=lambda x: x, h=lambda x: x, process_noise=np.eye(1), measurement_noise=np.eye(1)
    )
    with pytest.raises(ValueError, match="at least one time step"):
        ekf.filter(np.zeros(1), np.eye(1), np.zeros((0, 1)))
