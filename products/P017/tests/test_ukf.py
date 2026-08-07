"""Tests for the scaled unscented transform and the unscented Kalman filter."""

from __future__ import annotations

import numpy as np
import pytest

from estimkit import (
    KalmanFilter,
    MerweSigmaPoints,
    UnscentedKalmanFilter,
    constant_velocity_cwna,
    unscented_transform,
)


def test_weight_known_answer_alpha1_kappa0():
    # n = 2, alpha = 1, beta = 2, kappa = 0  =>  lambda = 1*(2+0) - 2 = 0,
    # n + lambda = 2.  W0^m = 0/2 = 0;  W0^c = 0 + (1 - 1 + 2) = 2;
    # Wi = 1/(2*2) = 0.25 for i = 1..4.  Sum of mean weights = 1.
    sp = MerweSigmaPoints(2, alpha=1.0, beta=2.0, kappa=0.0)
    assert sp.lambda_ == pytest.approx(0.0)
    assert sp.wm[0] == pytest.approx(0.0)
    assert sp.wc[0] == pytest.approx(2.0)
    assert sp.wm[1:] == pytest.approx(np.full(4, 0.25))
    assert sp.wm.sum() == pytest.approx(1.0)


def test_weight_known_answer_alpha1_kappa1():
    # n = 2, alpha = 1, beta = 2, kappa = 1  =>  lambda = 3 - 2 = 1,
    # n + lambda = 3.  W0^m = 1/3;  Wi = 1/6;  sum = 1/3 + 4/6 = 1.
    # W0^c = 1/3 + (1 - 1 + 2) = 7/3.
    sp = MerweSigmaPoints(2, alpha=1.0, beta=2.0, kappa=1.0)
    assert sp.lambda_ == pytest.approx(1.0)
    assert sp.wm[0] == pytest.approx(1.0 / 3.0)
    assert sp.wm[1] == pytest.approx(1.0 / 6.0)
    assert sp.wc[0] == pytest.approx(7.0 / 3.0)
    assert sp.wm.sum() == pytest.approx(1.0)


def test_sigma_points_known_answer_scalar():
    # n = 1, mean 0, P = 1, alpha = 1, kappa = 2  =>  lambda = 3 - 1 = 2,
    # n + lambda = 3, so the points are 0 and +- sqrt(3).
    sp = MerweSigmaPoints(1, alpha=1.0, beta=2.0, kappa=2.0)
    pts = sp.generate(np.array([0.0]), np.array([[1.0]]))
    assert pts.points.ravel() == pytest.approx([0.0, np.sqrt(3.0), -np.sqrt(3.0)])
    assert pts.wm[0] == pytest.approx(2.0 / 3.0)
    assert pts.wm[1] == pytest.approx(1.0 / 6.0)


def test_sigma_points_reconstruct_mean_and_covariance():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((4, 4))
    p = a @ a.T + 4.0 * np.eye(4)
    mean = rng.standard_normal(4) * 10.0
    sp = MerweSigmaPoints(4, alpha=0.7, beta=2.0, kappa=-1.0)
    pts = sp.generate(mean, p)
    m2, p2 = unscented_transform(pts.points, pts.wm, pts.wc)
    assert m2 == pytest.approx(mean, abs=1e-10)
    assert p2 == pytest.approx(p, abs=1e-10)


@pytest.mark.parametrize(
    ("alpha", "beta", "kappa"),
    [(1.0, 2.0, 0.0), (0.5, 0.0, 1.0), (1e-2, 2.0, 0.0), (1e-3, 2.0, -1.0)],
)
def test_unscented_transform_is_exact_for_affine_maps(alpha, beta, kappa):
    rng = np.random.default_rng(13)
    n = 3
    root = rng.standard_normal((n, n))
    p = root @ root.T + np.eye(n)
    mean = rng.standard_normal(n)
    a = rng.standard_normal((2, n))
    b = rng.standard_normal(2)

    sp = MerweSigmaPoints(n, alpha=alpha, beta=beta, kappa=kappa)
    pts = sp.generate(mean, p)
    transformed = np.array([a @ x + b for x in pts.points])
    m_ut, p_ut = unscented_transform(transformed, pts.wm, pts.wc)
    scale = max(1.0, float(np.max(np.abs(a @ p @ a.T))))
    assert m_ut == pytest.approx(a @ mean + b, rel=1e-9, abs=1e-9 * scale / alpha**2)
    assert p_ut == pytest.approx(a @ p @ a.T, abs=1e-9 * scale / alpha**2)


def test_unscented_transform_adds_noise_covariance():
    sp = MerweSigmaPoints(1, alpha=1.0, beta=2.0, kappa=2.0)
    pts = sp.generate(np.array([0.0]), np.array([[1.0]]))
    _, cov = unscented_transform(pts.points, pts.wm, pts.wc, noise_cov=np.array([[3.0]]))
    assert cov == pytest.approx(np.array([[4.0]]))


def test_ukf_matches_kf_on_linear_gaussian_system():
    f, q = constant_velocity_cwna(1.0, 0.05)
    h = np.array([[1.0, 0.0]])
    r = np.array([[9.0]])
    rng = np.random.default_rng(21)
    zs = np.cumsum(rng.standard_normal((150, 1)) * 2.0, axis=0)

    kf = KalmanFilter(f, h, q, r)
    ukf = UnscentedKalmanFilter(
        f=lambda x: f @ x, h=lambda x: h @ x,
        process_noise=q, measurement_noise=r,
        alpha=1.0, beta=2.0, kappa=0.0,
    )
    a = kf.filter(np.zeros(2), np.eye(2) * 100.0, zs)
    b = ukf.filter(np.zeros(2), np.eye(2) * 100.0, zs)
    scale = float(np.max(np.abs(a.x_post)))
    assert np.max(np.abs(b.x_post - a.x_post)) / scale < 1e-12
    assert np.max(np.abs(b.p_post - a.p_post)) < 1e-10
    assert np.max(np.abs(b.gain - a.gain)) < 1e-12


def test_ukf_effective_transition_recovers_true_transition_for_linear_f():
    f, q = constant_velocity_cwna(1.0, 0.05)
    h = np.array([[1.0, 0.0]])
    r = np.array([[9.0]])
    ukf = UnscentedKalmanFilter(
        f=lambda x: f @ x, h=lambda x: h @ x,
        process_noise=q, measurement_noise=r, alpha=1.0, beta=2.0, kappa=0.0,
    )
    res = ukf.filter(np.zeros(2), np.eye(2) * 10.0, np.zeros((20, 1)))
    assert res.transition[-1] == pytest.approx(f, abs=1e-9)


def test_ukf_zero_measurement_noise_snaps_onto_measurement():
    ukf = UnscentedKalmanFilter(
        f=lambda x: x, h=lambda x: x,
        process_noise=np.eye(2) * 0.1, measurement_noise=np.zeros((2, 2)),
        alpha=1.0, beta=2.0, kappa=0.0,
    )
    res = ukf.update(np.array([1.0, -2.0]), np.eye(2) * 3.0, np.array([5.0, 6.0]))
    assert res.x == pytest.approx([5.0, 6.0], abs=1e-9)


def test_ukf_beats_ekf_on_a_strongly_nonlinear_measurement():
    # Quadratic measurement h(x) = x^2 with a wide prior: the EKF Jacobian
    # 2x is zero at x = 0 and the linearisation is badly wrong over +-sigma.
    # The UKF sees curvature through the sigma points.
    from estimkit import ExtendedKalmanFilter

    prior_mean = np.array([0.5])
    prior_cov = np.array([[4.0]])
    truth = 3.0
    z = np.array([truth**2])
    r = np.array([[0.25]])
    q = np.array([[1e-9]])

    ekf = ExtendedKalmanFilter(
        f=lambda x: x, h=lambda x: x**2, process_noise=q, measurement_noise=r,
        f_jac=lambda x: np.eye(1), h_jac=lambda x: np.array([[2.0 * x[0]]]),
    )
    ukf = UnscentedKalmanFilter(
        f=lambda x: x, h=lambda x: x**2, process_noise=q, measurement_noise=r,
        alpha=1.0, beta=2.0, kappa=2.0,
    )
    e = ekf.update(prior_mean, prior_cov, z)
    u = ukf.update(prior_mean, prior_cov, z)
    assert abs(u.x[0] - truth) < abs(e.x[0] - truth)


# --------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n": 0}, "n must be an integer"),
        ({"n": 2, "alpha": 0.0}, "alpha must satisfy"),
        ({"n": 2, "alpha": 2.0}, "alpha must satisfy"),
        ({"n": 2, "beta": -1.0}, "beta must be finite"),
        ({"n": 2, "kappa": -2.0}, "kappa must be finite"),
        ({"n": 2, "kappa": np.nan}, "kappa must be finite"),
    ],
)
def test_sigma_point_parameter_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MerweSigmaPoints(**kwargs)


def test_sigma_points_reject_wrong_mean_length():
    sp = MerweSigmaPoints(2)
    with pytest.raises(ValueError, match="mean must have 2 elements"):
        sp.generate(np.zeros(3), np.eye(2))


def test_sigma_points_reject_wrong_cov_shape():
    sp = MerweSigmaPoints(2)
    with pytest.raises(ValueError, match="cov must be 2x2"):
        sp.generate(np.zeros(2), np.eye(3))


def test_sigma_points_report_covariance_collapse():
    sp = MerweSigmaPoints(2)
    with pytest.raises(ValueError, match="covariance collapse"):
        sp.generate(np.zeros(2), np.diag([1.0, -1e-6]))


def test_unscented_transform_weight_length_mismatch():
    with pytest.raises(ValueError, match="weights must match"):
        unscented_transform(np.zeros((3, 2)), np.ones(2), np.ones(3))


def test_unscented_transform_noise_shape_mismatch():
    with pytest.raises(ValueError, match="noise_cov must be"):
        unscented_transform(np.zeros((3, 2)), np.ones(3) / 3, np.ones(3) / 3,
                            noise_cov=np.eye(3))


def test_ukf_rejects_non_callable():
    with pytest.raises(TypeError, match="f and h must be callables"):
        UnscentedKalmanFilter(f=1, h=lambda x: x, process_noise=np.eye(1),
                              measurement_noise=np.eye(1))


def test_ukf_rejects_bad_h_output_length():
    ukf = UnscentedKalmanFilter(
        f=lambda x: x, h=lambda x: np.zeros(3),
        process_noise=np.eye(2), measurement_noise=np.eye(1), alpha=1.0,
    )
    with pytest.raises(ValueError, match="h must return 1 elements"):
        ukf.update(np.zeros(2), np.eye(2), np.zeros(1))


def test_ukf_rejects_empty_measurements():
    ukf = UnscentedKalmanFilter(
        f=lambda x: x, h=lambda x: x, process_noise=np.eye(1),
        measurement_noise=np.eye(1), alpha=1.0,
    )
    with pytest.raises(ValueError, match="at least one time step"):
        ukf.filter(np.zeros(1), np.eye(1), np.zeros((0, 1)))
