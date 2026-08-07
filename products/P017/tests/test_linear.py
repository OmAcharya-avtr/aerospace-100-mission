"""Unit, known-answer, validation and edge-case tests for the linear Kalman filter."""

from __future__ import annotations

import numpy as np
import pytest

from estimkit import KalmanFilter, constant_velocity_cwna, random_walk, steady_state


@pytest.fixture
def scalar_kf() -> KalmanFilter:
    f, h, q, r = random_walk(1.0, 1.0)
    return KalmanFilter(f, h, q, r)


def test_predict_known_answer_constant_velocity():
    # F = [[1, 1], [0, 1]], x = [0, 1], Q = 0, P = I.
    #   x^- = F x = [1, 1]
    #   P^- = F I F^T = [[1,1],[0,1]] [[1,0],[1,1]] = [[2, 1], [1, 1]]
    f = np.array([[1.0, 1.0], [0.0, 1.0]])
    h = np.array([[1.0, 0.0]])
    kf = KalmanFilter(f, h, np.zeros((2, 2)), np.eye(1))
    x, p = kf.predict(np.array([0.0, 1.0]), np.eye(2))
    assert x == pytest.approx([1.0, 1.0])
    assert p == pytest.approx(np.array([[2.0, 1.0], [1.0, 1.0]]))


def test_update_known_answer_scalar(scalar_kf):
    # x^- = 0, P^- = 1, H = 1, R = 1, z = 2.
    #   S = 1*1*1 + 1 = 2
    #   K = 1*1/2 = 0.5
    #   x^+ = 0 + 0.5 * (2 - 0) = 1.0
    #   P^+ = 0.5^2 * 1 + 0.5^2 * 1 = 0.5
    #   NIS = y^2 / S = 4 / 2 = 2
    res = scalar_kf.update(np.array([0.0]), np.array([[1.0]]), np.array([2.0]))
    assert res.x == pytest.approx([1.0])
    assert res.p == pytest.approx(np.array([[0.5]]))
    assert res.gain == pytest.approx(np.array([[0.5]]))
    assert res.innovation == pytest.approx([2.0])
    assert res.innovation_cov == pytest.approx(np.array([[2.0]]))
    assert res.nis == pytest.approx(2.0)


def test_control_input_shifts_prediction():
    f = np.eye(1)
    kf = KalmanFilter(f, np.eye(1), np.zeros((1, 1)), np.eye(1), control=np.array([[2.0]]))
    x, _ = kf.predict(np.array([1.0]), np.eye(1), u=np.array([3.0]))
    assert x == pytest.approx([7.0])  # 1 + 2*3


def test_control_without_b_matrix_raises(scalar_kf):
    with pytest.raises(ValueError, match="no B matrix"):
        scalar_kf.predict(np.zeros(1), np.eye(1), u=np.array([1.0]))


def test_steady_state_scalar_matches_hand_solution():
    # p^2 - q p - q r = 0  =>  p = (q + sqrt(q^2 + 4 q r)) / 2.
    # For q = r = 1 this is the golden ratio (1 + sqrt(5))/2, and
    # K = p/(p+r) = 1/phi = (sqrt(5) - 1)/2.
    f, h, q, r = random_walk(1.0, 1.0)
    p_prior, p_post, gain, _ = steady_state(f, h, q, r, tol=1e-15)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    assert float(p_prior[0, 0]) == pytest.approx(phi, abs=1e-12)
    assert float(gain[0, 0]) == pytest.approx(1.0 / phi, abs=1e-12)
    assert float(p_post[0, 0]) == pytest.approx(phi - 1.0, abs=1e-12)


@pytest.mark.parametrize(("q", "r"), [(0.25, 4.0), (2.0, 0.5), (1e-3, 1.0)])
def test_steady_state_scalar_closed_form(q, r):
    f, h, qm, rm = random_walk(q, r)
    p_prior, _, gain, _ = steady_state(f, h, qm, rm, tol=1e-15)
    p_hand = 0.5 * (q + np.sqrt(q * q + 4.0 * q * r))
    assert float(p_prior[0, 0]) == pytest.approx(p_hand, rel=1e-10)
    assert float(gain[0, 0]) == pytest.approx(p_hand / (p_hand + r), rel=1e-10)


def test_steady_state_constant_velocity_matches_kalata_alpha_beta():
    # Kalata, IEEE Trans. AES-20(2), 1984: with tracking index
    # Lambda = sigma_a T^2 / sigma_v,
    #   rho = (4 + L - sqrt(8L + L^2))/4, alpha = 1 - rho^2,
    #   beta = 2(2 - alpha) - 4 sqrt(1 - alpha),  K_inf = [alpha, beta/T].
    from estimkit import constant_velocity_dwna

    dt, sigma_a, sigma_v = 1.0, 0.1, 2.0
    f, q = constant_velocity_dwna(dt, sigma_a)
    h = np.array([[1.0, 0.0]])
    r = np.array([[sigma_v**2]])
    _, _, gain, _ = steady_state(f, h, q, r, tol=1e-15)
    lam = sigma_a * dt**2 / sigma_v
    rho = (4.0 + lam - np.sqrt(8.0 * lam + lam * lam)) / 4.0
    alpha = 1.0 - rho**2
    beta = 2.0 * (2.0 - alpha) - 4.0 * np.sqrt(1.0 - alpha)
    assert gain.ravel() == pytest.approx(np.array([alpha, beta / dt]), abs=1e-12)


def test_filter_converges_to_steady_state_gain():
    f, q = constant_velocity_cwna(1.0, 0.05)
    h = np.array([[1.0, 0.0]])
    r = np.array([[9.0]])
    kf = KalmanFilter(f, h, q, r)
    rng = np.random.default_rng(3)
    zs = rng.standard_normal((400, 1)) * 3.0
    res = kf.filter(np.zeros(2), np.eye(2) * 100.0, zs)
    _, _, gain_ss, _ = steady_state(f, h, q, r, tol=1e-15)
    assert res.gain[-1] == pytest.approx(gain_ss, abs=1e-9)


def test_filter_covariance_independent_of_measurements():
    # The linear KF covariance recursion does not depend on the data.
    f, q = constant_velocity_cwna(1.0, 0.05)
    h = np.array([[1.0, 0.0]])
    r = np.array([[9.0]])
    kf = KalmanFilter(f, h, q, r)
    rng = np.random.default_rng(11)
    a = kf.filter(np.zeros(2), np.eye(2) * 10.0, rng.standard_normal((50, 1)))
    b = kf.filter(np.zeros(2), np.eye(2) * 10.0, rng.standard_normal((50, 1)) * 100.0)
    assert a.p_post == pytest.approx(b.p_post)


def test_zero_measurement_noise_snaps_onto_measurement():
    # Fully observable, R = 0 => x^+ must equal z exactly.
    kf = KalmanFilter(np.eye(2), np.eye(2), np.eye(2) * 0.1, np.zeros((2, 2)))
    res = kf.update(np.array([1.0, -2.0]), np.eye(2) * 5.0, np.array([3.0, 4.0]))
    assert res.x == pytest.approx([3.0, 4.0], abs=1e-12)
    assert np.max(np.abs(res.p)) < 1e-12


def test_nis_is_dimensionless_and_scale_invariant():
    # Scaling the measurement units by c scales z, H, R^(1/2) consistently;
    # the NIS must not change.
    kf1 = KalmanFilter(np.eye(1), np.array([[1.0]]), np.array([[0.5]]), np.array([[2.0]]))
    c = 100.0
    kf2 = KalmanFilter(np.eye(1), np.array([[c]]), np.array([[0.5]]), np.array([[2.0 * c * c]]))
    r1 = kf1.update(np.array([0.0]), np.array([[1.0]]), np.array([1.0]))
    r2 = kf2.update(np.array([0.0]), np.array([[1.0]]), np.array([c]))
    assert r1.nis == pytest.approx(r2.nis)


# --------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------- #
def test_non_square_transition_raises():
    with pytest.raises(ValueError, match="square"):
        KalmanFilter(np.ones((2, 3)), np.eye(2), np.eye(2), np.eye(2))


def test_measurement_shape_mismatch_raises():
    with pytest.raises(ValueError, match="H must have shape"):
        KalmanFilter(np.eye(2), np.ones((1, 3)), np.eye(2), np.eye(1))


def test_negative_definite_q_raises():
    with pytest.raises(ValueError, match="Q must be positive semi-definite"):
        KalmanFilter(np.eye(1), np.eye(1), np.array([[-1.0]]), np.eye(1))


def test_negative_definite_r_raises():
    with pytest.raises(ValueError, match="R must be positive semi-definite"):
        KalmanFilter(np.eye(1), np.eye(1), np.eye(1), np.array([[-2.0]]))


def test_wrong_state_length_raises(scalar_kf):
    with pytest.raises(ValueError, match="x must have 1 elements"):
        scalar_kf.predict(np.zeros(3), np.eye(1))


def test_wrong_measurement_length_raises(scalar_kf):
    with pytest.raises(ValueError, match="z must have 1 elements"):
        scalar_kf.update(np.zeros(1), np.eye(1), np.zeros(2))


def test_empty_measurement_sequence_raises(scalar_kf):
    with pytest.raises(ValueError, match="at least one time step"):
        scalar_kf.filter(np.zeros(1), np.eye(1), np.zeros((0, 1)))


def test_measurement_sequence_wrong_width_raises(scalar_kf):
    with pytest.raises(ValueError, match="measurements must have shape"):
        scalar_kf.filter(np.zeros(1), np.eye(1), np.zeros((5, 3)))


def test_control_sequence_length_mismatch_raises():
    kf = KalmanFilter(np.eye(1), np.eye(1), np.eye(1), np.eye(1), control=np.eye(1))
    with pytest.raises(ValueError, match="same number of steps"):
        kf.filter(np.zeros(1), np.eye(1), np.zeros((4, 1)), controls=np.zeros((3, 1)))


def test_steady_state_non_convergence_raises():
    # Unstable, unobservable second state: the Riccati recursion diverges.
    f = np.array([[1.0, 0.0], [0.0, 2.0]])
    h = np.array([[1.0, 0.0]])
    with pytest.raises(RuntimeError, match="did not converge"):
        steady_state(f, h, np.eye(2), np.eye(1), max_iter=200)
