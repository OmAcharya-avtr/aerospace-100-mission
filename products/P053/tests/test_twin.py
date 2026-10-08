"""Tests for the declared twin and its steady-state residual generator."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from twininvalidate import (
    LinearGaussianTwin,
    attitude_channel,
    cwna_process_noise,
    reference_excitation,
    reference_twin,
    zoh_discretise,
)


def test_reference_twin_shapes(twin):
    assert twin.n_states == 2
    assert twin.n_inputs == 1
    assert twin.C.shape == (1, 2)
    assert twin.dt == pytest.approx(0.05)


def test_riccati_solution_satisfies_its_equation(filt):
    # Equation (5) must hold at the returned P. The residual is a pure
    # round-off quantity; max|P| is about 2e-5 here, so 1e-15 is generous.
    assert filt.riccati_residual() < 1e-15


def test_prediction_error_dynamics_are_stable(filt):
    assert filt.spectral_radius() < 1.0


def test_predictor_gain_includes_the_A_factor(twin, filt):
    # K = A P C^T / S, not P C^T / S. The two differ because A is not the
    # identity, and the difference is what made the residual autocorrelated
    # in the first version of this package.
    expected = (twin.A @ filt.P @ twin.C.T) / filt.S
    assert np.allclose(filt.K, expected, rtol=1e-12, atol=0.0)
    naive = (filt.P @ twin.C.T) / filt.S
    assert not np.allclose(filt.K, naive, rtol=1e-6, atol=1e-9)


def test_innovation_variance_matches_its_definition(twin, filt):
    assert filt.S == pytest.approx(float((twin.C @ filt.P @ twin.C.T + twin.R)[0, 0]), rel=1e-14)


def test_zoh_of_zero_dynamics_is_identity_and_b_dt():
    # xdot = 0 * x + b u over dt: A = I, B = b dt exactly.
    a_c = np.zeros((2, 2))
    b_c = np.array([[1.0], [2.0]])
    a_d, b_d = zoh_discretise(a_c, b_c, 0.25)
    assert np.allclose(a_d, np.eye(2))
    assert np.allclose(b_d, b_c * 0.25)


def test_zoh_scalar_known_answer():
    # xdot = a x + b u, a = -2, b = 3, dt = 0.1.
    # A = exp(-0.2) = 0.818730753...
    # B = b (exp(a dt) - 1) / a = 3 * (0.818730753 - 1) / (-2) = 0.271903870...
    a_d, b_d = zoh_discretise(np.array([[-2.0]]), np.array([[3.0]]), 0.1)
    assert a_d[0, 0] == pytest.approx(0.8187307530779818, rel=1e-12)
    assert b_d[0, 0] == pytest.approx(0.2719038703830273, rel=1e-12)


def test_zoh_matches_direct_matrix_exponential(twin):
    a_c = np.array([[0.0, 1.0], [-4.0, -0.4]])
    b_c = np.array([[0.0], [4.0]])
    a_d, b_d = zoh_discretise(a_c, b_c, 0.05)
    block = np.zeros((3, 3))
    block[:2, :2] = a_c
    block[:2, 2:] = b_c
    e = expm(block * 0.05)
    assert np.allclose(a_d, e[:2, :2], rtol=1e-13, atol=0.0)
    assert np.allclose(b_d, e[:2, 2:], rtol=1e-13, atol=0.0)


def test_cwna_process_noise_known_answer():
    # Q = sigma_a^2 [[dt^3/3, dt^2/2], [dt^2/2, dt]] with dt = 0.05, sigma_a = 0.02.
    # dt^3/3 = 1.25e-4/3 = 4.1666...e-5; times 4e-4 = 1.6666...e-8
    # dt^2/2 = 2.5e-3/2 = 1.25e-3;       times 4e-4 = 5.0e-7
    # dt     = 0.05;                     times 4e-4 = 2.0e-5
    q = cwna_process_noise(0.05, 0.02)
    assert q[0, 0] == pytest.approx(1.6666666666666667e-08, rel=1e-12)
    assert q[0, 1] == pytest.approx(5.0e-07, rel=1e-12)
    assert q[1, 0] == pytest.approx(5.0e-07, rel=1e-12)
    assert q[1, 1] == pytest.approx(2.0e-05, rel=1e-12)


def test_cwna_is_positive_definite():
    q = cwna_process_noise(0.05, 0.02)
    assert np.linalg.eigvalsh(q).min() > 0.0


def test_excitation_has_the_declared_constant_and_dither():
    u = reference_excitation(400, dt=0.05)
    assert u.shape == (400, 1)
    # u = 1 + 0.2 sin(2 pi 0.15 k dt); mean over whole periods is 1.
    assert u[0, 0] == pytest.approx(1.0)
    assert u.min() >= 0.8 - 1e-12
    assert u.max() <= 1.2 + 1e-12


def test_with_offset_returns_a_copy(twin):
    other = twin.with_offset(0.01)
    assert other.offset == pytest.approx(0.01)
    assert twin.offset == pytest.approx(0.0)
    assert np.allclose(other.A, twin.A)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"omega_n": 0.0}, "omega_n must be positive"),
        ({"omega_n": -1.0}, "omega_n must be positive"),
        ({"zeta": 1.0}, "zeta must lie"),
        ({"zeta": -0.1}, "zeta must lie"),
        ({"sigma_meas": 0.0}, "sigma_meas must be positive"),
    ],
)
def test_attitude_channel_rejects_bad_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        attitude_channel(**kwargs)


def test_zoh_rejects_non_positive_dt():
    with pytest.raises(ValueError, match="dt must be positive"):
        zoh_discretise(np.zeros((1, 1)), np.ones((1, 1)), 0.0)


def test_cwna_rejects_bad_arguments():
    with pytest.raises(ValueError, match="dt must be positive"):
        cwna_process_noise(-1.0, 0.1)
    with pytest.raises(ValueError, match="sigma_a must be non-negative"):
        cwna_process_noise(0.1, -1.0)


def test_excitation_rejects_negative_length():
    with pytest.raises(ValueError, match="n_samples must be non-negative"):
        reference_excitation(-1)


def _base_kwargs():
    t = reference_twin()
    return {"A": t.A, "B": t.B, "C": t.C, "Q": t.Q, "R": t.R, "dt": t.dt}


@pytest.mark.parametrize(
    "override, message",
    [
        ({"A": np.zeros((2, 3))}, "A must be square"),
        ({"B": np.zeros((3, 1))}, "B must have 2 rows"),
        ({"C": np.zeros((2, 2))}, r"C must have shape \(1, 2\)"),
        ({"Q": np.zeros((3, 3))}, r"Q must have shape \(2, 2\)"),
        ({"R": np.zeros((2, 2))}, r"R must have shape \(1, 1\)"),
        ({"R": np.array([[0.0]])}, "R must be strictly positive"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"Q": np.array([[1.0, 2.0], [0.0, 1.0]])}, "Q must be symmetric"),
        ({"Q": np.array([[-1.0, 0.0], [0.0, 1.0]])}, "Q must be positive semi-definite"),
    ],
)
def test_twin_validates_its_inputs(override, message):
    kwargs = _base_kwargs()
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        LinearGaussianTwin(**kwargs)


def test_twin_rejects_one_dimensional_matrices():
    kwargs = _base_kwargs()
    kwargs["A"] = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="A must be a 2-D array"):
        LinearGaussianTwin(**kwargs)
