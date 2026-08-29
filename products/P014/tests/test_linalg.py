"""Unit, KAT, edge-case and property tests for wavelab.linalg."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from wavelab.linalg import (
    noise_propagation_coefficients,
    null_space,
    tikhonov_solve,
    tsvd_solve,
)

# --------------------------------------------------------------------- tikhonov_solve


def test_tikhonov_solve_hand_calc_identity():
    # G = I, s = [1, 2], lambda = 0 -> u = s exactly (normal equations reduce
    # to u = s when G is orthonormal and unregularized).
    g = np.eye(2)
    s = np.array([1.0, 2.0])
    u = tikhonov_solve(g, s, lam=0.0)
    np.testing.assert_allclose(u, s)


def test_tikhonov_solve_hand_calc_scalar():
    # G = [[2]], s = [4], lambda = 0 -> normal eq: 4u = 8 -> u = 2.
    # With lambda = 2: (4 + 4)u = 8 -> u = 1.
    g = np.array([[2.0]])
    s = np.array([4.0])
    assert tikhonov_solve(g, s, lam=0.0)[0] == pytest.approx(2.0)
    assert tikhonov_solve(g, s, lam=2.0)[0] == pytest.approx(1.0)


def test_tikhonov_solve_rejects_negative_lambda():
    with pytest.raises(ValueError):
        tikhonov_solve(np.eye(2), np.ones(2), lam=-1.0)


def test_tikhonov_solve_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        tikhonov_solve(np.eye(3), np.ones(2), lam=0.1)


def test_tikhonov_solve_rejects_non_finite():
    g = np.eye(2)
    with pytest.raises(ValueError):
        tikhonov_solve(g, np.array([np.nan, 1.0]), lam=0.1)
    with pytest.raises(ValueError):
        tikhonov_solve(np.array([[np.inf, 0], [0, 1]]), np.ones(2), lam=0.1)


def test_tikhonov_solve_reduces_to_ols_for_well_conditioned_system():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(20, 4))
    u_true = rng.normal(size=4)
    s = g @ u_true
    u_hat = tikhonov_solve(g, s, lam=1e-10)
    np.testing.assert_allclose(u_hat, u_true, atol=1e-5)


def test_tikhonov_regularization_shrinks_norm():
    rng = np.random.default_rng(1)
    g = rng.normal(size=(20, 4))
    s = rng.normal(size=20)
    u_small_lam = tikhonov_solve(g, s, lam=0.01)
    u_large_lam = tikhonov_solve(g, s, lam=10.0)
    assert np.linalg.norm(u_large_lam) < np.linalg.norm(u_small_lam)


# --------------------------------------------------------------------- tsvd_solve


def test_tsvd_solve_hand_calc_rank_deficient():
    # G maps (u1, u2) -> u1 + u2 (rank 1); minimum-norm solution to s=[2] is
    # u = [1, 1] (equal split, the min-norm point on the line u1+u2=2).
    g = np.array([[1.0, 1.0]])
    s = np.array([2.0])
    u = tsvd_solve(g, s, rel_tol=1e-6)
    np.testing.assert_allclose(u, [1.0, 1.0], atol=1e-8)


def test_tsvd_solve_rejects_bad_rel_tol():
    with pytest.raises(ValueError):
        tsvd_solve(np.eye(2), np.ones(2), rel_tol=-0.1)
    with pytest.raises(ValueError):
        tsvd_solve(np.eye(2), np.ones(2), rel_tol=1.0)


def test_tsvd_solve_rejects_zero_matrix():
    with pytest.raises(ValueError):
        tsvd_solve(np.zeros((3, 2)), np.zeros(3), rel_tol=1e-6)


def test_tsvd_solve_matches_pinv_when_full_rank():
    rng = np.random.default_rng(2)
    g = rng.normal(size=(10, 4))
    s = rng.normal(size=10)
    u_tsvd = tsvd_solve(g, s, rel_tol=1e-10)
    u_pinv = np.linalg.pinv(g) @ s
    np.testing.assert_allclose(u_tsvd, u_pinv, atol=1e-8)


# --------------------------------------------------------------------- null_space


def test_null_space_hand_calc():
    g = np.array([[1.0, 1.0]])  # null space spanned by (1, -1)/sqrt(2)
    ns = null_space(g, rel_tol=1e-6)
    assert ns.shape == (2, 1)
    v = ns[:, 0]
    np.testing.assert_allclose(np.abs(v), [1 / np.sqrt(2), 1 / np.sqrt(2)], atol=1e-8)
    assert v[0] * v[1] < 0  # opposite signs, i.e. proportional to (1, -1)


def test_null_space_full_rank_matrix_is_empty():
    ns = null_space(np.eye(3), rel_tol=1e-6)
    assert ns.shape == (3, 0)


def test_null_space_vectors_are_orthonormal():
    g = np.zeros((1, 4))  # everything is null
    ns = null_space(g, rel_tol=1e-6)
    assert ns.shape == (4, 4)
    gram = ns.T @ ns
    np.testing.assert_allclose(gram, np.eye(4), atol=1e-8)


# --------------------------------------------------------------------- noise_propagation_coefficients


def test_noise_propagation_hand_calc_orthonormal():
    # G = I: R = I, coeff_k = sum_p R[k,p]^2 = 1 for each k.
    coeffs = noise_propagation_coefficients(np.eye(3), rel_tol=1e-6)
    np.testing.assert_allclose(coeffs, np.ones(3))


def test_noise_propagation_scales_with_scalar_gain():
    # G = c*I -> pseudo-inverse = (1/c) I -> coeff = 1/c^2.
    c = 4.0
    coeffs = noise_propagation_coefficients(c * np.eye(2), rel_tol=1e-6)
    np.testing.assert_allclose(coeffs, np.full(2, 1.0 / c**2))


def test_noise_propagation_matches_monte_carlo():
    # Var(u_hat_k) = coeff_k * sigma^2 -- verify empirically for a small
    # overdetermined, well-conditioned system.
    rng = np.random.default_rng(3)
    g = rng.normal(size=(15, 3))
    sigma = 0.2
    coeffs = noise_propagation_coefficients(g, rel_tol=1e-8)

    n_trials = 4000
    u_true = np.zeros(3)
    s0 = g @ u_true
    trial_rng = np.random.default_rng(4)
    noise = trial_rng.normal(0.0, sigma, size=(n_trials, 15))
    pinv = np.linalg.pinv(g)
    u_hats = (pinv @ (s0 + noise).T).T
    empirical_var = u_hats.var(axis=0)
    predicted_var = coeffs * sigma**2
    np.testing.assert_allclose(empirical_var, predicted_var, rtol=0.15)


def test_noise_propagation_rejects_zero_matrix():
    with pytest.raises(ValueError):
        noise_propagation_coefficients(np.zeros((3, 2)))


# --------------------------------------------------------------------- Hypothesis property tests


@given(
    arrays(dtype=np.float64, shape=(6, 3), elements=st.floats(-3, 3, allow_nan=False)),
    arrays(dtype=np.float64, shape=(6,), elements=st.floats(-3, 3, allow_nan=False)),
)
@settings(max_examples=40)
def test_tikhonov_solve_residual_normal_equations_identity(g, s):
    # Algebraic identity: for any lambda >= 0, u solves
    # (G^T G + lambda^2 I) u = G^T s exactly, by construction of the closed form.
    lam = 0.5
    u = tikhonov_solve(g, s, lam)
    lhs = (g.T @ g + lam**2 * np.eye(3)) @ u
    rhs = g.T @ s
    np.testing.assert_allclose(lhs, rhs, atol=1e-6)


@given(st.integers(min_value=2, max_value=8))
@settings(max_examples=15)
def test_null_space_dimension_plus_rank_equals_n_cols(m):
    rng = np.random.default_rng(m)
    rank = max(1, m - 1)
    a = rng.normal(size=(m, rank))
    b = rng.normal(size=(rank, m))
    g = a @ b  # (m, m), rank <= rank
    ns = null_space(g, rel_tol=1e-8)
    numerical_rank = np.linalg.matrix_rank(g, tol=1e-8)
    assert ns.shape[1] + numerical_rank == m
