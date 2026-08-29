"""Unit, KAT, edge-case and property tests for wavelab.zernike."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wavelab.zernike import (
    fit_zernike,
    noll_to_nm,
    nm_to_noll,
    normalization,
    radial_order_from_noll,
    radial_polynomial,
    unit_disc_grid,
    validate_nm,
    zernike,
    zernike_basis_matrix,
    zernike_gradient,
    zernike_gradient_noll,
    zernike_noll,
    zernike_slope_matrix,
)

# --------------------------------------------------------------------- validate_nm


def test_validate_nm_rejects_non_integer():
    with pytest.raises(TypeError):
        validate_nm(2.0, 0)
    with pytest.raises(TypeError):
        validate_nm(2, "0")
    with pytest.raises(TypeError):
        validate_nm(True, 0)  # bool must not silently pass as int


def test_validate_nm_rejects_negative_n():
    with pytest.raises(ValueError):
        validate_nm(-1, 0)


def test_validate_nm_rejects_m_out_of_range():
    with pytest.raises(ValueError):
        validate_nm(2, 3)


def test_validate_nm_rejects_odd_n_minus_m():
    with pytest.raises(ValueError):
        validate_nm(3, 0)  # n - |m| = 3, odd


def test_validate_nm_accepts_legal_pairs():
    for n, m in [(0, 0), (1, 1), (1, -1), (2, 0), (2, 2), (4, 0), (4, 4)]:
        validate_nm(n, m)  # must not raise


# --------------------------------------------------------------------- Noll indexing
# Hand calculation (Noll 1976, Table I):
# j : 1     2      3      4     5      6     7      8     9      10    11
# n : 0     1      1      2     2      2     3      3     3       3     4
# m : 0    +1     -1      0    -2     +2    -1     +1    -3      +3     0


@pytest.mark.parametrize(
    "j,expected",
    [
        (1, (0, 0)),
        (2, (1, 1)),
        (3, (1, -1)),
        (4, (2, 0)),
        (5, (2, -2)),
        (6, (2, 2)),
        (7, (3, -1)),
        (8, (3, 1)),
        (9, (3, -3)),
        (10, (3, 3)),
        (11, (4, 0)),
    ],
)
def test_noll_to_nm_matches_noll_table_i(j, expected):
    assert noll_to_nm(j) == expected


def test_noll_to_nm_rejects_j_less_than_one():
    with pytest.raises(ValueError):
        noll_to_nm(0)


def test_noll_to_nm_rejects_non_integer():
    with pytest.raises(TypeError):
        noll_to_nm(2.5)


def test_radial_order_from_noll_hand_calc():
    # j=1..3 -> n=0 (piston,1 mode), j=2,3 -> wait j=1 alone is n=0, then n=1 spans j=2,3.
    assert radial_order_from_noll(1) == 0
    assert radial_order_from_noll(2) == 1
    assert radial_order_from_noll(3) == 1
    assert radial_order_from_noll(4) == 2
    assert radial_order_from_noll(11) == 4


@given(st.integers(min_value=1, max_value=300))
def test_noll_nm_round_trip(j):
    n, m = noll_to_nm(j)
    assert nm_to_noll(n, m) == j


@given(
    st.integers(min_value=0, max_value=15).flatmap(
        lambda n: st.tuples(st.just(n), st.integers(min_value=-n, max_value=n))
    )
)
def test_nm_noll_round_trip_over_legal_pairs(nm):
    n, m = nm
    if (n - abs(m)) % 2 != 0:
        return
    j = nm_to_noll(n, m)
    assert noll_to_nm(j) == (n, m)


def test_nm_to_noll_sign_convention_even_j_is_cosine():
    # Within an order, even j carries m > 0 (cosine); e.g. j=2 -> m=+1, j=6 -> m=+2.
    assert nm_to_noll(1, 1) == 2
    assert nm_to_noll(1, -1) == 3
    assert nm_to_noll(2, 2) == 6
    assert nm_to_noll(2, -2) == 5


# --------------------------------------------------------------------- normalization


def test_normalization_hand_values():
    # N_0^0 = sqrt(1) = 1; N_1^1 = sqrt(2*2) = 2; N_2^0 = sqrt(3)
    assert normalization(0, 0) == pytest.approx(1.0)
    assert normalization(1, 1) == pytest.approx(2.0)
    assert normalization(2, 0) == pytest.approx(np.sqrt(3.0))


def test_normalization_unnormalized_is_one():
    assert normalization(4, 2, normalized=False) == 1.0


# --------------------------------------------------------------------- radial polynomial known closed forms


def test_radial_polynomial_known_closed_forms():
    rho = np.linspace(0, 1, 11)
    # R_0^0 = 1
    np.testing.assert_allclose(radial_polynomial(0, 0, rho), np.ones_like(rho))
    # R_1^1 = rho
    np.testing.assert_allclose(radial_polynomial(1, 1, rho), rho)
    # R_2^0 = 2 rho^2 - 1
    np.testing.assert_allclose(radial_polynomial(2, 0, rho), 2 * rho**2 - 1)
    # R_2^2 = rho^2
    np.testing.assert_allclose(radial_polynomial(2, 2, rho), rho**2)
    # R_4^0 = 6 rho^4 - 6 rho^2 + 1
    np.testing.assert_allclose(radial_polynomial(4, 0, rho), 6 * rho**4 - 6 * rho**2 + 1)


def test_radial_polynomial_rejects_negative_rho():
    with pytest.raises(ValueError):
        radial_polynomial(2, 0, -0.1)


@given(st.integers(min_value=0, max_value=10))
@settings(max_examples=20)
def test_radial_polynomial_edge_value_is_one(n):
    # R_n^m(1) = 1 for every legal (n, m) -- Born & Wolf property.
    for m in range(-n, n + 1, 2):
        assert radial_polynomial(n, m, 1.0) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------- zernike / gradient known values


def test_defocus_known_value():
    # Z_2^0 normalised = sqrt(3) * (2 rho^2 - 1); at rho=0 -> -sqrt(3)
    val = zernike(2, 0, 0.0, 0.0)
    assert val == pytest.approx(-np.sqrt(3.0))


def test_tilt_x_gradient_is_constant_two():
    # Noll j=2 (n=1, m=1): Z = 2x (normalised), so dZ/dx = 2, dZ/dy = 0 everywhere.
    x = np.array([-0.5, 0.0, 0.3, 0.9])
    y = np.array([0.2, 0.0, -0.4, 0.1])
    gx, gy = zernike_gradient_noll(2, x, y)
    np.testing.assert_allclose(gx, 2.0)
    np.testing.assert_allclose(gy, 0.0, atol=1e-12)


def test_defocus_gradient_known_value():
    # j=4 (n=2, m=0): Z = sqrt(3)(2 rho^2 - 1) = sqrt(3)(2x^2+2y^2-1)
    # dZ/dx = 4 sqrt(3) x
    x = np.array([0.3, -0.2, 0.0])
    y = np.array([0.1, 0.5, 0.0])
    gx, gy = zernike_gradient_noll(4, x, y)
    np.testing.assert_allclose(gx, 4 * np.sqrt(3.0) * x, atol=1e-10)
    np.testing.assert_allclose(gy, 4 * np.sqrt(3.0) * y, atol=1e-10)


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.8, 0.8, 20)
    y = rng.uniform(-0.8, 0.8, 20)
    h = 1e-6
    for n, m in [(2, -2), (3, 1), (4, 4), (5, -3)]:
        gx, gy = zernike_gradient(n, m, x, y)
        fd_x = (zernike(n, m, np.hypot(x + h, y), np.arctan2(y, x + h)) - zernike(n, m, np.hypot(x, y), np.arctan2(y, x))) / h
        fd_y = (zernike(n, m, np.hypot(x, y + h), np.arctan2(y + h, x)) - zernike(n, m, np.hypot(x, y), np.arctan2(y, x))) / h
        np.testing.assert_allclose(gx, fd_x, atol=2e-3, rtol=2e-3)
        np.testing.assert_allclose(gy, fd_y, atol=2e-3, rtol=2e-3)


def test_gradient_at_origin_is_finite_no_nan():
    for n, m in [(1, 1), (1, -1), (3, 1), (3, -3), (5, 5)]:
        gx, gy = zernike_gradient(n, m, 0.0, 0.0)
        assert np.isfinite(gx)
        assert np.isfinite(gy)


def test_zernike_noll_j1_is_piston_constant():
    x = np.array([-0.5, 0.0, 0.9])
    y = np.array([0.1, 0.0, -0.2])
    vals = zernike_noll(1, np.hypot(x, y), np.arctan2(y, x))
    np.testing.assert_allclose(vals, 1.0)


# --------------------------------------------------------------------- orthonormality (numerical integral)


def test_orthonormality_low_order_modes():
    # (1/pi) int Zi Zj dA = delta_ij (unit disc, area element dA); approximate
    # the integral by a fine regular-grid Riemann sum, dA = grid_spacing^2.
    n_pix = 400
    x, y, mask = unit_disc_grid(n_pix)
    d_area = (2.0 / (n_pix - 1)) ** 2
    rho, theta = np.hypot(x, y)[mask], np.arctan2(y, x)[mask]
    js = [2, 3, 4, 5, 6]
    vals = {j: zernike_noll(j, rho, theta) for j in js}
    for j in js:
        for k in js:
            integral = np.sum(vals[j] * vals[k]) * d_area / np.pi
            expected = 1.0 if j == k else 0.0
            assert integral == pytest.approx(expected, abs=0.03)


# --------------------------------------------------------------------- unit_disc_grid


def test_unit_disc_grid_rejects_small_n():
    with pytest.raises(ValueError):
        unit_disc_grid(1)


def test_unit_disc_grid_rejects_non_integer():
    with pytest.raises(TypeError):
        unit_disc_grid(4.5)


def test_unit_disc_grid_mask_shape_and_bounds():
    x, y, mask = unit_disc_grid(20)
    assert x.shape == (20, 20)
    assert mask.dtype == bool
    assert np.all(np.hypot(x[mask], y[mask]) <= 1.0 + 1e-12)


# --------------------------------------------------------------------- basis / slope matrix


def test_zernike_basis_matrix_rejects_empty_indices():
    with pytest.raises(ValueError):
        zernike_basis_matrix([], [0.0], [0.0])


def test_zernike_basis_matrix_rejects_mismatched_xy():
    with pytest.raises(ValueError):
        zernike_basis_matrix([2], [0.0, 0.1], [0.0])


def test_zernike_slope_matrix_shape():
    x = np.linspace(-0.5, 0.5, 5)
    y = np.zeros(5)
    mat = zernike_slope_matrix([2, 3, 4], x, y)
    assert mat.shape == (10, 3)


def test_zernike_slope_matrix_column_matches_analytic_gradient():
    x = np.array([0.1, -0.2, 0.3])
    y = np.array([0.05, 0.4, -0.1])
    mat = zernike_slope_matrix([4], x, y)
    gx, gy = zernike_gradient_noll(4, x, y)
    np.testing.assert_allclose(mat[:3, 0], gx)
    np.testing.assert_allclose(mat[3:, 0], gy)


# --------------------------------------------------------------------- fit_zernike


def test_fit_zernike_recovers_known_coefficients_noise_free():
    x, y, mask = unit_disc_grid(60)
    xm, ym = x[mask], y[mask]
    noll = [2, 3, 4, 5, 6, 7, 8]
    true_coeffs = np.array([0.3, -0.1, 0.2, 0.05, -0.15, 0.02, 0.08])
    values = zernike_basis_matrix(noll, xm, ym) @ true_coeffs
    fitted = fit_zernike(noll, xm, ym, values)
    np.testing.assert_allclose(fitted, true_coeffs, atol=1e-9)


def test_fit_zernike_rejects_underdetermined_system():
    with pytest.raises(ValueError):
        fit_zernike([2, 3, 4, 5, 6], [0.1, 0.2], [0.0, 0.1], [0.0, 0.0])


def test_fit_zernike_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        fit_zernike([2], [0.1, 0.2], [0.0, 0.1], [0.0, 0.0, 0.1])
