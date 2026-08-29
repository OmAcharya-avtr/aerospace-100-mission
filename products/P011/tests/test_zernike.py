"""Zernike tests: Noll ordering, closed-form values, orthonormality, statistics."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from waveforge.pupil import Pupil
from waveforge.zernike import (
    NOLL_RESIDUALS,
    ZernikeBasis,
    kolmogorov_residual_variance,
    nm_to_noll,
    noll_mode_variance,
    noll_residual,
    noll_to_nm,
    orthonormality_matrix,
    radial_polynomial,
    zernike,
    zernike_filter,
)


# Noll (1976) Table I, first eleven polynomials. These are the published
# closed forms; each is checked at a hand-chosen point.
@pytest.mark.parametrize(
    "j, rho, theta, expected",
    [
        # Z1 = 1
        (1, 0.5, 0.7, 1.0),
        # Z2 = 2 rho cos(theta); rho = 0.5, theta = 0 -> 1.0
        (2, 0.5, 0.0, 1.0),
        # Z3 = 2 rho sin(theta); rho = 0.5, theta = pi/2 -> 1.0
        (3, 0.5, np.pi / 2, 1.0),
        # Z4 = sqrt(3)(2 rho^2 - 1); rho = 0 -> -sqrt(3)
        (4, 0.0, 0.0, -np.sqrt(3.0)),
        # Z4 at rho = 1 -> +sqrt(3)
        (4, 1.0, 1.234, np.sqrt(3.0)),
        # Z5 = sqrt(6) rho^2 sin(2 theta); rho = 1, theta = pi/4 -> sqrt(6)
        (5, 1.0, np.pi / 4, np.sqrt(6.0)),
        # Z6 = sqrt(6) rho^2 cos(2 theta); rho = 1, theta = 0 -> sqrt(6)
        (6, 1.0, 0.0, np.sqrt(6.0)),
        # Z7 = sqrt(8)(3 rho^3 - 2 rho) sin(theta); rho = 1, theta = pi/2 -> sqrt(8)
        (7, 1.0, np.pi / 2, np.sqrt(8.0)),
        # Z8 = sqrt(8)(3 rho^3 - 2 rho) cos(theta); rho = 1, theta = 0 -> sqrt(8)
        (8, 1.0, 0.0, np.sqrt(8.0)),
        # Z9 = sqrt(8) rho^3 sin(3 theta); rho = 1, theta = pi/6 -> sqrt(8)
        (9, 1.0, np.pi / 6, np.sqrt(8.0)),
        # Z10 = sqrt(8) rho^3 cos(3 theta); rho = 1, theta = 0 -> sqrt(8)
        (10, 1.0, 0.0, np.sqrt(8.0)),
        # Z11 = sqrt(5)(6 rho^4 - 6 rho^2 + 1); rho = 0 -> sqrt(5)
        (11, 0.0, 0.0, np.sqrt(5.0)),
    ],
)
def test_noll_table_i_closed_forms(j, rho, theta, expected):
    assert zernike(j, np.array(rho), np.array(theta)) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "j, nm",
    [
        (1, (0, 0)), (2, (1, 1)), (3, (1, 1)), (4, (2, 0)), (5, (2, 2)), (6, (2, 2)),
        (7, (3, 1)), (8, (3, 1)), (9, (3, 3)), (10, (3, 3)), (11, (4, 0)), (12, (4, 2)),
        (15, (4, 4)), (16, (5, 1)), (21, (5, 5)),
    ],
)
def test_noll_index_mapping(j, nm):
    assert noll_to_nm(j) == nm


def test_nm_to_noll_round_trip():
    for j in range(1, 40):
        n, m = noll_to_nm(j)
        from waveforge.zernike import noll_is_cosine

        assert nm_to_noll(n, m, cosine=noll_is_cosine(j)) == j


def test_radial_polynomial_edge_value_is_one():
    # R_n^m(1) = 1 for every allowed (n, m) -- a standard identity.
    for n in range(0, 9):
        for m in range(n % 2, n + 1, 2):
            assert radial_polynomial(n, m, np.array(1.0)) == pytest.approx(1.0, abs=1e-9)


def test_orthonormality_on_a_fine_grid():
    # Noll (1976) eq. 4: <Z_i Z_j> = delta_ij over the unit disc.
    g = orthonormality_matrix(Pupil(1.0, 256), 21)
    err = np.abs(g - np.eye(21))
    assert err.max() < 5e-3


def test_orthonormality_improves_with_sampling():
    coarse = np.abs(orthonormality_matrix(Pupil(1.0, 64), 15) - np.eye(15)).max()
    fine = np.abs(orthonormality_matrix(Pupil(1.0, 256), 15) - np.eye(15)).max()
    assert fine < coarse


def test_basis_round_trip_is_exact_for_a_modal_wavefront():
    pupil = Pupil(1.0, 128)
    basis = ZernikeBasis(pupil, 15)
    rng = np.random.default_rng(3)
    coeffs = rng.standard_normal(15)
    recovered = basis.project(basis.to_phase(coeffs))
    assert np.allclose(recovered, coeffs, atol=1e-10)
    assert np.abs(basis.residual(basis.to_phase(coeffs))).max() < 1e-10


def test_orthonormalized_basis_is_orthonormal_on_an_annulus():
    pupil = Pupil(1.0, 128, obscuration=0.3)
    basis = ZernikeBasis(pupil, 15, orthonormalize=True)
    g = (basis.matrix.T @ basis.matrix) / basis.matrix.shape[0]
    assert np.abs(g - np.eye(15)).max() < 1e-10


def test_noll_residual_table_and_asymptote():
    assert noll_residual(1) == pytest.approx(1.0299)
    assert noll_residual(21) == pytest.approx(0.0208)
    # Noll (1976) eq. 34 asymptote, used above J = 21.
    assert noll_residual(100) == pytest.approx(0.2944 * 100 ** (-np.sqrt(3) / 2), rel=1e-12)
    assert noll_residual(22) < noll_residual(21)


def test_noll_residuals_are_monotone():
    values = [NOLL_RESIDUALS[j] for j in sorted(NOLL_RESIDUALS)]
    assert all(b < a for a, b in zip(values[:-1], values[1:]))


def test_mode_variance_matches_table_differences():
    # <a_j^2> should equal Delta_(j-1) - Delta_j from Noll's Table IV. The
    # table is quoted to 4 decimal places, so a difference of two entries
    # carries up to 1e-4 of pure rounding error -- which at j = 19
    # (difference 0.0011) is 9 % on its own. The tolerance is therefore the
    # looser of 5 % relative and 1.1e-4 absolute.
    for j in range(2, 22):
        expected = NOLL_RESIDUALS[j - 1] - NOLL_RESIDUALS[j]
        assert noll_mode_variance(j) == pytest.approx(expected, rel=0.05, abs=1.1e-4)


def test_modes_of_equal_radial_order_share_a_variance():
    assert noll_mode_variance(5) == pytest.approx(noll_mode_variance(6), rel=1e-12)
    assert noll_mode_variance(7) == pytest.approx(noll_mode_variance(10), rel=1e-12)


def test_zernike_filter_limits():
    # Piston filter -> 1 at f = 0; every other mode -> 0.
    assert zernike_filter(1, 0.0, 1.0) == pytest.approx(1.0)
    assert zernike_filter(4, 0.0, 1.0) == pytest.approx(0.0)
    # Piston filter is [2 J1(pi D f)/(pi D f)]^2, which is 0 at the first
    # zero of J1, u = 3.8317059702... -> f = 3.8317.../(pi D).
    f0 = 3.8317059702075125 / np.pi
    assert zernike_filter(1, f0, 1.0) == pytest.approx(0.0, abs=1e-18)


@pytest.mark.parametrize("n_modes", [1, 3, 11, 21])
def test_kolmogorov_residual_matches_noll_table(n_modes):
    # Direct integration of the Kolmogorov PSD against the residual Zernike
    # filter, compared with Noll (1976) Table IV. Tolerance 2 %: the published
    # table is 3-figure and the PSD coefficient 0.023 is itself rounded.
    d, r0 = 1.0, 0.2
    got = kolmogorov_residual_variance(n_modes, d, r0)
    ref = noll_residual(n_modes) * (d / r0) ** (5.0 / 3.0)
    assert got == pytest.approx(ref, rel=0.02)


@given(j=st.integers(min_value=1, max_value=45), rho=st.floats(0.0, 1.0))
@settings(max_examples=40, deadline=None)
def test_zernike_is_bounded_on_the_disc(j, rho):
    n, _ = noll_to_nm(j)
    theta = np.linspace(0, 2 * np.pi, 17)
    values = zernike(j, np.full_like(theta, rho), theta)
    # |Z_j| <= sqrt(2(n+1)) on the unit disc, since |R_n^m| <= 1.
    assert np.abs(values).max() <= np.sqrt(2.0 * (n + 1)) + 1e-9


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_noll_index_raises(bad):
    with pytest.raises(ValueError):
        noll_to_nm(bad)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        radial_polynomial(2, 3, np.array(0.5))
    with pytest.raises(ValueError):
        radial_polynomial(3, 2, np.array(0.5))
    with pytest.raises(ValueError):
        noll_mode_variance(1)
    with pytest.raises(ValueError):
        noll_residual(0)
    with pytest.raises(ValueError):
        ZernikeBasis(Pupil(1.0, 16), 0)
    with pytest.raises(ValueError):
        zernike_filter(2, -1.0, 1.0)
    with pytest.raises(ValueError):
        zernike_filter(2, 1.0, 0.0)
    with pytest.raises(ValueError):
        kolmogorov_residual_variance(0, 1.0, 0.2)


def test_basis_shape_validation():
    basis = ZernikeBasis(Pupil(1.0, 16), 6)
    with pytest.raises(ValueError):
        basis.to_phase(np.zeros(5))
    with pytest.raises(ValueError):
        basis.project(np.zeros((8, 8)))
