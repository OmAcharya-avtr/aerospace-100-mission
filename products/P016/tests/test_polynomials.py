"""Tests for radial polynomials, normalisation and orthonormality."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from zernkit import (
    normalization,
    radial_coefficients,
    radial_polynomial,
    unit_disc_grid,
    zernike,
    zernike_cartesian,
    zernike_noll,
    zernike_osa,
)
from zernkit.indexing import noll_to_nm

RHO = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
THETA = np.array([0.0, 0.7, 1.3, 2.9, 4.4])


def _legal_nm(max_n: int = 12) -> list[tuple[int, int]]:
    return [(n, m) for n in range(max_n + 1) for m in range(-n, n + 1, 2)]


# --- known answers, hand-written closed forms from Noll (1976) Table I ----


def test_z1_piston() -> None:
    assert np.allclose(zernike_noll(1, RHO, THETA), 1.0)


def test_z2_z3_tip_tilt() -> None:
    # Z2 = 2 rho cos(theta), Z3 = 2 rho sin(theta)
    assert np.allclose(zernike_noll(2, RHO, THETA), 2 * RHO * np.cos(THETA))
    assert np.allclose(zernike_noll(3, RHO, THETA), 2 * RHO * np.sin(THETA))


def test_z4_defocus() -> None:
    # Z4 = sqrt(3) (2 rho^2 - 1); at rho = 0 this is -sqrt(3) = -1.7320508...
    assert np.allclose(zernike_noll(4, RHO, THETA), np.sqrt(3) * (2 * RHO**2 - 1))
    assert zernike_noll(4, 0.0, 0.0) == pytest.approx(-np.sqrt(3.0))
    assert zernike_noll(4, 1.0, 0.0) == pytest.approx(np.sqrt(3.0))


def test_z5_z6_astigmatism() -> None:
    assert np.allclose(zernike_noll(5, RHO, THETA), np.sqrt(6) * RHO**2 * np.sin(2 * THETA))
    assert np.allclose(zernike_noll(6, RHO, THETA), np.sqrt(6) * RHO**2 * np.cos(2 * THETA))


def test_z7_z8_coma() -> None:
    radial = 3 * RHO**3 - 2 * RHO
    assert np.allclose(zernike_noll(7, RHO, THETA), np.sqrt(8) * radial * np.sin(THETA))
    assert np.allclose(zernike_noll(8, RHO, THETA), np.sqrt(8) * radial * np.cos(THETA))


def test_z11_primary_spherical() -> None:
    # Z11 = sqrt(5) (6 rho^4 - 6 rho^2 + 1); by hand at rho = 0.5:
    # 6(0.0625) - 6(0.25) + 1 = 0.375 - 1.5 + 1 = -0.125, times sqrt(5).
    assert np.allclose(zernike_noll(11, RHO, THETA), np.sqrt(5) * (6 * RHO**4 - 6 * RHO**2 + 1))
    assert zernike_noll(11, 0.5, 0.0) == pytest.approx(-0.125 * np.sqrt(5.0))


def test_unnormalized_forms_peak_at_one() -> None:
    for n, m in _legal_nm(8):
        assert radial_polynomial(n, m, 1.0) == pytest.approx(1.0, abs=1e-9)
        assert zernike(n, m, 1.0, 0.0, normalized=False) == pytest.approx(
            np.cos(m * 0.0) if m >= 0 else 0.0, abs=1e-9
        )


def test_normalization_factors() -> None:
    assert normalization(0, 0) == pytest.approx(1.0)
    assert normalization(1, 1) == pytest.approx(2.0)
    assert normalization(2, 0) == pytest.approx(np.sqrt(3.0))
    assert normalization(2, 2) == pytest.approx(np.sqrt(6.0))
    assert normalization(3, 1) == pytest.approx(np.sqrt(8.0))
    assert normalization(4, 0) == pytest.approx(np.sqrt(5.0))
    assert normalization(4, 2) == pytest.approx(np.sqrt(10.0))
    assert normalization(4, 2, normalized=False) == 1.0


def test_radial_coefficients_hand_calculated() -> None:
    # R_4^0 = 6 rho^4 - 6 rho^2 + 1 (Born & Wolf, Principles of Optics, Sec 9.2)
    assert np.allclose(radial_coefficients(4, 0), [1.0, 0.0, -6.0, 0.0, 6.0])
    # R_3^1 = 3 rho^3 - 2 rho
    assert np.allclose(radial_coefficients(3, 1), [0.0, -2.0, 0.0, 3.0])
    # R_2^2 = rho^2
    assert np.allclose(radial_coefficients(2, 2), [0.0, 0.0, 1.0])


def test_radial_coefficients_symmetric_in_sign_of_m() -> None:
    for n, m in _legal_nm(9):
        assert np.allclose(radial_coefficients(n, m), radial_coefficients(n, -m))


def test_osa_indexing_gives_the_same_modes() -> None:
    for j in range(1, 30):
        n, m = noll_to_nm(j)
        from zernkit import nm_to_osa

        assert np.allclose(
            zernike_noll(j, RHO, THETA), zernike_osa(nm_to_osa(n, m), RHO, THETA)
        )


def test_zernike_cartesian_matches_polar() -> None:
    x = np.array([0.0, 0.3, -0.5, 0.2])
    y = np.array([0.0, -0.4, 0.5, 0.9])
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    for n, m in _legal_nm(6):
        assert np.allclose(zernike_cartesian(n, m, x, y), zernike(n, m, rho, theta))


# --- orthonormality -------------------------------------------------------


def test_orthonormality_on_polar_quadrature() -> None:
    """(1/pi) int Z_i Z_j rho drho dtheta = delta_ij, Gauss-Legendre in rho."""
    n_modes = 21
    nodes, weights = np.polynomial.legendre.leggauss(80)
    rho = 0.5 * (nodes + 1.0)
    w_rho = 0.5 * weights
    n_theta = 256
    theta = 2 * np.pi * np.arange(n_theta) / n_theta
    w_theta = 2 * np.pi / n_theta

    rr, tt = np.meshgrid(rho, theta, indexing="ij")
    weight = (w_rho[:, None] * rr) * w_theta / np.pi

    modes = np.array([zernike_noll(j, rr, tt).ravel() for j in range(1, n_modes + 1)])
    gram = (modes * weight.ravel()) @ modes.T
    assert np.max(np.abs(gram - np.eye(n_modes))) < 1e-12


def test_unnormalized_norm_matches_analytic() -> None:
    """int |Z_n^m|^2 with W=1/pi equals 1/N^2 in the unnormalised convention."""
    nodes, weights = np.polynomial.legendre.leggauss(80)
    rho = 0.5 * (nodes + 1.0)
    w_rho = 0.5 * weights
    n_theta = 256
    theta = 2 * np.pi * np.arange(n_theta) / n_theta
    rr, tt = np.meshgrid(rho, theta, indexing="ij")
    weight = (w_rho[:, None] * rr) * (2 * np.pi / n_theta) / np.pi
    for n, m in _legal_nm(6):
        z = zernike(n, m, rr, tt, normalized=False)
        expected = 1.0 / normalization(n, m) ** 2
        assert np.sum(weight * z * z) == pytest.approx(expected, rel=1e-10)


# --- input validation -----------------------------------------------------


def test_invalid_indices_raise() -> None:
    with pytest.raises(ValueError):
        zernike(3, 0, 0.5, 0.0)
    with pytest.raises(ValueError):
        radial_coefficients(2, 4)
    with pytest.raises(ValueError, match="non-negative"):
        radial_polynomial(2, 0, -0.1)


def test_unit_disc_grid_validation() -> None:
    x, y, mask = unit_disc_grid(9)
    assert x.shape == (9, 9)
    assert mask[4, 4] and not mask[0, 0]
    assert mask.sum() < 81
    _, _, strict = unit_disc_grid(9, include_edge=False)
    assert strict.sum() <= mask.sum()
    with pytest.raises(ValueError):
        unit_disc_grid(1)
    with pytest.raises(TypeError):
        unit_disc_grid(9.0)  # type: ignore[arg-type]


# --- property-based tests -------------------------------------------------

_NM = st.integers(min_value=0, max_value=20).flatmap(
    lambda n: st.tuples(st.just(n), st.sampled_from(list(range(-n, n + 1, 2)) or [0]))
)


@given(_NM, st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=250, deadline=None)
def test_property_radial_parity(nm: tuple[int, int], rho: float) -> None:
    """R_n^m has the parity of n: R(-rho) = (-1)^n R(rho)."""
    n, m = nm
    coeffs = radial_coefficients(n, m)
    powers = np.nonzero(coeffs)[0]
    assert np.all(powers % 2 == n % 2)
    assert np.all(powers >= abs(m))
    val = radial_polynomial(n, m, rho)
    mirrored = np.polyval(coeffs[::-1], -rho)
    assert mirrored == pytest.approx((-1) ** n * val, abs=1e-9, rel=1e-9)


@given(_NM)
@settings(max_examples=200, deadline=None)
def test_property_radial_is_zero_when_n_minus_m_odd(nm: tuple[int, int]) -> None:
    """R_n^m is identically zero unless n - m is even -- here enforced as an error."""
    n, _ = nm
    for m_bad in range(-n, n + 1):
        if (n - abs(m_bad)) % 2 == 1:
            with pytest.raises(ValueError, match="even"):
                radial_coefficients(n, m_bad)


@given(_NM)
@settings(max_examples=250, deadline=None)
def test_property_radial_at_one_is_one(nm: tuple[int, int]) -> None:
    """R_n^m(1) = 1 for every legal (n, m)."""
    n, m = nm
    assert radial_polynomial(n, m, 1.0) == pytest.approx(1.0, abs=1e-8)


@given(_NM, st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=250, deadline=None)
def test_property_radial_bounded_on_disc(nm: tuple[int, int], rho: float) -> None:
    """|R_n^m(rho)| <= 1 on the unit disc (Born & Wolf, Sec. 9.2)."""
    n, m = nm
    assert abs(float(radial_polynomial(n, m, rho))) <= 1.0 + 1e-9
