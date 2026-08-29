"""Tests for waveforge.zernike.

Known answers are hand-evaluated closed forms from Born & Wolf, Appendix VII,
in the Noll orthonormal convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from waveforge.pupil import PupilGrid
from waveforge.zernike import (
    fit_zernike,
    n_modes_for_order,
    nm_to_noll,
    noll_indices,
    noll_to_nm,
    radial_polynomial,
    zernike_basis,
    zernike_cartesian,
    zernike_gradient_basis,
    zernike_mode_count_check,
    zernike_norm,
    zernike_polar,
)

# Noll (1976) Table 1, indices 1-15.
NOLL_TABLE = {
    1: (0, 0),
    2: (1, 1),
    3: (1, -1),
    4: (2, 0),
    5: (2, -2),
    6: (2, 2),
    7: (3, -1),
    8: (3, 1),
    9: (3, -3),
    10: (3, 3),
    11: (4, 0),
    12: (4, 2),
    13: (4, -2),
    14: (4, 4),
    15: (4, -4),
}


class TestIndexing:
    @pytest.mark.parametrize(("j", "nm"), sorted(NOLL_TABLE.items()))
    def test_noll_table(self, j, nm):
        assert noll_to_nm(j) == nm

    @pytest.mark.parametrize(("j", "nm"), sorted(NOLL_TABLE.items()))
    def test_round_trip(self, j, nm):
        assert nm_to_noll(*nm) == j

    def test_round_trip_many(self):
        for j in range(1, 200):
            assert nm_to_noll(*noll_to_nm(j)) == j

    def test_order_block_boundaries(self):
        # order n occupies n(n+1)/2 + 1 .. (n+1)(n+2)/2
        for n in range(6):
            first = n * (n + 1) // 2 + 1
            last = (n + 1) * (n + 2) // 2
            assert noll_to_nm(first)[0] == n
            assert noll_to_nm(last)[0] == n
            assert last - first + 1 == n + 1

    def test_even_j_is_cosine(self):
        for j in range(2, 60):
            n, m = noll_to_nm(j)
            if m != 0:
                assert (m > 0) == (j % 2 == 0)

    @pytest.mark.parametrize("j", [0, -1, 1.5])
    def test_invalid_index(self, j):
        with pytest.raises(ValueError, match="Noll index"):
            noll_to_nm(j)

    @pytest.mark.parametrize(("n", "m"), [(2, 1), (1, 3), (2, 3)])
    def test_invalid_nm(self, n, m):
        with pytest.raises(ValueError, match="invalid Zernike pair"):
            nm_to_noll(n, m)

    def test_negative_n_rejected(self):
        with pytest.raises(ValueError, match="radial order"):
            nm_to_noll(-1, 0)

    def test_non_integer_m_rejected(self):
        with pytest.raises(ValueError, match="azimuthal order"):
            nm_to_noll(2, 0.5)

    def test_noll_indices_length(self):
        assert len(noll_indices(15)) == 15
        assert noll_indices(3) == [(0, 0), (1, 1), (1, -1)]

    def test_noll_indices_rejects_zero(self):
        with pytest.raises(ValueError, match="j_max"):
            noll_indices(0)

    def test_n_modes_for_order(self):
        assert n_modes_for_order(0) == 1
        assert n_modes_for_order(1) == 3
        assert n_modes_for_order(4) == 15

    def test_n_modes_rejects_negative(self):
        with pytest.raises(ValueError, match="n_max"):
            n_modes_for_order(-1)

    def test_mode_count_check(self):
        assert zernike_mode_count_check(15) == 4
        assert zernike_mode_count_check(6) == 2


class TestRadialPolynomials:
    def test_r_at_unity_is_one(self):
        for n, m in NOLL_TABLE.values():
            if n == 0:
                continue
            assert radial_polynomial(n, m, 1.0) == pytest.approx(1.0)

    def test_odd_parity_returns_zero(self):
        assert np.allclose(radial_polynomial(2, 1, np.linspace(0, 1, 7)), 0.0)

    def test_bounded_on_disc(self):
        rho = np.linspace(0.0, 1.0, 101)
        for n in range(8):
            for m in range(-n, n + 1):
                if (n - abs(m)) % 2:
                    continue
                assert np.max(np.abs(radial_polynomial(n, m, rho))) <= 1.0 + 1e-12

    def test_defocus_closed_form(self):
        # R_2^0 = 2 rho^2 - 1
        rho = np.linspace(0, 1, 11)
        assert np.allclose(radial_polynomial(2, 0, rho), 2 * rho**2 - 1)

    def test_coma_closed_form(self):
        # R_3^1 = 3 rho^3 - 2 rho
        rho = np.linspace(0, 1, 11)
        assert np.allclose(radial_polynomial(3, 1, rho), 3 * rho**3 - 2 * rho)

    def test_spherical_closed_form(self):
        # R_4^0 = 6 rho^4 - 6 rho^2 + 1
        rho = np.linspace(0, 1, 11)
        assert np.allclose(radial_polynomial(4, 0, rho), 6 * rho**4 - 6 * rho**2 + 1)

    def test_trefoil_closed_form(self):
        rho = np.linspace(0, 1, 11)
        assert np.allclose(radial_polynomial(3, 3, rho), rho**3)


class TestNormalisation:
    def test_norm_values(self):
        assert zernike_norm(0, 0) == pytest.approx(1.0)
        assert zernike_norm(1, 1) == pytest.approx(2.0)  # sqrt(2*2)
        assert zernike_norm(2, 0) == pytest.approx(np.sqrt(3.0))
        assert zernike_norm(2, 2) == pytest.approx(np.sqrt(6.0))

    def test_piston_is_unity(self):
        rho = np.linspace(0, 1, 5)
        theta = np.zeros_like(rho)
        assert np.allclose(zernike_polar(1, rho, theta), 1.0)

    def test_tilt_closed_form(self):
        # Z_2 = 2 rho cos(theta) = 2 x
        x = np.linspace(-0.7, 0.7, 9)
        y = np.zeros_like(x)
        assert np.allclose(zernike_cartesian(2, x, y), 2 * x)

    def test_tilt_y_closed_form(self):
        # Z_3 = 2 rho sin(theta) = 2 y
        y = np.linspace(-0.7, 0.7, 9)
        x = np.zeros_like(y)
        assert np.allclose(zernike_cartesian(3, x, y), 2 * y)

    def test_defocus_closed_form(self):
        # Z_4 = sqrt(3) (2 rho^2 - 1)
        rho = np.linspace(0, 1, 9)
        theta = np.full_like(rho, 0.3)
        assert np.allclose(zernike_polar(4, rho, theta), np.sqrt(3) * (2 * rho**2 - 1))

    def test_astigmatism_closed_form(self):
        # Z_6 = sqrt(6) rho^2 cos(2 theta)
        rho, theta = np.array([0.4, 0.9]), np.array([0.2, 1.1])
        assert np.allclose(zernike_polar(6, rho, theta), np.sqrt(6) * rho**2 * np.cos(2 * theta))

    def test_astigmatism_sine_closed_form(self):
        rho, theta = np.array([0.4, 0.9]), np.array([0.2, 1.1])
        assert np.allclose(zernike_polar(5, rho, theta), np.sqrt(6) * rho**2 * np.sin(2 * theta))

    def test_spherical_closed_form(self):
        rho = np.array([0.0, 0.5, 1.0])
        theta = np.zeros_like(rho)
        expected = np.sqrt(5) * (6 * rho**4 - 6 * rho**2 + 1)
        assert np.allclose(zernike_polar(11, rho, theta), expected)


class TestBasis:
    def test_shape_with_piston(self):
        grid = PupilGrid(16, 1.0)
        rho, theta = grid.polar()
        basis = zernike_basis(10, rho, theta, mask=grid.mask)
        assert basis.shape == (10, grid.n_valid)

    def test_shape_without_piston(self):
        grid = PupilGrid(16, 1.0)
        rho, theta = grid.polar()
        basis = zernike_basis(10, rho, theta, mask=grid.mask, include_piston=False)
        assert basis.shape == (9, grid.n_valid)

    def test_orthonormality_on_fine_grid(self):
        grid = PupilGrid(128, 1.0)
        rho, theta = grid.polar()
        basis = zernike_basis(15, rho, theta, mask=grid.mask)
        gram = basis @ basis.T / grid.n_valid
        assert np.max(np.abs(gram - np.eye(15))) < 0.02

    def test_diagonal_close_to_one(self):
        grid = PupilGrid(128, 1.0)
        rho, theta = grid.polar()
        basis = zernike_basis(10, rho, theta, mask=grid.mask)
        diag = np.diag(basis @ basis.T / grid.n_valid)
        assert np.allclose(diag, 1.0, atol=0.02)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="rho shape"):
            zernike_basis(3, np.zeros(4), np.zeros(5))

    def test_mask_shape_mismatch(self):
        with pytest.raises(ValueError, match="mask shape"):
            zernike_basis(3, np.zeros(4), np.zeros(4), mask=np.zeros(5, dtype=bool))

    def test_j_max_too_small_without_piston(self):
        with pytest.raises(ValueError, match="j_max"):
            zernike_basis(1, np.zeros(4), np.zeros(4), include_piston=False)


class TestGradients:
    def test_tilt_gradient_is_constant(self):
        x = np.linspace(-0.8, 0.8, 9)
        xx, yy = np.meshgrid(x, x)
        dzdx, dzdy = zernike_gradient_basis(3, xx, yy)
        # Z_2 = 2x -> dZ/dx = 2, dZ/dy = 0
        assert np.allclose(dzdx[1], 2.0)
        assert np.allclose(dzdy[1], 0.0)
        # Z_3 = 2y
        assert np.allclose(dzdx[2], 0.0)
        assert np.allclose(dzdy[2], 2.0)

    def test_defocus_gradient_closed_form(self):
        x = np.linspace(-0.8, 0.8, 9)
        xx, yy = np.meshgrid(x, x)
        dzdx, dzdy = zernike_gradient_basis(4, xx, yy)
        # Z_4 = sqrt(3)(2 rho^2 - 1) = sqrt(3)(2x^2 + 2y^2 - 1)
        assert np.allclose(dzdx[3], np.sqrt(3) * 4 * xx.ravel())
        assert np.allclose(dzdy[3], np.sqrt(3) * 4 * yy.ravel())

    def test_no_nan_at_origin(self):
        dzdx, dzdy = zernike_gradient_basis(21, np.zeros(1), np.zeros(1))
        assert np.all(np.isfinite(dzdx))
        assert np.all(np.isfinite(dzdy))

    def test_matches_finite_differences(self):
        rng = np.random.default_rng(3)
        pts = rng.uniform(-0.6, 0.6, size=(2, 40))
        h = 1e-6
        dzdx, dzdy = zernike_gradient_basis(21, pts[0], pts[1])
        for j in range(1, 22):
            fx = (
                zernike_cartesian(j, pts[0] + h, pts[1])
                - zernike_cartesian(j, pts[0] - h, pts[1])
            ) / (2 * h)
            fy = (
                zernike_cartesian(j, pts[0], pts[1] + h)
                - zernike_cartesian(j, pts[0], pts[1] - h)
            ) / (2 * h)
            assert np.max(np.abs(dzdx[j - 1] - fx)) < 1e-5
            assert np.max(np.abs(dzdy[j - 1] - fy)) < 1e-5

    def test_piston_gradient_is_zero(self):
        dzdx, dzdy = zernike_gradient_basis(1, np.linspace(-1, 1, 5), np.zeros(5))
        assert np.allclose(dzdx[0], 0.0)
        assert np.allclose(dzdy[0], 0.0)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="x shape"):
            zernike_gradient_basis(3, np.zeros(4), np.zeros(5))

    def test_mask_shape_mismatch(self):
        with pytest.raises(ValueError, match="mask shape"):
            zernike_gradient_basis(3, np.zeros(4), np.zeros(4), mask=np.zeros(5, dtype=bool))

    def test_j_max_validation(self):
        with pytest.raises(ValueError, match="j_max"):
            zernike_gradient_basis(0, np.zeros(4), np.zeros(4))


class TestFitting:
    def test_recovers_single_mode(self):
        grid = PupilGrid(64, 1.0)
        rho, theta = grid.polar()
        phase = np.zeros((64, 64))
        phase[grid.mask] = 0.37 * zernike_polar(6, rho[grid.mask], theta[grid.mask])
        coeffs = fit_zernike(phase, rho, theta, 10, mask=grid.mask)
        assert coeffs[5] == pytest.approx(0.37, rel=1e-9)
        assert np.max(np.abs(np.delete(coeffs, 5))) < 1e-9

    def test_recovers_combination(self):
        grid = PupilGrid(64, 1.0)
        rho, theta = grid.polar()
        truth = np.array([0.0, 0.2, -0.1, 0.05, 0.0, 0.3, 0.0, 0.0, 0.0, -0.02])
        phase = np.zeros((64, 64))
        for j, a in enumerate(truth, start=1):
            phase[grid.mask] += a * zernike_polar(j, rho[grid.mask], theta[grid.mask])
        coeffs = fit_zernike(phase, rho, theta, 10, mask=grid.mask)
        assert np.allclose(coeffs, truth, atol=1e-9)

    def test_piston_is_recovered_when_included(self):
        grid = PupilGrid(32, 1.0)
        rho, theta = grid.polar()
        phase = np.zeros((32, 32))
        phase[grid.mask] = 1.0 + 0.5 * zernike_polar(4, rho[grid.mask], theta[grid.mask])
        coeffs = fit_zernike(phase, rho, theta, 6, mask=grid.mask)
        assert coeffs[0] == pytest.approx(1.0, rel=1e-9)
        assert coeffs[3] == pytest.approx(0.5, rel=1e-9)

    def test_without_piston_on_piston_free_phase(self):
        grid = PupilGrid(32, 1.0)
        rho, theta = grid.polar()
        phase = np.zeros((32, 32))
        phase[grid.mask] = 0.5 * zernike_polar(4, rho[grid.mask], theta[grid.mask])
        coeffs = fit_zernike(phase, rho, theta, 6, mask=grid.mask, include_piston=False)
        assert coeffs[2] == pytest.approx(0.5, rel=1e-6)

    def test_piston_leaks_when_excluded_from_the_basis(self):
        # Discrete sampling means the modes are not exactly orthogonal to
        # piston, so dropping piston from the basis biases the other
        # coefficients. This documents the size of that effect rather than
        # pretending it is absent.
        grid = PupilGrid(32, 1.0)
        rho, theta = grid.polar()
        phase = np.zeros((32, 32))
        phase[grid.mask] = 1.0 + 0.5 * zernike_polar(4, rho[grid.mask], theta[grid.mask])
        coeffs = fit_zernike(phase, rho, theta, 6, mask=grid.mask, include_piston=False)
        assert coeffs[2] != pytest.approx(0.5, rel=1e-6)
        assert coeffs[2] == pytest.approx(0.5, rel=0.05)

    def test_sample_count_mismatch(self):
        grid = PupilGrid(16, 1.0)
        rho, theta = grid.polar()
        with pytest.raises(ValueError, match="samples"):
            fit_zernike(np.zeros((8, 8)), rho, theta, 5)
