"""Unit, KAT, edge-case tests for wavelab.zonal.ZonalReconstructor."""

from __future__ import annotations

import numpy as np
import pytest

from wavelab.geometry import PupilGrid, fried_matrix, prune_unconstrained
from wavelab.zernike import zernike_noll
from wavelab.zonal import ZonalReconstructor


def test_rejects_bad_geometry():
    with pytest.raises(ValueError):
        ZonalReconstructor(PupilGrid(7), geometry="bogus")


def test_rejects_bad_method():
    with pytest.raises(ValueError):
        ZonalReconstructor(PupilGrid(7), geometry="hudgin", method="bogus")


def test_reconstruct_rejects_wrong_slope_shape():
    zr = ZonalReconstructor(PupilGrid(7), geometry="hudgin")
    with pytest.raises(ValueError):
        zr.reconstruct(np.zeros(zr.n_slopes + 1))


def test_reconstruct_rejects_non_finite():
    zr = ZonalReconstructor(PupilGrid(7), geometry="hudgin")
    s = np.zeros(zr.n_slopes)
    s[0] = np.inf
    with pytest.raises(ValueError):
        zr.reconstruct(s)


def test_n_used_le_n_active_and_hudgin_keeps_all():
    grid = PupilGrid(9)
    zr_h = ZonalReconstructor(grid, geometry="hudgin")
    assert zr_h.n_used == grid.n_active
    zr_f = ZonalReconstructor(grid, geometry="fried")
    assert zr_f.n_used <= grid.n_active


def test_hudgin_noise_free_recovers_known_ramp_exactly():
    grid = PupilGrid(9)
    zr = ZonalReconstructor(grid, geometry="hudgin", method="tsvd", reg=1e-10)
    x, _y = grid.active_coords()
    phi_true = 0.6 * x  # pure x-ramp, no piston/waffle ambiguity for Hudgin
    phi_true = phi_true - phi_true.mean()
    s = zr.matrix @ phi_true
    phi_hat = zr.reconstruct(s)
    np.testing.assert_allclose(phi_hat, phi_true, atol=1e-8)


def test_hudgin_noise_free_recovers_zernike_combination():
    grid = PupilGrid(11)
    zr = ZonalReconstructor(grid, geometry="hudgin", method="tsvd", reg=1e-10)
    x, y = grid.active_coords()
    rho, theta = np.hypot(x, y), np.arctan2(y, x)
    phi_true = 0.3 * zernike_noll(2, rho, theta) - 0.15 * zernike_noll(4, rho, theta)
    phi_true = phi_true - phi_true.mean()
    s = zr.matrix @ phi_true
    phi_hat = zr.reconstruct(s)
    np.testing.assert_allclose(phi_hat, phi_true, atol=1e-7)


def test_fried_noise_free_recovers_phase_modulo_waffle():
    # The Fried geometry cannot see the waffle component (module docstring);
    # recovery must match the true phase with its own waffle component
    # removed, to numerical tolerance -- not the raw true phase.
    grid = PupilGrid(11)
    zr = ZonalReconstructor(grid, geometry="fried", method="tsvd", reg=1e-8)
    x, y = grid.active_coords()
    _, keep_idx = prune_unconstrained(fried_matrix(grid))
    rho, theta = np.hypot(x, y), np.arctan2(y, x)
    phi_full = 0.3 * zernike_noll(2, rho, theta) - 0.15 * zernike_noll(4, rho, theta)
    phi_true = phi_full[keep_idx]
    phi_true = phi_true - phi_true.mean()

    s = zr.matrix @ phi_true
    phi_hat = zr.reconstruct(s)
    wc = zr.waffle_component(phi_true)
    residual = phi_hat - phi_true
    # Residual must equal exactly -wc * (unit waffle pattern), i.e. its norm
    # equals |wc| and it has zero component along everything else.
    assert np.linalg.norm(residual) == pytest.approx(abs(wc), abs=2e-3)
    assert zr.waffle_component(phi_hat) == pytest.approx(0.0, abs=2e-3)


def test_fried_reconstruction_of_pure_waffle_input_is_zero():
    grid = PupilGrid(9)
    zr = ZonalReconstructor(grid, geometry="fried", method="tsvd", reg=1e-8)
    n = grid.n_grid
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
    checker = np.where((ii + jj) % 2 == 0, 1.0, -1.0)
    w = checker[grid.mask][zr.keep_idx]
    s = zr.matrix @ w  # pure waffle produces exactly zero Fried slopes
    np.testing.assert_allclose(s, 0.0, atol=1e-10)
    phi_hat = zr.reconstruct(s)
    np.testing.assert_allclose(phi_hat, 0.0, atol=1e-8)


def test_null_space_dimension_hudgin_vs_fried():
    grid = PupilGrid(9)
    zr_h = ZonalReconstructor(grid, geometry="hudgin", method="tsvd", reg=1e-6)
    zr_f = ZonalReconstructor(grid, geometry="fried", method="tsvd", reg=1e-6)
    assert zr_h.null_space_dimension() == 1
    assert zr_f.null_space_dimension() == 2


def test_waffle_component_zero_for_hudgin():
    grid = PupilGrid(9)
    zr = ZonalReconstructor(grid, geometry="hudgin")
    assert zr.waffle_component(np.ones(zr.n_used)) == 0.0


def test_reconstruct_output_is_piston_free():
    grid = PupilGrid(9)
    for geom in ("hudgin", "fried"):
        zr = ZonalReconstructor(grid, geometry=geom, method="tsvd", reg=1e-8)
        rng = np.random.default_rng(0)
        s = rng.normal(size=zr.n_slopes)
        phi_hat = zr.reconstruct(s)
        assert phi_hat.mean() == pytest.approx(0.0, abs=1e-9)


def test_tikhonov_reconstruction_runs_and_is_piston_free():
    grid = PupilGrid(9)
    zr = ZonalReconstructor(grid, geometry="fried", method="tikhonov", reg=0.1)
    rng = np.random.default_rng(1)
    s = rng.normal(size=zr.n_slopes)
    phi_hat = zr.reconstruct(s)
    assert phi_hat.shape == (zr.n_used,)
    assert phi_hat.mean() == pytest.approx(0.0, abs=1e-9)
