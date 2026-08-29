"""Unit, KAT, edge-case tests for wavelab.modal.ModalReconstructor."""

from __future__ import annotations

import numpy as np
import pytest

from wavelab.geometry import PupilGrid
from wavelab.modal import ModalReconstructor
from wavelab.zernike import zernike_slope_matrix


def _make_layout(n_side=8):
    grid = PupilGrid(n_side)
    sub_x, sub_y = grid.active_coords()
    return sub_x, sub_y


def test_rejects_piston_index():
    sub_x, sub_y = _make_layout()
    with pytest.raises(ValueError):
        ModalReconstructor([1, 2], sub_x, sub_y)


def test_rejects_empty_indices():
    sub_x, sub_y = _make_layout()
    with pytest.raises(ValueError):
        ModalReconstructor([], sub_x, sub_y)


def test_rejects_bad_method():
    sub_x, sub_y = _make_layout()
    with pytest.raises(ValueError):
        ModalReconstructor([2, 3], sub_x, sub_y, method="bogus")


def test_rejects_mismatched_xy():
    with pytest.raises(ValueError):
        ModalReconstructor([2, 3], np.zeros(5), np.zeros(4))


def test_rejects_underdetermined_layout():
    with pytest.raises(ValueError):
        ModalReconstructor(list(range(2, 30)), np.array([0.1, 0.2]), np.array([0.0, 0.0]))


def test_matrix_shape_and_n_sub():
    sub_x, sub_y = _make_layout()
    noll = list(range(2, 12))
    recon = ModalReconstructor(noll, sub_x, sub_y)
    assert recon.n_sub == sub_x.size
    assert recon.matrix.shape == (2 * sub_x.size, len(noll))


def test_reconstruct_rejects_wrong_slope_shape():
    sub_x, sub_y = _make_layout()
    recon = ModalReconstructor([2, 3], sub_x, sub_y)
    with pytest.raises(ValueError):
        recon.reconstruct(np.zeros(3))


def test_reconstruct_rejects_non_finite_slopes():
    sub_x, sub_y = _make_layout()
    recon = ModalReconstructor([2, 3], sub_x, sub_y)
    s = np.zeros(2 * sub_x.size)
    s[0] = np.nan
    with pytest.raises(ValueError):
        recon.reconstruct(s)


def test_reconstruct_rejects_bad_active_mask_shape():
    sub_x, sub_y = _make_layout()
    recon = ModalReconstructor([2, 3], sub_x, sub_y)
    s = np.zeros(2 * sub_x.size)
    with pytest.raises(ValueError):
        recon.reconstruct(s, active=np.ones(3, dtype=bool))


def test_reconstruct_rejects_all_inactive():
    sub_x, sub_y = _make_layout()
    recon = ModalReconstructor([2, 3], sub_x, sub_y)
    s = np.zeros(2 * sub_x.size)
    with pytest.raises(ValueError):
        recon.reconstruct(s, active=np.zeros(sub_x.size, dtype=bool))


def test_noise_free_reconstruction_hand_calc_tilt_only():
    # Two subapertures at x = -0.5 and x = +0.5 (y=0), reconstructing tilt-x
    # (Noll j=2) alone: dZ2/dx = 2 everywhere, so slope = 2*a for every
    # subaperture. With a = 0.3, slope = 0.6 at both; least squares must
    # recover a = 0.3 exactly (well-determined, noise free).
    sub_x = np.array([-0.5, 0.5])
    sub_y = np.array([0.0, 0.0])
    recon = ModalReconstructor([2], sub_x, sub_y, method="tikhonov", reg=0.0)
    a_true = 0.3
    slopes = np.array([2 * a_true, 2 * a_true, 0.0, 0.0])  # x-block then y-block
    a_hat = recon.reconstruct(slopes)
    assert a_hat[0] == pytest.approx(a_true, abs=1e-10)


def test_noise_free_reconstruction_recovers_multi_mode_coefficients():
    sub_x, sub_y = _make_layout(8)
    noll = list(range(2, 16))
    recon = ModalReconstructor(noll, sub_x, sub_y, method="tsvd", reg=1e-10)
    rng = np.random.default_rng(0)
    a_true = rng.normal(0.0, 0.1, size=len(noll))
    matrix = zernike_slope_matrix(noll, sub_x, sub_y)
    slopes = matrix @ a_true
    a_hat = recon.reconstruct(slopes)
    np.testing.assert_allclose(a_hat, a_true, atol=1e-7)


def test_dropout_uses_only_active_rows():
    sub_x, sub_y = _make_layout(8)
    noll = list(range(2, 10))
    recon = ModalReconstructor(noll, sub_x, sub_y, method="tsvd", reg=1e-10)
    rng = np.random.default_rng(1)
    a_true = rng.normal(0.0, 0.1, size=len(noll))
    matrix = zernike_slope_matrix(noll, sub_x, sub_y)
    slopes = matrix @ a_true
    active = np.ones(sub_x.size, dtype=bool)
    active[::3] = False  # drop every third subaperture
    # Corrupt the dropped rows arbitrarily -- reconstruction must ignore them.
    row_mask = np.concatenate([active, active])
    corrupted = slopes.copy()
    corrupted[~row_mask] = 999.0
    a_hat = recon.reconstruct(corrupted, active=active)
    np.testing.assert_allclose(a_hat, a_true, atol=1e-6)


def test_tikhonov_regularization_reduces_noise_sensitivity():
    sub_x, sub_y = _make_layout(8)
    noll = list(range(2, 16))
    matrix = zernike_slope_matrix(noll, sub_x, sub_y)
    rng = np.random.default_rng(2)
    a_true = rng.normal(0.0, 0.1, size=len(noll))
    s_true = matrix @ a_true
    noisy = s_true + rng.normal(0.0, 0.5, size=s_true.shape)

    low_reg = ModalReconstructor(noll, sub_x, sub_y, method="tikhonov", reg=1e-6)
    high_reg = ModalReconstructor(noll, sub_x, sub_y, method="tikhonov", reg=1.0)
    a_low = low_reg.reconstruct(noisy)
    a_high = high_reg.reconstruct(noisy)
    # Heavier regularization shrinks the solution norm (bias-variance trade).
    assert np.linalg.norm(a_high) < np.linalg.norm(a_low)


def test_noise_propagation_coefficients_positive_and_finite():
    sub_x, sub_y = _make_layout(8)
    noll = list(range(2, 12))
    recon = ModalReconstructor(noll, sub_x, sub_y, method="tsvd", reg=1e-6)
    coeffs = recon.noise_propagation()
    assert coeffs.shape == (len(noll),)
    assert np.all(np.isfinite(coeffs))
    assert np.all(coeffs > 0)


def test_noise_propagation_tikhonov_matches_tsvd_at_tiny_reg():
    sub_x, sub_y = _make_layout(8)
    noll = list(range(2, 10))
    tik = ModalReconstructor(noll, sub_x, sub_y, method="tikhonov", reg=1e-8)
    tsvd = ModalReconstructor(noll, sub_x, sub_y, method="tsvd", reg=1e-8)
    np.testing.assert_allclose(tik.noise_propagation(), tsvd.noise_propagation(), rtol=1e-3)
