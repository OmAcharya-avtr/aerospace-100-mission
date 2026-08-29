"""Unit, KAT, edge-case tests for wavelab.dataset."""

from __future__ import annotations

import numpy as np
import pytest

from wavelab.dataset import build_modal_geometry, generate_batch


def test_build_modal_geometry_rejects_piston():
    with pytest.raises(ValueError):
        build_modal_geometry([1, 2], 8)


def test_build_modal_geometry_rejects_too_few_subapertures():
    with pytest.raises(ValueError):
        build_modal_geometry(list(range(2, 40)), 3)


def test_build_modal_geometry_shapes():
    geo = build_modal_geometry(list(range(2, 12)), 8)
    assert geo.n_modes == 10
    assert geo.matrix.shape == (2 * geo.n_sub, 10)


def test_generate_batch_rejects_bad_n_samples():
    geo = build_modal_geometry(list(range(2, 8)), 6)
    with pytest.raises(ValueError):
        generate_batch(geo, 0, photon_flux=100.0, dropout_rate=0.0, seed=0)


def test_generate_batch_rejects_bad_r0_range():
    geo = build_modal_geometry(list(range(2, 8)), 6)
    with pytest.raises(ValueError):
        generate_batch(geo, 2, photon_flux=100.0, dropout_rate=0.0, seed=0, r0_over_d_range=(0.3, 0.1))


def test_generate_batch_shapes():
    geo = build_modal_geometry(list(range(2, 8)), 6)
    batch = generate_batch(geo, 5, photon_flux=1000.0, dropout_rate=0.1, seed=0)
    assert len(batch) == 5
    assert batch.slopes.shape == (5, 2 * geo.n_sub)
    assert batch.active.shape == (5, geo.n_sub)
    assert batch.coeffs.shape == (5, geo.n_modes)


def test_generate_batch_deterministic_given_seed():
    geo = build_modal_geometry(list(range(2, 8)), 6)
    b1 = generate_batch(geo, 4, photon_flux=1000.0, dropout_rate=0.2, seed=5)
    b2 = generate_batch(geo, 4, photon_flux=1000.0, dropout_rate=0.2, seed=5)
    np.testing.assert_array_equal(b1.slopes, b2.slopes)
    np.testing.assert_array_equal(b1.active, b2.active)
    np.testing.assert_array_equal(b1.coeffs, b2.coeffs)


def test_generate_batch_different_seed_differs():
    geo = build_modal_geometry(list(range(2, 8)), 6)
    b1 = generate_batch(geo, 4, photon_flux=1000.0, dropout_rate=0.2, seed=5)
    b2 = generate_batch(geo, 4, photon_flux=1000.0, dropout_rate=0.2, seed=6)
    assert not np.allclose(b1.coeffs, b2.coeffs)


def test_generate_batch_zero_dropout_keeps_all_active():
    geo = build_modal_geometry(list(range(2, 8)), 6)
    batch = generate_batch(geo, 10, photon_flux=1000.0, dropout_rate=0.0, seed=1)
    assert np.all(batch.active)


def test_generate_batch_inactive_rows_are_zeroed():
    geo = build_modal_geometry(list(range(2, 8)), 6)
    batch = generate_batch(geo, 20, photon_flux=1000.0, dropout_rate=0.4, seed=2)
    for i in range(len(batch)):
        row_mask = np.concatenate([batch.active[i], batch.active[i]])
        assert np.all(batch.slopes[i][~row_mask] == 0.0)


def test_generate_batch_higher_flux_gives_smaller_noise_floor():
    # At dropout=0, slopes = s_true + noise; comparing residual of a direct
    # least-squares solve at two flux levels should show the higher-flux
    # batch has a smaller RMS deviation from the noise-free forward model.
    geo = build_modal_geometry(list(range(2, 10)), 8)
    low = generate_batch(geo, 30, photon_flux=50.0, dropout_rate=0.0, seed=3)
    high = generate_batch(geo, 30, photon_flux=50000.0, dropout_rate=0.0, seed=3)
    resid_low = low.slopes - (geo.matrix @ low.coeffs.T).T
    resid_high = high.slopes - (geo.matrix @ high.coeffs.T).T
    assert resid_high.std() < resid_low.std()
