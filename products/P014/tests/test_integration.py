"""End-to-end integration tests: screen -> slopes -> reconstruction -> error metric."""

from __future__ import annotations

import numpy as np

from wavelab.dataset import build_modal_geometry, generate_batch
from wavelab.geometry import PupilGrid
from wavelab.modal import ModalReconstructor
from wavelab.ml import ZernikeSlopeEnsemble
from wavelab.screens import kolmogorov_screen
from wavelab.zernike import fit_zernike, unit_disc_grid
from wavelab.zonal import ZonalReconstructor


def test_full_pipeline_screen_to_modal_reconstruction_low_error_at_high_flux():
    noll = list(range(2, 16))
    geometry = build_modal_geometry(noll, 8)
    recon = ModalReconstructor(noll, geometry.sub_x, geometry.sub_y, method="tsvd", reg=1e-6)

    x, y, mask = unit_disc_grid(64)
    screen = kolmogorov_screen(64, 0.15, seed=99)
    a_true = fit_zernike(noll, x[mask], y[mask], screen[mask])
    s_true = geometry.matrix @ a_true

    rng = np.random.default_rng(1)
    from wavelab.noise import add_slope_noise

    s_noisy = add_slope_noise(s_true, 1e6, rng, sigma_ref=1.0, flux_ref=100.0)
    a_hat = recon.reconstruct(s_noisy)

    rms_true = np.sqrt(np.mean(a_true**2))
    rms_err = np.sqrt(np.mean((a_hat - a_true) ** 2))
    assert rms_err < 0.05 * rms_true  # high flux -> reconstruction error << signal


def test_full_pipeline_zonal_reconstruction_of_a_real_screen():
    grid = PupilGrid(9)
    zr = ZonalReconstructor(grid, geometry="hudgin", method="tsvd", reg=1e-8)
    x, y = grid.active_coords()

    # Evaluate the (fitted, truncated) screen directly at the zonal grid
    # points rather than the raw pixel grid, so the "true" phase used for
    # comparison is exactly representable by the Zernike fit used elsewhere
    # in the pipeline.
    x_pix, y_pix, mask = unit_disc_grid(64)
    screen = kolmogorov_screen(64, 0.15, seed=5)
    noll = list(range(2, 16))
    a_true = fit_zernike(noll, x_pix[mask], y_pix[mask], screen[mask])

    from wavelab.zernike import zernike_basis_matrix

    phi_true = zernike_basis_matrix(noll, x, y) @ a_true
    phi_true = phi_true - phi_true.mean()
    s = zr.matrix @ phi_true
    phi_hat = zr.reconstruct(s)
    err = np.sqrt(np.mean((phi_hat - phi_true) ** 2))
    signal = np.sqrt(np.mean(phi_true**2))
    assert err < 0.02 * signal  # Hudgin geometry has no waffle blind spot


def test_full_pipeline_ml_vs_baseline_dropout_sweep_runs_and_reports_both():
    """Integration test of the exact benchmark structure used in validation:
    generate data, fit both reconstructors, sweep dropout, and confirm the
    comparison produces finite, well-formed numbers at every point (this test
    does not assert who wins -- that is an empirical question answered only
    by the full validation run; see validation/VALIDATION.md)."""
    noll = list(range(2, 10))
    geometry = build_modal_geometry(noll, 7)
    baseline = ModalReconstructor(noll, geometry.sub_x, geometry.sub_y, method="tikhonov", reg=1e-3)
    model = ZernikeSlopeEnsemble(geometry.n_sub, geometry.n_modes, n_estimators=3, max_iter=150)

    train = generate_batch(geometry, 150, photon_flux=500.0, dropout_rate=0.2, seed=1)
    model.fit(train.slopes, train.active, train.coeffs)

    for dropout in (0.0, 0.3, 0.6):
        test = generate_batch(geometry, 40, photon_flux=500.0, dropout_rate=dropout, seed=2)
        base_pred = np.array(
            [
                baseline.reconstruct(test.slopes[i], active=test.active[i])
                for i in range(len(test))
            ]
        )
        ml_pred = model.predict(test.slopes, test.active)
        base_rms = np.sqrt(np.mean((base_pred - test.coeffs) ** 2))
        ml_rms = np.sqrt(np.mean((ml_pred - test.coeffs) ** 2))
        assert np.isfinite(base_rms) and base_rms >= 0.0
        assert np.isfinite(ml_rms) and ml_rms >= 0.0
