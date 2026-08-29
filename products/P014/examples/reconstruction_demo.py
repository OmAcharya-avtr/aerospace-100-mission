#!/usr/bin/env python3
"""Example: reconstruct one synthetic Kolmogorov-screen wavefront.

Generates one phase screen, fits it to Zernike coefficients, adds photon
noise and subaperture dropout, reconstructs with the regularized modal
least-squares baseline, and plots the true vs. reconstructed wavefront maps
and their residual over the pupil. Saves
``../screenshots/reconstruction_demo.png``.

Run: ``python examples/reconstruction_demo.py`` from ``products/P014``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wavelab.dataset import build_modal_geometry
from wavelab.modal import ModalReconstructor
from wavelab.noise import add_slope_noise, apply_dropout
from wavelab.screens import kolmogorov_screen
from wavelab.zernike import fit_zernike, unit_disc_grid, zernike_basis_matrix

NOLL = list(range(2, 16))


def main() -> None:
    x, y, mask = unit_disc_grid(64)
    screen = kolmogorov_screen(64, r0_over_d=0.14, seed=7)
    a_true = fit_zernike(NOLL, x[mask], y[mask], screen[mask])

    geometry = build_modal_geometry(NOLL, n_side=10)
    recon = ModalReconstructor(NOLL, geometry.sub_x, geometry.sub_y, method="tikhonov", reg=2e-3)
    s_true = geometry.matrix @ a_true

    rng = np.random.default_rng(3)
    s_noisy = add_slope_noise(s_true, photon_flux=500.0, rng=rng, sigma_ref=1.0, flux_ref=100.0)
    active = apply_dropout(geometry.n_sub, dropout_rate=0.15, rng=rng)
    a_hat = recon.reconstruct(s_noisy, active=active)

    basis = zernike_basis_matrix(NOLL, x[mask], y[mask])
    true_map = np.full(mask.shape, np.nan)
    recon_map = np.full(mask.shape, np.nan)
    true_map[mask] = basis @ a_true
    recon_map[mask] = basis @ a_hat
    resid_map = recon_map - true_map

    rms_true = float(np.sqrt(np.mean(a_true**2)))
    rms_err = float(np.sqrt(np.mean((a_hat - a_true) ** 2)))

    vlim = np.nanmax(np.abs(true_map))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, arr, title in zip(
        axes,
        (true_map, recon_map, resid_map),
        (
            f"true wavefront (RMS = {rms_true:.3f} rad)",
            f"modal LSQ reconstruction\n(active subaps = {int(active.sum())}/{geometry.n_sub})",
            f"residual (RMS err = {rms_err:.3f} rad)",
        ),
    ):
        lim = vlim if arr is not resid_map else np.nanmax(np.abs(resid_map))
        im = ax.imshow(arr, origin="lower", extent=(-1, 1, -1, 1), cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x / pupil radius")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="phase [rad]")
    axes[0].set_ylabel("y / pupil radius")
    fig.suptitle("WaveLab: modal Zernike least-squares reconstruction of a Kolmogorov screen")
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "screenshots"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "reconstruction_demo.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")
    print(f"true coefficient RMS = {rms_true:.4f} rad, reconstruction RMS error = {rms_err:.4f} rad")


if __name__ == "__main__":
    main()
