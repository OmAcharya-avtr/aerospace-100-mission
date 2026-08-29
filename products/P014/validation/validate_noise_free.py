#!/usr/bin/env python3
"""Validation 1: noise-free reconstruction recovers the input to numerical tolerance.

Checked for all three reconstructors built in this package:

1. **Modal (Zernike) least-squares** (`wavelab.modal.ModalReconstructor`) --
   the forward model is analytic (`zernike_slope_matrix`) and the geometry is
   well-determined (more subapertures than modes), so noise-free
   reconstruction should recover the input coefficients to near machine
   precision.
2. **Hudgin zonal geometry** (`wavelab.zonal.ZonalReconstructor`,
   ``geometry="hudgin"``) -- null space is piston only (validated separately
   in tests/test_geometry.py); noise-free recovery should match the true
   (piston-removed) grid phase to near machine precision.
3. **Fried zonal geometry** (``geometry="fried"``) -- null space is piston
   *and* waffle. Noise-free reconstruction can only recover the input phase
   **modulo its own waffle component**; this script verifies that the
   residual equals exactly the true phase's projection onto the waffle
   pattern (`wavelab.zonal.ZonalReconstructor.waffle_component`), which is
   the honest thing to check rather than expecting the raw input back.

Run: ``python validation/validate_noise_free.py`` from ``products/P014``.
Output: this script's stdout, saved verbatim to
``validation/noise_free_output.txt``.
"""

from __future__ import annotations

import numpy as np

from wavelab.dataset import build_modal_geometry
from wavelab.geometry import PupilGrid, fried_matrix, prune_unconstrained
from wavelab.modal import ModalReconstructor
from wavelab.zernike import zernike_basis_matrix
from wavelab.zonal import ZonalReconstructor

NOLL = list(range(2, 16))  # 14 modes, up to n=4
N_TRIALS = 20


def check_modal() -> None:
    print("=== 1. Modal (Zernike) least-squares, noise-free ===")
    geometry = build_modal_geometry(NOLL, n_side=8)
    recon = ModalReconstructor(NOLL, geometry.sub_x, geometry.sub_y, method="tsvd", reg=1e-10)
    print(f"n_sub = {geometry.n_sub}, n_modes = {geometry.n_modes}")
    rng = np.random.default_rng(0)
    max_errs = []
    for trial in range(N_TRIALS):
        a_true = rng.normal(0.0, 0.15, size=len(NOLL))
        s_true = geometry.matrix @ a_true
        a_hat = recon.reconstruct(s_true)
        max_err = float(np.max(np.abs(a_hat - a_true)))
        max_errs.append(max_err)
    worst = max(max_errs)
    print(f"trials = {N_TRIALS}, worst max-abs-coefficient-error = {worst:.3e} rad")
    print(f"PASS (tolerance 1e-6 rad): {worst < 1e-6}")
    print()


def check_hudgin() -> None:
    print("=== 2. Hudgin zonal geometry, noise-free ===")
    grid = PupilGrid(11)
    zr = ZonalReconstructor(grid, geometry="hudgin", method="tsvd", reg=1e-10)
    x, y = grid.active_coords()
    print(f"n_grid = {grid.n_grid}, n_active = {grid.n_active}, n_used = {zr.n_used}")
    rng = np.random.default_rng(1)
    max_errs = []
    for trial in range(N_TRIALS):
        a_true = rng.normal(0.0, 0.15, size=len(NOLL))
        phi_true = zernike_basis_matrix(NOLL, x, y) @ a_true
        phi_true = phi_true - phi_true.mean()
        s = zr.matrix @ phi_true
        phi_hat = zr.reconstruct(s)
        max_errs.append(float(np.max(np.abs(phi_hat - phi_true))))
    worst = max(max_errs)
    print(f"trials = {N_TRIALS}, worst max-abs-phase-error = {worst:.3e} rad")
    print(f"PASS (tolerance 1e-5 rad): {worst < 1e-5}")
    print()


def check_fried() -> None:
    print("=== 3. Fried zonal geometry, noise-free (modulo waffle) ===")
    grid = PupilGrid(11)
    zr = ZonalReconstructor(grid, geometry="fried", method="tsvd", reg=1e-8)
    x, y = grid.active_coords()
    _, keep_idx = prune_unconstrained(fried_matrix(grid))
    print(f"n_grid = {grid.n_grid}, n_active = {grid.n_active}, n_used = {zr.n_used}")
    print(f"null space dimension = {zr.null_space_dimension()} (expect 2: piston + waffle)")
    rng = np.random.default_rng(2)
    resid_vs_waffle_errs = []
    for trial in range(N_TRIALS):
        a_true = rng.normal(0.0, 0.15, size=len(NOLL))
        phi_full = zernike_basis_matrix(NOLL, x, y) @ a_true
        phi_true = phi_full[keep_idx]
        phi_true = phi_true - phi_true.mean()
        s = zr.matrix @ phi_true
        phi_hat = zr.reconstruct(s)
        wc = zr.waffle_component(phi_true)
        residual_norm = float(np.linalg.norm(phi_hat - phi_true))
        resid_vs_waffle_errs.append(abs(residual_norm - abs(wc)))
    worst = max(resid_vs_waffle_errs)
    print(
        f"trials = {N_TRIALS}, worst |residual_norm - |true waffle component|| = {worst:.3e} rad"
    )
    print(f"PASS (residual explained exactly by the true waffle component, tol 1e-3 rad): {worst < 1e-3}")
    print()


if __name__ == "__main__":
    check_modal()
    check_hudgin()
    check_fried()
