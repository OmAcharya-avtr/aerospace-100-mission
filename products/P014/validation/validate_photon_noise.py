#!/usr/bin/env python3
"""Validation 2: reconstruction error vs photon flux vs the analytic noise-propagation coefficient.

For i.i.d. slope noise of variance ``sigma_s^2`` (`wavelab.noise.slope_sigma`),
the reconstructed-coefficient variance of a linear pseudo-inverse
reconstructor is predicted by

    Var(a_hat_k) = coeff_k * sigma_s^2

(`wavelab.linalg.noise_propagation_coefficients` /
`wavelab.modal.ModalReconstructor.noise_propagation`; Wallner 1983, *J. Opt.
Soc. Am.* **73**, 1771). This script fixes one true coefficient vector, draws
many independent noise realizations at each of several photon-flux levels,
reconstructs with the regularized modal least-squares baseline, and compares
the empirical per-mode variance (averaged over modes) with the analytic
prediction. It also checks the ``sigma^2 propto 1/N`` scaling directly against
the aggregate RMS coefficient error.

Run: ``python validation/validate_photon_noise.py`` from ``products/P014``.
Output saved to ``validation/photon_noise_output.txt``.
"""

from __future__ import annotations

import numpy as np

from wavelab.dataset import build_modal_geometry
from wavelab.modal import ModalReconstructor
from wavelab.noise import add_slope_noise, slope_sigma

NOLL = list(range(2, 16))
N_TRIALS = 800
FLUX_LEVELS = (100.0, 300.0, 1000.0, 3000.0, 10000.0)
SIGMA_REF, FLUX_REF = 1.0, 100.0


def main() -> None:
    geometry = build_modal_geometry(NOLL, n_side=8)
    recon = ModalReconstructor(NOLL, geometry.sub_x, geometry.sub_y, method="tsvd", reg=1e-8)
    coeffs = recon.noise_propagation()
    mean_coeff = float(np.mean(coeffs))
    print(f"n_sub = {geometry.n_sub}, n_modes = {geometry.n_modes}")
    print(f"mean per-mode noise propagation coefficient = {mean_coeff:.6f} (1/slope-unit^2)")
    print()

    rng = np.random.default_rng(0)
    a_true = rng.normal(0.0, 0.1, size=len(NOLL))
    s_true = geometry.matrix @ a_true

    print(f"{'flux':>8}  {'sigma_slope':>11}  {'predicted var':>13}  {'empirical var':>13}  {'ratio':>7}")
    ratios = []
    empirical_rms = []
    for flux in FLUX_LEVELS:
        sigma_s = slope_sigma(flux, sigma_ref=SIGMA_REF, flux_ref=FLUX_REF)
        predicted_var = mean_coeff * sigma_s**2
        trial_rng = np.random.default_rng(int(flux) + 1)
        errs = np.empty((N_TRIALS, len(NOLL)))
        for t in range(N_TRIALS):
            s_noisy = add_slope_noise(s_true, flux, trial_rng, sigma_ref=SIGMA_REF, flux_ref=FLUX_REF)
            a_hat = recon.reconstruct(s_noisy)
            errs[t] = a_hat - a_true
        empirical_var = float(np.mean(errs.var(axis=0)))
        ratio = empirical_var / predicted_var
        ratios.append(ratio)
        empirical_rms.append(float(np.sqrt(np.mean(errs**2))))
        print(f"{flux:8.0f}  {sigma_s:11.6f}  {predicted_var:13.6e}  {empirical_var:13.6e}  {ratio:7.3f}")

    print()
    worst_ratio_dev = max(abs(r - 1.0) for r in ratios)
    print(f"worst |empirical/predicted - 1| over all flux levels = {worst_ratio_dev:.3f}")
    print(f"PASS (agreement within 25%, {N_TRIALS} trials/point): {worst_ratio_dev < 0.25}")
    print()

    # Direct 1/sqrt(N) scaling check on the aggregate RMS error.
    print("Direct sigma(N) propto 1/sqrt(N) check on aggregate RMS coefficient error:")
    ref_rms, ref_flux = empirical_rms[0], FLUX_LEVELS[0]
    for flux, rms in zip(FLUX_LEVELS, empirical_rms):
        predicted_rms = ref_rms * np.sqrt(ref_flux / flux)
        rel_err = abs(rms - predicted_rms) / predicted_rms
        print(
            f"  flux={flux:8.0f}  measured RMS={rms:.6f}  predicted RMS={predicted_rms:.6f}  "
            f"rel.err={rel_err:.3f}"
        )


if __name__ == "__main__":
    main()
