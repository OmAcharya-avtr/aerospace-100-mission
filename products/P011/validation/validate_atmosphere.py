"""Validation 5 — phase-screen statistics.

The Fourier phase-screen method is checked in two separable steps, because
lumping them together hides which of the two is at fault:

1. **Implementation.** The measured structure function of an ensemble of
   screens against the *exact* expectation for the discrete spectrum they are
   built from (:func:`waveforge.atmosphere.band_limited_structure_function`).
   Any disagreement here is a coding error.
2. **Method bias.** That exact discrete expectation against the continuous
   Kolmogorov result ``D(r) = 6.8839 (r/r0)^(5/3)`` (Fried 1965; Roddier 1981).
   The gap is the known band-limitation of the method, and the effect of
   subharmonic augmentation (Lane, Glindemann and Dainty, *Waves in Random
   Media* **2**, 209, 1992) on it is measured.

A third check compares the modal content of pupil-sized cut-outs against the
analytic Kolmogorov Zernike variances.

Run from products/P011:
    PYTHONPATH=src python validation/validate_atmosphere.py
"""

from __future__ import annotations

import time

import numpy as np

from waveforge.atmosphere import (
    band_limited_structure_function,
    phase_screen,
    structure_function,
)
from waveforge.pupil import PupilGrid, piston_removed
from waveforge.statistics import phase_structure_function, zernike_variance
from waveforge.zernike import zernike_basis

R0 = 0.10
PIXEL_SCALE = 0.02
N_SCREEN = 128
N_ENSEMBLE = 60
MAX_LAG = N_SCREEN // 8


def main() -> None:
    start = time.perf_counter()
    print("=" * 78)
    print("WaveForge validation 5 — Kolmogorov phase-screen statistics")
    print("=" * 78)
    print(f"screen           : {N_SCREEN} x {N_SCREEN}, d = {PIXEL_SCALE} m, r0 = {R0} m")
    print(f"ensemble         : {N_ENSEMBLE} screens, seeds 41000 + 31 k")
    print(f"lags examined    : 1 .. {MAX_LAG} samples "
          f"(estimator degrades beyond ~N/8 on a finite window)")
    print(f"grid Nyquist     : {1 / (2 * PIXEL_SCALE):.1f} cycles/m")
    print(f"lowest frequency : {1 / (N_SCREEN * PIXEL_SCALE):.4f} cycles/m")
    print()

    lags = np.arange(1, MAX_LAG + 1)
    separations = lags * PIXEL_SCALE
    continuous = phase_structure_function(separations, R0)
    band_limited = band_limited_structure_function(N_SCREEN, PIXEL_SCALE, R0, lags)

    print("--- 1. Implementation check: measured vs the exact discrete expectation ---")
    print("Screens without subharmonics, whose expectation is known in closed form.")
    accumulator = np.zeros(MAX_LAG)
    for k in range(N_ENSEMBLE):
        screen = phase_screen(
            N_SCREEN, PIXEL_SCALE, R0, n_subharmonics=0, rng=41_000 + 31 * k
        )
        accumulator += structure_function(screen, max_lag=MAX_LAG)[1]
    measured = accumulator / N_ENSEMBLE
    print(f"{'r [m]':>8} {'measured':>12} {'exact':>12} {'ratio':>8}")
    ratios = measured / band_limited
    for index in range(0, MAX_LAG, max(1, MAX_LAG // 8)):
        print(
            f"{separations[index]:>8.3f} {measured[index]:>12.5f} "
            f"{band_limited[index]:>12.5f} {ratios[index]:>8.4f}"
        )
    worst = float(np.max(np.abs(ratios - 1.0)))
    print()
    print(f"worst deviation from the exact expectation : {worst * 100:.2f}%")
    print(f"Monte-Carlo standard error at {N_ENSEMBLE} screens : ~{100 / np.sqrt(N_ENSEMBLE):.1f}%")
    print("tolerance                                  : 5%")
    print(f"result                                     : {'PASS' if worst < 0.05 else 'FAIL'}")
    print()

    print("--- 2. Method bias: exact discrete expectation vs continuous theory ---")
    print(f"{'r [m]':>8} {'r/r0':>7} {'band-limited':>14} {'6.8839(r/r0)^(5/3)':>20} {'ratio':>8}")
    for index in range(0, MAX_LAG, max(1, MAX_LAG // 8)):
        print(
            f"{separations[index]:>8.3f} {separations[index] / R0:>7.3f} "
            f"{band_limited[index]:>14.5f} {continuous[index]:>20.5f} "
            f"{band_limited[index] / continuous[index]:>8.4f}"
        )
    print()
    print("The Fourier screen is short of the continuous theory at every")
    print("separation: it carries no power below 1/(N d) or above the grid")
    print("Nyquist. The deficit grows with separation because the missing power")
    print("is at low frequency. This is the method, not the implementation.")
    print()

    print("--- 3. Effect of subharmonic augmentation ---")
    print("Measured structure function relative to the CONTINUOUS theory, for")
    print("increasing numbers of Lane et al. (1992) subharmonic levels.")
    print(f"{'levels':>7}", end="")
    show = [0, MAX_LAG // 4, MAX_LAG // 2, MAX_LAG - 1]
    for index in show:
        print(f" {'r=' + format(separations[index], '.2f'):>10}", end="")
    print()
    for levels in (0, 1, 2, 3, 4, 6):
        accumulator = np.zeros(MAX_LAG)
        for k in range(N_ENSEMBLE):
            screen = phase_screen(
                N_SCREEN, PIXEL_SCALE, R0, n_subharmonics=levels, rng=52_000 + 17 * k
            )
            accumulator += structure_function(screen, max_lag=MAX_LAG)[1]
        ratio = accumulator / N_ENSEMBLE / continuous
        print(f"{levels:>7}", end="")
        for index in show:
            print(f" {ratio[index]:>10.3f}", end="")
        print()
    print()
    print("Subharmonics recover a large part of the deficit and the recovery is")
    print("strongest at the largest separations, which is what they are for. The")
    print("remaining shortfall at small r is the high-frequency truncation at the")
    print("grid Nyquist and cannot be fixed by subharmonics.")
    print()

    print("--- 4. Modal content of pupil-sized cut-outs vs the analytic variances ---")
    n_pix, diameter = 64, 0.5
    pupil = PupilGrid(n_pix, diameter)
    rho, theta = pupil.polar()
    basis = zernike_basis(21, rho, theta, mask=pupil.mask, include_piston=False)
    pinv = np.linalg.pinv(basis.T)
    n_modal = 150
    coefficients = np.zeros(20)
    for k in range(n_modal):
        raw = phase_screen(
            512, pupil.sample_spacing_m, R0, n_subharmonics=6, rng=70_000 + 23 * k
        )
        window = piston_removed(raw[:n_pix, :n_pix], pupil.mask)
        coefficients += (pinv @ window[pupil.mask]) ** 2
    coefficients /= n_modal
    d_over_r0 = diameter / R0
    print(f"pupil {n_pix} x {n_pix} over {diameter} m, D/r0 = {d_over_r0:.1f}, "
          f"{n_modal} screens (512^2, 6 subharmonic levels)")
    print(f"{'j':>4} {'measured':>12} {'analytic':>12} {'ratio':>8} {'MC s.e.':>9}")
    for j in (2, 3, 4, 6, 8, 11, 15, 21):
        analytic = zernike_variance(j, d_over_r0)
        ratio = coefficients[j - 2] / analytic
        stderr = np.sqrt(2.0 / n_modal)  # chi-square with n_modal degrees of freedom
        print(
            f"{j:>4} {coefficients[j - 2]:>12.5f} {analytic:>12.5f} "
            f"{ratio:>8.3f} {stderr:>9.3f}"
        )
    high_order = np.array(
        [coefficients[j - 2] / zernike_variance(j, d_over_r0) for j in range(4, 22)]
    )
    print()
    print(f"mean ratio over j = 4..21 (tip/tilt excluded) : {high_order.mean():.3f}")
    print(f"ratio for tip and tilt (j = 2, 3)             : "
          f"{coefficients[0] / zernike_variance(2, d_over_r0):.3f}, "
          f"{coefficients[1] / zernike_variance(3, d_over_r0):.3f}")
    print("The higher-order modes agree with the analytic Kolmogorov variances,")
    print("while tip and tilt carry the whole of the low-frequency deficit. Any")
    print("study of overall tip/tilt magnitude with these screens should use a")
    print("larger generating screen or a von Karman outer scale.")
    print()
    print(f"elapsed: {time.perf_counter() - start:.1f} s")
    print("=" * 78)


if __name__ == "__main__":
    main()
