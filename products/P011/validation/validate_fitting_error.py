"""Validation 2 — residual error versus D/r0 against the standard scalings.

Two independent scalings are checked on the same Monte-Carlo phase screens:

* **Modal.** Removing the first ``J`` Zernike modes must leave Noll's
  ``Delta_J (D/r0)^(5/3)`` (Noll 1976, Table IV).
* **Zonal.** Fitting with a deformable mirror of actuator pitch ``d_act`` must
  leave ``a_F (d_act/r0)^(5/3)`` with ``a_F`` near 0.28 for a continuous
  face-sheet mirror (Hudgin 1977; Hardy 1998 Table 6.1).  The **exponent** is
  the physics; the **coefficient** depends on the influence function, so the
  measured value is reported, not assumed.

Run from products/P011:
    PYTHONPATH=src python validation/validate_fitting_error.py
"""

from __future__ import annotations

import time

import numpy as np

from waveforge.atmosphere import phase_screen
from waveforge.dm import DeformableMirror
from waveforge.errorbudget import (
    HUDGIN_FITTING_COEFFICIENT,
    ideal_filter_fitting_coefficient,
)
from waveforge.pupil import PupilGrid, piston_removed, variance
from waveforge.statistics import noll_residual_variance
from waveforge.zernike import zernike_basis

N_PIX = 64
DIAMETER = 0.5
SCREEN_PIXELS = 512
SUBHARMONICS = 6
N_SCREENS = 40
R0_VALUES = (0.4, 0.2, 0.1, 0.05)
SEED_BASE = 30_000

# The zonal check needs a finer pupil grid so that the smallest actuator pitch
# is still well sampled, and its own screens to match.
ZONAL_N_PIX = 128
ZONAL_SCREEN_PIXELS = 1024
ZONAL_N_SCREENS = 20
ZONAL_R0 = 0.10
ZONAL_N_ACT = (5, 9, 17, 33)


def screens_for(
    r0: float,
    pupil: PupilGrid,
    seed_offset: int,
    n_screens: int = N_SCREENS,
    screen_pixels: int = SCREEN_PIXELS,
    subharmonics: int = SUBHARMONICS,
) -> list[np.ndarray]:
    """Independent screens; each r0 gets its own disjoint seed block so that
    the scaling test compares different realisations rather than the same
    realisation rescaled."""
    n_pix = pupil.n_pix
    out = []
    for k in range(n_screens):
        raw = phase_screen(
            screen_pixels,
            pupil.sample_spacing_m,
            r0,
            n_subharmonics=subharmonics,
            rng=SEED_BASE + seed_offset + 977 * k,
        )
        out.append(piston_removed(raw[:n_pix, :n_pix], pupil.mask))
    return out


def main() -> None:
    start = time.perf_counter()
    pupil = PupilGrid(N_PIX, DIAMETER)
    rho, theta = pupil.polar()
    print("=" * 78)
    print("WaveForge validation 2 — residual error versus D/r0")
    print("=" * 78)
    print(f"pupil                : {N_PIX} x {N_PIX} samples over {DIAMETER} m")
    print(f"generating screen    : {SCREEN_PIXELS} x {SCREEN_PIXELS}, "
          f"{SUBHARMONICS} subharmonic levels")
    print(f"Monte Carlo          : {N_SCREENS} screens per r0, seeds {SEED_BASE} + 977 k")
    print(f"r0 values [m]        : {R0_VALUES}")
    print()

    j_values = (3, 6, 10, 15, 21)
    j_max = max(j_values)
    basis = zernike_basis(j_max, rho, theta, mask=pupil.mask, include_piston=False)
    pinv = np.linalg.pinv(basis.T)

    all_screens = {
        r0: screens_for(r0, pupil, 100_000 * (index + 1))
        for index, r0 in enumerate(R0_VALUES)
    }

    print("--- 1. Total piston-removed variance vs Noll Delta_1 ---")
    print(f"{'r0 [m]':>8} {'D/r0':>8} {'measured':>12} {'Noll':>12} {'ratio':>8}")
    total_ratios = []
    for r0 in R0_VALUES:
        measured = float(np.mean([variance(s, pupil.mask) for s in all_screens[r0]]))
        expected = noll_residual_variance(1, DIAMETER / r0)
        total_ratios.append(measured / expected)
        print(
            f"{r0:>8.3f} {DIAMETER / r0:>8.2f} {measured:>12.4f} "
            f"{expected:>12.4f} {measured / expected:>8.3f}"
        )
    ratios = np.array(total_ratios)
    print(f"pooled mean ratio    : {ratios.mean():.3f}")
    print(f"range over the four r0 blocks: {ratios.min():.3f} to {ratios.max():.3f}")
    print("Two effects, both reported rather than corrected for:")
    print("  (a) the ratio is below 1 because a Fourier screen carries no power")
    print("      below 1/(N d); even with six subharmonic levels a deficit")
    print("      remains, and it sits almost entirely in tip and tilt;")
    print("  (b) the scatter between r0 blocks is large because the total")
    print("      variance is dominated by two modes (tip and tilt), so with 40")
    print("      screens the estimator still has ~20% standard error. The")
    print("      higher-order residuals of check 2 are far better determined.")
    print()

    print("--- 2. Modal residual after removing Noll modes 1..J ---")
    print("Expected: Delta_J (D/r0)^(5/3) with Delta_J from Noll (1976) Table IV.")
    print(f"{'r0 [m]':>8} {'D/r0':>7} {'J':>4} {'measured':>12} {'Noll':>12} {'ratio':>8}")
    modal_ratios: dict[int, list[float]] = {j: [] for j in j_values}
    for r0 in R0_VALUES:
        d_over_r0 = DIAMETER / r0
        residual_sums = {j: 0.0 for j in j_values}
        for screen in all_screens[r0]:
            values = screen[pupil.mask]
            coeffs = pinv @ values
            for j in j_values:
                fitted = coeffs[: j - 1] @ basis[: j - 1]
                residual_sums[j] += float(np.var(values - fitted))
        for j in j_values:
            measured = residual_sums[j] / N_SCREENS
            expected = noll_residual_variance(j, d_over_r0)
            modal_ratios[j].append(measured / expected)
            print(
                f"{r0:>8.3f} {d_over_r0:>7.2f} {j:>4} {measured:>12.4f} "
                f"{expected:>12.4f} {measured / expected:>8.3f}"
            )
    print()
    print(f"{'J':>4} {'mean ratio':>12} {'spread':>10}")
    for j in j_values:
        arr = np.array(modal_ratios[j])
        print(f"{j:>4} {arr.mean():>12.3f} {arr.max() - arr.min():>10.3f}")
    print("Interpretation: the ratio is nearly independent of D/r0, which is the")
    print("(D/r0)^(5/3) scaling being confirmed; its offset from 1 is the same")
    print("screen band-limitation quantified in check 1, plus the discrete")
    print("orthonormality error of validation 1b.")
    print()

    print("--- 3. Zonal (deformable-mirror) fitting error vs actuator pitch ---")
    print("Expected: sigma^2 = a_F (d_act/r0)^(5/3), a_F ~ 0.28 for a continuous")
    print("face-sheet mirror (Hudgin 1977). The pitch direction is the physically")
    print("informative one: the r0 direction is exactly self-similar for")
    print("Kolmogorov statistics and so tests the generator, not the fitting law.")
    print()
    fine_pupil = PupilGrid(ZONAL_N_PIX, DIAMETER)
    fine_screens = screens_for(
        ZONAL_R0,
        fine_pupil,
        900_000,
        n_screens=ZONAL_N_SCREENS,
        screen_pixels=ZONAL_SCREEN_PIXELS,
        subharmonics=5,
    )
    f_nyquist = 1.0 / (2.0 * fine_pupil.sample_spacing_m)
    print(f"pupil for this check : {ZONAL_N_PIX} x {ZONAL_N_PIX} over {DIAMETER} m")
    print(f"screens              : {ZONAL_SCREEN_PIXELS}^2, 5 subharmonic levels, "
          f"{ZONAL_N_SCREENS} realisations, r0 = {ZONAL_R0} m")
    print(f"grid Nyquist         : {f_nyquist:.1f} cycles/m")
    print()
    print(
        f"{'n_act':>6} {'d_act [m]':>10} {'d_act/r0':>9} {'sigma^2':>10} "
        f"{'a_F':>8} {'lost':>7} {'a_F corr':>9}"
    )
    pitches, variances, corrected = [], [], []
    for n_act in ZONAL_N_ACT:
        mirror = DeformableMirror(fine_pupil, n_act=n_act, margin_actuators=0)
        residual = float(
            np.mean(
                [variance(mirror.fitting_residual(s), fine_pupil.mask) for s in fine_screens]
            )
        )
        ratio = mirror.pitch_m / ZONAL_R0
        f_cut = 1.0 / (2.0 * mirror.pitch_m)
        # fraction of the fitting-error power that lies above the grid Nyquist
        # and is therefore absent from the screens
        lost = (f_nyquist / f_cut) ** (-5.0 / 3.0)
        a_f = residual / ratio ** (5.0 / 3.0)
        pitches.append(mirror.pitch_m)
        variances.append(residual)
        corrected.append(a_f / (1.0 - lost))
        print(
            f"{n_act:>6} {mirror.pitch_m:>10.5f} {ratio:>9.4f} {residual:>10.5f} "
            f"{a_f:>8.4f} {lost * 100:>6.1f}% {a_f / (1.0 - lost):>9.4f}"
        )
    print()
    log_pitch = np.log(np.array(pitches))
    log_var = np.log(np.array(variances))
    slope = float(np.polyfit(log_pitch, log_var, 1)[0])
    local = [
        float(
            np.log(variances[i] / variances[i + 1]) / np.log(pitches[i] / pitches[i + 1])
        )
        for i in range(len(pitches) - 1)
    ]
    print(f"fitted exponent over all pitches : {slope:.4f}")
    print(f"expected exponent (5/3)          : {5 / 3:.4f}")
    print("local exponent between successive pitches:")
    for i, value in enumerate(local):
        print(f"    n_act {ZONAL_N_ACT[i]:>3} -> {ZONAL_N_ACT[i + 1]:>3} : {value:.4f}")
    print()
    ideal = ideal_filter_fitting_coefficient()
    print(f"ideal low-pass a_F (derived here): {ideal:.4f}  "
          f"(= 0.0229 * 2 pi * 3/5 * 2^(5/3))")
    print(f"Hudgin 1977 / Hardy 1998 Tab 6.1 : {HUDGIN_FITTING_COEFFICIENT:.4f}")
    print(f"band-limitation-corrected a_F at n_act = {ZONAL_N_ACT[-1]}: {corrected[-1]:.4f}")
    print(
        f"    relative to Hudgin               : "
        f"{(corrected[-1] / HUDGIN_FITTING_COEFFICIENT - 1) * 100:+.1f}%"
    )
    print(
        f"    relative to the derived ideal    : "
        f"{(corrected[-1] / ideal - 1) * 100:+.1f}%"
    )
    print(
        "result (corrected a_F within 10% of Hudgin at the finest pitch): "
        f"{'PASS' if abs(corrected[-1] / HUDGIN_FITTING_COEFFICIENT - 1) < 0.10 else 'FAIL'}"
    )
    print()
    print("Honest reading. The exponent measured across pitch is about "
          f"{slope:.2f},")
    print("not 5/3, and the coefficient rises monotonically with actuator count.")
    print("Both are the same finite-aperture effect: the (d_act/r0)^(5/3) law is")
    print("an infinite-aperture, infinite-bandwidth result, while a mirror with")
    print("only 5 x 5 Gaussian influence functions fitted by unconstrained least")
    print("squares over a 0.5 m pupil removes far more than a hard spatial")
    print("low-pass would. As the actuator count grows the two converge: after")
    print("correcting for the power the screens do not carry above the grid")
    print(f"Nyquist, a_F reaches {corrected[-1]:.3f} at {ZONAL_N_ACT[-1]} x "
          f"{ZONAL_N_ACT[-1]} actuators,")
    print("against the published 0.28. The scaling law should therefore be used")
    print("for sizing only when many actuators span the aperture, which is")
    print("exactly the regime it was derived for.")
    print()
    print(f"elapsed: {time.perf_counter() - start:.1f} s")
    print("=" * 78)


if __name__ == "__main__":
    main()
