"""Level 2 (Research) validation evidence for shacksim.

Run from products/P018/:

    PYTHONPATH=src python validation/run_validation.py

Writes validation/validation_output.txt (verbatim console output) and three
PNGs into validation/. Every number quoted in validation/VALIDATION.md comes
from this script.

Checks
------
1. Known global tilt -> analytically predicted uniform slope vector.
2. Zero wavefront -> zero slopes; correlation-estimator S-curve bias.
3. Slope error vs photon count against the standard centroid noise-propagation
   expression (shacksim.slopes.cog_noise_sigma).
4. Centre-of-gravity bias under an unsubtracted background offset, against the
   analytic shrinkage factor S / (S + B p^2).
5. Learned estimator vs thresholded centre of gravity across photon counts,
   round and elongated spots — crossover reported wherever it falls.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from shacksim import (  # noqa: E402
    LensletArray,
    MLSlopeEstimator,
    cog_displacement,
    cog_noise_sigma,
    cog_slopes,
    correlation_displacement,
    correlation_slopes,
    generate_subaperture_dataset,
    reference_template,
    simulate_frame,
    subaperture_spot,
    tilt_slopes,
)

HERE = Path(__file__).resolve().parent

# Figures are reported by file name only. Printing the absolute path bakes the
# build machine's directory layout into committed evidence (this file previously
# carried the build container's absolute home path), which leaks the build
# environment and makes the output non-reproducible across machines for no
# benefit. The release gate rejects such a path anywhere in tracked content --
# including in a comment like this one, which is why the path is described
# rather than quoted.
OUT = HERE / "validation_output.txt"

ARRAY = LensletArray()
BACKGROUND = 1.0
READ_NOISE = 3.0
TUNE_SEED0 = 300
TRAIN_SEED = 100
TEST_SEED0 = 9000
PHOTON_LEVELS = (30.0, 50.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
THRESHOLD_GRID = (0.0, 4.0, 7.0, 10.0, 13.0, 16.0, 20.0, 30.0)


class Tee:
    """Write to stdout and to the transcript file at the same time."""

    def __init__(self, path: Path) -> None:
        self.handle = path.open("w", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        self.handle.write(text)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.handle.flush()


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------- 1
def check_known_tilt() -> None:
    header("1. Known global tilt -> analytically predicted uniform slope vector")
    a = ARRAY
    print(f"Geometry: {a.n_lenslets}x{a.n_lenslets} lenslets, pitch {a.pitch * 1e6:.0f} um, "
          f"f = {a.focal_length * 1e3:.0f} mm, {a.pixels_per_sub} px/subap, "
          f"lambda = {a.wavelength * 1e9:.0f} nm")
    print(f"Derived: pixel {a.pixel_size * 1e6:.3f} um, pixel angle {a.pixel_angle * 1e6:.1f} urad, "
          f"spot FWHM {a.spot_fwhm * 1e6:.2f} um = {a.spot_fwhm_px:.3f} px, "
          f"sigma {a.spot_sigma_px:.4f} px")
    print(f"Illuminated subapertures: {a.n_valid} of {a.n_lenslets ** 2}")
    print()
    print("Wavefront W(X,Y) = gx*X + gy*Y  =>  dW/dX = gx everywhere (a constant),")
    print("so every subaperture must report exactly (gx, gy) and the spot must sit at")
    print("d = gx * f / p pixels.  Hand check for gx = 1.000e-03 rad:")
    hand = 1.0e-3 * ARRAY.focal_length / ARRAY.pixel_size
    print(f"  d = 1.000e-03 * 50.0e-3 m / 31.25e-6 m = {hand:.6f} px  (exact: 1.6)")
    print()
    print("Noise-free frames (no shot noise, no read noise, no background):")
    print(f"{'gx [rad]':>12} {'gy [rad]':>12} {'max|CoG-true| [rad]':>21} "
          f"{'[px]':>10} {'max|corr-true| [px]':>20}")
    worst_cog = 0.0
    worst_corr = 0.0
    tpl = reference_template(a)
    for gx, gy in [(0.0, 0.0), (1.0e-3, 0.0), (0.0, -1.5e-3), (2.0e-3, 1.0e-3),
                   (-2.5e-3, -2.5e-3)]:
        truth = tilt_slopes(a, gx, gy)
        frame = simulate_frame(a, truth, photons=5.0e4, shot_noise=False, seed=1)
        est = cog_slopes(frame, a)
        err = np.abs(est - truth).max()
        est_c = correlation_slopes(frame, a, template=tpl)
        err_c = np.abs(a.slope_to_displacement(est_c - truth)).max()
        worst_cog = max(worst_cog, float(err))
        worst_corr = max(worst_corr, float(err_c))
        print(f"{gx:12.3e} {gy:12.3e} {err:21.3e} "
              f"{a.slope_to_displacement(err):10.2e} {err_c:20.4f}")
    print()
    print(f"Worst-case noise-free CoG slope error: {worst_cog:.3e} rad "
          f"= {a.slope_to_displacement(worst_cog):.3e} px "
          f"(tolerance 1e-8 rad) -> {'PASS' if worst_cog < 1e-8 else 'FAIL'}")
    print(f"Worst-case noise-free correlation error: {worst_corr:.4f} px "
          f"(tolerance 0.05 px) -> {'PASS' if worst_corr < 0.05 else 'FAIL'}")
    print()
    print("Uniformity across the pupil, noisy frame (N = 3000 e-/subap, B = 1.0, R = 3.0):")
    truth = tilt_slopes(a, 1.0e-3, -0.5e-3)
    frame = simulate_frame(a, truth, photons=3000.0, background=BACKGROUND,
                           read_noise=READ_NOISE, seed=42)
    est = cog_slopes(frame, a, threshold=BACKGROUND + 3 * READ_NOISE)
    mean = est.mean(axis=0)
    spread = est.std(axis=0)
    print(f"  true    (gx, gy) = ({truth[0, 0]:.6e}, {truth[0, 1]:.6e}) rad")
    print(f"  mean    (gx, gy) = ({mean[0]:.6e}, {mean[1]:.6e}) rad")
    print(f"  bias             = ({mean[0] - truth[0, 0]:+.3e}, "
          f"{mean[1] - truth[0, 1]:+.3e}) rad "
          f"= ({a.slope_to_displacement(mean[0] - truth[0, 0]):+.4f}, "
          f"{a.slope_to_displacement(mean[1] - truth[0, 1]):+.4f}) px")
    print(f"  subap-to-subap std = ({spread[0]:.3e}, {spread[1]:.3e}) rad "
          f"= ({a.slope_to_displacement(spread[0]):.4f}, "
          f"{a.slope_to_displacement(spread[1]):.4f}) px")
    pred = cog_noise_sigma(a, 3000.0, BACKGROUND, READ_NOISE,
                           displacement_px=a.slope_to_displacement(1.0e-3))
    print(f"  linear-CoG noise prediction (un-thresholded) = {pred:.3e} rad "
          f"= {a.slope_to_displacement(pred):.4f} px "
          "(thresholding beats it, see section 3)")


# --------------------------------------------------------------------------- 2
def check_zero_and_scurve() -> None:
    header("2. Zero wavefront -> zero slopes; correlation S-curve bias")
    a = ARRAY
    zero = np.zeros((a.n_valid, 2))
    frame = simulate_frame(a, zero, photons=5.0e4, shot_noise=False, seed=0)
    cog = cog_slopes(frame, a)
    corr = correlation_slopes(frame, a)
    print(f"Noise-free zero wavefront: max|CoG| = {np.abs(cog).max():.3e} rad, "
          f"max|corr| = {np.abs(corr).max():.3e} rad "
          f"-> {'PASS' if max(np.abs(cog).max(), np.abs(corr).max()) < 1e-12 else 'FAIL'} "
          "(tolerance 1e-12 rad)")
    print()
    print("Sub-pixel S-curve of the 3-point parabolic correlation interpolator")
    print("(noise-free, round diffraction-limited spot, matched template):")
    tpl = reference_template(a)
    shifts = np.linspace(-1.0, 1.0, 41)
    errs = []
    for d in shifts:
        stamp = subaperture_spot(a, d, 0.0, 1.0e5)
        est = correlation_displacement(stamp, tpl)[0, 0]
        errs.append(est - d)
    errs = np.asarray(errs)
    print(f"  max |error| over |d| <= 1 px: {np.abs(errs).max():.4f} px")
    print(f"  RMS error                  : {rms(errs):.4f} px")
    print(f"  error at d = 0             : {errs[len(errs) // 2]:.3e} px")
    cog_errs = np.array(
        [cog_displacement(subaperture_spot(a, d, 0.0, 1.0e5))[0, 0] - d for d in shifts]
    )
    print(f"  same sweep, CoG max |error|: {np.abs(cog_errs).max():.3e} px")
    print("  => the correlation estimator carries a systematic sub-pixel bias that the")
    print("     centre of gravity does not; it is the price of peak interpolation.")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(shifts, errs, "o-", ms=3, label="correlation (3-pt parabolic)")
    ax.plot(shifts, cog_errs, "s-", ms=3, label="centre of gravity")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("true spot displacement [px]")
    ax.set_ylabel("estimate - truth [px]")
    ax.set_title("Noise-free sub-pixel bias (S-curve), matched template")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "scurve_bias.png", dpi=130)
    plt.close(fig)
    print("  figure -> scurve_bias.png")


# --------------------------------------------------------------------------- 3
def check_noise_propagation() -> dict[str, list[float]]:
    header("3. Slope error vs photon count vs the standard noise-propagation expression")
    a = ARRAY
    print("Expression (Hardy 1998 ch. 5; Thomas et al. 2006, MNRAS 371, 323; photon-limited")
    print("form also in Winick 1986, JOSA A 3, 1809), derived in slopes.cog_noise_sigma:")
    print("    Var(x) = M2/N + (B + R^2)/N^2 * [p^2(p^2-1)/12 + p^2 d^2]   [px^2]")
    print("    sigma_g = sqrt(Var(x)) * p_pix / f                          [rad]")
    print("M2 is evaluated numerically from the pixel-integrated spot profile.")
    print("It describes the LINEAR (un-thresholded, un-clipped) CoG on background-")
    print("subtracted data, which is the estimator measured in the first table.")
    print(f"\nB = {BACKGROUND} e-/px, R = {READ_NOISE} e- RMS, 4000 stamps per point, "
          "slopes uniform over +/- 0.6 of the field.")
    print()
    print(f"{'N [e-]':>8} {'measured sx [px]':>17} {'predicted [px]':>15} {'ratio':>7} "
          f"{'measured sg [rad]':>18} {'predicted [rad]':>16}")
    levels = [100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]
    meas_all, pred_all = [], []
    for n_ph in levels:
        stamps, slopes = generate_subaperture_dataset(
            a, 4000, photons=n_ph, background=BACKGROUND, read_noise=READ_NOISE,
            seed=int(7000 + n_ph),
        )
        d_true = a.slope_to_displacement(slopes)
        est = cog_displacement(stamps - BACKGROUND, threshold=0.0, clip_negative=False)
        err_x = est[:, 0] - d_true[:, 0]
        meas = rms(err_x)
        pred = rms(cog_noise_sigma(a, n_ph, BACKGROUND, READ_NOISE,
                                   displacement_px=d_true[:, 0]) / a.pixel_angle)
        meas_all.append(meas)
        pred_all.append(pred)
        print(f"{n_ph:8.0f} {meas:17.4f} {pred:15.4f} {meas / pred:7.3f} "
              f"{meas * a.pixel_angle:18.3e} {pred * a.pixel_angle:16.3e}")
    ratios = np.array(meas_all) / np.array(pred_all)
    ok = ratios[1:]  # N >= 300
    print()
    print(f"Agreement for N >= 300 e-: ratio in [{ok.min():.3f}, {ok.max():.3f}] "
          f"(tolerance 0.85-1.15) -> {'PASS' if (ok.min() > 0.85 and ok.max() < 1.15) else 'FAIL'}")
    print(f"At N = 100 e- the measured error is {ratios[0]:.1f}x the prediction: the "
          "first-order")
    print("linearization assumes the CoG denominator does not fluctuate, and at this flux")
    print("it can approach zero. This is the documented validity boundary of the")
    print("expression, reported as measured, not tuned away.")

    print()
    print("Same sweep for the practical estimators (RMS over both slope axes, px):")
    print(f"{'N [e-]':>8} {'linear CoG':>11} {'thresh CoG':>11} {'threshold':>10} "
          f"{'correlation':>12}")
    tpl = reference_template(a)
    thr_curve, cog_curve, corr_curve = [], [], []
    for n_ph in PHOTON_LEVELS:
        stamps, slopes = generate_subaperture_dataset(
            a, 2000, photons=n_ph, background=BACKGROUND, read_noise=READ_NOISE,
            seed=int(TEST_SEED0 + n_ph),
        )
        d_true = a.slope_to_displacement(slopes)
        lin = rms(cog_displacement(stamps - BACKGROUND, 0.0, clip_negative=False) - d_true)
        thr = tune_threshold(a, n_ph)
        cogv = rms(cog_displacement(stamps, threshold=thr) - d_true)
        corrv = rms(correlation_displacement(stamps, tpl) - d_true)
        thr_curve.append(thr)
        cog_curve.append(cogv)
        corr_curve.append(corrv)
        print(f"{n_ph:8.0f} {lin:11.4f} {cogv:11.4f} {thr:10.1f} {corrv:12.4f}")
    return {"cog": cog_curve, "corr": corr_curve, "threshold": thr_curve}


def tune_threshold(array: LensletArray, n_photons: float, elongation: float = 1.0) -> float:
    """Pick the CoG threshold minimising RMS on a *tuning* dataset (never the test set).

    Tuned separately for each (flux, elongation) operating point so that the
    classical baseline is given its best available configuration.
    """
    stamps, slopes = generate_subaperture_dataset(
        array, 1500, photons=n_photons, background=BACKGROUND, read_noise=READ_NOISE,
        elongation=elongation, seed=int(TUNE_SEED0 + n_photons + 11 * elongation),
    )
    d_true = array.slope_to_displacement(slopes)
    scores = [rms(cog_displacement(stamps, threshold=t) - d_true) for t in THRESHOLD_GRID]
    return float(THRESHOLD_GRID[int(np.argmin(scores))])


# --------------------------------------------------------------------------- 4
def check_background_bias() -> None:
    header("4. Centre-of-gravity bias under an unsubtracted background offset")
    a = ARRAY
    print("Analytic prediction. With a uniform background B added to every pixel and no")
    print("threshold, the CoG numerator gains B*sum(x_i) = 0 (the pixel grid is centred)")
    print("and the denominator gains B*p^2, so")
    print("    x_hat = S*d / (S + B*p^2) = d * kappa,   kappa = S / (S + B p^2)")
    print("i.e. a pure multiplicative shrinkage of every measured slope toward zero — a")
    print("gain error on the whole wavefront, not a random error. Noise-free frames.")
    print()
    p2 = a.pixels_per_sub**2
    print(f"p^2 = {p2} pixels per subaperture")
    print(f"{'S [e-]':>8} {'B [e-/px]':>10} {'kappa pred':>11} {'d=4 px: pred':>13} "
          f"{'measured':>10} {'|err| [px]':>11} {'kappa meas':>11}")
    worst = 0.0
    measured: dict[tuple[float, float], float] = {}
    for signal in (1000.0, 5000.0):
        for bkg in (0.0, 0.5, 2.0, 10.0, 50.0):
            kappa = signal / (signal + bkg * p2)
            d = 4.0
            stamp = subaperture_spot(a, d, 0.0, signal) + bkg
            est = cog_displacement(stamp, threshold=0.0)[0, 0]
            err = abs(est - kappa * d)
            worst = max(worst, err)
            measured[(signal, bkg)] = est / d
            print(f"{signal:8.0f} {bkg:10.1f} {kappa:11.4f} {kappa * d:13.4f} "
                  f"{est:10.4f} {err:11.2e} {est / d:11.4f}")
    print()
    print(f"Worst deviation from the analytic shrinkage: {worst:.2e} px "
          f"(tolerance 1e-3 px) -> {'PASS' if worst < 1e-3 else 'FAIL'}")
    print("Magnitude of the effect, quoted as measured-slope / true-slope:")
    for signal, bkg in ((1000.0, 0.5), (1000.0, 2.0), (1000.0, 10.0),
                        (5000.0, 2.0), (5000.0, 10.0)):
        k = measured[(signal, bkg)]
        print(f"  S = {signal:5.0f} e-, B = {bkg:4.1f} e-/px -> {k * 100:5.1f} % of truth, "
              f"i.e. a {100 * (1 - k):4.1f} % wavefront gain error")
    print("Subtracting B, or thresholding at B, removes it (see the threshold column in")
    print("section 3). This is why an uncorrected background offset is a systematic")
    print("wavefront error and not merely extra noise.")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bkgs = np.linspace(0.0, 50.0, 26)
    for signal, style in ((1000.0, "o-"), (5000.0, "s-")):
        meas = [cog_displacement(subaperture_spot(a, 4.0, 0.0, signal) + b)[0, 0] / 4.0
                for b in bkgs]
        ax.plot(bkgs, meas, style, ms=3, label=f"measured, S = {signal:.0f} e-")
        ax.plot(bkgs, signal / (signal + bkgs * p2), "k--", lw=1.0,
                label="analytic S/(S+Bp^2)" if signal == 1000.0 else None)
    ax.set_xlabel("background B [e-/px]")
    ax.set_ylabel("measured slope / true slope  [-]")
    ax.set_title("Centre-of-gravity shrinkage under an unsubtracted background")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "background_bias.png", dpi=130)
    plt.close(fig)
    print("figure -> background_bias.png")


# --------------------------------------------------------------------------- 5
def check_ml_benchmark(classical: dict[str, list[float]]) -> None:
    header("5. Learned slope estimator vs thresholded centre of gravity")
    a = ARRAY
    n_train = 9000
    print(f"Training set: {n_train} single-subaperture stamps, photons log-uniform in")
    print("[30, 30000] e-, elongation uniform in [1, 3] along x, B = 1.0 e-/px, "
          "R = 3.0 e- RMS,")
    print(f"seed = {TRAIN_SEED}. Test sets use disjoint seeds ({TEST_SEED0}+N) and are never")
    print("seen during training or threshold tuning (tuning seeds are 300+N).")
    t0 = time.time()
    x_tr, y_tr = generate_subaperture_dataset(
        a, n_train, photons=(30.0, 30000.0), background=BACKGROUND, read_noise=READ_NOISE,
        elongation=(1.0, 3.0), seed=TRAIN_SEED,
    )
    t_gen = time.time() - t0
    t0 = time.time()
    model = MLSlopeEstimator(a, n_estimators=5, hidden_layer_sizes=(96, 48),
                             random_state=0).fit(x_tr, y_tr)
    t_fit = time.time() - t0
    print(f"Data generation {t_gen:.1f} s; training 5 x MLP(96,48) {t_fit:.1f} s "
          "on 2 CPU cores.")
    tpl = reference_template(a)

    results: dict[float, dict[str, list[float]]] = {}
    for elong in (1.0, 3.0):
        print()
        label = "round diffraction-limited spot" if elong == 1.0 else "spot elongated 3x along x"
        print(f"--- {label} (elongation = {elong:.1f}) ---")
        print(f"{'N [e-]':>8} {'thr':>5} {'CoG [px]':>9} {'corr [px]':>10} {'ML [px]':>8} "
              f"{'ML/CoG':>7} {'ML std [px]':>12} {'std/err':>8}")
        cog_l, corr_l, ml_l, std_l = [], [], [], []
        for n_ph in PHOTON_LEVELS:
            stamps, slopes = generate_subaperture_dataset(
                a, 2000, photons=n_ph, background=BACKGROUND, read_noise=READ_NOISE,
                elongation=elong, seed=int(TEST_SEED0 + n_ph + 7 * elong),
            )
            d_true = a.slope_to_displacement(slopes)
            thr = tune_threshold(a, n_ph, elongation=elong)
            cog_e = rms(cog_displacement(stamps, threshold=thr) - d_true)
            corr_e = rms(correlation_displacement(stamps, tpl) - d_true)
            ml, std = model.predict(stamps, return_std=True)
            ml_e = rms(a.slope_to_displacement(ml) - d_true)
            std_px = float(np.mean(a.slope_to_displacement(std)))
            cog_l.append(cog_e)
            corr_l.append(corr_e)
            ml_l.append(ml_e)
            std_l.append(std_px)
            print(f"{n_ph:8.0f} {thr:5.0f} {cog_e:9.4f} {corr_e:10.4f} {ml_e:8.4f} "
                  f"{ml_e / cog_e:7.3f} {std_px:12.4f} {std_px / ml_e:8.3f}")
        results[elong] = {"cog": cog_l, "corr": corr_l, "ml": ml_l, "std": std_l}
        wins = [n for n, c, m in zip(PHOTON_LEVELS, cog_l, ml_l) if m < c]
        if wins:
            print(f"  ML beats the thresholded CoG at N = "
                  f"{[f'{n:.0f}' for n in wins]} e- of "
                  f"{[f'{n:.0f}' for n in PHOTON_LEVELS]} tested.")
        else:
            print("  ML does not beat the thresholded CoG at ANY tested photon count.")
        ratios = np.array(std_l) / np.array(ml_l)
        print(f"  ensemble-spread / actual-RMS ratio spans {ratios.min():.2f} to "
              f"{ratios.max():.2f} — NOT a calibrated 1-sigma bound.")

    print()
    print("Crossover summary (thresholded CoG is the baseline):")
    for elong in (1.0, 3.0):
        cog_l = np.array(results[elong]["cog"])
        ml_l = np.array(results[elong]["ml"])
        better = ml_l < cog_l
        ratio = ml_l / cog_l
        tag = f"  elongation {elong:.0f}x:"
        if better.all():
            print(f"{tag} ML better at EVERY tested flux.")
        elif not better.any():
            print(f"{tag} the thresholded CoG wins at EVERY tested flux — the learned "
                  "estimator never helps here.")
        else:
            idx = np.where(better)[0]
            contiguous = bool(idx[0] == 0 and np.all(np.diff(idx) == 1))
            if contiguous:
                last = int(idx[-1])
                print(f"{tag} crossover between N = {PHOTON_LEVELS[last]:.0f} and "
                      f"{PHOTON_LEVELS[last + 1]:.0f} e- — ML wins below, CoG above.")
            else:
                print(f"{tag} NO single crossover — ML wins at N = "
                      f"{[f'{PHOTON_LEVELS[i]:.0f}' for i in idx]} e- and loses elsewhere.")
        print(f"{tag} best ML/CoG ratio {ratio.min():.2f} at "
              f"N = {PHOTON_LEVELS[int(np.argmin(ratio))]:.0f} e-, worst "
              f"{ratio.max():.2f} at N = {PHOTON_LEVELS[int(np.argmax(ratio))]:.0f} e-.")
    print("  Against the CORRELATION estimator the picture differs — see the table above:")
    for elong in (1.0, 3.0):
        corr_l = np.array(results[elong]["corr"])
        ml_l = np.array(results[elong]["ml"])
        n_better = int((ml_l < corr_l).sum())
        print(f"    elongation {elong:.0f}x: ML beats correlation at {n_better} of "
              f"{len(PHOTON_LEVELS)} flux levels.")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, elong in zip(axes, (1.0, 3.0)):
        r = results[elong]
        ax.loglog(PHOTON_LEVELS, r["cog"], "o-", label="thresholded CoG (tuned)")
        ax.loglog(PHOTON_LEVELS, r["corr"], "^-", label="correlation")
        ax.loglog(PHOTON_LEVELS, r["ml"], "s-", label="ML ensemble")
        ax.fill_between(PHOTON_LEVELS,
                        np.array(r["ml"]) - np.array(r["std"]),
                        np.array(r["ml"]) + np.array(r["std"]),
                        alpha=0.18, color="tab:green", label="ML ensemble spread")
        pred = [cog_noise_sigma(ARRAY, n, BACKGROUND, READ_NOISE, elongation=elong,
                                displacement_px=0.6 * ARRAY.pixels_per_sub / 2 / np.sqrt(3))
                / ARRAY.pixel_angle for n in PHOTON_LEVELS]
        ax.loglog(PHOTON_LEVELS, pred, "k--", lw=1.0, label="linear-CoG noise theory")
        ax.set_title(f"elongation {elong:.0f}x")
        ax.set_xlabel("photons per subaperture [e-]")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("RMS slope error [px of spot displacement]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Slope error vs flux: classical baselines and the learned estimator")
    fig.tight_layout()
    fig.savefig(HERE / "ml_vs_classical.png", dpi=130)
    plt.close(fig)
    print("figure -> ml_vs_classical.png")
    print()
    print("Classical-only curves from section 3 (round spot, for cross-reference):")
    print(f"  tuned thresholds {classical['threshold']}")


def main() -> None:
    tee = Tee(OUT)
    sys.stdout = tee  # type: ignore[assignment]
    try:
        started = time.time()
        print("shacksim 0.1.0 — Level 2 validation run")
        print(f"numpy {np.__version__}, matplotlib {matplotlib.__version__}, "
              f"python {sys.version.split()[0]}")
        check_known_tilt()
        check_zero_and_scurve()
        classical = check_noise_propagation()
        check_background_bias()
        check_ml_benchmark(classical)
        print(f"\nTotal wall-clock: {time.time() - started:.1f} s")
    finally:
        sys.stdout = sys.__stdout__
        tee.flush()


if __name__ == "__main__":
    main()
