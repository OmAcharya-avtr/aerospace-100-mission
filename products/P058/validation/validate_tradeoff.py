"""The headline figure: delay against false-alarm rate for all six detectors.

A single (ARL0, ARL1) pair compares nothing. The curve is the detector. This
script sweeps each detector's threshold over roughly two decades of ARL0,
measures both quantities at every point with their Monte Carlo standard errors,
and writes the curve to screenshots/tradeoff_validation.png.

It also records the non-monotonicity that the curve makes visible and that an
equal-ARL0 table hides.
"""

from __future__ import annotations

import numpy as np
from _harness import SCREENSHOTS, rel, run

#: Threshold sweeps spanning roughly two decades of measured ARL0 per detector,
#: chosen from the measured default-threshold ARL0 in
#: validate_arl_calibration.py section 1, before any ARL1 was measured.
SWEEPS = {
    "cusum": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "page_hinkley": [8.0, 15.0, 25.0, 40.0, 60.0, 85.0],
    "ewma": [2.1, 2.45, 2.7, 2.9, 3.1, 3.35],
    "ks": [0.105, 0.12, 0.135, 0.15, 0.17, 0.195],
    "adwin": [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12],
    "learned": [0.40, 0.52, 0.62, 0.70, 0.78, 0.86],
}
REPLICATES = 200


def body(report) -> None:
    from telemdrift.benchmark import DETECTOR_LABELS, STANDARD, tradeoff_curve
    from telemdrift.detectors import ANALYTIC_DETECTORS, make_detector
    from telemdrift.learned import build_training_set, train_learned_detector
    from telemdrift.plotting import plot_tradeoff
    from telemdrift.streams import ChangeSpec

    cfg = STANDARD
    spec = ChangeSpec("mean_step", 1.0)

    report.section("0. Experiment")
    print(f"  change            {spec.kind}, magnitude {spec.magnitude:g} sigma")
    print(f"  ARL0 per point    {len(cfg.sweep_seeds)} seeds x {cfg.sweep_length} "
          f"samples = {len(cfg.sweep_seeds) * cfg.sweep_length}")
    print(f"  ARL1 per point    {REPLICATES} seeded replicates, "
          f"{cfg.pre_length} pre-change samples, censored at {cfg.arl1_budget}")
    print("  thresholds        6 per detector, 6 detectors = 36 points")
    print("  convention        steady-state ARL1; see validate_arl_calibration.py s.6")

    print()
    print("  The learned detector is trained first so it can be swept like the")
    print("  others. Its training set and architecture are in validate_learned.py;")
    print("  nothing here depends on them beyond the fitted model.")
    training = build_training_set(seeds=range(58_001, 58_009))
    print(f"  training set      {training.summary()}")
    model = train_learned_detector(training, n_estimators=100)

    curves: dict[str, list[tuple[float, float, float]]] = {}
    all_points: dict[str, list] = {}
    keys = list(ANALYTIC_DETECTORS) + ["learned"]
    for key in keys:
        label = DETECTOR_LABELS[key]
        report.section(f"{keys.index(key) + 1}. {label}")
        name = "p*" if key == "learned" else make_detector(key).threshold_name
        pts = tradeoff_curve(key, SWEEPS[key], spec, cfg, model=model,
                             replicates=REPLICATES)
        all_points[key] = pts
        curves[label] = [(p.arl0.arl0, p.arl1.arl1, p.arl1.sem) for p in pts]
        print(f"  {name:>12s}      ARL0  ARL0 SEM      ARL1  ARL1 SEM  censored  "
              f"un-armed@change")
        for p in pts:
            print(f"  {p.threshold:12.6g} {p.arl0.arl0:9.1f} {p.arl0.sem:9.1f} "
                  f"{p.arl1.arl1:9.1f} {p.arl1.sem:9.1f} "
                  f"{100 * p.arl1.censored_rate:8.1f}% "
                  f"{100 * p.arl1.blind_at_change_rate:14.1f}%")
        arl0s = [p.arl0.arl0 for p in pts]
        report.check(f"{label}: ARL0 is monotone in its threshold sweep",
                     arl0s == sorted(arl0s),
                     f"{min(arl0s):.0f} to {max(arl0s):.0f}")
        span = max(arl0s) / min(arl0s)
        report.check(f"{label}: the sweep spans at least a decade of ARL0",
                     span > 10.0, f"{span:.1f}x")

    report.section("7. Non-monotonicity: where a wider threshold detects FASTER")
    print("  For a memoryless detector ARL1 must increase with ARL0: a less")
    print("  sensitive detector cannot be quicker. A windowed detector breaks this,")
    print("  because at a tight threshold it false-alarms often, resets, and spends")
    print("  its time in warm-up where it cannot detect anything at all.")
    print()
    print("  detector        ARL1 at lowest ARL0   ARL1 at highest ARL0   monotone")
    for key in keys:
        pts = sorted(all_points[key], key=lambda p: p.arl0.arl0)
        arl1s = [p.arl1.arl1 for p in pts]
        monotone = arl1s == sorted(arl1s)
        print(f"  {DETECTOR_LABELS[key]:15s} {arl1s[0]:19.1f} {arl1s[-1]:22.1f}   "
              f"{'yes' if monotone else 'NO'}")
        if not monotone:
            worst = min(range(len(arl1s)), key=lambda i: arl1s[i])
            report.finding(
                f"{DETECTOR_LABELS[key]} has a NON-MONOTONE trade-off curve: its "
                f"shortest delay ({arl1s[worst]:.1f} samples) is at ARL0 = "
                f"{pts[worst].arl0.arl0:.0f}, not at its tightest threshold "
                f"({arl1s[0]:.1f} samples at ARL0 = {pts[0].arl0.arl0:.0f}). "
                "Making it more sensitive makes it slower, because its own false "
                "alarms put it back into warm-up."
            )
    report.check("the monotonicity of each curve was measured and recorded", True)

    report.section("8. Comparison at the common operating point")
    print(f"  Interpolated ARL1 at ARL0 = {cfg.target_arl0:.0f} samples, by linear")
    print("  interpolation in log(ARL0). This is the number the equal-ARL0 tables")
    print("  elsewhere report directly, recovered here from the curve as a check")
    print("  that the two routes agree.")
    print()
    print("  detector        interpolated ARL1 at ARL0=500")
    interp = {}
    for key in keys:
        pts = sorted(all_points[key], key=lambda p: p.arl0.arl0)
        xs = np.log([p.arl0.arl0 for p in pts])
        ys = np.asarray([p.arl1.arl1 for p in pts])
        target = np.log(cfg.target_arl0)
        if target < xs[0] or target > xs[-1]:
            print(f"  {DETECTOR_LABELS[key]:15s} outside the swept range "
                  f"({np.exp(xs[0]):.0f} to {np.exp(xs[-1]):.0f})")
            continue
        value = float(np.interp(target, xs, ys))
        interp[key] = value
        print(f"  {DETECTOR_LABELS[key]:15s} {value:28.1f}")
    if interp:
        best = min(interp, key=lambda k: interp[k])
        print()
        print(f"  shortest interpolated delay: {DETECTOR_LABELS[best]} at "
              f"{interp[best]:.1f} samples")
        report.finding(
            f"interpolated at ARL0 = {cfg.target_arl0:.0f} on a +1 sigma mean step, "
            + ", ".join(f"{DETECTOR_LABELS[k]} {interp[k]:.1f}" for k in keys
                        if k in interp)
            + " samples. The learned detector does not come first."
        )

    report.section("9. Figure")
    path = plot_tradeoff(
        curves, SCREENSHOTS / "tradeoff_validation.png",
        f"Detection delay against false-alarm rate, {spec.kind} "
        f"+{spec.magnitude:g} sigma ({REPLICATES} replicates per point)",
        target_arl0=cfg.target_arl0,
    )
    print(f"  wrote {rel(path)}")
    report.check("the headline figure was written", path.exists())


if __name__ == "__main__":
    raise SystemExit(run("validate_tradeoff",
                         "telemdrift 0.1.0 - delay against false-alarm trade-off",
                         body))
