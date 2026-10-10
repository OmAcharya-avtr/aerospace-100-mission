"""Where the calibration stops being true, measured.

Section 1 is the limitation that matters most in practice: the thresholds in this
package are calibrated on an i.i.d. Gaussian stream, and real spacecraft
telemetry is autocorrelated. Section 2 records the ADWIN structural findings:
the min_sub saturation that forced the shipped configuration, the published
window-shrinking convention measured against the full-reset convention the
harness uses, and the gap between ADWIN's nominal delta and its measured ARL0.
"""

from __future__ import annotations

import numpy as np
from _harness import run

PHIS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9)


def body(report) -> None:
    from telemdrift.benchmark import (
        DETECTOR_LABELS,
        STANDARD,
        analytic_factory,
        ar1_stream_fn,
        calibrate_all_analytic,
        stationary_stream_fn,
    )
    from telemdrift.detectors import ADWIN, ANALYTIC_DETECTORS
    from telemdrift.scoring import measure_arl0
    from telemdrift.streams import ar1_stationary, stationary

    cfg = STANDARD

    report.section("1. Autocorrelation destroys an i.i.d.-calibrated threshold")
    print("  Every threshold in this package is fitted on an i.i.d. Gaussian stream.")
    print("  Housekeeping telemetry is sampled faster than the physical time")
    print("  constants it measures and is strongly autocorrelated. The AR(1) stream")
    print("  used here has the SAME marginal distribution N(0,1) as the calibration")
    print("  stream - only the dependence differs - so anything that moves below is")
    print("  caused by dependence alone.")
    print()
    cals = calibrate_all_analytic(cfg)
    print("  Measured lag-one autocorrelation of the generated streams:")
    for phi in PHIS:
        x = ar1_stationary(200_000, 58_901, phi=phi)
        print(f"    phi = {phi:.1f}: measured lag-1 = "
              f"{np.corrcoef(x[:-1], x[1:])[0, 1]:+.4f}, "
              f"sd = {x.std():.4f}")
    print()
    print("  ARL0 at the i.i.d.-calibrated thresholds, "
          f"{len(cfg.eval_seeds[:4])} seeds x 30000 samples:")
    print()
    header = "  phi   " + "".join(f"{DETECTOR_LABELS[k]:>16s}" for k in ANALYTIC_DETECTORS)
    print(header)
    table = {}
    for phi in PHIS:
        row = []
        for key in ANALYTIC_DETECTORS:
            fn = stationary_stream_fn if phi == 0.0 else ar1_stream_fn(phi)
            r = measure_arl0(analytic_factory(key, cals[key].threshold), fn,
                             cfg.eval_seeds[:4], 30_000)
            row.append(r)
        table[phi] = row
        print(f"  {phi:.1f}   " + "".join(f"{r.arl0:10.0f}+/-{r.sem:<4.0f}" for r in row))
    print()
    print("  Degradation factor, ARL0 at phi = 0 divided by ARL0 at phi:")
    print(header)
    for phi in PHIS[1:]:
        factors = [table[0.0][i].arl0 / table[phi][i].arl0
                   for i in range(len(ANALYTIC_DETECTORS))]
        print(f"  {phi:.1f}   " + "".join(f"{f:15.1f}x" for f in factors))
    worst_phi = 0.9
    factors = {k: table[0.0][i].arl0 / table[worst_phi][i].arl0
               for i, k in enumerate(ANALYTIC_DETECTORS)}
    print()
    print("  A factor of F means the detector false-alarms F times more often than")
    print("  the operating point it was calibrated to, on a stream whose marginal")
    print("  distribution is identical. This is not a small correction.")
    report.check("autocorrelation degrades at least one detector by more than 5x",
                 max(factors.values()) > 5.0,
                 f"worst {max(factors, key=lambda k: factors[k])} "
                 f"{max(factors.values()):.1f}x")
    report.finding(
        "AUTOCORRELATION BREAKS THE CALIBRATION. At lag-one autocorrelation 0.9, "
        "with the marginal distribution unchanged, the measured ARL0 falls by "
        + ", ".join(f"{DETECTOR_LABELS[k]} {factors[k]:.1f}x" for k in ANALYTIC_DETECTORS)
        + ". Any threshold from this package applied to real telemetry without "
        "re-calibrating on that channel's own quiet data will false-alarm far more "
        "often than its nominal ARL0."
    )
    print()
    print("  The windowed KS test degrades least, which is a property of its")
    print("  statistic rather than a virtue: it compares empirical distributions,")
    print("  and an AR(1) stream has the same marginal distribution as the")
    print("  i.i.d. one it was calibrated against. It is blind to the change in")
    print("  dependence for exactly the reason it is robust to it.")

    report.section("2A. ADWIN's delta scaling, which set the shipped min_sub")
    print("  eps_cut grows only as sqrt(log(1/delta)), so driving delta down raises")
    print("  the cut threshold extremely slowly. With a small min_sub the cut test")
    print("  is applied to tiny subwindows where the Hoeffding bound behind eps_cut")
    print("  is at its loosest, and the ARL0 reachable at any sane delta is short.")
    print()
    print("  Measured ARL0, 3 seeds x 30000 stationary samples:")
    print()
    print("  min_sub   delta=1e-2   1e-4   1e-6   1e-8   1e-10   1e-12")
    deltas = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
    sat = {}
    for ms in (5, 15, 30, 50):
        row = []
        for d in deltas:
            r = measure_arl0(lambda dd=d, mm=ms: ADWIN(delta=dd, min_sub=mm),
                             lambda L, s: stationary(L, s), [58_911, 58_912, 58_913],
                             30_000)
            row.append(r.arl0)
        sat[ms] = row
        print(f"  {ms:7d}   " + "".join(f"{v:7.0f}" for v in row))
    print()
    span5 = sat[5][-1] / sat[5][0]
    print(f"  Ten orders of magnitude of delta buy a factor of {span5:.1f} in ARL0 at")
    print(f"  min_sub = 5 ({sat[5][0]:.0f} to {sat[5][-1]:.0f} samples). That is the")
    print("  sqrt(log(1/delta)) scaling, measured.")
    print()
    print(f"  At min_sub = 5 the {cfg.target_arl0:.0f}-sample target is crossed only")
    print(f"  somewhere between delta = 1e-10 (ARL0 {sat[5][4]:.0f}) and delta = 1e-12")
    print(f"  (ARL0 {sat[5][-1]:.0f}): a confidence parameter nine to ten orders of")
    print("  magnitude below the 0.002 the common implementations ship. At")
    print(f"  min_sub = 30 the target is crossed near delta = 1e-7 (ARL0 "
          f"{sat[30][2]:.0f} at 1e-6, {sat[30][3]:.0f} at 1e-8).")
    report.check(
        "at min_sub = 5 the target ARL0 lies between delta = 1e-10 and 1e-12, as "
        "documented",
        sat[5][4] < cfg.target_arl0 < sat[5][5],
        f"ARL0 {sat[5][4]:.0f} at 1e-10, {sat[5][-1]:.0f} at 1e-12, "
        f"target {cfg.target_arl0:.0f}",
    )
    report.check(
        "ten orders of magnitude of delta change ARL0 by less than 20x at "
        "min_sub = 5, as sqrt(log(1/delta)) predicts",
        span5 < 20.0, f"{span5:.1f}x",
    )
    print()
    print("  The practical consequence, measured rather than argued: the calibration")
    print("  procedure's bracket search walks outward from the detector's declared")
    print("  default by a factor of 1.6 per step for at most 26 steps, which reaches")
    print("  delta = 0.002 / 1.6^26 = 1e-8. At min_sub = 5 that is not far enough")
    print("  and the calibration reports a bracketing FAILURE. Measured:")
    print()
    from telemdrift.thresholds import calibrate_threshold

    for ms in (5, 30):
        res = calibrate_threshold(
            name=f"ADWIN min_sub={ms}",
            factory_from_threshold=lambda th, mm=ms: ADWIN(delta=th, min_sub=mm),
            default_threshold=0.002,
            stream_fn=lambda L, s: stationary(L, s),
            target_arl0=cfg.target_arl0,
            calibration_seeds=cfg.cal_seeds,
            evaluation_seeds=cfg.eval_seeds[:4],
            calibration_length=cfg.cal_length,
            evaluation_length=30_000,
            clip=(1e-12, 0.999_999),
        )
        print(f"    min_sub = {ms:2d}: bracketing_failed = {res.bracketing_failed}, "
              f"delta = {res.threshold:.4g}, held-out ARL0 = "
              f"{res.achieved.arl0:.1f} +/- {res.achieved.sem:.1f} "
              f"(target error {100 * res.target_error:+.1f} %)")
        if ms == 5:
            report.check("min_sub = 5 fails to bracket the target from the shipped "
                         "default within the bracket budget",
                         res.bracketing_failed,
                         f"delta={res.threshold:.3g}, ARL0={res.achieved.arl0:.0f}")
        else:
            report.check("min_sub = 30 brackets the target from the shipped default",
                         not res.bracketing_failed,
                         f"delta={res.threshold:.3g}, ARL0={res.achieved.arl0:.0f}")
    print()
    print("  min_sub = 30 is therefore the shipped default. It was declared BEFORE")
    print("  any ARL1 was measured, which makes it a configuration choice and not a")
    print("  tuned result, and the alternative was an ADWIN that could not be placed")
    print("  at the common operating point at all.")
    report.finding(
        f"ADWIN's delta is a very weak control on its false-alarm rate: ten orders of "
        f"magnitude of delta move the measured ARL0 by only {span5:.1f}x at min_sub = 5 "
        f"({sat[5][0]:.0f} to {sat[5][-1]:.0f} samples), because eps_cut grows as "
        f"sqrt(log(1/delta)). At min_sub = 5 the {cfg.target_arl0:.0f}-sample target "
        "needs delta near 1e-11 and the bracket search from the shipped default of "
        "0.002 does not reach it, so the calibration reports a failure. min_sub = 30 "
        "is the shipped default for that reason, declared before any delay was "
        "measured."
    )

    report.section("2B. The harness reset convention against published ADWIN")
    print("  Published ADWIN drops the older part of its window on a cut and keeps")
    print("  the newer part. This harness resets the whole window, so that all five")
    print("  analytic detectors are treated identically. The obvious objection is")
    print("  that the harness convention is unfair to ADWIN. It is measured here,")
    print("  and it is not: the published convention gives a SHORTER ARL0, because")
    print("  after shrinking the detector immediately re-tests the window it kept.")
    print()
    print("  delta        full reset (harness)   shrink (published)   ratio")
    for d in (1e-2, 1e-4, 1e-6, 1e-8):
        a = measure_arl0(lambda dd=d: ADWIN(delta=dd, shrink_on_detect=False),
                         lambda L, s: stationary(L, s), [58_921, 58_922, 58_923], 30_000)
        b = measure_arl0(lambda dd=d: ADWIN(delta=dd, shrink_on_detect=True),
                         lambda L, s: stationary(L, s), [58_921, 58_922, 58_923], 30_000)
        print(f"  {d:<12.0e} {a.arl0:20.1f} {b.arl0:20.1f} {a.arl0 / b.arl0:7.2f}x")
        report.check(f"the harness convention is not the pessimistic one at "
                     f"delta = {d:.0e}", a.arl0 >= b.arl0,
                     f"reset {a.arl0:.0f} vs shrink {b.arl0:.0f}")
    report.finding(
        "the full-reset convention this harness uses for ADWIN gives a LONGER "
        "measured ARL0 than published ADWIN's window-shrinking convention at every "
        "delta tested, so the convention does not disadvantage ADWIN. Both columns "
        "are in section 2B of validation/outputs/validate_robustness.txt."
    )

    report.section("2C. ADWIN's nominal delta is not a false-alarm rate")
    print("  The published bound is per cut test. The detector performs O(log n) cut")
    print("  tests per check and a check every check_every = 10 samples, so the")
    print("  nominal delta and the measured false-alarm rate are different")
    print("  quantities. Here is how different.")
    print()
    print("  delta       nominal 1/delta   measured ARL0   measured/nominal")
    for d in (1e-2, 1e-3, 1e-4, 1e-6):
        r = measure_arl0(lambda dd=d: ADWIN(delta=dd),
                         lambda L, s: stationary(L, s), [58_931, 58_932, 58_933], 30_000)
        print(f"  {d:<11.0e} {1.0 / d:15.0f} {r.arl0:15.1f} {r.arl0 * d:17.4f}")
    report.finding(
        "ADWIN's nominal delta is not a stream-level false-alarm rate: the measured "
        "ARL0 is orders of magnitude shorter than 1/delta at every delta tested. "
        "Anyone reading delta = 0.002 as 'one false alarm per 500 samples' is wrong "
        "by a large factor in the unsafe direction; the measured ARL0 at that delta "
        "is in section 1 of validate_arl_calibration.py."
    )


if __name__ == "__main__":
    raise SystemExit(run("validate_robustness",
                         "telemdrift 0.1.0 - where the calibration stops being true",
                         body))
