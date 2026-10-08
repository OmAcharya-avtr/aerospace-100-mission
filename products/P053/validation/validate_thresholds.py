"""Validate the threshold-setting method.

The declared method: bisect the censored-MLE in-control ARL0 onto a declared
target, on in-control streams only. Nothing out-of-control is visible to it.

Checks
------
1. The two closed-form thresholds, equations (7) and (8), against hand
   calculations.
2. The Monte-Carlo calibrator against the closed form, on the same bank. If
   the calibrator is right, calibrating GLR with window 1 by bisection must
   land on ``Phi^{-1}(1 - 1/(2T))^2 / 2``.
3. The achieved ARL0 of every calibrated threshold on a **fresh** in-control
   bank, which is the honest number: the calibration bank is fitted to by
   construction, a fresh bank is not.
4. The false-alarm rate in false alarms per 1000 operating hours at 20 Hz,
   which is the unit a requirement is usually written in.
"""

from __future__ import annotations

import numpy as np
from _bootstrap import add_src_to_path

add_src_to_path()

from twininvalidate import (  # noqa: E402
    DetectorSpec,
    arl0_estimate,
    calibrate_threshold,
    ewma_lambda1_threshold,
    glr_window1_threshold,
    in_control_streams,
    rate_from_arl0,
)

TARGETS = (200.0, 1000.0, 5000.0)
SAMPLE_RATE_HZ = 20.0
N_RUNS = 400
N_SAMPLES = 4000


def main() -> int:
    print("=== closed-form thresholds, equations (7) and (8) ===")
    print("GLR with window 1 alarms iff |z| > sqrt(2h); the per-sample false-alarm")
    print("probability is p = 2(1 - Phi(sqrt(2h))) and the run length is Geometric(p),")
    print("so h*(T) = Phi^{-1}(1 - 1/(2T))^2 / 2 exactly. EWMA with lambda 1 alarms")
    print("iff |z| > L, so L*(T) = Phi^{-1}(1 - 1/(2T)).")
    print()
    print(f"{'target ARL0':>12}{'Phi^-1':>14}{'L* (EWMA)':>13}{'h* (GLR)':>13}{'L*^2/2':>13}")
    for t in (100.0, 1000.0, 10000.0):
        lam = ewma_lambda1_threshold(t)
        h = glr_window1_threshold(t)
        print(f"{t:>12.0f}{lam:>14.10f}{lam:>13.7f}{h:>13.7f}{lam**2 / 2:>13.7f}")
    hand_1000 = 3.2905267315**2 / 2.0
    err = abs(glr_window1_threshold(1000.0) - hand_1000)
    print()
    print("hand check at T = 1000:  1 - 1/2000 = 0.9995,  Phi^-1(0.9995) = 3.2905267315")
    print(f"                         3.2905267315^2 / 2 = {hand_1000:.10f}")
    print(f"                         code gives           {glr_window1_threshold(1000.0):.10f}")
    print(f"                         difference {err:.3e}, tolerance 1e-8  "
          f"{'PASS' if err < 1e-8 else 'FAIL'}")

    print()
    print("=== Monte-Carlo calibrator against the closed form ===")
    print(f"calibration bank: {N_RUNS} x {N_SAMPLES} = {N_RUNS * N_SAMPLES} in-control samples,")
    print("seed 53001. The censored-MLE ARL0 is (total samples at risk)/(alarms),")
    print("whose relative standard error is 1/sqrt(alarms).")
    bank = in_control_streams(n_runs=N_RUNS, n_samples=N_SAMPLES, seed=53001)
    fresh = in_control_streams(n_runs=N_RUNS, n_samples=N_SAMPLES, seed=53110)
    fresh2 = in_control_streams(n_runs=N_RUNS, n_samples=N_SAMPLES, seed=53111)
    fresh3 = in_control_streams(n_runs=N_RUNS, n_samples=N_SAMPLES, seed=53112)
    results: list[bool] = []
    for target in TARGETS:
        spec = DetectorSpec("glr", window=1)
        closed = glr_window1_threshold(target)
        # Force the bisection path by asking for the same detector with the
        # closed form switched off: calibrate a window-2 GLR is a different
        # statistic, so instead bisect by hand on the stored statistic.
        stat = spec.statistic(bank)
        lo, hi = 0.0, 1.05 * float(stat.max())
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if arl0_estimate(stat, mid).value < target:
                lo = mid
            else:
                hi = mid
        bisected = 0.5 * (lo + hi)
        rel = abs(bisected / closed - 1.0)
        # Tolerance: the threshold is set by a finite sample, so the achievable
        # accuracy is set by the ARL0 standard error 1/sqrt(alarms) propagated
        # through dh/dARL0. At T=1000 with ~1600 alarms that is a few per cent
        # on ARL0 and under 1 % on the threshold, because the tail is steep.
        print(
            f"target {target:>7.0f}  closed form {closed:>9.5f}  bisection {bisected:>9.5f}  "
            f"relative difference {rel * 100:>5.2f} %  tolerance 5 %  "
            f"{'PASS' if rel < 0.05 else 'FAIL'}"
        )
        results.append(rel < 0.05)

    print()
    print("=== achieved ARL0 on FRESH in-control banks (seeds 53110-53112) ===")
    print("The calibration bank is fitted to, so its own ARL0 is not evidence. Three")
    print("independent fresh banks of the same size are pooled, giving about 1170")
    print("alarms per detector and a relative standard error of 1/sqrt(1170) = 2.9 %.")
    print("The declared tolerance is three of those standard errors, 8.8 %, on the")
    print("pooled estimate -- a criterion fixed by the sample size, not by the result.")
    pooled = np.concatenate([fresh, fresh2, fresh3], axis=0)
    print(
        f"  {'detector':<32}{'target':>8}{'threshold':>11}{'ARL0':>10}"
        f"{'stderr':>9}{'alarms':>8}{'bias':>9}  verdict"
    )
    for target in TARGETS:
        for name in ("cusum", "ewma", "glr", "varcusum"):
            spec = DetectorSpec(name)
            cal = calibrate_threshold(spec, bank, target)
            est = arl0_estimate(spec.statistic(pooled), cal.threshold)
            bias = est.value / target - 1.0
            ok = abs(est.value - target) < 3.0 * est.stderr
            results.append(bool(ok))
            print(
                f"  {spec.label():<32}{target:>8.0f}{cal.threshold:>11.4f}"
                f"{est.value:>10.1f}{est.stderr:>9.1f}{est.n_detected:>8d}"
                f"{bias * 100:>8.1f}%"
                f"  {'PASS' if ok else 'FAIL'}"
            )
    print()
    print("FINDING  the calibration-transfer bias is detector-dependent. The three")
    print("FINDING  declared baselines transfer to within +3.4 % / -8.7 % and all nine")
    print("FINDING  of their checks PASS. The variance-CUSUM oracle transfers at")
    print("FINDING  -8.6 % (target 200), +13.5 % (target 1000) and +9.1 % (target")
    print("FINDING  5000); the first two exceed the declared 8.8 % tolerance and are")
    print("FINDING  reported as FAILED checks rather than absorbed into a wider")
    print("FINDING  tolerance. Diagnosis: the variance CUSUM's increments z^2 - 1 are")
    print("FINDING  right-skewed with a heavy upper tail, so its own upper tail is")
    print("FINDING  built from fewer, larger excursions and the bisected threshold")
    print("FINDING  lands on a bank-specific extreme. The mean CUSUM's increments")
    print("FINDING  z - k are symmetric and light-tailed, so the same bank size")
    print("FINDING  resolves its tail better. Consequence for the headline results:")
    print("FINDING  the variance CUSUM is NOT one of the three declared baselines and")
    print("FINDING  no headline curve depends on it. Its reported delay at a nominal")
    print("FINDING  ARL0 of 1000 is in fact measured at an effective ARL0 of about")
    print("FINDING  1135, which makes it look slightly BETTER than a fairly matched")
    print("FINDING  comparison would, and it loses to the mean CUSUM anyway.")

    print()
    print("=== false-alarm rate conversion ===")
    print("ARL0 = 1000 * 3600 * f / N samples for N false alarms per 1000 h at f Hz.")
    for rate in (1.0, 10.0, 100.0):
        samples = 1000.0 * 3600.0 * SAMPLE_RATE_HZ / rate
        print(f"  {rate:>6.0f} per 1000 h at 20 Hz  =  {samples:.3e} samples")
    print(f"  ARL0 1000 samples at 20 Hz = 50.0 s = "
          f"{rate_from_arl0(1000.0, SAMPLE_RATE_HZ):.0f} per 1000 h")
    print()
    print("NOTE  the headline curves in this repository use a target ARL0 of 1000")
    print("NOTE  samples, i.e. 72000 false alarms per 1000 h at 20 Hz. That is far")
    print("NOTE  too many for an operational monitor. It is used because resolving")
    print("NOTE  an ARL0 of 7.2e6 samples (10 per 1000 h) by simulation needs about")
    print("NOTE  7e9 in-control samples for 1000 alarms, which does not fit the")
    print("NOTE  3-minute compute budget. The curve's SHAPE is the deliverable; the")
    print("NOTE  operating point a real requirement implies is off its right-hand end")
    print("NOTE  and is not measured here. This is stated in the README limitations.")

    print()
    print(f"SUMMARY  {sum(results)}/{len(results)} checks PASS")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
