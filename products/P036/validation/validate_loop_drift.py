"""Validation 4: the loop driver's reported drift matches an injected clock skew.

The skew is injected through SkewedSimulatedTimebase, whose sleep primitive
advances the timestamping clock by ``d*(1+eps) + wake_delay`` for a requested
sleep of ``d``. That clock is deterministic, so the comparison is against a
closed form and not against a measurement on a shared CPU.

Closed forms derived in rtclock.loop, with a = T*eps + wake_delay:
    relative mode:  drift_k = k * a                                   (exact)
    absolute mode:  drift_k = a (1 - (-eps)^k) / (1 + eps),  k >= 1   (exact)

Checks
  4a. Relative mode, a grid of 4 skews x 3 wake delays x 5 periods, every
      iteration of 500 compared against k*a.
  4b. Absolute mode, the same grid, every iteration compared against the
      closed form.
  4c. Recovered skew: fit the drift slope by least squares in relative mode
      and compare the recovered skew in ppm against the injected value.
  4d. The qualitative claim: relative-mode drift grows linearly and
      absolute-mode drift does not. Reported as a ratio over 50 000
      iterations.
  4e. A real-clock run on THIS machine, for the record only. The machine is
      shared by four concurrent build agents, so the number is a sample of a
      loaded host and is NOT asserted against any bound. It is printed with
      the measured clock resolution as its error term.

Run from products/P036/:  python validation/validate_loop_drift.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtclock.loop import (  # noqa: E402
    FixedRateLoop,
    absolute_mode_drift,
    relative_mode_drift,
)
from rtclock.timebase import (  # noqa: E402
    MonotonicTimebase,
    SkewedSimulatedTimebase,
    measure_clock_resolution,
)

failures: list[str] = []
SKEWS_PPM = (-1000.0, -1.0, 1.0, 250.0)
DELAYS_S = (0.0, 1e-7, 5e-6)
PERIODS_S = (1e-4, 1e-3, 2.5e-3, 1e-2, 0.1)
ITERS = 500

print("=" * 78)
print("VALIDATION 4 -- reported drift vs injected clock skew")
print("=" * 78)
print()
print("Injection: SkewedSimulatedTimebase.sleep(d) advances the timestamping")
print("clock by d*(1 + skew_ppm*1e-6) + wake_delay_s. Deterministic, noise-free.")
print(f"Grid: skews {SKEWS_PPM} ppm x delays {DELAYS_S} s x periods {PERIODS_S} s")
print(f"Iterations per run: {ITERS}. Every iteration is compared, not just the last.")
print()


def tolerance(period: float, index: int) -> float:
    """Combined tolerance: binary64 accumulation noise over ``index`` periods.

    The driver's clock value after ``index`` periods is about
    ``index * period``, and the drift is a difference of two such accumulated
    sums, so the roundoff floor is of order ``index * ulp(index * period)``.
    Four times that is used as the absolute floor; it is model-independent and
    is quoted alongside every result below.
    """
    return 4.0 * max(index, 1) * math.ulp(max(index, 1) * period)


def sweep(mode: str, closed) -> tuple[float, float, int]:
    """Return (worst deviation / tolerance, worst absolute deviation, count)."""
    worst_ratio = 0.0
    worst_abs = 0.0
    n = 0
    for skew in SKEWS_PPM:
        for delay in DELAYS_S:
            for period in PERIODS_S:
                tb = SkewedSimulatedTimebase(skew_ppm=skew, wake_delay_s=delay)
                report = FixedRateLoop(period_s=period, timebase=tb, mode=mode).run(
                    iterations=ITERS
                )
                for rec in report.records:
                    exp = closed(rec.index, period, skew, delay)
                    n += 1
                    dev = abs(rec.drift_s - exp)
                    tol = 1e-9 * abs(exp) + tolerance(period, rec.index)
                    worst_abs = max(worst_abs, dev)
                    worst_ratio = max(worst_ratio, dev / tol)
                    if dev > tol:
                        failures.append(
                            f"{mode} skew={skew} delay={delay} T={period} k={rec.index}: "
                            f"expected {exp!r} got {rec.drift_s!r} dev {dev:.3e} tol {tol:.3e}"
                        )
    return worst_ratio, worst_abs, n


print("Tolerance per comparison: 1e-9 * |closed form| + 4*k*ulp(k*T), the second")
print("term being binary64 accumulation noise over k periods. A ratio of")
print("deviation to tolerance below 1 is a pass.")
print()

print("4a. Relative mode: drift_k = k * (T*eps + d)")
ratio_rel, abs_rel, n_rel = sweep("relative", relative_mode_drift)
ok_4a = ratio_rel <= 1.0
if not ok_4a:
    failures.append(f"4a worst deviation/tolerance ratio {ratio_rel}")
print(f"    comparisons                 : {n_rel}")
print(f"    worst absolute deviation    : {abs_rel:.3e} s")
print(f"    worst deviation / tolerance : {ratio_rel:.3e}  (must be <= 1)")
print(f"    verdict                     : {'PASS' if ok_4a else 'FAIL'}")
print()

print("4b. Absolute mode: drift_k = a (1 - (-eps)^k) / (1 + eps)")
ratio_abs, abs_abs, n_abs = sweep("absolute", absolute_mode_drift)
ok_4b = ratio_abs <= 1.0
if not ok_4b:
    failures.append(f"4b worst deviation/tolerance ratio {ratio_abs}")
print(f"    comparisons                 : {n_abs}")
print(f"    worst absolute deviation    : {abs_abs:.3e} s")
print(f"    worst deviation / tolerance : {ratio_abs:.3e}  (must be <= 1)")
print(f"    verdict                     : {'PASS' if ok_4b else 'FAIL'}")
print()

print("4c. Recovering the injected skew from the drift slope (relative mode)")
print("    slope = T * eps, so eps_recovered = slope / T, in ppm = 1e6 * slope / T")
print(f"{'injected [ppm]':>15}  {'T [s]':>9}  {'slope [s/iter]':>17}  "
      f"{'recovered [ppm]':>16}  {'rel err':>10}  verdict")
for skew in (-1000.0, -10.0, 0.5, 1.0, 100.0, 1000.0):
    for period in (1e-3, 1e-2):
        tb = SkewedSimulatedTimebase(skew_ppm=skew, wake_delay_s=0.0)
        report = FixedRateLoop(period_s=period, timebase=tb, mode="relative").run(
            iterations=2000
        )
        slope = report.drift_slope_s_per_iteration()
        recovered = 1e6 * slope / period
        rel_err = abs(recovered - skew) / abs(skew)
        ok = rel_err <= 1e-6
        if not ok:
            failures.append(f"4c skew={skew} T={period}: recovered {recovered}")
        print(f"{skew:>15.3f}  {period:>9.4f}  {slope:>17.9e}  {recovered:>16.9f}  "
              f"{rel_err:>10.2e}  {'PASS' if ok else 'FAIL'}")
print()

print("4d. Relative mode accumulates, absolute mode does not (50 000 iterations)")
period, skew = 1e-3, 200.0
a = period * skew * 1e-6
rel_rep = FixedRateLoop(
    period_s=period, timebase=SkewedSimulatedTimebase(skew_ppm=skew), mode="relative"
).run(iterations=50_000)
abs_rep = FixedRateLoop(
    period_s=period, timebase=SkewedSimulatedTimebase(skew_ppm=skew), mode="absolute"
).run(iterations=50_000)
exp_rel = 49_999 * a
exp_abs = a / (1.0 + skew * 1e-6)
ratio = rel_rep.final_drift_s / abs_rep.final_drift_s
# 50 000 iterations of 1 ms accumulate a clock value of 50 s, so the
# roundoff floor here is larger than in 4a/4b: 1e-6 relative is used, and the
# measured deviations are printed so the reader can see the actual size.
dev_rel = abs(rel_rep.final_drift_s - exp_rel) / exp_rel
dev_abs = abs(abs_rep.final_drift_s - exp_abs) / exp_abs
ok_4d = dev_rel <= 1e-6 and dev_abs <= 1e-6 and ratio > 40_000.0
if not ok_4d:
    failures.append(f"4d ratio {ratio}")
print(f"    period                      : {period:.6e} s")
print(f"    injected skew               : {skew:.3f} ppm  -> a = T*eps = {a:.6e} s")
print(f"    relative final drift        : {rel_rep.final_drift_s:.12e} s "
      f"(closed form {exp_rel:.12e} s)")
print(f"    absolute final drift        : {abs_rep.final_drift_s:.12e} s "
      f"(closed form {exp_abs:.12e} s)")
print(f"    relative deviation from form: {dev_rel:.3e}  (tolerance 1e-6)")
print(f"    absolute deviation from form: {dev_abs:.3e}  (tolerance 1e-6)")
print(f"    relative / absolute         : {ratio:.3f}")
print(f"    relative drift slope        : {rel_rep.drift_slope_s_per_iteration():.9e} s/iter")
print(f"    absolute drift slope        : {abs_rep.drift_slope_s_per_iteration():.9e} s/iter")
print(f"    verdict                     : {'PASS' if ok_4d else 'FAIL'}")
print()

print("4e. For the record only: a real-clock run on THIS machine.")
print("    NOT ASSERTED. The build container has 1 CPU core and four concurrent")
print("    build agents, so what follows is a sample of a loaded general-purpose")
print("    host, not a performance claim and not a bound. CPython on a")
print("    general-purpose OS is not a real-time platform.")
res = measure_clock_resolution(samples=20_000)
tb = MonotonicTimebase(resolution=res)
real = FixedRateLoop(period_s=2e-3, timebase=tb, mode="absolute").run(
    iterations=300, clock_quantum_s=res.measured_tick_s
)
hist = real.duration_histogram()
print(f"    measured clock tick         : {res.measured_tick_s:.6e} s")
print(f"    worst-case duration error   : +/-{res.worst_case_duration_error_s:.6e} s")
print("    period                      : 2.000000e-03 s, 300 iterations, absolute mode")
print(f"    max |drift|                 : {real.max_abs_drift_s:.6e} s "
      f"+/- {res.worst_case_duration_error_s:.1e} s (clock quantization)")
print(f"    final drift                 : {real.final_drift_s:.6e} s")
print(f"    drift slope                 : {real.drift_slope_s_per_iteration():.6e} s/iter")
print(f"    late releases               : {real.late_releases} of 300")
print(f"    body duration p50 / p99 / max (nearest_rank): "
      f"{hist.percentile(50.0):.3e} / {hist.percentile(99.0):.3e} / "
      f"{hist.maximum():.3e} s")
print()

print("=" * 78)
if failures:
    print(f"RESULT: {len(failures)} FAILURE(S)")
    for f in failures[:20]:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: all asserted checks PASS (4e is reported, not asserted)")
print("=" * 78)
