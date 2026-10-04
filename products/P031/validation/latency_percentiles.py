"""Check 2 — latency percentiles reproduce an injected known distribution.

Requirement: "latency histogram percentiles reproduce an injected known
latency distribution to within its sampling error (state the error)".

The injected distribution
-------------------------
A shifted exponential, which is the standard minimal model of an execution
time: an irreducible minimum plus an exponential tail.

    F(x) = 1 - exp(-(x - a) / b),    x >= a
    q(p) = a - b ln(1 - p)                                         [s]
    f(q(p)) = (1 - p) / b                                          [1/s]

with ``a`` the offset [s] and ``b`` the scale [s]. The quantile and density
are elementary; the exponential distribution and its quantile function are in
any standard reference (e.g. Johnson, Kotz & Balakrishnan, *Continuous
Univariate Distributions*, Vol. 1, 2nd ed., Wiley 1994, Chapter 19).

The stated sampling error
-------------------------
The sample quantile is asymptotically normal with standard error

    se(p) = sqrt(p (1 - p) / n) / f(q(p)) = b sqrt(p / ((1 - p) n))  [s]

(Serfling, *Approximation Theorems of Mathematical Statistics*, Wiley 1980,
§2.3.3, Corollary 2.3.3B). The tolerance used below is **3 se**, a two-sided
99.7 % band for a normal deviate. The band is computed from the distribution
and the sample size before the comparison; it is not tuned to the result.

Three things are checked
------------------------
1. The exact-mode histogram's percentiles against ``q(p)``, within 3 se.
2. The loop's ``total`` histogram against the same reference, to show the loop
   neither loses nor distorts samples when the durations are injected.
3. The binned-mode histogram's error against the exact mode, which must lie in
   ``[0, one bin width]`` by construction.

Run: ``python validation/latency_percentiles.py``
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hilforge.backends import make_backend_pair
from hilforge.loop import STAGES, HilLoop, LoopConfig
from hilforge.timing import LatencyHistogram, PeriodSpec, quantile_standard_error

SEED = 7310
N = 200_000
OFFSET_S = 1.20e-3
SCALE_S = 0.90e-3
PROBS = (0.50, 0.90, 0.99, 0.999)
K_SIGMA = 3.0
BIN_WIDTH_S = 2.0e-5
N_LOOP = 20_000
PERIOD = 0.010


def analytic_quantile(p: float) -> float:
    """``q(p) = a - b ln(1 - p)`` [s]."""
    return OFFSET_S - SCALE_S * math.log(1.0 - p)


def analytic_density(p: float) -> float:
    """``f(q(p)) = (1 - p) / b`` [1/s]."""
    return (1.0 - p) / SCALE_S


def main() -> int:
    rng = np.random.Generator(np.random.PCG64(SEED))
    samples = OFFSET_S + rng.exponential(SCALE_S, size=N)

    print("HilForge validation 2 — latency percentiles against a known distribution")
    print("=" * 78)
    print(f"injected law      : shifted exponential, a = {OFFSET_S:.6e} s, "
          f"b = {SCALE_S:.6e} s")
    print(f"samples           : {N}   seed {SEED}")
    print(f"tolerance         : {K_SIGMA:.1f} x se, se(p) = b sqrt(p / ((1-p) n)) "
          f"[Serfling 1980 §2.3.3]")
    print(f"sample mean       : {float(samples.mean()):.9e} s "
          f"(analytic a + b = {OFFSET_S + SCALE_S:.9e} s)")
    print()

    hist = LatencyHistogram()
    hist.extend(samples)

    print("2a. exact-mode histogram vs the analytic quantile")
    print(f"  {'p':>7} {'analytic [s]':>15} {'measured [s]':>15} "
          f"{'deviation [s]':>15} {'3 se [s]':>13} {'|dev|/se':>9}  result")
    failures = 0
    for p in PROBS:
        want = analytic_quantile(p)
        got = hist.percentile(p, method="linear")
        se = quantile_standard_error(p, N, analytic_density(p))
        dev = got - want
        ok = abs(dev) <= K_SIGMA * se
        failures += 0 if ok else 1
        print(f"  {p:>7.3f} {want:>15.6e} {got:>15.6e} {dev:>15.3e} "
              f"{K_SIGMA * se:>13.3e} {abs(dev) / se:>9.3f}  "
              f"{'PASS' if ok else 'FAIL'}")
    print()

    # 2b — the same samples through the loop, injected as iteration durations.
    sim, _ = make_backend_pair(seed=SEED, sample_dt_s=PERIOD)
    loop_durations = tuple(samples[:N_LOOP])
    cfg = LoopConfig(
        period=PeriodSpec(period_s=PERIOD),
        n_iterations=N_LOOP,
        injected_durations_s=loop_durations,
        record_signals=False,
    )
    record = HilLoop(sim, cfg).run()
    total_hist = record.stage_histograms["total"]
    direct = LatencyHistogram()
    direct.extend(samples[:N_LOOP])

    print(f"2b. the loop's own 'total' histogram over the first {N_LOOP} samples")
    print(f"  {'p':>7} {'direct [s]':>15} {'loop [s]':>15} {'difference [s]':>16}  result")
    for p in PROBS:
        a = direct.percentile(p, method="linear")
        b = total_hist.percentile(p, method="linear")
        same = abs(b - a) <= 1e-15 * max(abs(a), 1.0)
        failures += 0 if same else 1
        print(f"  {p:>7.3f} {a:>15.6e} {b:>15.6e} {b - a:>16.3e}  "
              f"{'PASS' if same else 'FAIL'}")
    print(f"  loop samples recorded : {len(total_hist)} (expected {N_LOOP})")
    print(f"  loop mean total       : {total_hist.mean:.9e} s")
    print()

    print("2c. the four stages sum to the total, iteration by iteration")
    stage_sum = sum(record.stage_durations_s(s) for s in STAGES)
    totals = record.durations_s()
    max_resid = float(np.max(np.abs(stage_sum - totals)))
    # Four float64 additions of values bounded by max(total): the round-off
    # bound is a small multiple of eps times that magnitude.
    bound = 8.0 * float(np.finfo(np.float64).eps) * float(np.max(totals))
    ok_sum = max_resid <= bound
    failures += 0 if ok_sum else 1
    print(f"  max |sum(stages) - total| : {max_resid:.3e} s  "
          f"(round-off bound {bound:.3e} s)  {'PASS' if ok_sum else 'FAIL'}")
    print("  (the stage split is exact by construction: the injected total is")
    print("   multiplied by fractions that sum to 1, and summed back in the loop)")
    print()

    print(f"2d. binned mode, bin width {BIN_WIDTH_S:.3e} s")
    binned = LatencyHistogram(mode="binned", bin_width_s=BIN_WIDTH_S, n_bins=8192)
    binned.extend(samples)
    print(f"  {'p':>7} {'exact [s]':>15} {'binned [s]':>15} {'error [s]':>13} "
          f"{'bound [s]':>13}  result")
    for p in PROBS:
        exact = hist.percentile(p)
        approx = binned.percentile(p)
        err = approx - exact
        ok = 0.0 <= err <= BIN_WIDTH_S
        failures += 0 if ok else 1
        print(f"  {p:>7.3f} {exact:>15.6e} {approx:>15.6e} {err:>13.3e} "
              f"{BIN_WIDTH_S:>13.3e}  {'PASS' if ok else 'FAIL'}")
    print(f"  overflow samples : {binned.overflow}")
    print()
    print(f"verdict: {'PASS' if failures == 0 else f'{failures} FAILURES'}")
    print("note: these are injected synthetic latencies, not measured hardware")
    print("      latencies. The check validates the histogram and the loop's")
    print("      sample handling, not any machine's timing.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
