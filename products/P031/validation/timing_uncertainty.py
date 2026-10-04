"""Check 6 — uncertainty analysis over timing measurements.

Requirement (Level 3): "uncertainty analysis over timing measurements,
including clock resolution as a stated error term".

The budget
----------
A measured duration is a difference of two quantised timestamps. Two terms:

* **Type A, statistical** (JCGM 100:2008 §4.2): from the observed scatter,
  ``u_A = s / sqrt(n)`` for the mean of ``n`` durations.
* **Type B, clock resolution** (JCGM 100:2008 §4.3.7; the uniform-quantisation
  variance ``delta^2/12`` is Bennett, "Spectra of Quantized Signals", *Bell
  System Technical Journal* 27(3):446-472, 1948): each timestamp carries
  variance ``delta^2/12``, a duration therefore ``delta^2/6``, and the mean of
  ``n`` independent durations ``delta^2/(6n)``, so
  ``u_B = delta / sqrt(6 n)``.

They are independent and combine in quadrature (JCGM 100:2008 §5.1.2):
``u_c = sqrt(u_A^2 + u_B^2)``, expanded with ``k = 2`` for about 95 % coverage
(§6.3.3).

The timing-call overhead is a **bias**, not an uncertainty: it shifts every
duration the same way. It is measured and reported separately, never added in
quadrature and never subtracted from individual samples (which would make
short durations negative).

Two parts
---------
6a verifies the arithmetic against a case computed by hand, and verifies the
scaling laws (``u_A ~ 1/sqrt(n)``, ``u_B ~ delta``).

6b applies the budget to an actual measured loop run on this machine, and
reports what fraction of the combined uncertainty each term contributes. On a
nanosecond-resolution clock the resolution term is negligible against the
scatter; the point of stating it is that on a 15.6 ms clock it would not be,
and the budget would say so instead of hiding it.

Run: ``python validation/timing_uncertainty.py``
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hilforge.backends import make_backend_pair
from hilforge.loop import STAGES, HilLoop, LoopConfig
from hilforge.timebase import (
    MonotonicTimebase,
    clock_report,
    duration_resolution_uncertainty,
)
from hilforge.timing import PeriodSpec, timing_uncertainty

PERIOD = 0.010
N_LOOP = 20_000
CLOCK_SAMPLES = 20_000
SEED = 20261004


def main() -> int:
    print("HilForge validation 6 — timing-measurement uncertainty")
    print("=" * 72)
    print()

    print("6a. the arithmetic, against a hand-computed case")
    # Four identical durations of 1 ms, delta = 1e-6 s.
    #   s = 0            -> u_A = 0
    #   u_B = delta / sqrt(6 n) = 1e-6 / sqrt(24) = 2.041241452e-07 s
    #   u_c = u_B,  U(k=2) = 4.082482905e-07 s
    unc = timing_uncertainty([1.0e-3] * 4, resolution_s=1.0e-6, call_overhead_s=5.0e-8)
    hand = {
        "u_statistical_s": 0.0,
        "u_resolution_s": 1.0e-6 / math.sqrt(24.0),
        "combined_s": 1.0e-6 / math.sqrt(24.0),
        "expanded_k2_s": 2.0e-6 / math.sqrt(24.0),
    }
    failures = 0
    for key, want in hand.items():
        got = getattr(unc, key)
        ok = abs(got - want) <= 1e-18 + 1e-12 * abs(want)
        failures += 0 if ok else 1
        print(f"  {key:<18} hand={want:.9e}  code={got:.9e}  "
              f"{'PASS' if ok else 'FAIL'}")
    # delta / sqrt(6) for a single duration.
    single = duration_resolution_uncertainty(1.0e-6)
    ok = abs(single - 1.0e-6 / math.sqrt(6.0)) < 1e-21
    failures += 0 if ok else 1
    print(f"  {'u(single duration)':<18} hand={1.0e-6 / math.sqrt(6.0):.9e}  "
          f"code={single:.9e}  {'PASS' if ok else 'FAIL'}")
    print()

    print("  scaling laws")
    rng = np.random.Generator(np.random.PCG64(SEED))
    base = rng.normal(1.0e-3, 5.0e-5, size=40_000)
    u_small = timing_uncertainty(base[:10_000], resolution_s=1.0e-7)
    u_large = timing_uncertainty(base[:40_000], resolution_s=1.0e-7)
    ratio_a = u_small.u_statistical_s / u_large.u_statistical_s
    ratio_b = u_small.u_resolution_s / u_large.u_resolution_s
    for label, got, want in (
        ("u_A ratio at 4x n (expect 2)", ratio_a, 2.0),
        ("u_B ratio at 4x n (expect 2)", ratio_b, 2.0),
    ):
        ok = abs(got - want) < 0.05
        failures += 0 if ok else 1
        print(f"  {label:<32} measured={got:.6f}  {'PASS' if ok else 'FAIL'}")
    u_coarse = timing_uncertainty(base[:10_000], resolution_s=1.0e-3)
    ratio_delta = u_coarse.u_resolution_s / u_small.u_resolution_s
    ok = abs(ratio_delta - 1.0e4) / 1.0e4 < 1e-9
    failures += 0 if ok else 1
    print(f"  {'u_B ratio at 1e4x delta':<32} measured={ratio_delta:.6e}  "
          f"{'PASS' if ok else 'FAIL'}")
    print()

    print("6b. the measured clock on this machine")
    clock = clock_report(CLOCK_SAMPLES)
    print(clock.as_text())
    tb = MonotonicTimebase()
    assert tb.resolution_s > 0.0
    print()

    print(f"6c. the budget applied to a measured {N_LOOP}-iteration loop run")
    sim, _ = make_backend_pair(seed=SEED, sample_dt_s=PERIOD)
    cfg = LoopConfig(
        period=PeriodSpec(period_s=PERIOD),
        n_iterations=N_LOOP,
        record_signals=False,
    )
    record = HilLoop(sim, cfg).run()
    print(f"  deterministic_timing : {record.deterministic_timing} "
          f"(False means a wall clock was read)")
    print(f"  is_hardware          : {record.is_hardware}")
    print()
    print(f"  {'stage':<10} {'n':>7} {'mean [s]':>13} {'stdev [s]':>13} "
          f"{'u_A [s]':>12} {'u_B [s]':>12} {'u_c [s]':>12} {'u_B/u_c':>9}")
    for stage in list(STAGES) + ["total"]:
        if stage == "total":
            data = record.durations_s()
        else:
            data = record.stage_durations_s(stage)
        unc = timing_uncertainty(
            data,
            resolution_s=clock.measured_resolution_s,
            call_overhead_s=clock.call_overhead_s,
        )
        print(f"  {stage:<10} {unc.n:>7} {unc.mean_s:>13.6e} {unc.stdev_s:>13.6e} "
              f"{unc.u_statistical_s:>12.3e} {unc.u_resolution_s:>12.3e} "
              f"{unc.combined_s:>12.3e} "
              f"{unc.u_resolution_s / unc.combined_s:>9.2e}")
    print()
    total_unc = timing_uncertainty(
        record.durations_s(),
        resolution_s=clock.measured_resolution_s,
        call_overhead_s=clock.call_overhead_s,
    )
    print("  full budget for the mean total iteration duration")
    for line in total_unc.as_text().splitlines():
        print(f"    {line}")
    print()
    bias_fraction = clock.call_overhead_s * len(STAGES) / total_unc.mean_s
    print(f"  timing-call bias on a 4-stage iteration : "
          f"{clock.call_overhead_s * len(STAGES):.3e} s")
    print(f"  as a fraction of the mean iteration     : {bias_fraction:.4f}")
    print("  (four stages, one call pair each; reported, never subtracted)")
    print()
    print("  INTERPRETATION")
    print("  The resolution term is negligible here because the clock is")
    print("  nanosecond-resolution; the scatter, which is contention on a")
    print("  shared single-core container, dominates by orders of magnitude.")
    print("  The term is carried anyway because on a coarse clock it would")
    print("  dominate instead, and a budget that drops it would be silently")
    print("  wrong there. The timing-call bias above is not negligible: it is")
    print("  a measurable fraction of a short iteration and is why this")
    print("  harness reports per-stage latency rather than claiming to measure")
    print("  the work alone.")
    print()
    print(f"verdict: {'PASS' if failures == 0 else f'{failures} FAILURES'}")
    print("note: every number in 6b and 6c is a host-side measurement on a")
    print("      shared, single-core cloud container. None of them is a")
    print("      hardware measurement and none supports a Level 4 claim.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
