"""Validation 5: clock resolution is measured on this machine, not assumed.

Method (rtclock.timebase.measure_clock_resolution):
  N back-to-back reads of time.monotonic_ns() into a preallocated list, then
  the forward differences. The smallest non-zero forward difference is the
  observable tick: no interval shorter than it can be distinguished from
  zero. The median forward difference is the per-read cost, an upper bound on
  the shortest interval the clock can time as opposed to merely represent.
  The fraction of identical successive readings says which of the two -- the
  clock or the read loop -- is the limiting factor.

Error term carried everywhere a timing number is reported:
  worst case on a duration    |dt - dt_true| <= q
  standard uncertainty        u(dt) = q / sqrt(6)
with q the measured observable tick. Reference: JCGM 100:2008 (GUM)
Sec. 4.3.7 (uniform rounding error, standard deviation q/sqrt(12)) and
Sec. 5.1.2 (combination in quadrature of two independent endpoints).

Checks
  5a. Measure monotonic and perf_counter at four sample counts and report.
  5b. Internal consistency: measured tick >= advertised resolution; median
      forward difference is 0 or >= one tick; identical-reading fraction in
      [0, 1]; the derived uncertainty equals q/sqrt(6).
  5c. Repeatability: ten independent measurements of the same clock, with
      their spread, so the reader sees how stable the number is on a loaded
      host rather than taking a single sample on trust.
  5d. The quantization error term at the loop periods this package is aimed
      at, as a fraction of the period.

Every number below is a measurement taken on the build container at the time
the file was written: 1 CPU core, four concurrent build agents. It describes
that machine and no other.

Run from products/P036/:  python validation/validate_clock_resolution.py
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtclock.timebase import duration_uncertainty_s, measure_clock_resolution  # noqa: E402

failures: list[str] = []

print("=" * 78)
print("VALIDATION 5 -- clock resolution MEASURED, with its error term")
print("=" * 78)
print()
print("Platform strings (for the record, not asserted):")
print(f"    sys.platform                : {sys.platform}")
print(f"    python                      : {sys.version.split()[0]}")
for name in ("monotonic", "perf_counter"):
    info = time.get_clock_info(name)
    print(f"    time.{name:<13} : implementation={info.implementation} "
          f"monotonic={info.monotonic} adjustable={info.adjustable} "
          f"advertised resolution={info.resolution:g} s")
print()

print("5a. Measurements")
print(f"{'clock':>13}  {'samples':>8}  {'advertised [s]':>16}  {'measured tick [s]':>19}  "
      f"{'median read [s]':>17}  {'identical frac':>15}")
results = {}
for clock in ("monotonic", "perf_counter"):
    for n in (2_000, 20_000, 100_000, 400_000):
        r = measure_clock_resolution(samples=n, clock_name=clock)
        results[(clock, n)] = r
        print(f"{clock:>13}  {n:>8}  {r.advertised_s:>16.6e}  {r.measured_tick_s:>19.6e}  "
              f"{r.median_call_delta_s:>17.6e}  {r.zero_delta_fraction:>15.6f}")
print()
primary = results[("monotonic", 400_000)]
print("    Primary reported value (monotonic, 400 000 samples):")
print(f"      observable tick            q = {primary.measured_tick_s:.6e} s")
print(f"      per-read cost (median)       = {primary.median_call_delta_s:.6e} s")
print(f"      worst-case duration error    = +/- {primary.worst_case_duration_error_s:.6e} s")
print(f"      standard uncertainty q/sqrt6 = {primary.standard_duration_uncertainty_s:.6e} s")
print(f"      identical-reading fraction   = {primary.zero_delta_fraction:.6f}")
print(f"      method                       = {primary.method}")
if primary.zero_delta_fraction == 0.0:
    print("      interpretation: no two successive reads were identical, so the")
    print("      read loop is slower than the clock and the measured tick is an")
    print("      UPPER BOUND on the hardware granularity, not the granularity.")
else:
    print("      interpretation: some successive reads were identical, so the")
    print("      clock granularity, not the read loop, is the limiting factor.")
print()

print("5b. Internal consistency")
for (clock, n), r in results.items():
    checks = {
        "measured tick >= advertised": r.measured_tick_s >= r.advertised_s,
        "median read is 0 or >= one tick": (
            r.median_call_delta_s == 0.0 or r.median_call_delta_s >= r.measured_tick_s
        ),
        "identical fraction in [0,1]": 0.0 <= r.zero_delta_fraction <= 1.0,
        "u = q/sqrt(6)": math.isclose(
            r.standard_duration_uncertainty_s,
            r.measured_tick_s / math.sqrt(6.0),
            rel_tol=1e-15,
        ),
        "worst case = q": r.worst_case_duration_error_s == r.measured_tick_s,
    }
    bad = [k for k, v in checks.items() if not v]
    if bad:
        failures.append(f"5b {clock}/{n}: {bad}")
    print(f"    {clock:>13} n={n:<7} {'PASS' if not bad else 'FAIL ' + str(bad)}")
print()

print("5c. Repeatability: 10 independent measurements, monotonic, 50 000 samples")
ticks = []
medians = []
for _ in range(10):
    r = measure_clock_resolution(samples=50_000)
    ticks.append(r.measured_tick_s)
    medians.append(r.median_call_delta_s)
print(f"    observable tick  min / median / max [s] : {min(ticks):.6e} / "
      f"{statistics.median(ticks):.6e} / {max(ticks):.6e}")
print(f"    per-read cost    min / median / max [s] : {min(medians):.6e} / "
      f"{statistics.median(medians):.6e} / {max(medians):.6e}")
spread = (max(ticks) - min(ticks)) / statistics.median(ticks)
print(f"    tick spread (max-min)/median            : {spread:.3f}")
print("    The spread is the honest statement of how well this quantity is")
print("    known on a shared, loaded host. It is not asserted against a bound.")
ok_5c = all(t > 0.0 for t in ticks)
if not ok_5c:
    failures.append("5c: a measurement returned a non-positive tick")
print(f"    all measurements strictly positive      : {'PASS' if ok_5c else 'FAIL'}")
print()

print("5d. The error term at the loop rates this package targets")
q = primary.measured_tick_s
u = duration_uncertainty_s(q)
print(f"    using q = {q:.6e} s and u = q/sqrt(6) = {u:.6e} s")
print(f"{'rate [Hz]':>11}  {'period [s]':>13}  {'q/T':>12}  {'u/T':>12}  "
      f"{'q/T [ppm]':>12}")
for rate in (1.0, 10.0, 100.0, 400.0, 1_000.0, 10_000.0, 100_000.0):
    period = 1.0 / rate
    print(f"{rate:>11.0f}  {period:>13.6e}  {q / period:>12.3e}  {u / period:>12.3e}  "
          f"{1e6 * q / period:>12.3f}")
print()
print("    Read this as: at 100 kHz the clock quantization alone is "
      f"{1e6 * q / 1e-5:.0f} ppm")
print("    of the period, so a drift figure quoted at that rate on this machine")
print("    cannot be trusted below that level, whatever the loop does.")
print()

print("=" * 78)
if failures:
    print(f"RESULT: {len(failures)} FAILURE(S)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: all consistency checks PASS; the measured values are reported,")
print("        not asserted, because they describe this machine at this moment.")
print("=" * 78)
