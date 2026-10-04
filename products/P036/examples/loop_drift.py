"""Example 4: drift from an injected clock skew, absolute vs relative waits.

Produces ../screenshots/loop_drift.png with two panels:

  left   drift against iteration for a +200 ppm injected skew under both
         scheduling modes, with the closed forms overplotted. Relative waits
         accumulate linearly; absolute waits do not. The y axis is symmetric
         log so both can share it.
  right  drift after 10 000 iterations against injected skew from -1000 to
         +1000 ppm, for both modes, against the closed forms. The ratio
         between the two modes is the whole argument for absolute scheduling.

The clock is SkewedSimulatedTimebase: deterministic and noise-free, so the
figure is a comparison against arithmetic and not a measurement of the host.
A real-clock sample from this machine is printed at the end for the record
and is not plotted, because the host is a shared 1-core container.

Run from products/P036/:  python examples/loop_drift.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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

PERIOD_S = 2.5e-3
SKEW_PPM = 200.0
ITERS = 10_000

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

runs = {}
ax1b = ax1.twinx()
handles = []
for mode, color, closed, axis in (
    ("relative", "#c62828", relative_mode_drift, ax1),
    ("absolute", "#1f4e79", absolute_mode_drift, ax1b),
):
    tb = SkewedSimulatedTimebase(skew_ppm=SKEW_PPM)
    report = FixedRateLoop(period_s=PERIOD_S, timebase=tb, mode=mode).run(iterations=ITERS)
    runs[mode] = report
    ks = np.array([r.index for r in report.records])
    drift_us = np.array(report.drifts_s) * 1e6
    expected_us = np.array([closed(int(k), PERIOD_S, SKEW_PPM, 0.0) for k in ks]) * 1e6
    (line,) = axis.plot(ks, drift_us, "-", color=color, lw=1.8,
                        label=f"{mode} waits, measured by the driver")
    (mark,) = axis.plot(ks[::500], expected_us[::500], "o", color=color, ms=6, mfc="none",
                        mew=1.4, label=f"{mode} closed form")
    handles += [line, mark]

# Two linear axes rather than one log axis: a straight line is the whole point
# of the relative-mode curve, and a log axis would hide it.
ax1.set_xlabel("iteration $k$")
ax1.set_ylabel("RELATIVE mode drift  [$\\mu$s]", color="#c62828")
ax1.tick_params(axis="y", labelcolor="#c62828")
ax1.set_ylim(-250, 5250)
ax1b.set_ylabel("ABSOLUTE mode drift  [$\\mu$s]", color="#1f4e79")
ax1b.tick_params(axis="y", labelcolor="#1f4e79")
ax1b.set_ylim(-0.05, 0.90)
ax1.set_title(f"Injected skew {SKEW_PPM:.0f} ppm, T = {PERIOD_S * 1e3:.1f} ms\n"
              "relative drift grows as $k\\,T\\epsilon$; absolute drift is flat at "
              "$T\\epsilon/(1+\\epsilon)$")
ax1.grid(alpha=0.3, lw=0.5)
ax1.legend(handles=handles, fontsize=8.5, loc="upper left", framealpha=0.95)
ax1.text(0.98, 0.04,
         "note the two y scales:\nleft 0-5250 $\\mu$s, right 0-0.90 $\\mu$s\n"
         "(a factor of 10$^4$)",
         transform=ax1.transAxes, ha="right", va="bottom", fontsize=8,
         bbox={"boxstyle": "round", "fc": "white", "ec": "#9e9e9e", "alpha": 0.9})

skews = np.linspace(-1000.0, 1000.0, 81)
for mode, color, closed in (
    ("relative", "#c62828", relative_mode_drift),
    ("absolute", "#1f4e79", absolute_mode_drift),
):
    measured = []
    for s in skews:
        tb = SkewedSimulatedTimebase(skew_ppm=float(s))
        rep = FixedRateLoop(period_s=PERIOD_S, timebase=tb, mode=mode).run(iterations=2000)
        measured.append(rep.final_drift_s)
    expected = [closed(1999, PERIOD_S, float(s), 0.0) for s in skews]
    ax2.plot(skews, np.array(measured) * 1e6, "-", color=color, lw=1.8,
             label=f"{mode}, measured")
    ax2.plot(skews[::6], np.array(expected[::6]) * 1e6, "o", color=color, ms=5,
             mfc="none", mew=1.3, label=f"{mode}, closed form")

ax2.set_yscale("symlog", linthresh=1e-2)
ax2.set_xlabel("injected sleep-primitive skew  [ppm]")
ax2.set_ylabel("drift after 2000 iterations  [$\\mu$s]")
ax2.set_title("Drift after 2000 iterations vs injected skew\n"
              "T = 2.5 ms; markers are the closed forms")
ax2.axhline(0.0, color="k", lw=0.8)
ax2.axvline(0.0, color="k", lw=0.8)
ax2.grid(alpha=0.3, lw=0.5, which="both")
ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.95)

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "screenshots" / "loop_drift.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
print()
a = PERIOD_S * SKEW_PPM * 1e-6
print(f"period T                        : {PERIOD_S:.6e} s")
print(f"injected skew eps               : {SKEW_PPM:.1f} ppm")
print(f"per-iteration error a = T*eps   : {a:.9e} s")
print()
for mode, closed in (("relative", relative_mode_drift), ("absolute", absolute_mode_drift)):
    rep = runs[mode]
    exp = closed(ITERS - 1, PERIOD_S, SKEW_PPM, 0.0)
    print(f"{mode:>9} mode after {ITERS} iterations:")
    print(f"            drift measured      : {rep.final_drift_s:.12e} s")
    print(f"            drift closed form   : {exp:.12e} s")
    print(f"            relative difference : {abs(rep.final_drift_s - exp) / abs(exp):.3e}")
    print(f"            max |drift|         : {rep.max_abs_drift_s:.12e} s")
    print(f"            drift slope         : {rep.drift_slope_s_per_iteration():.9e} s/iter")
ratio = runs["relative"].final_drift_s / runs["absolute"].final_drift_s
print()
print(f"relative / absolute final drift : {ratio:.1f}x")
print()
res = measure_clock_resolution(samples=20_000)
real = FixedRateLoop(
    period_s=PERIOD_S, timebase=MonotonicTimebase(resolution=res), mode="absolute"
).run(iterations=200, clock_quantum_s=res.measured_tick_s)
print("For the record only, on this shared 1-core container (NOT a bound and")
print("NOT plotted): a 200-iteration real-clock run at the same period.")
print(f"  measured clock tick           : {res.measured_tick_s:.6e} s")
print(f"  max |drift|                   : {real.max_abs_drift_s:.6e} s "
      f"+/- {res.worst_case_duration_error_s:.1e} s")
print(f"  late releases                 : {real.late_releases} of 200")
