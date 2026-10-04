"""Example 3: latency histogram with both percentile definitions marked.

Produces ../screenshots/latency_histogram.png with two panels:

  left   the histogram of a synthetic 20 000-sample control-loop latency
         trace with the deadline, the two p99.9 values (nearest-rank and
         linear) and the maximum marked, so the difference between the two
         definitions is a visible distance on the x axis rather than a
         footnote;
  right  the full percentile curve from p0 to p100 under both definitions,
         with the region where they differ shaded, plus the overrun count as
         a function of the budget.

Run from products/P036/:  python examples/latency_histogram.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtclock.histogram import LatencyHistogram  # noqa: E402

PERIOD_S = 2.5e-3  # a 400 Hz loop
rng = np.random.default_rng(20261004)
n = 20_000
# Synthetic, not measured: a lognormal body around 1.2 ms with a 3 per cent
# heavy-tail contamination. Fixed seed, so the figure regenerates exactly.
base = rng.lognormal(mean=math.log(1.2e-3), sigma=0.25, size=n)
spikes = rng.random(n) < 0.03
base[spikes] *= 1.0 + 2.0 * rng.random(int(spikes.sum()))

hist = LatencyHistogram(label="loop latency")
hist.extend(base.tolist())

p999_nr = hist.percentile(99.9, "nearest_rank")
p999_lin = hist.percentile(99.9, "linear")
p50 = hist.percentile(50.0, "nearest_rank")
rep = hist.overruns(PERIOD_S)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

ax1.hist(np.asarray(hist.samples) * 1e3, bins=120, color="#90a4ae",
         edgecolor="#546e7a", lw=0.3)
for value, color, style, label in (
    (p50, "#1f4e79", "-", f"p50 (nearest-rank) = {p50 * 1e3:.4f} ms"),
    (p999_nr, "#c62828", "-", f"p99.9 nearest-rank = {p999_nr * 1e3:.4f} ms"),
    (p999_lin, "#ef6c00", "--", f"p99.9 linear (type 7) = {p999_lin * 1e3:.4f} ms"),
    (hist.maximum(), "#4a148c", ":", f"max = {hist.maximum() * 1e3:.4f} ms"),
):
    ax1.axvline(value * 1e3, color=color, ls=style, lw=1.6, label=label)
ax1.axvline(PERIOD_S * 1e3, color="k", ls="-.", lw=1.8,
            label=f"period / budget = {PERIOD_S * 1e3:.3f} ms")
ax1.set_yscale("log")
ax1.set_xlabel("loop latency  [ms]")
ax1.set_ylabel("count (log scale)")
ax1.set_title(f"Synthetic 400 Hz loop latency, N = {hist.count}\n"
              f"{rep.count} overruns ({rep.fraction * 100:.2f} %), "
              f"longest cascade {rep.longest_consecutive_run}")
ax1.legend(fontsize=8, framealpha=0.95)
ax1.grid(alpha=0.3, lw=0.5)

ps = np.concatenate([np.linspace(0.0, 99.0, 400), np.linspace(99.0, 100.0, 600)])
nr = np.array([hist.percentile(float(p), "nearest_rank") for p in ps])
lin = np.array([hist.percentile(float(p), "linear") for p in ps])
gap_us = (nr - lin) * 1e6

ax2.plot(ps, gap_us, "-", color="#6a1b9a", lw=1.4)
ax2.axhline(0.0, color="k", lw=0.9)
ax2.plot([99.9], [(p999_nr - p999_lin) * 1e6], "o", color="#c62828", ms=9, mec="k", mew=1.0,
         zorder=5, label=f"p99.9: {(p999_nr - p999_lin) * 1e6:+.1f} $\\mu$s "
                        f"({100 * (p999_nr - p999_lin) / p999_lin:+.2f} %)")
gap99_us = (hist.percentile(99.0, "nearest_rank") - hist.percentile(99.0, "linear")) * 1e6
ax2.plot([99.0], [gap99_us], "s", color="#1f4e79", ms=8, mec="k", mew=1.0, zorder=5,
         label=f"p99: {gap99_us:+.2f} $\\mu$s")
ax2.set_xlabel("percentile $p$")
ax2.set_ylabel("nearest-rank $-$ linear  [$\\mu$s]")
ax2.set_title("The two definitions differ, and by how much\n"
              "p99 to p100, where the samples thin out")
ax2.set_xlim(99.0, 100.0)
ax2.grid(alpha=0.3, lw=0.5)
ax2.legend(fontsize=9, loc="upper left", framealpha=0.95)
ax2.text(0.02, 0.04,
         "both are exact over the stored samples;\n"
         "they are answers to different questions,\n"
         "not an approximation of one another",
         transform=ax2.transAxes, ha="left", va="bottom", fontsize=8.5,
         bbox={"boxstyle": "round", "fc": "white", "ec": "#9e9e9e", "alpha": 0.9})

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "screenshots" / "latency_histogram.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
print()
print("Synthetic trace (seed 20261004, lognormal body + 3 % heavy tail):")
for key, value in hist.summary(
    percentiles=(50.0, 90.0, 99.0, 99.9, 100.0), method="nearest_rank"
).items():
    print(f"  {key:<28} {value:.9e}" if key != "count" else f"  {key:<28} {int(value)}")
print()
print("Same percentiles under the linear (type 7) definition:")
for key, value in hist.summary(percentiles=(50.0, 90.0, 99.0, 99.9), method="linear").items():
    if key != "count":
        print(f"  {key:<28} {value:.9e}")
print()
print(f"p99.9 nearest-rank minus linear : {(p999_nr - p999_lin):+.9e} s")
print(f"overruns against the {PERIOD_S * 1e3:.3f} ms period : {rep.count} of {hist.count} "
      f"({rep.fraction * 100:.4f} %)")
print(f"longest consecutive overrun run : {rep.longest_consecutive_run}")
print(f"worst overshoot                 : {rep.worst_overshoot_s:.9e} s")
