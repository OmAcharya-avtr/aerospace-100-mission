"""Fade statistics of a correlated lognormal link, and what the definitions cost.

Four panels: a slice of the amplitude record with the threshold and the fades
shaded; the fade-duration histogram against the exponential that a textbook
would assume; outage probability and availability against threshold with the
analytic curves; and the level-crossing rate and mean fade duration against
threshold.

What to notice: the fade-duration histogram is not exponential, and it is not
close. A third of the fades are one sample long and a few last four hundred
times the mean, which no one-parameter memoryless law can do at once. The
availability panel is the one a link budget uses, and it is the panel least
sensitive to the definitional choices; the crossing-rate panel is the most
sensitive.

Writes ``../screenshots/fade_statistics_overview.png``.

Runtime: about 10 s on one core.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from linkoutage.channel import (  # noqa: E402
    analytic_level_crossing_rate,
    analytic_mean_fade_duration,
    analytic_outage_fraction,
    lognormal_amplitude_series,
)
from linkoutage.fade import (  # noqa: E402
    FadeDefinitions,
    fade_durations,
    fade_runs,
    fade_statistics,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "screenshots", "fade_statistics_overview.png")

FS = 1.0e6
TAU = 2.0e-4
SI = 0.6
N = 1_000_000
THRESHOLD = 0.6

series = lognormal_amplitude_series(N, fs_hz=FS, tau_s=TAU, si=SI, seed=41)
stats = fade_statistics(series.amplitude, THRESHOLD, FS)

print(f"record            : {N} samples at {FS:.3g} Hz = {N / FS:.4g} s")
print(f"channel           : SI = {SI}, tau = {TAU:.3g} s, "
      f"correlation length {TAU * FS:.0f} samples")
print(f"threshold         : amplitude {THRESHOLD}")
print(f"level-crossing rate : {stats.level_crossing_rate_hz:.4f} Hz")
print(f"mean fade duration  : {stats.mean_fade_duration_s:.6e} s")
print(f"outage fraction     : {stats.outage_fraction:.6f}")
print(f"availability        : {stats.availability:.6f}")
print(f"complete fades      : {stats.n_complete_fades}")
print(f"single-sample fades : {stats.n_single_sample_fades} "
      f"({stats.n_single_sample_fades / stats.n_complete_fades:.1%} of them)")

figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))

# --- panel 1: a slice of the record -----------------------------------------
ax = axes[0, 0]
lo, hi = 20_000, 26_000
t_ms = np.arange(lo, hi) / FS * 1e3
ax.plot(t_ms, series.amplitude[lo:hi], lw=0.7, color="#1f4e79")
ax.axhline(THRESHOLD, color="#c00000", lw=1.2, ls="--", label=f"threshold {THRESHOLD}")
runs = fade_runs(series.amplitude, THRESHOLD, FS)
shown = 0
for start, stop in zip(runs.start, runs.stop, strict=True):
    if start >= lo and stop <= hi:
        ax.axvspan(start / FS * 1e3, stop / FS * 1e3, color="#c00000", alpha=0.16, lw=0)
        shown += 1
ax.set_xlabel("time [ms]")
ax.set_ylabel("amplitude")
ax.set_title(
    f"6 ms of the record: {shown} fades, mean "
    f"{stats.mean_fade_duration_s * 1e6:.1f} us over the whole 1 s",
    fontsize=10,
)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.25)

# --- panel 2: fade-duration distribution ------------------------------------
ax = axes[0, 1]
complete, censored = fade_durations(series.amplitude, THRESHOLD, FS)
lengths = np.asarray(np.rint(complete * FS), dtype=int)
counts = np.bincount(lengths)[1:]
support = np.arange(1, counts.size + 1)
ax.loglog(support, counts / counts.sum(), ".", ms=4, color="#1f4e79", label="measured")
mean_len = float(lengths.mean())
p = 1.0 / mean_len
geo = (1.0 - p) ** (support - 1.0) * p
ax.loglog(support, geo, "-", lw=1.4, color="#c00000", label=f"geometric, p = {p:.4f}")
ax.set_xlabel("fade duration [samples]")
ax.set_ylabel("probability")
cv = float(complete.std(ddof=1) / complete.mean())
ax.set_title(
    f"Fade durations are not memoryless: cv = {cv:.2f} against 1.00",
    fontsize=10,
)
ax.legend(fontsize=8)
ax.grid(alpha=0.25, which="both")
ax.set_ylim(0.2 / counts.sum(), 2.0)
tail_x = float(support[-1]) * 0.45
ax.annotate(
    "measured fades out here;\nthe geometric puts less than\n1e-9 of its mass past 200",
    xy=(tail_x, 1.2 / counts.sum()),
    xytext=(3.0, 3.0 / counts.sum()),
    arrowprops={"arrowstyle": "->", "lw": 0.9},
    fontsize=8,
)

# --- panel 3: outage and availability vs threshold --------------------------
ax = axes[1, 0]
thresholds = np.linspace(0.2, 1.2, 41)
measured = [fade_statistics(series.amplitude, t, FS).outage_fraction for t in thresholds]
analytic = [analytic_outage_fraction(t, SI) for t in thresholds]
ax.semilogy(thresholds, measured, "o", ms=3.4, color="#1f4e79", label="measured outage")
ax.semilogy(thresholds, analytic, "-", lw=1.4, color="#c00000", label="analytic Phi(u)")
ax.axvline(THRESHOLD, color="0.4", lw=0.9, ls=":")
ax.set_xlabel("amplitude threshold")
ax.set_ylabel("outage probability")
ax.set_title("Outage probability against threshold, measured and analytic", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.25, which="both")
twin = ax.twinx()
twin.plot(thresholds, [1.0 - m for m in measured], lw=1.0, color="#2e7d32", alpha=0.8)
twin.set_ylabel("availability", color="#2e7d32")
twin.tick_params(axis="y", labelcolor="#2e7d32")

# --- panel 4: rate and mean fade duration vs threshold ----------------------
ax = axes[1, 1]
rate_measured = [
    fade_statistics(series.amplitude, t, FS).level_crossing_rate_hz for t in thresholds
]
rate_analytic = [
    analytic_level_crossing_rate(t, si=SI, tau_s=TAU, fs_hz=FS) for t in thresholds
]
defs_nosingle = FadeDefinitions(count_single_sample_fades=False)
rate_nosingle = [
    fade_statistics(
        series.amplitude, t, FS, definitions=defs_nosingle
    ).level_crossing_rate_hz
    for t in thresholds
]
ax.plot(thresholds, rate_measured, "o", ms=3.4, color="#1f4e79", label="measured LCR")
ax.plot(thresholds, rate_analytic, "-", lw=1.4, color="#c00000", label="analytic LCR")
ax.plot(
    thresholds,
    rate_nosingle,
    "--",
    lw=1.2,
    color="#7b1fa2",
    label="LCR, single-sample fades not counted",
)
ax.set_xlabel("amplitude threshold")
ax.set_ylabel("level-crossing rate [Hz]")
ax.set_title(
    "The definitional choice costs more than the model error", fontsize=10
)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25)
twin2 = ax.twinx()
# The mean fade duration needs enough complete fades to mean anything; below
# 30 it is noise and is not drawn.
mfd_t, mfd_measured, mfd_analytic = [], [], []
for t in thresholds:
    s_t = fade_statistics(series.amplitude, t, FS)
    if s_t.n_complete_fades < 30:
        continue
    mfd_t.append(t)
    mfd_measured.append(s_t.mean_fade_duration_s * 1e6)
    mfd_analytic.append(analytic_mean_fade_duration(t, si=SI, tau_s=TAU, fs_hz=FS) * 1e6)
twin2.plot(mfd_t, mfd_measured, lw=1.0, color="#2e7d32", alpha=0.85)
twin2.plot(mfd_t, mfd_analytic, lw=1.0, ls=":", color="#2e7d32", alpha=0.85)
twin2.set_ylabel("mean fade duration [us]", color="#2e7d32")
twin2.tick_params(axis="y", labelcolor="#2e7d32")

figure.suptitle(
    "Fade statistics of a correlated lognormal optical link "
    f"(SI = {SI}, tau = {TAU * 1e6:.0f} us, fs = {FS / 1e6:.0f} MHz, "
    f"{N} samples, seed 41)",
    fontsize=11,
)
figure.tight_layout()
figure.savefig(OUT, dpi=130)
print()
print(f"censored runs excluded from the mean: {censored.size}")
print(f"wrote screenshots/{os.path.basename(OUT)}")
