"""Fade-duration survival: what the exponential assumption gets wrong, and where.

Three panels: the survival function on a log axis, where an exponential is a
straight line and the measured curve is not; the quantile-quantile plot of the
measured durations against the fitted exponential; and the ratio of measured to
exponential survival, which is the number that matters to anyone sizing an
interleaver or a retransmission timer.

What to notice: the measured survival is above the exponential at both ends and
below it in the middle. A one-parameter memoryless law cannot have a third of
its mass at one sample and a tail reaching four hundred times the mean, so it
splits the difference and is wrong everywhere. The ratio panel shows the
exponential under-predicting long fades by orders of magnitude; a timer sized
from it fires too early, and an interleaver sized from it is too shallow.

Writes ``../screenshots/fade_duration_fit.png``.

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
from scipy import stats as sps  # noqa: E402

from linkoutage.channel import lognormal_amplitude_series  # noqa: E402
from linkoutage.distributions import (  # noqa: E402
    compare_fade_duration_models,
    kaplan_meier_survival,
)
from linkoutage.fade import fade_durations  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "screenshots", "fade_duration_fit.png")

FS = 1.0e6
TAU = 2.0e-4
SI = 0.6
N = 2_000_000
THRESHOLD = 0.6

series = lognormal_amplitude_series(N, fs_hz=FS, tau_s=TAU, si=SI, seed=41)
complete, censored = fade_durations(series.amplitude, THRESHOLD, FS)
comparison = compare_fade_duration_models(complete, fs_hz=FS, censored_s=censored)

print(comparison.report())

times, survival = kaplan_meier_survival(complete, censored)
mean = float(complete.mean())

figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))

# --- panel 1: survival -------------------------------------------------------
ax = axes[0]
x_us = times * 1e6
ax.semilogy(x_us, survival, lw=1.6, color="#1f4e79", label="measured (Kaplan-Meier)")
grid = np.linspace(times.min(), times.max(), 400)
ax.semilogy(
    grid * 1e6,
    np.exp(-grid / mean),
    lw=1.4,
    color="#c00000",
    ls="--",
    label=f"exponential, mean {mean * 1e6:.2f} us",
)
for name, colour in (("lognormal", "#2e7d32"), ("weibull", "#7b1fa2")):
    fit = next(f for f in comparison.fits if f.name == name)
    dist = sps.lognorm if name == "lognormal" else sps.weibull_min
    ax.semilogy(
        grid * 1e6,
        dist.sf(grid, *fit.params),
        lw=1.1,
        ls=":",
        color=colour,
        label=f"{name} (AIC {fit.aic:.0f})",
    )
ax.set_xlabel("fade duration [us]")
ax.set_ylabel("P(fade longer than t)")
ax.set_xlim(0, 120)
ax.set_ylim(1e-4, 1.2)
ax.set_title("Survival: an exponential would be a straight line", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.25, which="both")

# --- panel 2: quantile-quantile against the exponential ----------------------
ax = axes[1]
q = np.linspace(0.001, 0.999, 500)
measured_q = np.quantile(complete, q) * 1e6
exp_q = sps.expon.ppf(q, scale=mean) * 1e6
ax.plot(exp_q, measured_q, lw=1.6, color="#1f4e79")
lim = max(exp_q.max(), measured_q.max())
ax.plot([0, lim], [0, lim], lw=1.0, ls="--", color="#c00000", label="y = x")
ax.set_xlabel("exponential quantile [us]")
ax.set_ylabel("measured quantile [us]")
ax.set_title(
    f"Quantile-quantile: cv = {comparison.extra['coefficient_of_variation']:.2f} "
    "against 1.00",
    fontsize=10,
)
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
q10, q50, q99 = (
    float(np.quantile(complete, v) / sps.expon.ppf(v, scale=mean)) for v in (0.1, 0.5, 0.99)
)
ax.annotate(
    "measured / exponential quantile\n"
    f"at the 10th percentile: {q10:.2f}\n"
    f"at the median:          {q50:.2f}\n"
    f"at the 99th percentile: {q99:.2f}",
    xy=(exp_q[-5], measured_q[-5]),
    xytext=(lim * 0.30, lim * 0.70),
    arrowprops={"arrowstyle": "->", "lw": 0.9},
    fontsize=8,
    family="monospace",
)

# --- panel 3: survival ratio -------------------------------------------------
ax = axes[2]
lengths = np.asarray(np.rint(complete * FS), dtype=int)
cuts = np.unique(np.asarray(np.round(np.logspace(0, np.log10(lengths.max()), 60)), dtype=int))
p_geom = 1.0 / float(lengths.mean())
empirical = np.array([float(np.mean(lengths > c)) for c in cuts])
geometric = (1.0 - p_geom) ** cuts
keep = empirical > 0
ax.loglog(
    cuts[keep],
    empirical[keep] / geometric[keep],
    lw=1.8,
    color="#1f4e79",
)
ax.axhline(1.0, lw=1.0, ls="--", color="#c00000")
ax.set_xlabel("fade length cut-off [samples]")
ax.set_ylabel("measured survival / memoryless survival")
ax.set_title("How wrong, as a factor: the tail is the problem", fontsize=10)
ax.grid(alpha=0.25, which="both")
ax.annotate(
    "an interleaver sized from a\nmemoryless fit is too shallow\nby this factor",
    xy=(cuts[keep][-1], (empirical[keep] / geometric[keep])[-1]),
    xytext=(2.0, 1e4),
    arrowprops={"arrowstyle": "->", "lw": 0.9},
    fontsize=8,
)

figure.suptitle(
    "Fade-duration distribution of a correlated lognormal link: "
    f"{comparison.n_complete} complete fades, geometric chi-square "
    f"{comparison.geometric.statistic:.0f} on {comparison.geometric.dof} dof, "
    f"memoryless {'REJECTED' if comparison.exponential_rejected else 'not rejected'}",
    fontsize=11,
)
figure.tight_layout()
figure.savefig(OUT, dpi=130)
print()
print(f"wrote screenshots/{os.path.basename(OUT)}")
