"""Markov channel-state fitting, and the exact place the Markov assumption fails.

Three panels: the dwell-time distribution of the fade state against the
geometric the fitted chain implies; the same for the good state; and the
departure from first-order Markov as a function of how finely the amplitude is
quantised into states.

What to notice: the fitted chain reproduces the mean dwell of both states
almost exactly -- maximum likelihood on a two-state chain is a fit to the mean
and nothing else -- and gets the shape wrong at both ends. The quantised state
of a continuous Gauss-Markov process is not Markov, because "below the
threshold" throws away how far below. Adding states reduces the departure but
does not remove it at any state count a link budget would use; the third panel
quantifies that with Cramer's V on the triplet table rather than with a
p-value, because at two million samples every p-value is zero.

Writes ``../screenshots/markov_dwell_fit.png``.

Runtime: about 20 s on one core.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from linkoutage.channel import lognormal_amplitude_series  # noqa: E402
from linkoutage.markov import (  # noqa: E402
    dwell_lengths,
    dwell_time_goodness_of_fit,
    fit_markov,
    fit_semi_markov,
    markov_order_test,
    state_sequence,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "screenshots", "markov_dwell_fit.png")

FS = 1.0e6
TAU = 2.0e-4
SI = 0.6
N = 2_000_000
THRESHOLD = 0.6

series = lognormal_amplitude_series(N, fs_hz=FS, tau_s=TAU, si=SI, seed=41)
states = state_sequence(series.amplitude, [THRESHOLD])
fit = fit_markov(states)
gof = dwell_time_goodness_of_fit(states, fit)
order = markov_order_test(states)
semi = fit_semi_markov(states)

print(fit.report())
print()
for g in gof:
    print(g.line())
print()
print(order.report())
print()
print(f"semi-Markov free parameters: {semi.n_parameters} against "
      f"{fit.n_parameters} for the Markov chain")

per_state, n_censored = dwell_lengths(states, n_states=2)

figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
labels = ("fade state (a < 0.6)", "good state (a >= 0.6)")

for i in (0, 1):
    ax = axes[i]
    d = per_state[i]
    counts = np.bincount(d)[1:]
    support = np.arange(1, counts.size + 1)
    pmf = counts / counts.sum()
    p = 1.0 - fit.transition_matrix[i, i]
    geo = (1.0 - p) ** (support - 1.0) * p
    ax.loglog(support, pmf, ".", ms=4, color="#1f4e79", label="measured dwell")
    ax.loglog(
        support,
        geo,
        "-",
        lw=1.4,
        color="#c00000",
        label=f"geometric from P[{i},{i}], p = {p:.5f}",
    )
    ax.set_ylim(0.2 / counts.sum(), 2.0)
    ax.set_xlabel("dwell [samples]")
    ax.set_ylabel("probability")
    g = gof[i]
    ax.set_title(
        f"{labels[i]}\nmean {g.mean_observed:.2f} vs {g.mean_geometric:.2f} samples, "
        f"cv {g.cv_observed:.2f} vs {g.cv_geometric:.2f}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    ax.annotate(
        "the mean is fitted;\nthe shape is not",
        xy=(support[len(support) // 12], pmf[len(support) // 12]),
        xytext=(2.0, 0.6 / counts.sum() * 40),
        arrowprops={"arrowstyle": "->", "lw": 0.9},
        fontsize=8,
    )

# --- panel 3: departure from Markov against state count ----------------------
ax = axes[2]
threshold_sets = [
    [0.6],
    [0.5, 0.8],
    [0.4, 0.6, 0.9],
    [0.4, 0.55, 0.7, 0.9],
    [0.35, 0.45, 0.55, 0.7, 0.85, 1.05],
]
# Beyond about K = 7 the triplet contingency table becomes structurally sparse:
# with rho = 0.995 a single sample almost never moves more than one state, so
# most cells are near-impossible and the conditional-independence test loses
# its usable middle states. Points past that are not comparable and are not
# plotted.
k_values = []
v_values = []
cv_ratios = []
for thresholds in threshold_sets:
    st = state_sequence(series.amplitude, thresholds)
    f = fit_markov(st)
    o = markov_order_test(st)
    gofs = dwell_time_goodness_of_fit(st, f)
    ratios = [
        g.cv_observed / g.cv_geometric
        for g in gofs
        if np.isfinite(g.cv_observed) and g.cv_geometric > 0
    ]
    k_values.append(f.n_states)
    v_values.append(o.cramers_v)
    cv_ratios.append(max(ratios) if ratios else np.nan)
    print(f"K = {f.n_states:2d}  Cramer V = {o.cramers_v:.4f}  "
          f"max cv ratio = {cv_ratios[-1]:.2f}  chi2 = {o.statistic:.0f} on {o.dof} dof"
          f"{'  [' + o.note + ']' if o.note else ''}")

ax.plot(k_values, v_values, "o-", lw=1.6, ms=5, color="#1f4e79", label="Cramer's V (order test)")
ax.set_xlabel("number of channel states K")
ax.set_ylabel("Cramer's V: departure from first-order Markov")
ax.set_title(
    "More states help, and do not fix it\n"
    "(p-value is 0 for every point; the effect size is the information)",
    fontsize=10,
)
ax.set_xticks(k_values)
ax.grid(alpha=0.25)
ax.set_ylim(0.0, max(v_values) * 1.25)
twin = ax.twinx()
twin.plot(k_values, cv_ratios, "s--", lw=1.2, ms=4, color="#2e7d32", alpha=0.85)
twin.axhline(1.0, lw=1.0, ls=":", color="#2e7d32")
twin.set_ylabel("worst dwell cv / geometric cv", color="#2e7d32")
twin.tick_params(axis="y", labelcolor="#2e7d32")
ax.legend(fontsize=8, loc="upper right")

figure.suptitle(
    "Two-state and N-state Markov fitting of a correlated lognormal link: "
    f"{N} samples, Cramer's V = {order.cramers_v:.3f} at K = 2",
    fontsize=11,
)
figure.tight_layout()
figure.savefig(OUT, dpi=130)
print()
print(f"censored dwell runs excluded: {n_censored}")
print(f"wrote screenshots/{os.path.basename(OUT)}")
