"""The learned gain scheduler against classical gain rules, with intervals.

A reduced-size version of ``validation/learned_vs_fixed_ci.py`` so it runs in
under a minute.  The numbers to cite are the validation script's, not these;
the point of this figure is the shape of the result, including the comparisons
the experiment cannot resolve.

    python examples/learned_vs_fixed.py
"""

from __future__ import annotations

import time

import numpy as np
from _plotstyle import COLORS, save

import matplotlib.pyplot as plt  # noqa: E402

from detumblesim import (  # noqa: E402
    ENERGY_WEIGHT,
    FixedGainPolicy,
    GainScheduler,
    PowerLawGainPolicy,
    ScheduledGainPolicy,
    fit_power_law_gain,
    mean_ci,
    oracle_gain,
    paired_difference_ci,
    run_policy,
    sample_scenarios,
    training_rows,
)

GAIN_GRID = np.geomspace(1.0e4, 1.0e6, 8)
SIM = {"duration_s": 23000.0, "control_dt_s": 2.0, "substeps": 1}
N_TRAIN, N_TEST = 14, 24
WEIGHTS = (0.0, ENERGY_WEIGHT, 2.0)
ORDER = ("fixed", "powerlaw", "learned")


def cost_at(scores, w):
    return np.array([s.time_orbits + w * (s.energy_term / ENERGY_WEIGHT) for s in scores])


def main() -> None:
    t0 = time.perf_counter()
    train = sample_scenarios(N_TRAIN, 1000)
    grid, best = [], []
    for s in train:
        bg, _, costs = oracle_gain(s, GAIN_GRID, **SIM)
        grid.append(costs)
        best.append(bg)
    k_fixed = float(GAIN_GRID[int(np.argmin(np.mean(grid, axis=0)))])
    coef, rms = fit_power_law_gain(train, np.array(best))

    xs, ys = [], []
    for s, bg in zip(train, best, strict=True):
        res, _ = run_policy(s, FixedGainPolicy(k_fixed), **SIM)
        x, y = training_rows(s, float(bg), k_fixed, res)
        if x.size:
            xs.append(x)
            ys.append(y)
    sch = GainScheduler().fit(np.vstack(xs), np.concatenate(ys))

    scores = {p: [] for p in ORDER}
    for s in sample_scenarios(N_TEST, 5000):
        m_max = float(np.min(s.magnetorquer.max_dipole_am2))
        j = s.inertia_scale_kgm2
        scores["fixed"].append(run_policy(s, FixedGainPolicy(k_fixed), **SIM)[1])
        scores["powerlaw"].append(
            run_policy(s, PowerLawGainPolicy(coef, m_max, j), **SIM)[1]
        )
        scores["learned"].append(
            run_policy(s, ScheduledGainPolicy(sch, k_fixed, m_max, j), **SIM)[1]
        )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    width = 0.26
    idx = np.arange(len(WEIGHTS))
    for i, p in enumerate(ORDER):
        means, errs = [], []
        for w in WEIGHTS:
            ci = mean_ci(cost_at(scores[p], w))
            means.append(ci.mean)
            errs.append(ci.half_width)
        ax.bar(idx + (i - 1) * width, means, width, yerr=errs, capsize=3,
               color=COLORS[p], label=p, alpha=0.9)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"w = {w}" for w in WEIGHTS])
    ax.set_ylabel("cost  (mean over scenarios, 95% CI)")
    ax.set_title(f"Held-out cost at three energy weights ({N_TEST} scenarios)\n"
                 "marginal intervals overlap because scenario difficulty varies",
                 fontsize=9.5)
    ax.legend(fontsize=8)

    ax = axes[1]
    pairs = [("learned", "fixed"), ("powerlaw", "fixed"), ("learned", "powerlaw")]
    labels, means, errs, colors = [], [], [], []
    for j_w, w in enumerate(WEIGHTS):
        for a, b in pairs:
            d = paired_difference_ci(cost_at(scores[a], w), cost_at(scores[b], w))
            labels.append(f"{a} - {b}\nw = {w}")
            means.append(d.mean)
            errs.append(d.half_width)
            colors.append(COLORS["powerlaw"] if d.excludes_zero else "#9e9e9e")
        del j_w
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=errs, color=colors, capsize=3, height=0.62)
    ax.axvline(0.0, color="black", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("paired cost difference (negative favours the first policy)")
    ax.set_title("Paired differences with 95% intervals\n"
                 "grey bars cross zero: this experiment cannot separate them",
                 fontsize=9.5)

    path = save(fig, "learned_vs_fixed.png")
    print(f"saved {path}")
    print(f"tuned fixed gain k = {k_fixed:.4e} A m^2 s / T")
    print(f"power law: log10 k = {coef[0]:.4f} + {coef[1]:.4f} log10(m_max) "
          f"+ {coef[2]:.4f} log10(j), RMS residual {rms:.4f} dex")
    for w in WEIGHTS:
        for a, b in pairs:
            d = paired_difference_ci(cost_at(scores[a], w), cost_at(scores[b], w))
            verdict = "resolved" if d.excludes_zero else "NOT RESOLVED"
            print(f"w={w}: {a:>9} - {b:<9} {d.mean:+8.4f} "
                  f"[{d.ci_low:+.4f}, {d.ci_high:+.4f}]  {verdict}")
    print(f"wall time {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
