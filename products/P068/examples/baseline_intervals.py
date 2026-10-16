"""Three analytic baselines fitted to the same window, with their crossing intervals.

What to notice: inside the observation window all three fits are almost
indistinguishable and all three sit within the noise, yet their extrapolated
crossing dates and 90 percent intervals do not overlap. The disagreement
between the models is far larger than any one model's stated uncertainty.
"""

from __future__ import annotations

import numpy as np

from _common import EXPONENTIAL, LINEAR, POWERLAW, TRUTH, add_src_to_path, save, style

add_src_to_path()

import matplotlib.pyplot as plt  # noqa: E402

from degradepredict.baselines import exponential_baseline, linear_baseline  # noqa: E402
from degradepredict.generate import DEFAULT_LIMIT, generate_population, mean_curve  # noqa: E402
from degradepredict.physics import powerlaw_fade_baseline  # noqa: E402

FITTERS = (
    ("linear", linear_baseline, LINEAR),
    ("exponential", exponential_baseline, EXPONENTIAL),
    ("powerlaw-fade", powerlaw_fade_baseline, POWERLAW),
)


def _predicted_curve(fit, grid):
    x = np.power(grid, fit.z)
    line = fit.intercept + fit.slope * x
    return np.exp(line) if fit.metric_link == "log" else line


def main() -> None:
    style()
    pop = generate_population(60, seed=20261019)
    units = [pop[i] for i in (20, 9, 0)]  # one unit of each true shape family
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0),
                             gridspec_kw={"height_ratios": [2.1, 1.0]})
    for col, tr in enumerate(units):
        ax, bx = axes[0, col], axes[1, col]
        horizon = max(tr.crossing_time_days * 1.6, tr.origin_days * 2.0)
        grid = np.linspace(1e-6, horizon, 600)
        ax.plot(grid, mean_curve(grid, tr.family, tr.params), color=TRUTH, lw=1.0,
                ls="--", label="true mean")
        ax.plot(tr.times_days, tr.values_percent, color="#444444", lw=0.0,
                marker=".", ms=2.5, label="telemetry")
        ax.axhline(DEFAULT_LIMIT.value, color="#b0344f", lw=0.9)
        ax.axvline(tr.crossing_time_days, color=TRUTH, lw=0.9, ls=":")
        for row, (name, fitter, colour) in enumerate(FITTERS):
            fit = fitter(tr.times_days, tr.values_percent, DEFAULT_LIMIT)
            ax.plot(grid, _predicted_curve(fit, grid), color=colour, lw=1.1, label=name)
            lo, hi = fit.fieller_interval_days
            lo_plot, hi_plot = max(lo, 0.0), min(hi, horizon)
            if lo_plot < horizon:
                bx.plot([lo_plot, hi_plot], [row, row], color=colour, lw=3.0,
                        solid_capstyle="butt")
            if fit.crossing_days <= horizon:
                bx.plot([fit.crossing_days], [row], marker="|", color=colour, ms=12)
            if not np.isfinite(hi) or hi > horizon:
                bx.annotate(
                    f"-> {fit.crossing_days:.0f} d" if np.isfinite(fit.crossing_days)
                    else "unbounded",
                    (horizon * 0.99, row), fontsize=7, color=colour, va="center",
                    ha="right",
                )
        bx.axvline(tr.crossing_time_days, color=TRUTH, lw=0.9, ls=":")
        bx.set_yticks(range(len(FITTERS)))
        bx.set_yticklabels([n for n, _, _ in FITTERS], fontsize=8)
        bx.set_xlim(0.0, horizon)
        bx.set_xlabel("crossing time [d]")
        bx.set_ylim(-0.6, len(FITTERS) - 0.4)
        ax.set_xlim(0.0, horizon)
        ax.set_ylim(70.0, 103.0)
        ax.set_title(
            f"unit {tr.traj_id}: true shape {tr.family}, true crossing "
            f"{tr.crossing_time_days:.0f} d",
            fontsize=9,
        )
        if col == 0:
            ax.set_ylabel("capacity [% BOL]")
            bx.set_ylabel("90 % Fieller interval")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Same window, three declared shapes: the model disagreement exceeds every "
        "model's own interval",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    print(f"wrote {save(fig, 'baseline_intervals.png')}")


if __name__ == "__main__":
    main()
