"""Twelve seeded trajectories with their limits, crossings and censoring status.

What to notice: the observation window (solid) is short relative to the
extrapolation to the crossing (dashed), the three mean shapes are not
distinguishable by eye inside the window, and more than half the units are
right-censored, so their crossing is never observed in the record.
"""

from __future__ import annotations

import numpy as np

from _common import CENSORED, COMPLETE, TRUTH, add_src_to_path, save, style

add_src_to_path()

import matplotlib.pyplot as plt  # noqa: E402

from degradepredict.generate import DEFAULT_LIMIT, generate_population, mean_curve  # noqa: E402


def main() -> None:
    style()
    pop = generate_population(12, seed=20261017)
    fig, axes = plt.subplots(3, 4, figsize=(13.0, 7.2), sharey=True)
    for ax, tr in zip(axes.ravel(), pop, strict=True):
        colour = COMPLETE if tr.event else CENSORED
        horizon = tr.crossing_time_days * 1.12
        grid = np.linspace(0.0, horizon, 400)
        ax.plot(grid, mean_curve(grid, tr.family, tr.params), color=TRUTH,
                lw=0.9, ls="--", label="true mean")
        ax.plot(tr.times_days, tr.values_percent, color=colour, lw=1.2,
                label="observed window")
        ax.axhline(DEFAULT_LIMIT.value, color="#b0344f", lw=0.9)
        ax.axvline(tr.crossing_time_days, color=TRUTH, lw=0.8, ls=":")
        ax.axvline(tr.origin_days + tr.followup_days, color=colour, lw=0.8, ls="-.")
        ax.set_title(
            f"{tr.family}  T={tr.crossing_time_days:.0f} d  "
            f"{'complete' if tr.event else 'censored'}",
            fontsize=8,
        )
        ax.set_xlim(0.0, horizon)
        ax.set_ylim(72.0, 103.0)
    for ax in axes[-1]:
        ax.set_xlabel("service age [d]")
    for ax in axes[:, 0]:
        ax.set_ylabel("capacity [% BOL]")
    handles = [
        plt.Line2D([], [], color=COMPLETE, lw=1.2, label="observed window, complete unit"),
        plt.Line2D([], [], color=CENSORED, lw=1.2, label="observed window, censored unit"),
        plt.Line2D([], [], color=TRUTH, lw=0.9, ls="--", label="true mean curve"),
        plt.Line2D([], [], color=TRUTH, lw=0.8, ls=":", label="true crossing"),
        plt.Line2D([], [], color="#777777", lw=0.8, ls="-.", label="end of follow-up"),
        plt.Line2D([], [], color="#b0344f", lw=0.9, label="limit, 80 % of BOL"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Twelve seeded degradation trajectories (seed 20261017): window, extrapolation, "
        "crossing and censoring",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    print(f"wrote {save(fig, 'trajectory_gallery.png')}")


if __name__ == "__main__":
    main()
