"""The attributed error budget as a bar chart, with the dominant term named.

Writes ``screenshots/budget_breakdown.png``. The figure is the product's
argument in one picture: six lines spanning fourteen orders of magnitude, and
the one that matters changes with the ephemeris age.
"""

from __future__ import annotations

import numpy as np

from _common import SERIES, bootstrap, plt, save, style_axes

bootstrap()

from pointahead.budget import BudgetInputs, build_budget  # noqa: E402
from pointahead.scenario import DEFAULT_BEAM, leo_crossing_pair  # noqa: E402

AGES = (0.0, 30.0, 300.0)


def main() -> int:
    link = leo_crossing_pair()
    budgets = [
        build_budget(link, BudgetInputs(0.05, age, 1, DEFAULT_BEAM)) for age in AGES
    ]
    names = [term.name for term in budgets[0].terms]

    print(f"geometry                : {link.label}, range {link.range_m / 1.0e3:.3f} km")
    print(f"point-ahead angle       : {budgets[0].point_ahead_rad * 1.0e6:.6f} urad")
    print(f"beam divergence         : "
          f"{budgets[0].beam.divergence_half_angle_rad * 1.0e6:.6f} urad half-angle")
    print()
    header = f"{'term':<32s}" + "".join(f"{f'age {age:.0f} s':>16s}" for age in AGES)
    print(header)
    print("-" * len(header))
    for index, name in enumerate(names):
        row = f"{name:<32s}"
        for budget in budgets:
            row += f"{budget.terms[index].value_rad * 1.0e6:>16.6e}"
        print(row)
    print("-" * len(header))
    row = f"{'total (root sum of squares)':<32s}"
    for budget in budgets:
        row += f"{budget.total_rss_rad * 1.0e6:>16.6e}"
    print(row)
    print()
    for age, budget in zip(AGES, budgets, strict=True):
        print(
            f"age {age:>6.0f} s  dominant {budget.dominant_term:<30s} "
            f"share {budget.dominant_fraction:.4f}  loss {budget.beam.loss_db:.6e} dB"
        )

    figure, axes = plt.subplots(figsize=(9.5, 5.0))
    positions = np.arange(len(names))
    width = 0.26
    floor = 1.0e-10
    for offset, (age, budget) in enumerate(zip(AGES, budgets, strict=True)):
        values = [max(term.value_rad * 1.0e6, floor) for term in budget.terms]
        axes.bar(
            positions + (offset - 1) * width,
            values,
            width=width,
            color=SERIES[offset],
            label=f"ephemeris age {age:.0f} s",
        )
    axes.set_yscale("log")
    axes.set_xticks(positions)
    axes.set_xticklabels([name.replace("_", "\n") for name in names], fontsize=7)
    axes.axhline(
        budgets[0].beam.divergence_half_angle_rad * 1.0e6,
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        label="beam divergence half-angle",
    )
    axes.axhline(
        budgets[0].beam.affordable_error_rad * 1.0e6,
        color="#8c3b3b",
        linestyle=":",
        linewidth=1.2,
        label="error affordable at 1 dB pointing loss",
    )
    style_axes(
        axes,
        f"Attributed point-ahead error budget, {link.label}, "
        f"range {link.range_m / 1.0e3:.0f} km",
        "budget line",
        "contribution / urad (log scale, values below 1e-10 urad clipped)",
    )
    axes.legend(fontsize=8, loc="upper right", framealpha=0.95)
    print()
    print(f"written: {save(figure, 'budget_breakdown.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
