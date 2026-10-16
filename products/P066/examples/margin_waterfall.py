"""Where the allocation goes, level by level, for the CubeSat power budget.

Writes ``screenshots/margin_waterfall.png``. Shows nominal, contingency
allowance and the margin that is left, and marks the k-sigma position so the
difference between closing on predicted values and closing under the declared
dispersion is visible in one picture.
"""

from __future__ import annotations

from pathlib import Path

from _bootstrap import CASES, SCREENSHOTS
from _plotstyle import CONTINGENCY, MARGIN, NOMINAL, OVERRUN, finish, plt

from budgetroll import load, rollup


def main() -> Path:
    """Build the figure and return the path written."""
    budget = load(CASES / "cubesat_power.yaml")
    roll = rollup(budget)
    children = [r for r in roll.results if r.depth == 1]
    root = roll.root

    fig, (ax_stack, ax_levels) = plt.subplots(
        1, 2, figsize=(12.0, 5.0), gridspec_kw={"width_ratios": [1.0, 1.25]}
    )

    bottom = 0.0
    for result in children:
        ax_stack.bar("load", result.nominal, bottom=bottom, color=NOMINAL, edgecolor="white")
        ax_stack.text(
            0, bottom + result.nominal / 2, result.name, ha="center", va="center",
            fontsize=7, color="white",
        )
        bottom += result.nominal
    ax_stack.bar(
        "load", root.allowance, bottom=bottom, color=CONTINGENCY, edgecolor="white",
        label=f"contingency {root.allowance:.4f} {budget.unit}",
    )
    bottom += root.allowance
    margin = root.margin_absolute or 0.0
    ax_stack.bar(
        "load", margin, bottom=bottom, color=MARGIN, edgecolor="white",
        label=f"margin {margin:.4f} {budget.unit}",
    )
    ax_stack.axhline(root.allocation, color="black", lw=1.2)
    ax_stack.text(
        0.45, root.allocation, f" allocation {root.allocation:.3f} {budget.unit}",
        va="top", fontsize=8,
    )
    k_line = root.predicted + budget.k_sigma * root.uncertainty
    ax_stack.axhline(k_line, color=OVERRUN, lw=1.2, ls="--")
    ax_stack.text(
        0.45, root.predicted,
        f" predicted + {budget.k_sigma:g} sigma\n = {k_line:.4f} {budget.unit},\n"
        f" which is {k_line - root.allocation:.4f} {budget.unit}\n"
        " above the allocation",
        va="top", color=OVERRUN, fontsize=8,
    )
    ax_stack.set_xlim(-0.55, 1.9)
    ax_stack.set_ylim(0, root.allocation * 1.12)
    ax_stack.set_ylabel(f"orbit-average power [{budget.unit}]")
    ax_stack.set_title("Allocation, consumption and what is left", fontsize=10)
    ax_stack.legend(fontsize=7, loc="lower right")
    ax_stack.grid(axis="y", alpha=0.25)

    labels = [("  " * r.depth) + r.name for r in roll.results]
    # The root has no ancestor allocation, so its share is taken against its
    # own allocation; every other node is reported against the nearest
    # ancestor allocation, which here is the root's.
    shares = [
        100.0
        * (
            r.consumption_of_parent_allocation
            if r.consumption_of_parent_allocation is not None
            else (r.predicted / r.allocation if r.allocation else 0.0)
        )
        for r in roll.results
    ]
    positions = range(len(labels))
    ax_levels.barh(list(positions), shares, color=NOMINAL)
    ax_levels.set_yticks(list(positions))
    ax_levels.set_yticklabels(labels, fontsize=7)
    ax_levels.invert_yaxis()
    ax_levels.set_xlabel("share of the root allocation consumed [%]")
    ax_levels.set_title(
        f"Every level against the {root.allocation:.3f} {budget.unit} allocation",
        fontsize=10,
    )
    ax_levels.set_xlim(0, max(shares) * 1.12)
    ax_levels.grid(axis="x", alpha=0.25)
    for pos, share in zip(positions, shares, strict=True):
        ax_levels.text(share + 0.6, pos, f"{share:.2f}", va="center", fontsize=7)

    fig.suptitle(
        "3U CubeSat orbit-average power: the budget closes on predicted values "
        "and does not close at two standard deviations.",
        fontsize=10,
    )
    return finish(fig, SCREENSHOTS / "margin_waterfall.png")


if __name__ == "__main__":
    print(f"wrote {main()}")
