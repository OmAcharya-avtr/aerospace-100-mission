"""Linear against root-sum-square against the exact correlated form.

Writes ``screenshots/combination_comparison.png``. The point of the figure is
that the three rules give three different numbers from the same leaves, and
that which one is right is a question about the declaration, not about the
arithmetic.
"""

from __future__ import annotations

from pathlib import Path

from _bootstrap import CASES, SCREENSHOTS
from _plotstyle import EXACT, LINEAR, MUTED, RSS, finish, plt

from budgetroll import linear_rollup_uncertainty, load, rollup, rss_rollup_uncertainty

CASE_FILES = ["clean_mass.yaml", "cubesat_power.yaml", "cubesat_link.yaml"]


def main() -> Path:
    """Build the figure and return the path written."""
    budgets = [load(CASES / name) for name in CASE_FILES]
    fig, axes = plt.subplots(1, len(budgets), figsize=(12.0, 4.4))
    for ax, budget in zip(axes, budgets, strict=True):
        roll = rollup(budget)
        values = [
            rss_rollup_uncertainty(budget),
            roll.root.uncertainty,
            roll.root.exact_uncertainty,
            linear_rollup_uncertainty(budget),
        ]
        labels = ["RSS\n(independent)", "declared\nhierarchy", "exact with\ndeclared R",
                  "linear\n(fully correlated)"]
        colours = [RSS, MUTED, EXACT, LINEAR]
        bars = ax.bar(labels, values, color=colours)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.4g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(f"{budget.id}\n1 sigma at the root", fontsize=9)
        ax.set_ylabel(f"uncertainty [{budget.unit}]")
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylim(0, max(values) * 1.22)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "The same leaves, four combination rules. RSS never exceeds the linear sum; "
        "the exact form can exceed both when the declared matrix is inadmissible.",
        fontsize=10,
    )
    return finish(fig, SCREENSHOTS / "combination_comparison.png")


if __name__ == "__main__":
    print(f"wrote {main()}")
