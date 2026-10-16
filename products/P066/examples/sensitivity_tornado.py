"""Which leaf moves the top-level margin most, and which moves the dispersion.

Writes ``screenshots/sensitivity_tornado.png``. The two rankings are
deliberately plotted side by side, because they are different questions and
the leaf that dominates one often does not dominate the other.
"""

from __future__ import annotations

from pathlib import Path

from _bootstrap import CASES, SCREENSHOTS
from _plotstyle import EXACT, OVERRUN, finish, plt

from budgetroll import load, sensitivity


def main() -> Path:
    """Build the figure and return the path written."""
    budget = load(CASES / "cubesat_power.yaml")
    rows = sensitivity(budget, step=0.10)

    fig, (ax_margin, ax_var) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    names = [r.name for r in rows]
    deltas = [r.margin_delta for r in rows]
    positions = range(len(rows))
    ax_margin.barh(list(positions), deltas, color=OVERRUN)
    ax_margin.set_yticks(list(positions))
    ax_margin.set_yticklabels(names, fontsize=7)
    ax_margin.invert_yaxis()
    ax_margin.set_xlabel(f"change in root margin at +10 % on the item [{budget.unit}]")
    ax_margin.set_title("Effect on the top-level margin", fontsize=10)
    ax_margin.set_xlim(min(deltas) * 1.30, 0.0)
    ax_margin.grid(axis="x", alpha=0.25)
    for pos, delta in zip(positions, deltas, strict=True):
        ax_margin.text(delta - 0.004, pos, f"{delta:.4f}", va="center", ha="right", fontsize=7)

    by_variance = sorted(rows, key=lambda r: -(r.variance_share or 0.0))
    var_names = [r.name for r in by_variance]
    var_shares = [100.0 * (r.variance_share or 0.0) for r in by_variance]
    positions = range(len(by_variance))
    ax_var.barh(list(positions), var_shares, color=EXACT)
    ax_var.set_yticks(list(positions))
    ax_var.set_yticklabels(var_names, fontsize=7)
    ax_var.invert_yaxis()
    ax_var.set_xlabel("share of the root variance under the declared correlations [%]")
    ax_var.set_title("Effect on the dispersion", fontsize=10)
    ax_var.set_xlim(0.0, max(var_shares) * 1.14)
    ax_var.grid(axis="x", alpha=0.25)
    for pos, share in zip(positions, var_shares, strict=True):
        ax_var.text(share + 0.5, pos, f"{share:.2f}", va="center", fontsize=7)

    fig.suptitle(
        "Two different questions about the same budget. The item that costs the most "
        "margin is not the item that carries the most uncertainty.",
        fontsize=10,
    )
    return finish(fig, SCREENSHOTS / "sensitivity_tornado.png")


if __name__ == "__main__":
    print(f"wrote {main()}")
