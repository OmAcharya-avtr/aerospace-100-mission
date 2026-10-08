"""Exact coverage of the Clopper-Pearson and Wilson intervals, and the zero-failure bounds.

Writes ../screenshots/interval_coverage.png.

The left panel is exact arithmetic, not simulation: coverage is the binomial
probability mass of the values of k whose interval contains the true p. The
right panel shows the three closed-form zero-failure upper bounds against the
3/n approximation.

Run from the product directory:

    python examples/interval_coverage.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rareverify.intervals import (  # noqa: E402
    exact_coverage,
    rule_of_three_upper,
    zero_failure_upper,
)

OUTPUT = Path(__file__).resolve().parent.parent / "screenshots" / "interval_coverage.png"


def main() -> None:
    n = 50
    probabilities = np.linspace(0.005, 0.5, 200)
    coverage_cp = [exact_coverage(n, p, method="clopper-pearson") for p in probabilities]
    coverage_wilson = [exact_coverage(n, p, method="wilson") for p in probabilities]

    sizes = np.unique(np.round(np.logspace(1, 5, 60)).astype(int))
    cp_one = [zero_failure_upper(int(m), 0.95, side="upper") for m in sizes]
    cp_two = [zero_failure_upper(int(m), 0.95, side="two-sided") for m in sizes]
    wilson_two = [
        zero_failure_upper(int(m), 0.95, method="wilson", side="two-sided") for m in sizes
    ]
    three_over_n = [rule_of_three_upper(int(m)) for m in sizes]

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    axes[0].plot(probabilities, coverage_cp, label="Clopper-Pearson", linewidth=1.6)
    axes[0].plot(probabilities, coverage_wilson, label="Wilson", linewidth=1.2)
    axes[0].axhline(0.95, color="black", linestyle="--", linewidth=0.9, label="nominal 0.95")
    axes[0].set_xlabel("true violation probability p")
    axes[0].set_ylabel("exact coverage")
    axes[0].set_title(f"Exact two-sided coverage, n = {n}, nominal 95 %")
    axes[0].set_ylim(0.88, 1.005)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(alpha=0.3)
    worst = float(np.min(coverage_wilson))
    axes[0].annotate(
        f"Wilson minimum {worst:.4f}",
        xy=(probabilities[int(np.argmin(coverage_wilson))], worst),
        xytext=(0.18, 0.905),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        fontsize=8,
    )

    # Every bound is proportional to 1/n, so plotting n * bound separates them.
    axes[1].semilogx(
        sizes, np.asarray(sizes) * np.asarray(cp_one),
        label="Clopper-Pearson one-sided", linewidth=1.6,
    )
    axes[1].semilogx(
        sizes, np.asarray(sizes) * np.asarray(cp_two),
        label="Clopper-Pearson two-sided upper", linewidth=1.2,
    )
    axes[1].semilogx(
        sizes, np.asarray(sizes) * np.asarray(wilson_two),
        label="Wilson two-sided upper", linewidth=1.2,
    )
    axes[1].semilogx(
        sizes, np.asarray(sizes) * np.asarray(three_over_n),
        linestyle=":", label="3/n (rule of three)", linewidth=1.6,
    )
    axes[1].axhline(
        2.9957322735539909, color="black", linewidth=0.8, linestyle="--",
        label="-ln(0.05) = 2.99573",
    )
    axes[1].set_xlabel("runs n with zero violations observed")
    axes[1].set_ylabel("n x upper confidence limit")
    axes[1].set_title("Zero-failure bounds at 95 %, scaled by n")
    axes[1].set_ylim(2.5, 4.4)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    axes[1].annotate(
        "3/n overstates the exact one-sided bound\nby 1.4246e-3 relative in the limit",
        xy=(1e4, 3.0), xytext=(2e1, 3.55),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8}, fontsize=8,
    )

    figure.suptitle(
        "rareverify: the probability bounded is the probability of the model, "
        "not of any vehicle",
        fontsize=9,
    )
    figure.tight_layout()
    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT, dpi=140)
    plt.close(figure)


if __name__ == "__main__":
    main()
