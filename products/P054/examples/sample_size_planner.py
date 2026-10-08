"""Sample-size planning: what a target violation probability actually costs.

Writes ../screenshots/sample_size_planner.png.

Run from the product directory:

    python examples/sample_size_planner.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rareverify.montecarlo import required_samples_for_cov  # noqa: E402
from rareverify.planner import (  # noqa: E402
    detection_probability,
    samples_for_relative_width,
    samples_for_zero_failure_bound,
)

OUTPUT = (
    Path(__file__).resolve().parent.parent / "screenshots" / "sample_size_planner.png"
)


def main() -> None:
    targets = np.logspace(-7, -2, 26)
    demonstration = [samples_for_zero_failure_bound(float(p), 0.95) for p in targets]
    demonstration99 = [samples_for_zero_failure_bound(float(p), 0.99) for p in targets]
    estimation = [
        samples_for_relative_width(float(p), 0.5, 0.95) for p in targets
    ]
    crude_cov = [required_samples_for_cov(float(p), 0.1) for p in targets]

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    axes[0].loglog(targets, demonstration, marker="o", markersize=3, label="demonstrate, 95 %")
    axes[0].loglog(
        targets, demonstration99, marker="s", markersize=3, label="demonstrate, 99 %"
    )
    axes[0].loglog(
        targets, estimation, marker="^", markersize=3, label="estimate to width 0.5 p, 95 %"
    )
    axes[0].loglog(
        targets, crude_cov, linestyle=":", label="crude run for 10 % coefficient of variation"
    )
    axes[0].set_xlabel("target violation probability")
    axes[0].set_ylabel("runs required")
    axes[0].set_title("Runs required, Clopper-Pearson")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3, which="both")
    axes[0].annotate(
        "1e-4 needs 29956 runs\nto demonstrate, 658443 to estimate",
        xy=(1e-4, 29956),
        xytext=(2e-7, 2e5),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        fontsize=8,
    )

    runs = np.unique(np.round(np.logspace(2, 6, 80)).astype(int))
    for p, style in ((1e-3, "-"), (1e-4, "--"), (1e-5, ":")):
        axes[1].semilogx(
            runs,
            [detection_probability(int(m), p) for m in runs],
            style,
            label=f"p = {p:.0e}",
        )
    axes[1].axhline(0.95, color="black", linewidth=0.9, linestyle="--", label="0.95")
    axes[1].set_xlabel("runs n")
    axes[1].set_ylabel("probability of at least one violation")
    axes[1].set_title("Probability a campaign sees anything at all")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    figure.suptitle(
        "rareverify: demonstration and estimation are different questions with "
        "different costs",
        fontsize=9,
    )
    figure.tight_layout()
    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT, dpi=140)
    plt.close(figure)


if __name__ == "__main__":
    main()
