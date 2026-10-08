"""Measured variance reduction of importance sampling and subset simulation.

Writes ../screenshots/variance_reduction.png.

Every bar is an empirical variance over independent replications, put on an
equal true-evaluation footing. Nothing is a formula prediction.

Run from the product directory:

    python examples/variance_reduction.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rareverify.benchmark import replicate, summarise, variance_reduction  # noqa: E402
from rareverify.limitstates import (  # noqa: E402
    LinearGaussianLimitState,
    LognormalRatioLimitState,
    RippledLimitState,
)
from rareverify.montecarlo import crude_monte_carlo  # noqa: E402
from rareverify.subset import subset_simulation  # noqa: E402
from rareverify.tilting import analytic_mean_shift, importance_sampling  # noqa: E402

OUTPUT = (
    Path(__file__).resolve().parent.parent / "screenshots" / "variance_reduction.png"
)
BUDGET = 100_000
REPLICATIONS = 40


def main() -> None:
    instances = [
        ("linear\nbeta=3.719", LinearGaussianLimitState(beta=3.719, dimension=2)),
        ("linear\nbeta=4.753", LinearGaussianLimitState(beta=4.753, dimension=2)),
        ("lognormal\nratio", LognormalRatioLimitState()),
        ("rippled\nA=0.8 w=1.5", RippledLimitState()),
        (
            "rippled\nb=5.5 A=2.5",
            RippledLimitState(beta=5.5, amplitude=2.5, frequency=2.0),
        ),
    ]
    labels, is_vrf, subset_vrf, covs_crude, covs_is, covs_subset = [], [], [], [], [], []
    for label, state in instances:
        reference = state.analytic_probability()
        crude = summarise(
            "crude",
            replicate(
                lambda rng, s=state: crude_monte_carlo(s, BUDGET, rng=rng),
                REPLICATIONS,
                seed=1001,
            ),
            reference,
        )
        tilted = summarise(
            "analytic-IS",
            replicate(
                lambda rng, s=state: importance_sampling(
                    s, analytic_mean_shift(s), BUDGET, rng=rng
                ),
                REPLICATIONS,
                seed=1002,
            ),
            reference,
        )
        subset = summarise(
            "subset",
            replicate(
                lambda rng, s=state: subset_simulation(s, n_per_level=2000, rng=rng),
                REPLICATIONS,
                seed=1004,
            ),
            reference,
        )
        labels.append(label)
        is_vrf.append(variance_reduction(tilted, crude, reference).vrf_measured)
        subset_vrf.append(variance_reduction(subset, crude, reference).vrf_measured)
        covs_crude.append(crude.empirical_cov)
        covs_is.append(tilted.empirical_cov)
        covs_subset.append(subset.empirical_cov)

    positions = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    axes[0].bar(positions - 0.2, is_vrf, width=0.38, label="analytic-IS")
    axes[0].bar(positions + 0.2, subset_vrf, width=0.38, label="subset simulation")
    axes[0].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("measured variance reduction factor vs crude")
    axes[0].set_title(
        f"Equal true-evaluation basis, {REPLICATIONS} replications, "
        f"{BUDGET} samples"
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y", which="both")
    for x, value in zip(positions - 0.2, is_vrf, strict=True):
        axes[0].text(x, value * 1.15, f"{value:.0f}", ha="center", fontsize=7)
    for x, value in zip(positions + 0.2, subset_vrf, strict=True):
        axes[0].text(x, value * 1.15, f"{value:.1f}", ha="center", fontsize=7)

    axes[1].bar(positions - 0.27, covs_crude, width=0.26, label="crude")
    axes[1].bar(positions, covs_is, width=0.26, label="analytic-IS")
    axes[1].bar(positions + 0.27, covs_subset, width=0.26, label="subset simulation")
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("measured coefficient of variation")
    axes[1].set_title(
        "Spread of the estimate across replications\n"
        "(subset simulation spends about 7500 evaluations, not 100000)"
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y", which="both")

    figure.suptitle(
        "rareverify: measured, not predicted. A factor below 1 would mean "
        "worse than plain Monte Carlo.",
        fontsize=9,
    )
    figure.tight_layout()
    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT, dpi=140)
    plt.close(figure)


if __name__ == "__main__":
    main()
