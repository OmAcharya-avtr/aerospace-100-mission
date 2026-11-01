"""The permutation null, and what a swept flight-test card does to it.

What to notice: the observed trend statistic of the exchangeable campaign
sits inside its own permutation null, and the swept campaign's sits in the far
tail of the same null. The null is identical in shape for both -- it is built
by permuting the sample, so it knows nothing about the campaign -- which is
why the comparison is exact in finite samples and needs no asymptotics.
"""

from __future__ import annotations

import numpy as np

from _common import (
    DISTRIBUTION_FREE,
    NEUTRAL,
    REFUSED,
    RESAMPLE,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from transfergap import synthetic  # noqa: E402
from transfergap.exchangeability import (  # noqa: E402
    null_distribution,
    permutation_test,
)

N = 120
PERMUTATIONS = 4000
ALPHA = 0.05


def main() -> int:
    style()
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 6.4))

    for row, statistic in enumerate(("trend", "half_split")):
        for column, (campaign, colour, title) in enumerate((
            ("random", DISTRIBUTION_FREE, "exchangeable campaign"),
            ("swept", REFUSED, "swept Mach card -- NOT exchangeable"),
        )):
            axis = axes[row][column]
            gaps = synthetic.generate(N, campaign, seed=303).gaps
            rng = np.random.default_rng(11)
            null = np.abs(null_distribution(gaps, statistic, PERMUTATIONS, rng))
            result = permutation_test(
                gaps, statistic, permutations=PERMUTATIONS, alpha=ALPHA, seed=11
            )
            axis.hist(null, bins=48, color=NEUTRAL, alpha=0.55,
                      label=f"permutation null, B = {PERMUTATIONS}")
            critical = float(np.quantile(null, 1.0 - ALPHA))
            axis.axvline(critical, color=RESAMPLE, ls=":", lw=1.4,
                         label=f"{1 - ALPHA:g} null quantile = {critical:.3f}")
            axis.axvline(abs(result.observed), color=colour, lw=2.0,
                         label=f"observed |T| = {abs(result.observed):.3f}, "
                               f"p = {result.p_value:.4f}")
            axis.set_title(f"{statistic} statistic, {title}")
            axis.set_xlabel("|statistic| under the permutation null")
            axis.set_ylabel("permutations")
            axis.legend(loc="upper right", fontsize=7.5)

    figure.suptitle(
        f"transfergap: the exchangeability assumption, TESTED -- n = {N}, "
        f"alpha = {ALPHA:g}",
        fontsize=11,
    )
    figure.subplots_adjust(top=0.90, hspace=0.42, wspace=0.22)
    path = save(figure, "exchangeability_test")
    print(f"written: {path}")
    for statistic in ("trend", "half_split"):
        for campaign in ("random", "swept"):
            result = permutation_test(
                synthetic.generate(N, campaign, seed=303).gaps, statistic,
                permutations=PERMUTATIONS, alpha=ALPHA, seed=11,
            )
            print(f"  {statistic:<11s} {campaign:<7s} observed {result.observed:+.4f} "
                  f"p = {result.p_value:.4f} -> "
                  f"{'REJECTED' if result.rejected else 'not rejected'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
