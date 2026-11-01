"""The per-term budget with its intervals, on a good and a restricted geometry.

What to notice: on the restricted geometry the intervals on the refused terms
are enormous and still do not contain the truth, because the error there is a
systematic consequence of the minimum-norm choice and not a random one. A wide
interval is not a safe interval.
"""

from __future__ import annotations

import numpy as np

from _common import LEAST, REFUSED, TRUTH, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

import pointmodel  # noqa: E402
from pointmodel import synthetic, terms  # noqa: E402

OBSERVATIONS = 300
SIGMA = 2.0


def main() -> int:
    style()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    positions = np.arange(terms.N_TERMS)
    for axis, geometry in zip(axes, ("full_sky", "narrow_el"), strict=True):
        residuals = synthetic.generate(
            geometry, OBSERVATIONS, sigma_arcsec=SIGMA, seed=20260101
        )
        budget = pointmodel.decompose(
            residuals.design, residuals.observed, sigma_arcsec=SIGMA, robust=False
        )
        truth = residuals.truth_mapping()
        for index, row in enumerate(budget.terms):
            colour = LEAST if row.separable else REFUSED
            axis.errorbar(
                index,
                row.coefficient,
                yerr=[
                    [row.coefficient - row.interval[0]],
                    [row.interval[1] - row.coefficient],
                ],
                fmt="o",
                ms=5,
                color=colour,
                capsize=3,
                lw=1.3,
            )
            axis.plot(
                [index - 0.3, index + 0.3],
                [truth[row.name], truth[row.name]],
                color=TRUTH,
                lw=1.8,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(terms.TERM_NAMES, rotation=45, ha="right")
        axis.set_ylabel("coefficient, arcsec")
        axis.set_yscale("symlog", linthresh=10.0)
        axis.set_title(
            f"{geometry}: rank {budget.identifiability.numerical_rank} of 9, "
            f"condition {budget.identifiability.condition_number:.3g}\n"
            f"refused: {', '.join(budget.refused_terms) or 'nothing'}",
            fontsize=9,
        )
    figure.suptitle(
        "the per-term budget with 95 per cent intervals; black bars are the truth\n"
        "red points are refused by the rank analysis and must not be read as "
        "measurements",
        fontsize=11,
    )
    figure.subplots_adjust(top=0.80, wspace=0.22)
    path = save(figure, "confidence_intervals")
    print(f"written: {path}")
    print("")
    for geometry in ("full_sky", "narrow_el"):
        residuals = synthetic.generate(
            geometry, OBSERVATIONS, sigma_arcsec=SIGMA, seed=20260101
        )
        budget = pointmodel.decompose(
            residuals.design, residuals.observed, sigma_arcsec=SIGMA, robust=False
        )
        truth = residuals.truth_mapping()
        covered = sum(
            1
            for row in budget.terms
            if row.interval[0] <= truth[row.name] <= row.interval[1]
        )
        print(f"{geometry}: {covered} of 9 intervals contain the truth")
        for row in budget.terms:
            inside = row.interval[0] <= truth[row.name] <= row.interval[1]
            print(
                f"  {row.name:<4s} {row.coefficient:12.4f} "
                f"[{row.interval[0]:12.4f}, {row.interval[1]:12.4f}] "
                f"truth {truth[row.name]:8.4f} "
                f"{'inside' if inside else 'OUTSIDE'} "
                f"{'separable' if row.separable else 'REFUSED'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
