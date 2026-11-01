"""Each term's contribution to the residual field, and why they do not add up.

What to notice: the sum of the squared contributions is not the squared total
unless the columns are orthogonal, and on a narrow elevation band the refused
terms contribute enormous cancelling amounts. The budget rows must not be added
in quadrature.
"""

from __future__ import annotations

import numpy as np

from _common import LEAST, NEUTRAL, REFUSED, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

import pointmodel  # noqa: E402
from pointmodel import synthetic, terms  # noqa: E402

OBSERVATIONS = 300
SIGMA = 2.0
ORDER = ("full_sky", "two_az", "low_el", "narrow_el", "single_az")


def main() -> int:
    style()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    positions = np.arange(terms.N_TERMS)
    width = 0.16
    closures = []
    for offset, name in enumerate(ORDER):
        residuals = synthetic.generate(name, OBSERVATIONS, sigma_arcsec=SIGMA, seed=20260101)
        budget = pointmodel.decompose(
            residuals.design, residuals.observed, sigma_arcsec=SIGMA, robust=False
        )
        closures.append((name, budget.quadrature_closure, budget.total_rms_arcsec))
        heights = [row.contribution_rms_arcsec for row in budget.terms]
        colours = [
            REFUSED if not row.separable else LEAST for row in budget.terms
        ]
        axes[0].bar(
            positions + (offset - 2) * width,
            heights,
            width=width,
            color=colours,
            edgecolor="none",
            alpha=0.85 if name == "full_sky" else 0.6,
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(terms.TERM_NAMES, rotation=45, ha="right")
    axes[0].set_ylabel("contribution rms over the observed rows, arcsec")
    axes[0].set_title(
        "per-term contribution, five geometries side by side\n"
        "red bars are refused; their split is arbitrary",
        fontsize=9,
    )

    names = [name for name, _closure, _total in closures]
    values = [closure for _name, closure, _total in closures]
    axes[1].bar(
        np.arange(len(names)),
        values,
        color=[REFUSED if value > 1.5 or value < 0.9 else LEAST for value in values],
    )
    axes[1].axhline(1.0, color=NEUTRAL, ls="--", lw=1.2)
    axes[1].text(
        0.1, 1.3, "1.0 is exact only for an orthogonal design", fontsize=8, color=NEUTRAL
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(np.arange(len(names)))
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set_ylabel("sum of squared contributions / squared total")
    axes[1].set_title("quadrature closure: the term correlation made visible", fontsize=9)

    figure.suptitle(
        "a pointing error budget does not add in quadrature, and the departure is "
        "reported",
        fontsize=11,
    )
    figure.subplots_adjust(top=0.84, wspace=0.26)
    path = save(figure, "budget_contributions")
    print(f"written: {path}")
    print("")
    print("geometry     quadrature closure   observed rms arcsec")
    for name, closure, total in closures:
        print(f"{name:<12s} {closure:18.6f}   {total:19.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
