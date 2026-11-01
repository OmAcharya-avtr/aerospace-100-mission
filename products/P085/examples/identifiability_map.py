"""Condition number and rank against the width of the observed elevation band.

What to notice: the condition number is flat and small once the band is wider
than about 20 degrees and rises without limit as it narrows. There is no sharp
edge, which is why the package reports the number and the named directions
rather than a yes or no.
"""

from __future__ import annotations

import numpy as np

from _common import DETERMINED, MARGINAL, REFUSED, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from pointmodel import identifiability, terms  # noqa: E402

CENTRE_DEG = 47.5
WIDTHS = np.linspace(2.0, 85.0, 56)
OBSERVATIONS = 300


def main() -> int:
    style()
    rng = np.random.default_rng(20260101)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, OBSERVATIONS)
    conditions = []
    ranks = []
    for width in WIDTHS:
        low = max(5.0, CENTRE_DEG - 0.5 * width)
        high = min(89.0, CENTRE_DEG + 0.5 * width)
        elevation = np.radians(rng.uniform(low, high, OBSERVATIONS))
        design = terms.stack_residuals(*terms.design_matrix(azimuth, elevation))
        report = identifiability.analyse(design, np.zeros(terms.N_TERMS))
        conditions.append(report.condition_number)
        ranks.append(report.numerical_rank)

    figure, axes = plt.subplots(
        2, 1, figsize=(8.0, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    axes[0].semilogy(WIDTHS, conditions, color=REFUSED, lw=1.8)
    axes[0].axhline(1e3, color=MARGINAL, ls="--", lw=1.2)
    axes[0].text(
        60.0, 1.3e3, "condition 1e3: the declared null threshold", fontsize=8, color=MARGINAL
    )
    axes[0].axhline(1e2, color=DETERMINED, ls=":", lw=1.2)
    axes[0].text(60.0, 1.1e2, "condition 1e2: the full-sky value", fontsize=8, color=DETERMINED)
    axes[0].set_ylabel("condition number of the normalised design")
    axes[0].set_title(
        "a pointing model's conditioning is set by the elevation coverage\n"
        f"{OBSERVATIONS} observations, all azimuths, band centred on {CENTRE_DEG:g} degrees"
    )

    axes[1].step(WIDTHS, ranks, where="mid", color=REFUSED, lw=1.8)
    axes[1].set_ylim(5.5, 9.5)
    axes[1].set_yticks([6, 7, 8, 9])
    axes[1].set_ylabel("rank of 9")
    axes[1].set_xlabel("width of the observed elevation band, degrees")
    figure.subplots_adjust(hspace=0.12)
    path = save(figure, "identifiability_map")
    print(f"written: {path}")
    print("")
    print("band width deg   condition      rank")
    for width, condition, rank in zip(WIDTHS, conditions, ranks, strict=True):
        if abs(width % 10.0) < 0.8 or width == WIDTHS[0] or width == WIDTHS[-1]:
            print(f"{width:14.2f}   {condition:12.4e}   {rank:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
