"""The near-null directions of three geometries, named in physical coefficients.

What to notice: the direction a narrow elevation band cannot determine is
almost exactly CA minus half of IA minus half of NPAE, and the one a single
azimuth cannot determine mixes the two azimuth-axis tilt components with NPAE
and the elevation index. These are statements about the instrument, not about
the solver.
"""

from __future__ import annotations

import numpy as np

from _common import DETERMINED, MARGINAL, REFUSED, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from pointmodel import identifiability, synthetic, terms, wls  # noqa: E402

GEOMETRIES = ("narrow_el", "single_az", "low_el")
OBSERVATIONS = 300
SIGMA = 2.0


def main() -> int:
    style()
    positions = np.arange(terms.N_TERMS)
    panels = []
    for name in GEOMETRIES:
        residuals = synthetic.generate(name, OBSERVATIONS, sigma_arcsec=SIGMA, seed=20260101)
        result = wls.fit(residuals.design, residuals.observed, SIGMA)
        report = identifiability.analyse_fit(residuals.design, result)
        panels.append((name, report))

    rows = max(len(report.null_directions) for _name, report in panels)
    figure, axes = plt.subplots(rows, len(panels), figsize=(12.0, 2.3 * rows + 1.2))
    axes = np.atleast_2d(axes)
    for column, (name, report) in enumerate(panels):
        for row in range(rows):
            axis = axes[row, column]
            if row < len(report.null_directions):
                direction = report.null_directions[row]
                colours = [
                    REFUSED if abs(value) >= 0.2 else MARGINAL
                    for value in direction.weights
                ]
                axis.bar(positions, direction.weights, color=colours)
                axis.set_ylim(-1.05, 1.05)
                axis.axhline(0.0, color="#1a1a1a", lw=0.8)
                axis.axhline(0.2, color=DETERMINED, ls=":", lw=0.8)
                axis.axhline(-0.2, color=DETERMINED, ls=":", lw=0.8)
                axis.set_title(
                    f"{name}  direction {direction.index}  "
                    f"s/s1 = {direction.ratio:.2e}",
                    fontsize=9,
                )
                axis.set_xticks(positions)
                axis.set_xticklabels(terms.TERM_NAMES, rotation=45, ha="right")
                axis.tick_params(labelbottom=True)
            else:
                axis.set_axis_off()
            if column == 0 and row < len(report.null_directions):
                axis.set_ylabel("unit weight")
    figure.suptitle(
        "directions the observed geometry cannot determine, in physical coefficients\n"
        "red bars carry 0.2 or more of a unit-norm nine-vector and are the material "
        "participants",
        fontsize=11,
    )
    figure.subplots_adjust(top=0.86, hspace=0.70, wspace=0.18)
    path = save(figure, "null_space_directions")
    print(f"written: {path}")
    print("")
    for name, report in panels:
        print(f"{name}: rank {report.numerical_rank} of 9, "
              f"condition {report.condition_number:.4e}")
        for direction in report.null_directions:
            print(f"  [{direction.index}] {direction.statement()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
