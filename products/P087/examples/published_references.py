"""Every published reference this package is scored against, pass and fail.

What to notice: the fifth bar. Bennett's own published accuracy claim of 0.07
arcminutes is not reproduced against this package's ray trace below about 19
degrees of apparent elevation, and the miss is drawn at full size rather than
clipped. The claim was made against Garfinkel (1967), which this package does
not implement, so the miss measures the difference between two reference
atmospheres and not an error in the published formula.
"""

from __future__ import annotations

import numpy as np

from _common import bootstrap, relative, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from refractpath.reference_checks import (  # noqa: E402
    bennett_departure_arcsec,
    run_reference_checks,
)
from refractpath.references import BENNETT_ACCURACY_CLAIM  # noqa: E402


def main() -> str:
    style()
    comparisons = run_reference_checks()
    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(8.6, 7.4), gridspec_kw={"height_ratios": [1.0, 1.0]}
    )

    labels = []
    ratios = []
    colours = []
    for comparison in comparisons:
        reference = comparison.reference
        labels.append(f"{reference.key}\n({reference.kind})")
        if reference.tolerance > 0.0:
            ratios.append(abs(comparison.difference) / reference.tolerance)
        else:
            ratios.append(comparison.measured / reference.value)
        colours.append("#3f8f4f" if comparison.within_tolerance else "#b0344f")
    positions = np.arange(len(labels))
    upper.barh(positions, ratios, color=colours, height=0.6)
    upper.axvline(1.0, color="#333333", linewidth=1.0)
    upper.set_yticks(positions)
    upper.set_yticklabels(labels, fontsize=7.5)
    upper.invert_yaxis()
    upper.set_xscale("log")
    upper.set_xlabel("measured departure divided by the published tolerance (1.0 is the limit)")
    upper.set_title(
        "Six published references, five reproduced and one not. "
        "Green is inside the published tolerance."
    )
    for position, ratio, comparison in zip(positions, ratios, comparisons, strict=True):
        upper.annotate(
            f"{ratio:.3g}x",
            xy=(ratio, position),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=("#3f8f4f" if comparison.within_tolerance else "#b0344f"),
        )

    grid, departure = bennett_departure_arcsec()
    lower.plot(grid, departure, color="#1f4e79", linewidth=1.5,
               label="|equation (15) - equation (19)|")
    lower.axhline(BENNETT_ACCURACY_CLAIM.value, color="#b0344f", linewidth=1.2,
                  linestyle="--",
                  label=f"published claim, {BENNETT_ACCURACY_CLAIM.value:.2f} arcsec")
    inside = departure <= BENNETT_ACCURACY_CLAIM.value
    last_violation = int(np.flatnonzero(~inside)[-1])
    holds_above = float(grid[last_violation + 1])
    lower.axvline(holds_above, color="#3f8f4f", linewidth=1.0, linestyle=":")
    lower.annotate(
        f"claim holds at and above\n{holds_above:.2f} deg",
        xy=(holds_above, 8.0),
        xytext=(holds_above + 8.0, 11.0),
        fontsize=8,
        color="#3f8f4f",
        arrowprops={"arrowstyle": "->", "color": "#3f8f4f", "linewidth": 0.8},
    )
    lower.set_yscale("log")
    lower.set_xscale("symlog", linthresh=1.0)
    lower.set_xlim(0.0, 90.0)
    lower.set_ylim(1e-2, 1e2)
    lower.set_xlabel("apparent elevation (deg, symmetric log below 1 deg)")
    lower.set_ylabel("absolute departure (arcsec)")
    lower.set_title(
        "The failing check, at full size: 1010 hPa, 10 C, dry, 550 nm"
    )
    lower.legend(loc="lower left", fontsize=8)

    figure.subplots_adjust(hspace=0.38)
    path = save(figure, "published_references")
    print(f"written: {path}")
    return relative(path)


if __name__ == "__main__":
    main()
