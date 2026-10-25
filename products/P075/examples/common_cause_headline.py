"""The headline figure: system unreliability with and without the declared beta.

Shows the whole smallsat bus over a three-year mission with every declared beta
factor suppressed and then applied, broken down by the group each contribution
comes from, and with the change in the cut-set structure annotated. The point
of the figure is that the second bar is not the first bar plus a percentage:
the declared common cause introduces single points of failure that the
independent diagram does not have.
"""

from __future__ import annotations

import numpy as np

from _common import SERIES, bootstrap, save, style_axes

bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from reliablock.cases import case, mission  # noqa: E402
from reliablock.commoncause import (  # noqa: E402
    apply_common_cause,
    group_beta_effects,
    strip_common_cause,
)
from reliablock.quantify import quantify  # noqa: E402
from reliablock.structure import minimal_cut_sets  # noqa: E402


def main() -> int:
    diagram = case("smallsat_bus")
    profile = mission("three_year")
    independent = strip_common_cause(diagram)
    with_cc = apply_common_cause(diagram)
    q_independent = quantify(independent, profile).unreliability
    q_with_cc = quantify(with_cc, profile).unreliability
    singles_before = sum(1 for c in minimal_cut_sets(independent) if len(c) == 1)
    singles_after = sum(1 for c in minimal_cut_sets(with_cc) if len(c) == 1)
    effects = group_beta_effects(diagram, profile)

    print(f"diagram                  : {diagram.name}")
    print(f"mission                  : {profile.name}, {profile.total_hours:.6g} h")
    print(f"Q assuming independence  : {q_independent:.9e}")
    print(f"Q with declared common cause : {q_with_cc:.9e}")
    print(f"ratio                    : {q_with_cc / q_independent:.6f}")
    print(f"single points of failure : {singles_before} -> {singles_after}")
    print()
    print(
        f"{'group':<18s}{'beta':>7s}{'k-of-n':>9s}"
        f"{'Q indep':>14s}{'Q with cc':>14s}{'ratio':>9s}"
    )
    print("-" * 71)
    for effect in effects:
        print(
            f"{effect.label:<18s}{effect.beta:>7.3g}"
            f"{f'{effect.k}-of-{effect.members}':>9s}"
            f"{effect.unreliability_independent:>14.6e}"
            f"{effect.unreliability_with_common_cause:>14.6e}{effect.ratio:>9.4f}"
        )

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    left.bar(
        [0, 1],
        [q_independent, q_with_cc],
        width=0.55,
        color=(SERIES[0], SERIES[1]),
        edgecolor="white",
    )
    left.set_xticks([0, 1])
    left.set_xticklabels(["independence\nassumed", "declared\ncommon cause"])
    for index, value in enumerate((q_independent, q_with_cc)):
        left.annotate(
            f"{value:.4e}",
            (index, value),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
        )
    left.annotate(
        f"x{q_with_cc / q_independent:.3f}",
        (0.5, 0.5 * (q_independent + q_with_cc)),
        ha="center",
        fontsize=11,
        color=SERIES[1],
    )
    left.set_ylim(0.0, 1.25 * q_with_cc)
    style_axes(
        left,
        f"system unreliability over {profile.total_hours:.0f} h",
        "",
        "Q, dimensionless",
    )
    left.text(
        0.02,
        0.92,
        f"single points of failure: {singles_before} -> {singles_after}",
        transform=left.transAxes,
        fontsize=9,
        color=SERIES[1],
    )

    labels = [e.label for e in effects]
    indices = np.arange(len(labels))
    width = 0.38
    right.bar(
        indices - width / 2,
        [e.unreliability_independent for e in effects],
        width,
        label="independence assumed",
        color=SERIES[0],
        edgecolor="white",
    )
    right.bar(
        indices + width / 2,
        [e.unreliability_with_common_cause for e in effects],
        width,
        label="declared common cause",
        color=SERIES[1],
        edgecolor="white",
    )
    for index, effect in enumerate(effects):
        right.annotate(
            f"x{effect.ratio:.2f}",
            (index + width / 2, effect.unreliability_with_common_cause),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )
    right.set_xticks(indices)
    right.set_xticklabels(
        [f"{e.label}\nbeta={e.beta:g}, {e.k}-of-{e.members}" for e in effects], fontsize=8
    )
    right.legend(fontsize=8, frameon=False)
    style_axes(right, "per redundant group", "", "group Q, dimensionless")

    written = save(figure, "common_cause_headline.png")
    print()
    print(f"written: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
