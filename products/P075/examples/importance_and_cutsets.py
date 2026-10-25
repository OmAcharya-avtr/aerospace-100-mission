"""Importance measures and the cut-set structure, before and after common cause.

Two panels on the voting triple: the importance table as a bar chart, and the
cut-set order histogram with and without the declared beta factor. The second
panel is the qualitative half of the headline: the declared common cause adds a
cut set of order one, which no amount of channel redundancy removes.
"""

from __future__ import annotations

import numpy as np

from _common import SERIES, bootstrap, save, style_axes

bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from reliablock.cases import case, mission  # noqa: E402
from reliablock.commoncause import apply_common_cause, strip_common_cause  # noqa: E402
from reliablock.importance import importance_table  # noqa: E402
from reliablock.quantify import quantify  # noqa: E402
from reliablock.structure import minimal_cut_sets  # noqa: E402


def main() -> int:
    diagram = case("voting_triple")
    profile = mission("three_year")
    independent = strip_common_cause(diagram)
    with_cc = apply_common_cause(diagram)
    rows = importance_table(with_cc, profile)

    print(f"diagram                  : {diagram.name}")
    print(f"mission                  : {profile.name}, {profile.total_hours:.6g} h")
    print(f"R assuming independence  : {quantify(independent, profile).reliability:.12f}")
    print(f"R with declared cc       : {quantify(with_cc, profile).reliability:.12f}")
    print()
    print(
        f"{'basic event':<26s}{'q':>14s}{'Birnbaum':>14s}"
        f"{'critical':>11s}{'Fus-Ves':>11s}{'cuts':>6s}"
    )
    print("-" * 82)
    for row in rows:
        print(
            f"{row.event:<26s}{row.unreliability:>14.6e}{row.birnbaum:>14.6e}"
            f"{row.criticality:>11.6f}{row.fussell_vesely:>11.6f}{row.cut_sets:>6d}"
        )

    orders_before: dict[int, int] = {}
    orders_after: dict[int, int] = {}
    for cut in minimal_cut_sets(independent):
        orders_before[len(cut)] = orders_before.get(len(cut), 0) + 1
    for cut in minimal_cut_sets(with_cc):
        orders_after[len(cut)] = orders_after.get(len(cut), 0) + 1
    print()
    print(f"cut sets by order, independence assumed : {dict(sorted(orders_before.items()))}")
    print(f"cut sets by order, declared common cause: {dict(sorted(orders_after.items()))}")
    print()
    print("minimal cut sets with the declared common cause:")
    for cut in minimal_cut_sets(with_cc):
        print("  {" + ", ".join(sorted(cut)) + "}")

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    labels = [r.event for r in rows]
    indices = np.arange(len(labels))
    width = 0.27
    left.barh(indices + width, [r.birnbaum for r in rows], width, label="Birnbaum",
              color=SERIES[0], edgecolor="white")
    left.barh(indices, [r.criticality for r in rows], width, label="criticality",
              color=SERIES[1], edgecolor="white")
    left.barh(indices - width, [r.fussell_vesely for r in rows], width,
              label="Fussell-Vesely", color=SERIES[2], edgecolor="white")
    left.set_yticks(indices)
    left.set_yticklabels(labels, fontsize=8)
    left.legend(fontsize=8, frameon=False, loc="lower right")
    style_axes(left, "importance measures, exact not rare-event", "dimensionless", "")

    orders = sorted(set(orders_before) | set(orders_after))
    positions = np.arange(len(orders))
    right.bar(
        positions - 0.19,
        [orders_before.get(o, 0) for o in orders],
        0.38,
        label="independence assumed",
        color=SERIES[0],
        edgecolor="white",
    )
    right.bar(
        positions + 0.19,
        [orders_after.get(o, 0) for o in orders],
        0.38,
        label="declared common cause",
        color=SERIES[1],
        edgecolor="white",
    )
    right.set_xticks(positions)
    right.set_xticklabels([f"order {o}" for o in orders])
    right.legend(fontsize=8, frameon=False)
    style_axes(right, "minimal cut sets by order", "", "count")
    right.text(
        0.04,
        0.80,
        "an order-1 cut set is a single point of failure;\n"
        "the declared common cause adds one that\n"
        "no amount of channel redundancy removes",
        transform=right.transAxes,
        fontsize=8,
        color=SERIES[1],
    )

    written = save(figure, "importance_and_cutsets.png")
    print()
    print(f"written: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
