"""A phased mission profile against a flat one, per block.

The same diagram over the same elapsed time, once as a single flat phase and
once as a declared three-phase profile with per-block duty cycles and stress
factors. The left panel is the effective age each block accumulates; the right
is what that does to the system reliability. A duty cycle of 0.12 on the
transmitter and a stress factor of 3 on the power strings during launch move
the answer in opposite directions, and the figure shows which wins.
"""

from __future__ import annotations

import numpy as np

from _common import SERIES, bootstrap, save, style_axes

bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from reliablock.cases import case, mission  # noqa: E402
from reliablock.commoncause import apply_common_cause  # noqa: E402
from reliablock.quantify import quantify  # noqa: E402


def main() -> int:
    diagram = case("smallsat_bus")
    flat = mission("three_year")
    phased = mission("leo_phased")

    flat_ages = flat.effective_ages(diagram)
    phased_ages = phased.effective_ages(diagram)
    names = list(diagram.block_names())

    print(f"diagram                  : {diagram.name}")
    print(f"flat profile             : {flat.name}, {flat.total_hours:.6g} h elapsed")
    print(f"phased profile           : {phased.name}, {phased.total_hours:.6g} h elapsed")
    print()
    print(f"{'block':<26s}{'flat eff h':>14s}{'phased eff h':>16s}{'ratio':>10s}")
    print("-" * 66)
    for name in names:
        ratio = phased_ages[name] / flat_ages[name] if flat_ages[name] else float("nan")
        print(f"{name:<26s}{flat_ages[name]:>14.6g}{phased_ages[name]:>16.6g}{ratio:>10.4f}")

    print()
    for label, profile in (("flat", flat), ("phased", phased)):
        independent = quantify(diagram, profile)
        with_cc = quantify(apply_common_cause(diagram), profile)
        print(
            f"{label:<8s} R independent {independent.reliability:.12f}  "
            f"R with cc {with_cc.reliability:.12f}  "
            f"Q ratio {with_cc.unreliability / independent.unreliability:.6f}"
        )

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.6))

    indices = np.arange(len(names))
    width = 0.38
    left.barh(
        indices + width / 2,
        [flat_ages[n] for n in names],
        width,
        label="flat, three years",
        color=SERIES[0],
        edgecolor="white",
    )
    left.barh(
        indices - width / 2,
        [phased_ages[n] for n in names],
        width,
        label="LEO phased",
        color=SERIES[1],
        edgecolor="white",
    )
    left.set_yticks(indices)
    left.set_yticklabels(names, fontsize=8)
    left.set_xscale("log")
    left.legend(fontsize=8, frameon=False, loc="lower right")
    style_axes(left, "effective age per block", "effective hours, log scale", "")

    hours = np.linspace(100.0, phased.total_hours, 50)
    for index, (label, suffix) in enumerate(
        (("independence assumed", ""), ("declared common cause", " (cc)"))
    ):
        del suffix
        structure = diagram if index == 0 else apply_common_cause(diagram)
        curve = [quantify(structure, float(t)).reliability for t in hours]
        right.plot(hours, curve, color=SERIES[index], linewidth=1.7, label=label)
    for index, (label, profile) in enumerate((("flat", flat), ("phased", phased))):
        value = quantify(apply_common_cause(diagram), profile).reliability
        right.scatter([profile.total_hours], [value], s=30, zorder=3,
                      color=SERIES[2 + index], marker="D")
        right.annotate(
            f"{label} profile\nR = {value:.6f}",
            (profile.total_hours, value),
            textcoords="offset points",
            xytext=(-96, 12 - 30 * index),
            fontsize=8,
            color=SERIES[2 + index],
        )
    right.legend(fontsize=8, frameon=False, loc="lower left")
    style_axes(
        right,
        "system reliability against elapsed time, flat duty",
        "elapsed hours",
        "system reliability, dimensionless",
    )

    written = save(figure, "mission_phases.png")
    print()
    print(f"written: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
