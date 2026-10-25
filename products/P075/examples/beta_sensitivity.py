"""How much of the answer rests on the beta factor, swept over the parameter.

A declared beta factor is a judgement, not a measurement, so the useful output
is not one number but the curve: how does the system unreliability move as the
assumed correlation goes from nothing to total? This example sweeps a common
beta across every declared group of the smallsat bus and plots the system
unreliability against it, with the independence point marked.

It also plots the k-of-n non-conservatism: for a 3-of-4 group the beta factor
can REDUCE the computed unreliability above a certain member hazard, because
the beta-factor model has exactly one common-cause mode and it fails all four
members where the group only needs two failures.
"""

from __future__ import annotations

import numpy as np

from _common import SERIES, bootstrap, save, style_axes

bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from reliablock.cases import case, mission  # noqa: E402
from reliablock.commoncause import apply_common_cause, strip_common_cause  # noqa: E402
from reliablock.diagram import Block, Diagram, KOfN, Parallel  # noqa: E402
from reliablock.distributions import Exponential  # noqa: E402
from reliablock.quantify import quantify  # noqa: E402

HOURS = 1000.0


def _retune(diagram: Diagram, beta: float) -> Diagram:
    """The same diagram with every declared beta replaced by ``beta``."""
    from dataclasses import replace

    def walk(item):  # type: ignore[no-untyped-def]
        if isinstance(item, Block):
            return item
        children = tuple(walk(c) for c in item.children)
        if isinstance(item, Parallel):
            return replace(item, items=children, beta_factor=beta)
        if isinstance(item, KOfN):
            return replace(item, items=children, beta_factor=beta)
        from reliablock.diagram import Series, SharedLoad, Standby

        if isinstance(item, Series):
            return replace(item, items=children)
        if isinstance(item, Standby):
            return replace(item, primary=children[0], standbys=children[1:])
        if isinstance(item, SharedLoad):
            return replace(item, units=tuple(c for c in children if isinstance(c, Block)))
        return item

    return Diagram(name=diagram.name, root=walk(diagram.root))


def main() -> int:
    diagram = case("smallsat_bus")
    profile = mission("three_year")
    betas = np.linspace(0.0, 0.5, 51)
    unreliabilities = []
    for beta in betas:
        retuned = _retune(diagram, float(beta))
        unreliabilities.append(quantify(apply_common_cause(retuned), profile).unreliability)
    unreliabilities = np.array(unreliabilities)
    baseline = quantify(strip_common_cause(diagram), profile).unreliability
    declared = quantify(apply_common_cause(diagram), profile).unreliability

    print(f"diagram                  : {diagram.name}")
    print(f"mission                  : {profile.name}")
    print(f"Q at beta = 0            : {unreliabilities[0]:.9e}")
    at_tenth = quantify(apply_common_cause(_retune(diagram, 0.1)), profile).unreliability
    print(f"Q at beta = 0.1          : {at_tenth:.9e}")
    print(f"Q at beta = 0.5          : {unreliabilities[-1]:.9e}")
    print(f"Q with declared betas    : {declared:.9e}")
    print(f"Q assuming independence  : {baseline:.9e}")
    print(f"ratio at beta = 0.5      : {unreliabilities[-1] / baseline:.6f}")
    print()

    # The k-of-n non-conservatism map.
    hazards = np.geomspace(0.003, 3.0, 60)
    ratios = {}
    for k, n in ((1, 2), (2, 3), (3, 4), (4, 5)):
        row = []
        for hazard in hazards:
            rate = float(hazard) / HOURS
            items = tuple(
                Block(name=f"u{i}", distribution=Exponential(rate)) for i in range(n)
            )
            node = (
                Parallel(items=items, beta_factor=0.1, label="g")
                if k == 1
                else KOfN(k=k, items=items, beta_factor=0.1, label="g")
            )
            group = Diagram("g", node)
            independent = quantify(strip_common_cause(group), HOURS).unreliability
            with_cc = quantify(apply_common_cause(group), HOURS).unreliability
            row.append(with_cc / independent)
        ratios[(k, n)] = np.array(row)
        crossing = "none"
        below = row[0] < 1.0
        for hazard, value in zip(hazards, row, strict=True):
            if (value < 1.0) != below:
                crossing = f"{hazard:.4f}"
                break
        print(
            f"{k}-of-{n}: ratio at H=0.003 {row[0]:>12.4f}  "
            f"at H=3 {row[-1]:>8.4f}  crosses 1 at H={crossing}"
        )

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    left.plot(betas, unreliabilities, color=SERIES[0], linewidth=1.8)
    left.axhline(baseline, color=SERIES[2], linewidth=1.0, linestyle="--")
    left.annotate(
        "independence assumed",
        (0.26, baseline),
        textcoords="offset points",
        xytext=(0, 6),
        fontsize=8,
        color=SERIES[2],
    )
    left.scatter([0.12], [declared], color=SERIES[1], zorder=3, s=28)
    left.annotate(
        "the diagram's own declared betas\n(0.12, 0.08, 0.05)",
        (0.12, declared),
        textcoords="offset points",
        xytext=(10, -22),
        fontsize=8,
        color=SERIES[1],
    )
    style_axes(
        left,
        "system unreliability against a common beta on every group",
        "beta applied to every declared group, dimensionless",
        "Q, dimensionless",
    )

    for (k, n), row in ratios.items():
        right.plot(hazards, row, linewidth=1.6, label=f"{k}-of-{n}")
    right.axhline(1.0, color="#1b1b1b", linewidth=1.0, linestyle="--")
    right.set_xscale("log")
    right.set_yscale("log")
    right.legend(fontsize=8, frameon=False, title="group", title_fontsize=8)
    style_axes(
        right,
        "ratio Q(with cc) / Q(independent) at beta = 0.1",
        "member cumulative hazard over the mission, dimensionless",
        "ratio, dimensionless",
    )
    right.text(
        0.03,
        0.08,
        "below the dashed line the beta factor\nREDUCES the computed unreliability",
        transform=right.transAxes,
        fontsize=8,
        color=SERIES[1],
    )

    written = save(figure, "beta_sensitivity.png")
    print()
    print(f"written: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
