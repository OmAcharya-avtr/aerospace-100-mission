"""Every structure's reliability over a mission, and the closed form against simulation.

Six declared structures with the same per-unit hazard rate, plotted against
mission duration: a single block, two in series, two in parallel, 2-of-3, a
cold standby pair with an imperfect switch, and a load-sharing pair. The
simulated points are overlaid on the closed-form curves so that a reader can
see the simulation is scored against the arithmetic and not the other way
round.
"""

from __future__ import annotations

import numpy as np

from _common import SERIES, bootstrap, save, style_axes

bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from reliablock.diagram import (  # noqa: E402
    Block,
    Diagram,
    KOfN,
    Parallel,
    Series,
    SharedLoad,
    Standby,
)
from reliablock.distributions import Exponential  # noqa: E402
from reliablock.montecarlo import simulate  # noqa: E402
from reliablock.quantify import quantify  # noqa: E402

RATE = 3.0e-4
SEED = 20261025
TRIALS = 200_000


def _block(name: str) -> Block:
    return Block(name=name, distribution=Exponential(RATE))


def structures() -> dict[str, Diagram]:
    return {
        "single block": Diagram("single", Series(items=(_block("a"),))),
        "two in series": Diagram("series", Series(items=(_block("a"), _block("b")))),
        "two in parallel": Diagram("parallel", Parallel(items=(_block("a"), _block("b")))),
        "2-of-3": Diagram(
            "2of3", KOfN(k=2, items=(_block("a"), _block("b"), _block("c")))
        ),
        "cold standby, switch 0.95": Diagram(
            "standby",
            Standby(
                primary=_block("p"),
                standbys=(_block("s"),),
                switch_reliability=0.95,
                label="sb",
            ),
        ),
        "shared load, factor 0.5": Diagram(
            "shared",
            SharedLoad(units=(_block("u1"), _block("u2")), shared_rate_factor=0.5),
        ),
    }


def main() -> int:
    hours = np.linspace(0.0, 6000.0, 61)
    sample_hours = (1000.0, 3000.0, 5000.0)
    cases = structures()

    print(f"per-unit hazard rate     : {RATE:.3e} /h")
    print(f"Monte Carlo trials       : {TRIALS} at seed {SEED}")
    print()
    print(f"{'structure':<28s}{'t, h':>8s}{'closed form R':>18s}{'simulated R':>16s}{'sigma':>8s}")
    print("-" * 78)
    worst_sigma = 0.0
    for name, diagram in cases.items():
        for t in sample_hours:
            closed = quantify(diagram, t)
            result = simulate(diagram, t, TRIALS, seed=SEED)
            sigma = result.standard_errors_from(closed.reliability)
            worst_sigma = max(worst_sigma, sigma)
            print(
                f"{name:<28s}{t:>8.0f}{closed.reliability:>18.9f}"
                f"{result.reliability:>16.9f}{sigma:>8.3f}"
            )
    print()
    print(f"worst distance in standard errors : {worst_sigma:.4f}")

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    for index, (name, diagram) in enumerate(cases.items()):
        colour = SERIES[index % len(SERIES)]
        curve = [quantify(diagram, float(t)).reliability for t in hours]
        left.plot(hours, curve, color=colour, linewidth=1.6, label=name)
        points = [simulate(diagram, t, TRIALS, seed=SEED).reliability for t in sample_hours]
        left.scatter(sample_hours, points, color=colour, s=18, zorder=3, marker="o")
    left.legend(fontsize=8, frameon=False)
    style_axes(
        left,
        "closed form (lines) with simulation overlaid (points)",
        "mission duration, hours",
        "system reliability, dimensionless",
    )

    for index, (name, diagram) in enumerate(cases.items()):
        colour = SERIES[index % len(SERIES)]
        curve = [max(quantify(diagram, float(t)).unreliability, 1e-18) for t in hours[1:]]
        right.plot(hours[1:], curve, color=colour, linewidth=1.6, label=name)
    right.set_yscale("log")
    right.legend(fontsize=8, frameon=False)
    style_axes(
        right,
        "the same thing in unreliability space, where redundancy is visible",
        "mission duration, hours",
        "system unreliability, dimensionless",
    )

    written = save(figure, "structure_comparison.png")
    print()
    print(f"written: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
