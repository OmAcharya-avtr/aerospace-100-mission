"""The greedy optimality gap as a distribution, not a mean.

What to notice: the mean gap is the least informative number on this figure.
The left panel is the empirical cumulative distribution over a seeded instance
family; the step at zero is the fraction of instances on which the operational
baseline is already optimal, and the right tail is the small number of instances
where it is not close. The right panel is the gap against how tight the contact
is: the dependency density matters more than the capacity fraction.
"""

from __future__ import annotations

import numpy as np

from _common import ANALYTIC, EXACT, GREEDY, LEARNED, ORACLE, bootstrap, save, style

bootstrap()

from downlinkpick.dataset import generate_instance  # noqa: E402
from downlinkpick.exact import exact_select  # noqa: E402
from downlinkpick.greedy import greedy_select  # noqa: E402
from downlinkpick.selection import SelectionProblem, sizes_from_ratios  # noqa: E402

N_INSTANCES = 120

FAMILIES = (
    ("capacity 20%", dict(capacity_fraction=0.20), ANALYTIC),
    ("capacity 45%", dict(capacity_fraction=0.45), EXACT),
    ("capacity 85%", dict(capacity_fraction=0.85), LEARNED),
    (
        "capacity 45%, dense dependencies",
        dict(capacity_fraction=0.45, dependency_fraction=0.6),
        GREEDY,
    ),
)


def _problem(instance) -> SelectionProblem:
    values = {
        name: float(instance.inventory[name].declared_value or 0.0)
        for name in instance.inventory.names
    }
    return SelectionProblem(
        instance.inventory,
        instance.plan,
        values,
        sizes_from_ratios(instance.inventory, instance.true_ratios),
    )


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    figure.subplots_adjust(wspace=0.26, bottom=0.14, top=0.88)

    summary: list[tuple[str, float, float, float, int]] = []
    for label, kwargs, colour in FAMILIES:
        gaps = []
        for seed in range(1, N_INSTANCES + 1):
            problem = _problem(
                generate_instance(seed, n_products=24, n_windows=3, **kwargs)
            )
            optimum = exact_select(problem).total_value
            heuristic = greedy_select(problem).total_value
            gaps.append(
                max((optimum - heuristic) / optimum, 0.0) if optimum > 0.0 else 0.0
            )
        array = np.sort(np.asarray(gaps))
        fraction = np.arange(1, array.size + 1) / array.size
        left.step(array, fraction, where="post", color=colour, linewidth=1.7, label=label)
        optimal = int(np.sum(array <= 1e-9))
        summary.append((label, float(array.mean()), float(array.max()), 0.0, optimal))
        right.scatter(
            np.full(array.size, len(summary) - 1) + np.linspace(-0.18, 0.18, array.size),
            array,
            s=5,
            color=colour,
            alpha=0.45,
        )
        right.plot(
            [len(summary) - 1 - 0.28, len(summary) - 1 + 0.28],
            [array.mean(), array.mean()],
            color=ORACLE,
            linewidth=1.6,
        )

    left.set_xlabel("greedy optimality gap, (optimum - greedy) / optimum")
    left.set_ylabel("fraction of instances at or below")
    left.set_title(f"empirical distribution, {N_INSTANCES} seeded instances per family")
    left.set_xlim(left=0.0)
    left.set_ylim(0.0, 1.02)
    left.legend(loc="lower right", fontsize=8)

    right.set_xticks(range(len(FAMILIES)))
    right.set_xticklabels(
        [label.replace(", ", "\n") for label, _k, _c in FAMILIES], fontsize=8
    )
    right.set_ylabel("greedy optimality gap")
    right.set_title("per instance, with the family mean as a bar")
    right.set_ylim(bottom=0.0)

    caption = "   ".join(
        f"{label}: mean {mean:.4f}, max {worst:.4f}, {optimal}/{N_INSTANCES} optimal"
        for label, mean, worst, _unused, optimal in summary
    )
    figure.text(0.5, 0.005, caption, ha="center", fontsize=7)
    return save(figure, "greedy_gap_distribution")


if __name__ == "__main__":
    print(main())
