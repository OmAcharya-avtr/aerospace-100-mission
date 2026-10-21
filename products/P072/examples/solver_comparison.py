"""Greedy against the proved optimum on one case, product by product.

What to notice: the two selectors agree about almost everything. The value of
the exact solver on this instance is one product -- the left panel marks where
the selections differ -- and the right panel puts that difference next to the
two upper bounds. Dantzig's fractional bound and the LP relaxation are both
valid upper bounds on the optimum and both are loose, which is why neither is a
substitute for solving the integer program when the gap matters.
"""

from __future__ import annotations

import numpy as np

from _common import EXACT, GREEDY, LEFT_BEHIND, ORACLE, bootstrap, save, style

bootstrap()

from downlinkpick.cases import load_case  # noqa: E402
from downlinkpick.exact import exact_select, lp_relaxation_bound  # noqa: E402
from downlinkpick.greedy import dantzig_bound, greedy_select  # noqa: E402
from downlinkpick.selection import SelectionProblem  # noqa: E402

CASE = "examples/cases/three_pass_declared.yaml"


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    case = load_case(CASE)
    problem = SelectionProblem.from_declared(case.inventory, case.plan)
    greedy = greedy_select(problem)
    exact = exact_select(problem)

    names = list(problem.inventory.names)
    density = np.array(
        [float(problem.values[name]) / (problem.sizes_bits[name] / 1.0e9) for name in names]
    )
    value = np.array([float(problem.values[name]) for name in names])
    in_greedy = np.array([name in greedy.selected_set for name in names])
    in_exact = np.array([name in exact.selected_set for name in names])

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(10.4, 4.3), gridspec_kw={"width_ratios": (1.5, 1.0)}
    )
    figure.subplots_adjust(wspace=0.26, bottom=0.16, top=0.88)

    both = in_greedy & in_exact
    neither = ~in_greedy & ~in_exact
    only_exact = in_exact & ~in_greedy
    only_greedy = in_greedy & ~in_exact
    left.scatter(density[both], value[both], s=34, color=ORACLE, label="both selectors")
    left.scatter(
        density[neither], value[neither], s=34, color=LEFT_BEHIND, marker="x",
        label="neither"
    )
    left.scatter(
        density[only_exact], value[only_exact], s=90, color=EXACT, marker="D",
        label="exact only"
    )
    left.scatter(
        density[only_greedy], value[only_greedy], s=90, color=GREEDY, marker="s",
        label="greedy only"
    )
    for index, name in enumerate(names):
        if only_exact[index] or only_greedy[index]:
            left.annotate(
                name, (density[index], value[index]), xytext=(7, 3),
                textcoords="offset points", fontsize=8
            )
    left.set_xscale("log")
    left.set_xlabel("value density [value units per Gbit]")
    left.set_ylabel("declared value [value units]")
    left.set_title("who is selected, against value density")
    left.legend(fontsize=8, loc="upper left")

    bars = {
        "greedy": (greedy.total_value, GREEDY),
        "exact": (exact.total_value, EXACT),
        "LP relaxation": (lp_relaxation_bound(problem), LEFT_BEHIND),
        "Dantzig bound": (dantzig_bound(problem), LEFT_BEHIND),
    }
    right.bar(list(bars), [v for v, _c in bars.values()], color=[c for _v, c in bars.values()])
    for position, (_label, (height, _colour)) in enumerate(bars.items()):
        right.text(position, height, f"{height:.3f}", ha="center", va="bottom", fontsize=8)
    gap = (exact.total_value - greedy.total_value) / exact.total_value
    right.set_ylabel("objective [value units]")
    right.set_title(f"objectives and upper bounds\ngreedy gap {gap:.4%}")
    right.set_ylim(0.0, max(v for v, _c in bars.values()) * 1.14)
    right.tick_params(axis="x", labelrotation=12)

    figure.text(
        0.5,
        0.01,
        f"exact status {exact.status}; both bounds are valid and loose, which is why "
        "the integer program is solved rather than bounded",
        ha="center",
        fontsize=7,
    )
    return save(figure, "solver_comparison")


if __name__ == "__main__":
    print(main())
