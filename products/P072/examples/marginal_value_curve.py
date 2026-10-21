"""What the next minute of contact is worth, and when it is worth less than nothing.

What to notice: the objective is a step function of capacity, not a smooth
concave curve, so the marginal value of contact time is a finite difference and
not a derivative. The left panel sweeps extra seconds onto one pass. The right
panel is the instance where extending the pass *lowers* the achievable value: a
window's end is also its delivery epoch, so a longer pass is too late for a
product whose deadline falls inside the extension, and that product's columns
leave the model. The marked step is the 12.89 per cent drop pinned as a
regression test.
"""

from __future__ import annotations

import numpy as np

from _common import ANALYTIC, EXACT, GREEDY, ORACLE, SHADE, bootstrap, save, style

bootstrap()

from downlinkpick.cases import load_case  # noqa: E402
from downlinkpick.dataset import generate_instance  # noqa: E402
from downlinkpick.exact import exact_select  # noqa: E402
from downlinkpick.greedy import greedy_select  # noqa: E402
from downlinkpick.marginal import capacity_sweep  # noqa: E402
from downlinkpick.selection import SelectionProblem, sizes_from_ratios  # noqa: E402

CASE = "examples/cases/three_pass_declared.yaml"
STEPS = tuple(float(step) for step in range(0, 481, 15))
DROP_STEPS = tuple(float(step) for step in range(380, 561, 10))


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.2, 4.3))
    figure.subplots_adjust(wspace=0.26, bottom=0.14, top=0.86)

    case = load_case(CASE)
    problem = SelectionProblem.from_declared(case.inventory, case.plan)
    for window, colour in zip(problem.plan.names, (EXACT, GREEDY, ANALYTIC), strict=True):
        points = capacity_sweep(problem, window, STEPS)
        extra = np.array([p.extra_s for p in points])
        values = np.array([p.total_value for p in points])
        left.step(extra, values, where="post", color=colour, linewidth=1.7, label=window)
    greedy_points = capacity_sweep(
        problem, problem.plan.names[0], STEPS, selector=greedy_select
    )
    left.step(
        [p.extra_s for p in greedy_points],
        [p.total_value for p in greedy_points],
        where="post",
        color=ORACLE,
        linewidth=1.2,
        linestyle="--",
        label="pass1, greedy",
    )
    left.set_xlabel("extra seconds added to one pass")
    left.set_ylabel("achievable value [value units]")
    left.set_title("three_pass_declared: value against added contact time")
    left.legend(loc="upper left", fontsize=8)

    instance = generate_instance(36, n_products=20, n_windows=3, capacity_fraction=0.4)
    values = {
        name: float(instance.inventory[name].declared_value or 0.0)
        for name in instance.inventory.names
    }
    drop_problem = SelectionProblem(
        instance.inventory,
        instance.plan,
        values,
        sizes_from_ratios(instance.inventory, instance.true_ratios),
    )
    points = capacity_sweep(drop_problem, "pass1", DROP_STEPS)
    extra = np.array([p.extra_s for p in points])
    objective = np.array([p.total_value for p in points])
    crossings = np.array([p.n_expiry_crossings for p in points])
    first_cross = int(np.argmax(crossings > crossings[0])) if crossings.max() > crossings[0] else 0
    if first_cross:
        right.axvspan(extra[first_cross], extra[-1], color=SHADE, label="an expiry crossed")
    right.step(extra, objective, where="post", color=EXACT, linewidth=1.8)
    fall = int(np.argmin(np.diff(objective)))
    right.annotate(
        f"{objective[fall]:.3f} -> {objective[fall + 1]:.3f}\n"
        f"{-(objective[fall + 1] - objective[fall]) / objective[fall]:.2%} drop",
        xy=(extra[fall + 1], objective[fall + 1]),
        xytext=(extra[fall + 1] + 12.0, objective[fall + 1] + 1.6),
        arrowprops={"arrowstyle": "->", "color": ORACLE, "linewidth": 1.0},
        fontsize=8,
    )
    right.set_xlabel("extra seconds added to pass1")
    right.set_ylabel("achievable value [value units]")
    right.set_title("seed 36: a longer pass delivers less")
    right.legend(loc="lower left", fontsize=8)

    nominal = exact_select(drop_problem).total_value
    figure.text(
        0.5,
        0.005,
        f"seed 36 unextended optimum {nominal:.4f}; the drop is a measured property of "
        "the model, not a solver failure",
        ha="center",
        fontsize=7,
    )
    return save(figure, "marginal_value_curve")


if __name__ == "__main__":
    print(main())
