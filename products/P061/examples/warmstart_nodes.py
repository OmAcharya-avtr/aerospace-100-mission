"""Does the warm start help the solver. Mostly there is nothing to help.

Left: the cold node count across the shipped instances and a slice of the seeded
family, with presolve on and off. The bars at 1 are instances HiGHS solves at
the root, where a primal bound has no tree to prune.
Right: cold against warm on the instances that actually branch.

Run: ``python examples/warmstart_nodes.py``
"""

from __future__ import annotations

import numpy as np
from _common import EXACT, GREEDY, LEARNED, add_src_to_path, save, style

add_src_to_path()

import matplotlib.pyplot as plt  # noqa: E402

from activityplan.exact import solve_exact  # noqa: E402
from activityplan.greedy import plan_greedy  # noqa: E402
from activityplan.instances import (  # noqa: E402
    INSTANCE_ORDER,
    RECOMMENDED_STEP_S,
    instance_family,
    named_instance,
)
from activityplan.timeline import RasterisedProblem  # noqa: E402
from activityplan.warmstart import compare_warm_start, warm_start_from_plan  # noqa: E402

FAMILY_SEED = 9000
FAMILY_SIZE = 8


def main() -> int:
    style()
    cases = [
        (name, RasterisedProblem.build(named_instance(name), RECOMMENDED_STEP_S[name]))
        for name in INSTANCE_ORDER
    ]
    cases += [
        (str(spec.seed), RasterisedProblem.build(spec.analytic_problem(), spec.step_s))
        for spec in instance_family(FAMILY_SIZE, FAMILY_SEED)
    ]

    labels: list[str] = []
    cold_on: list[int] = []
    cold_off: list[int] = []
    branching: list[tuple[str, RasterisedProblem, bool]] = []
    for label, rasterised in cases:
        on = solve_exact(rasterised, time_limit_s=150.0, presolve=True)
        off = solve_exact(rasterised, time_limit_s=150.0, presolve=False)
        labels.append(label)
        cold_on.append(on.node_count)
        cold_off.append(off.node_count)
        if on.node_count > 1:
            branching.append((label, rasterised, True))
        if off.node_count > 1:
            branching.append((label, rasterised, False))

    rows = []
    for label, rasterised, presolve in branching:
        candidate = plan_greedy(rasterised).plan
        warm_start = warm_start_from_plan(rasterised, candidate, source="greedy")
        comparison = compare_warm_start(
            rasterised, warm_start, presolve=presolve, time_limit_s=200.0
        )
        if comparison.warm is None:
            continue
        rows.append(
            (
                f"{label}\npresolve {'on' if presolve else 'off'}",
                comparison.cold.node_count,
                comparison.warm.node_count,
                comparison.cold.solve_s,
                comparison.warm.solve_s,
                comparison.optimum_preserved,
            )
        )

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), width_ratios=(2.0, 1.0))

    ax = axes[0]
    positions = np.arange(len(labels))
    ax.bar(positions - 0.19, cold_on, width=0.36, color=GREEDY, label="presolve on")
    ax.bar(positions + 0.19, cold_off, width=0.36, color=EXACT, label="presolve off")
    ax.axhline(1.0, color=LEARNED, linewidth=1.0, linestyle="--", label="solved at the root")
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("branch-and-bound nodes (log scale)")
    ax.set_title(
        "cold node count. A bar at 1 means HiGHS closed the gap at the root,\n"
        "so there was no search for a warm start to prune."
    )
    ax.legend(fontsize=8)

    ax = axes[1]
    if rows:
        positions = np.arange(len(rows))
        ax.bar(
            positions - 0.19,
            [row[1] for row in rows],
            width=0.36,
            color=EXACT,
            label="cold nodes",
        )
        ax.bar(
            positions + 0.19,
            [row[2] for row in rows],
            width=0.36,
            color=LEARNED,
            label="warm nodes",
        )
        for index, row in enumerate(rows):
            ax.text(
                index,
                max(row[1], row[2]) * 1.25,
                f"{row[4] / row[3]:.2f}x time",
                ha="center",
                fontsize=7.5,
                color=LEARNED,
            )
        ax.set_yscale("log")
        ax.set_xticks(positions)
        ax.set_xticklabels([row[0] for row in rows], fontsize=7)
        ax.set_ylabel("nodes (log scale)")
        ax.set_title(
            "the instances that branch. The warm start is a primal-bound\n"
            "cut, not a MIP start; scipy.optimize.milp has no MIP start."
        )
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no instance branched", ha="center", va="center")

    fig.suptitle(
        "activityplan: the warm start helps only where there is a search to prune",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.84, wspace=0.26)
    path = save(fig, "warmstart_nodes.png")
    print(f"wrote {path.relative_to(add_src_to_path())}")
    print("")
    root_solved = sum(1 for value in cold_on + cold_off if value <= 1)
    print(f"cold solves                : {len(cold_on) + len(cold_off)}")
    print(f"solved at the root         : {root_solved}")
    print(f"instances with a real tree : {len(branching)}")
    print("")
    print(
        f"{'case':28s} {'cold nodes':>11s} {'warm nodes':>11s} {'cold s':>8s} "
        f"{'warm s':>8s} {'optimum kept':>13s}"
    )
    for label, cold_nodes, warm_nodes, cold_s, warm_s, kept in rows:
        print(
            f"{label.replace(chr(10), ' '):28s} {cold_nodes:11d} {warm_nodes:11d} "
            f"{cold_s:8.3f} {warm_s:8.3f} {str(kept):>13s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
