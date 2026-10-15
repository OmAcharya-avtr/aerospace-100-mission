"""What makes each shipped instance hard, in one picture per instance.

For every instance: the declared energy budget (generated, fixed load, activity
demand, usable battery) and the declared data budget (produced, recorder
capacity, contact capacity), side by side with what the exact optimum actually
achieved. The point is that the five instances are hard for different reasons,
which is why the feasibility report names a different constraint on each.

Run: ``python examples/instance_portraits.py``
"""

from __future__ import annotations

import numpy as np
from _common import (
    ANALYTIC,
    BATTERY,
    EXACT,
    GREEDY,
    LEARNED,
    RECORDER,
    add_src_to_path,
    save,
    style,
)

add_src_to_path()

import matplotlib.pyplot as plt  # noqa: E402

from activityplan.exact import solve_exact  # noqa: E402
from activityplan.feasibility import evaluate  # noqa: E402
from activityplan.greedy import plan_greedy  # noqa: E402
from activityplan.instances import INSTANCE_ORDER, RECOMMENDED_STEP_S, named_instance  # noqa: E402
from activityplan.timeline import RasterisedProblem  # noqa: E402


def main() -> int:
    style()
    fig, axes = plt.subplots(3, len(INSTANCE_ORDER), figsize=(15.0, 8.4))
    summary = []
    for column, name in enumerate(INSTANCE_ORDER):
        problem = named_instance(name)
        rasterised = RasterisedProblem.build(problem, RECOMMENDED_STEP_S[name])
        greedy = plan_greedy(rasterised)
        exact = solve_exact(rasterised, time_limit_s=150.0)
        report = evaluate(rasterised, exact.plan)
        generated, fixed = problem.baseline_energy_balance_wh()

        ax = axes[0][column]
        energy_labels = ["generated", "fixed load", "usable\nbattery", "activity\ndemand"]
        energy_values = [
            generated,
            fixed,
            problem.resources.usable_wh,
            problem.total_activity_energy_wh,
        ]
        ax.bar(
            range(4),
            energy_values,
            color=[ANALYTIC, GREEDY, BATTERY, EXACT],
            width=0.68,
        )
        ax.set_xticks(range(4))
        ax.set_xticklabels(energy_labels, fontsize=6)
        ax.set_title(name, fontsize=9)
        if column == 0:
            ax.set_ylabel("energy [W*h]")

        ax = axes[1][column]
        data_labels = ["produced", "recorder", "contacts"]
        data_values = [
            problem.total_data_mbit,
            problem.resources.recorder_capacity_mbit,
            problem.resources.total_contact_capacity_mbit,
        ]
        ax.bar(range(3), data_values, color=[EXACT, RECORDER, LEARNED], width=0.6)
        ax.set_xticks(range(3))
        ax.set_xticklabels(data_labels, fontsize=6)
        if column == 0:
            ax.set_ylabel("data [Mbit]")

        ax = axes[2][column]
        ax.bar(
            [0, 1, 2],
            [problem.total_priority, exact.value, greedy.value],
            color=["#ececec", EXACT, GREEDY],
            width=0.6,
        )
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["requested", "exact", "greedy"], fontsize=6)
        ax.set_title(
            f"binds: {report.binding.name}\n{report.binding.normalised:+.4f}", fontsize=7
        )
        if column == 0:
            ax.set_ylabel("priority units")
        summary.append(
            (
                name,
                problem.n_activities,
                problem.total_priority,
                greedy.value,
                exact.value,
                report.binding.name,
                report.binding.normalised,
            )
        )

    fig.suptitle(
        "activityplan: the five shipped instances are hard for five different reasons",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.90, hspace=0.55, wspace=0.35)
    path = save(fig, "instance_portraits.png")
    print(f"wrote {path.relative_to(add_src_to_path())}")
    print("")
    print(
        f"{'instance':18s} {'acts':>5s} {'requested':>10s} {'greedy':>7s} {'exact':>7s} "
        f"{'binding constraint':28s} {'normalised':>11s}"
    )
    for row in summary:
        print(
            f"{row[0]:18s} {row[1]:5d} {row[2]:10.1f} {row[3]:7.1f} {row[4]:7.1f} "
            f"{row[5]:28s} {row[6]:+11.6f}"
        )
    print("")
    print(f"mean greedy/exact ratio : {np.mean([r[3] / r[4] for r in summary]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
