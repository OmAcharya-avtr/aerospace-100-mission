"""What the selection cost: the identity of everything left behind, with reasons.

What to notice: a selection is an instruction and a left-behind report is an
argument. Only one of the four reasons -- capacity -- can be fixed by asking the
network for more contact time. A product left behind because it expired before
any pass, because it is larger than any single window, or because something it
depends on is unreachable will not be helped by another minute, and the report
separates those cases so that the operator does not ask for the wrong thing.
"""

from __future__ import annotations

import numpy as np

from _common import (
    LEFT_BEHIND,
    ORACLE,
    REASON_COLOURS,
    bootstrap,
    save,
    style,
)

bootstrap()

from downlinkpick.cases import load_case  # noqa: E402
from downlinkpick.dataset import generate_instance  # noqa: E402
from downlinkpick.exact import exact_select  # noqa: E402
from downlinkpick.marginal import (  # noqa: E402
    LEFT_BEHIND_REASONS,
    left_behind_report,
    value_left_behind,
)
from downlinkpick.selection import SelectionProblem, sizes_from_ratios  # noqa: E402

CASE = "examples/cases/three_pass_declared.yaml"
N_INSTANCES = 60


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

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    figure.subplots_adjust(wspace=0.3, bottom=0.16, top=0.88)

    case = load_case(CASE)
    problem = SelectionProblem.from_declared(case.inventory, case.plan)
    selection = exact_select(problem)
    rows = left_behind_report(problem, selection)

    names = [row.product for row in rows]
    values = [row.value for row in rows]
    colours = [REASON_COLOURS[row.reason] for row in rows]
    positions = np.arange(len(rows))
    left.barh(positions, values, color=colours)
    for position, row in zip(positions, rows, strict=True):
        annotation = (
            f"{row.extra_seconds_needed:.0f} s on {row.best_window}"
            if row.reason == "capacity" and row.best_window
            else row.reason
        )
        left.text(row.value + 0.08, position, annotation, va="center", fontsize=7.5)
    left.set_yticks(positions)
    left.set_yticklabels(names, fontsize=8)
    left.invert_yaxis()
    left.set_xlabel("value left behind [value units]")
    left.set_xlim(0.0, max(values) * 1.6 if values else 1.0)
    left.set_title(
        f"three_pass_declared: {len(rows)} of {len(problem.inventory)} left behind, "
        f"worth {sum(values):.2f}"
    )

    totals = {reason: [] for reason in LEFT_BEHIND_REASONS}
    sent = []
    for seed in range(1, N_INSTANCES + 1):
        instance = generate_instance(
            seed, n_products=24, n_windows=3, capacity_fraction=0.45, dependency_fraction=0.4
        )
        instance_problem = _problem(instance)
        instance_selection = exact_select(instance_problem)
        per_reason = value_left_behind(
            left_behind_report(instance_problem, instance_selection)
        )
        available = instance_problem.total_value_available()
        for reason in LEFT_BEHIND_REASONS:
            totals[reason].append(per_reason[reason] / available)
        # The raw declared value of the selected products, NOT the
        # reliability-weighted objective, so that the bar sums to exactly one:
        # selected value plus left-behind value is the whole inventory, which is
        # the accounting identity validate_marginal.py checks.
        sent.append(
            sum(float(instance_problem.values[name]) for name in instance_selection.selected_set)
            / available
        )

    bottom = np.zeros(1)
    order = [*LEFT_BEHIND_REASONS]
    means = [float(np.mean(totals[reason])) for reason in order]
    sent_mean = float(np.mean(sent))
    right.bar(["mean over 60 instances"], [sent_mean], color=ORACLE, label="selected")
    bottom = np.array([sent_mean])
    for reason, mean in zip(order, means, strict=True):
        right.bar(
            ["mean over 60 instances"],
            [mean],
            bottom=bottom,
            color=REASON_COLOURS[reason],
            label=f"left behind: {reason}",
        )
        if mean > 0.01:
            right.text(
                0,
                bottom[0] + mean / 2.0,
                f"{mean:.1%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
        bottom = bottom + mean
    right.text(0, sent_mean / 2.0, f"{sent_mean:.1%}", ha="center", va="center",
               fontsize=8, color="white")
    right.set_ylabel("fraction of the inventory's declared value")
    right.set_xlim(-0.6, 0.6)
    right.set_ylim(0.0, 1.05)
    right.set_title("where the value goes, over a seeded instance family")
    right.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    right.set_facecolor("white")

    figure.text(
        0.5,
        0.01,
        "only the capacity slice can be bought back with more contact time; the rest is "
        "decided by the deadlines, the sizes and the dependency graph",
        ha="center",
        fontsize=7,
        color=LEFT_BEHIND,
    )
    return save(figure, "left_behind_report")


if __name__ == "__main__":
    print(main())
