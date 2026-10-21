"""The contact plan and what the selection put into each pass.

What to notice: the achievable rate is not flat across a pass, so a window's
capacity is the integral of the rate and not the product of a rate and a
duration. The shaded area under each profile is the capacity actually committed
by the exact solver, and the unshaded remainder is the spare capacity reported
in the selection table. The right-hand panel is the queue against the contact:
the recorder holds more than the plan can carry, which is the whole problem.
"""

from __future__ import annotations

import numpy as np

from _common import EXACT, LEFT_BEHIND, ORACLE, SHADE, bootstrap, save, style

bootstrap()

from downlinkpick.cases import load_case  # noqa: E402
from downlinkpick.exact import exact_select  # noqa: E402
from downlinkpick.selection import SelectionProblem  # noqa: E402

CASE = "examples/cases/three_pass_declared.yaml"


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    case = load_case(CASE)
    problem = SelectionProblem.from_declared(case.inventory, case.plan)
    selection = exact_select(problem)
    committed = selection.bits_by_window()
    capacity = problem.capacity_bits()

    figure = plt.figure(figsize=(10.4, 4.2))
    grid = figure.add_gridspec(1, 4, width_ratios=(1.0, 1.0, 1.0, 0.85), wspace=0.32)

    for index, window in enumerate(problem.plan):
        axis = figure.add_subplot(grid[0, index])
        times = np.linspace(window.start_s, window.end_s, 400)
        rates = np.array([window.rate_bps(float(t)) for t in times]) / 1.0e6
        axis.plot(times - window.start_s, rates, color=EXACT, linewidth=1.8)
        used = committed.get(window.name, 0.0)
        boundary = window.time_for_bits(used) if used > 0.0 else 0.0
        inside = times - window.start_s <= boundary
        axis.fill_between(
            (times - window.start_s)[inside],
            0.0,
            rates[inside],
            color=SHADE,
            label="committed",
        )
        axis.axvline(boundary, color=ORACLE, linewidth=1.0, linestyle="--")
        axis.set_title(
            f"{window.name} ({window.station})\n"
            f"{capacity[window.name] / 1e9:.2f} Gbit, reliability {window.reliability:.2f}"
        )
        axis.set_xlabel("seconds into the pass")
        if index == 0:
            axis.set_ylabel("achievable rate [Mbit/s]")
            axis.legend(loc="lower center")
        axis.set_xlim(0.0, window.duration_s)
        axis.set_ylim(bottom=0.0)

    axis = figure.add_subplot(grid[0, 3])
    queue = problem.inventory.total_downlink_bits() / 1.0e9
    contact = problem.plan.total_capacity_bits() / 1.0e9
    delivered = selection.total_bits() / 1.0e9
    axis.bar(
        ["queue", "contact", "sent"],
        [queue, contact, delivered],
        color=[LEFT_BEHIND, EXACT, ORACLE],
    )
    for position, value in enumerate((queue, contact, delivered)):
        axis.text(position, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    axis.set_ylabel("Gbit")
    axis.set_title(
        f"queue against contact\n{len(selection.assignments)} of "
        f"{len(problem.inventory)} products sent"
    )
    axis.set_ylim(0.0, queue * 1.18)

    path = save(figure, "contact_timeline")
    return path


if __name__ == "__main__":
    print(main())
