"""Greedy against the exact optimum on one instance, as a timeline.

What to notice: the two schedules differ in which requests get the contested
slots, not in how busy the antennas are. The greedy baseline leaves the same
antennas nearly as full and still delivers a lower objective and a different set
of losers, which is why an occupancy chart is not a fairness measurement.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from _common import EXACT, GREEDY, STARVED, bootstrap, hours, save, style

bootstrap()

from contactsched.benchmark import optimality_gap  # noqa: E402
from contactsched.dataset import generate_instance  # noqa: E402
from contactsched.exact import solve_exact  # noqa: E402
from contactsched.fairness import analyse  # noqa: E402
from contactsched.greedy import schedule_greedy  # noqa: E402

SEED = 7
HORIZON_S = 43200.0


def draw(axis, problem, assignment, title, colour) -> None:
    """One timeline panel: antennas on the y axis, contacts as bars."""
    stations = [s.name for s in problem.stations]
    report = analyse(problem, assignment)
    starved = {o.request_id for o in report.starved}
    for window in problem.windows:
        axis.barh(
            stations.index(window.station),
            hours(window.duration_s),
            left=hours(window.rise_s),
            height=0.62,
            color="#eef3f8",
            zorder=1,
        )
    for start, end, station, _spacecraft, request_id in assignment.timeline(problem):
        axis.barh(
            stations.index(station),
            hours(end - start),
            left=hours(start),
            height=0.42,
            color=STARVED if request_id in starved else colour,
            zorder=3,
        )
    axis.set_yticks(range(len(stations)))
    axis.set_yticklabels(stations, fontsize=8)
    axis.set_xlim(0.0, hours(HORIZON_S))
    axis.set_title(
        f"{title}: objective {assignment.objective:.4f}, "
        f"{assignment.contact_count} contacts, {len(report.starved)} starved, "
        f"Jain {report.jain:.4f}, max-min {report.max_min:.4f}"
    )


def main() -> None:
    style()
    instance = generate_instance(SEED, horizon_s=HORIZON_S)
    problem = instance.problem
    greedy = schedule_greedy(problem)
    exact = solve_exact(problem)

    figure, axes = plt.subplots(2, 1, figsize=(11.0, 6.0), sharex=True)
    draw(axes[0], problem, greedy, "priority-ordered first-fit", GREEDY)
    draw(axes[1], problem, exact, "exact optimum on HiGHS", EXACT)
    axes[1].set_xlabel("hours after the scenario epoch")
    handles = [
        plt.Line2D([0], [0], color="#eef3f8", lw=8),
        plt.Line2D([0], [0], color=GREEDY, lw=8),
        plt.Line2D([0], [0], color=EXACT, lw=8),
        plt.Line2D([0], [0], color=STARVED, lw=8),
    ]
    axes[1].legend(
        handles,
        ["visibility window", "greedy contact", "exact contact", "contact of a starved request"],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        fontsize=8,
    )
    figure.suptitle(
        f"Instance seed {SEED}, {len(problem.requests)} requests, "
        f"{len(problem.candidates)} candidate placements",
        y=0.99,
    )
    figure.tight_layout()
    path = save(figure, "schedule_comparison")

    print(f"instance seed      : {SEED}")
    print(f"greedy objective   : {greedy.objective:.6f} ({greedy.contact_count} contacts)")
    print(f"exact objective    : {exact.objective:.6f} ({exact.contact_count} contacts)")
    print(f"optimality gap     : {optimality_gap(greedy, exact):.6f}")
    for label, assignment in (("greedy", greedy), ("exact", exact)):
        report = analyse(problem, assignment)
        print(
            f"{label:<18} starved {len(report.starved)}, short "
            f"{report.total_seconds_short:.0f} s, Jain {report.jain:.6f}, "
            f"max-min {report.max_min:.6f}, inversions {report.inversion_count}"
        )
    print(f"figure written     : {path}")


if __name__ == "__main__":
    main()
