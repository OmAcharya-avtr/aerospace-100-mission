"""Per-request satisfaction and the fairness indices, by method.

What to notice: the exact optimum raises the objective and the Jain index and
leaves the same request on zero. A better schedule is not a fairer one, and the
max-min share is the number that says so.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import EXACT, FAIR, GREEDY, STARVED, bootstrap, save, style

bootstrap()

from contactsched.benchmark import run_method  # noqa: E402
from contactsched.dataset import generate_instance  # noqa: E402
from contactsched.fairness import analyse  # noqa: E402
from contactsched.problem import ContactRequest, build_problem  # noqa: E402
from contactsched.visibility import all_windows  # noqa: E402

SEED = 7
HORIZON_S = 43200.0


def declared_instance():
    """Seed 7's network and fleet with requests whose floors are reachable.

    The generator's own floors are mostly unreachable, so the fairness-
    constrained method is infeasible on them and there would be nothing to
    compare. Here each floor is set to the request's own standalone capacity,
    capped at three, which is the renegotiated form the worked example arrives
    at. The unmodified instance is in schedule_comparison.py.
    """
    from contactsched.exact import standalone_capacity

    instance = generate_instance(SEED, horizon_s=HORIZON_S)
    problem = instance.problem
    requests = []
    for request in problem.requests:
        capacity = standalone_capacity(problem, request.request_id)
        floor = max(1, min(request.min_contacts, capacity, 3))
        requests.append(
            ContactRequest(
                request_id=request.request_id,
                mission=request.mission,
                spacecraft=request.spacecraft,
                priority=request.priority,
                duration_s=request.duration_s,
                min_contacts=floor,
                max_contacts=max(floor, request.max_contacts),
                stations=request.stations,
            )
        )
    windows = tuple(
        all_windows(instance.scenario, instance.stations, instance.fleet, 45.0, 0.5)
    )
    return build_problem(instance.stations, tuple(requests), windows, slot_step_s=240.0)


def main() -> None:
    style()
    problem = declared_instance()
    methods = (
        ("greedy", GREEDY),
        ("exact", EXACT),
        ("exact-fair", FAIR),
    )
    results = {name: run_method(problem, name) for name, _c in methods}

    ids = [r.request_id for r in problem.requests]
    priorities = [r.priority for r in problem.requests]
    figure, (bars, indices) = plt.subplots(
        1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [3, 2]}
    )

    width = 0.26
    positions = np.arange(len(ids))
    for offset, (name, colour) in enumerate(methods):
        result = results[name]
        if result.assignment.status == "infeasible":
            continue
        report = analyse(problem, result.assignment)
        lookup = {o.request_id: o.satisfaction for o in report.outcomes}
        values = [lookup[request_id] for request_id in ids]
        bars.bar(
            positions + (offset - 1) * width,
            values,
            width=width,
            color=colour,
            label=name,
        )
        for index, value in enumerate(values):
            if value <= 1e-12:
                bars.plot(
                    positions[index] + (offset - 1) * width,
                    0.02,
                    marker="v",
                    color=STARVED,
                    markersize=5,
                )
    bars.set_xticks(positions)
    bars.set_xticklabels([f"{i}\np{p}" for i, p in zip(ids, priorities, strict=True)], fontsize=7)
    bars.set_ylabel("satisfaction (allocated / declared need, capped at 1)")
    bars.set_ylim(0.0, 1.12)
    bars.axhline(1.0, color="#9a9a9a", lw=0.8, ls="--")
    bars.set_title("Per-request satisfaction; a red marker is a request on zero")
    bars.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    labels = ["Jain (request)", "Jain (mission)", "max-min share"]
    offsets = np.arange(len(labels))
    for offset, (name, colour) in enumerate(methods):
        result = results[name]
        if result.assignment.status == "infeasible":
            continue
        report = analyse(problem, result.assignment)
        values = [report.jain, report.jain_by_mission, report.max_min]
        indices.bar(offsets + (offset - 1) * width, values, width=width, color=colour)
        for index, value in enumerate(values):
            indices.text(
                offsets[index] + (offset - 1) * width,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                fontsize=6.5,
            )
    indices.set_xticks(offsets)
    indices.set_xticklabels(labels, fontsize=8)
    indices.set_ylim(0.0, 1.15)
    indices.set_title("Fairness indices; 1.0 is perfectly equal, not perfectly good")

    figure.tight_layout()
    path = save(figure, "fairness_metrics")

    for name, _colour in methods:
        result = results[name]
        if result.assignment.status == "infeasible":
            print(f"{name:<12} infeasible: {result.assignment.notes[-1][:80]}")
            continue
        report = analyse(problem, result.assignment)
        print(
            f"{name:<12} objective {result.assignment.objective:8.4f}  "
            f"starved {len(report.starved)}  short {report.total_seconds_short:7.0f} s  "
            f"Jain {report.jain:.6f}  max-min {report.max_min:.6f}  "
            f"inversions {report.inversion_count}"
        )
    print(f"figure written: {path}")


if __name__ == "__main__":
    main()
