"""The worked example quoted in README.md, run, with its real output captured.

The code between the two markers below is the code quoted in README.md, and the
text this script writes is the output quoted beneath it. Both are replaced from a
run of this script as the last step before hand-off, so they cannot drift from
the package.

The scenario is declared, not looked up: two antennas at one high-latitude site,
one at the equator, four spacecraft across three missions, a twelve-hour horizon,
and five contact requests whose priorities and minimum contact counts are written
out below. Nothing here is a real facility, a real spacecraft or a real request.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

# --- README EXAMPLE BEGINS ---
from contactsched import (  # noqa: E402
    ContactRequest,
    FairnessPolicy,
    Scenario,
    all_windows,
    analyse,
    build_problem,
    default_fleet,
    default_network,
    format_fairness,
    format_schedule,
    schedule_greedy,
    solve_exact,
)

scenario = Scenario.from_calendar(2026, 4, 30, duration_s=43200.0)
stations = default_network()[:3]
fleet = default_fleet()[:4]
windows = tuple(all_windows(scenario, stations, fleet, step_s=30.0, tolerance_s=0.25))

requests = (
    ContactRequest("POL-A", "POLARIS", "POLARIS-A", 1, 420.0, min_contacts=3, max_contacts=4),
    ContactRequest("POL-B", "POLARIS", "POLARIS-B", 2, 420.0, min_contacts=3, max_contacts=4),
    ContactRequest("POL-C", "POLARIS", "POLARIS-C", 3, 300.0, min_contacts=2, max_contacts=3),
    ContactRequest("TER-1", "TERRASCAN", "TERRASCAN-1", 1, 600.0, min_contacts=3, max_contacts=3),
    ContactRequest("TER-2", "TERRASCAN", "TERRASCAN-1", 4, 180.0, min_contacts=2, max_contacts=2),
)

problem = build_problem(stations, requests, windows, slot_step_s=120.0)
greedy = schedule_greedy(problem)
exact = solve_exact(problem)
fair = solve_exact(problem, policy=FairnessPolicy(enforce_min_contacts=True))
# --- README EXAMPLE ENDS ---


def main() -> None:
    out = Tee("worked_example_output.txt")
    out.rule("contactsched worked_example")
    for line in environment_lines():
        out(line)

    out.rule("The declared scenario")
    out("epoch               : 2026-04-30T00:00:00")
    out(f"horizon             : {scenario.duration_s:.0f} s")
    out(f"antennas            : {', '.join(s.name for s in stations)}")
    out(f"spacecraft          : {', '.join(s.name for s in fleet)}")
    out(f"visibility windows  : {len(windows)}")
    out(f"total visible time  : {sum(w.duration_s for w in windows):.1f} s")
    out(f"requests            : {len(requests)}")
    out(f"declared demand     : {sum(r.requested_seconds for r in requests):.1f} s")
    out(f"candidate placements: {len(problem.candidates)}")

    out.rule("Baseline 1: priority-ordered first-fit")
    out(format_schedule(problem, greedy))

    out.rule("Baseline 2: the exact optimum on HiGHS")
    out(format_schedule(problem, exact))

    out.rule("What the greedy baseline cost, in objective")
    gap = (exact.objective - greedy.objective) / exact.objective
    out(f"greedy objective    : {greedy.objective:.6f}")
    out(f"exact objective     : {exact.objective:.6f}")
    out(f"optimality gap      : {gap:.6f}")
    out(f"contacts, greedy    : {greedy.contact_count}")
    out(f"contacts, exact     : {exact.contact_count}")

    out.rule("What it cost in contention and fairness: the greedy schedule")
    out(format_fairness(analyse(problem, greedy)))

    out.rule("What it cost in contention and fairness: the exact schedule")
    out(format_fairness(analyse(problem, exact)))

    out.rule("Asking for the declared minimum contact counts to be guaranteed")
    out(f"status: {fair.status}")
    for note in fair.notes:
        out(f"  {note}")
    if fair.status != "infeasible":
        out("")
        out(format_fairness(analyse(problem, fair)))
        out("")
        out(f"objective given up : {exact.objective - fair.objective:.6f}")

    out.rule("Renegotiating the binding floor")
    out("The infeasibility diagnostic names a request and the number of mutually")
    out("compatible candidates it actually has. Lowering that one floor to the named")
    out("capacity and re-solving is the renegotiation loop this product is for.")
    out("")
    from contactsched import standalone_capacity

    overrides = {
        request.request_id: min(
            request.min_contacts, standalone_capacity(problem, request.request_id)
        )
        for request in requests
    }
    for request in requests:
        capacity = standalone_capacity(problem, request.request_id)
        out(
            f"  {request.request_id}: declared {request.min_contacts}, standalone "
            f"capacity {capacity}, renegotiated floor {overrides[request.request_id]}"
        )
    renegotiated = solve_exact(
        problem,
        policy=FairnessPolicy(
            enforce_min_contacts=True, min_contacts_floor_override=overrides
        ),
    )
    out("")
    out(f"status after renegotiation : {renegotiated.status}")
    if renegotiated.status != "infeasible":
        out(f"objective                  : {renegotiated.objective:.6f}")
        out(f"objective given up against the free optimum : "
            f"{exact.objective - renegotiated.objective:.6f}")
        out("")
        out(format_fairness(analyse(problem, renegotiated)))
    else:
        out("Still infeasible: the floors are individually reachable but not jointly.")
        for note in renegotiated.notes:
            out(f"  {note}")

    out.rule("Result")
    out(f"written to {out.save()}")


if __name__ == "__main__":
    main()
