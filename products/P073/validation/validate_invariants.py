"""Invariants over an instance family, and negative controls that must fire.

Three classes of check, kept separate because they catch different things.

**Invariants.** No antenna serves two contacts at once once its turnaround is
counted, no spacecraft talks to two antennas at once, no request exceeds its
ceiling, every contact lies inside a visibility window. Checked for every method
on every instance in the family.

**Independently derived bounds.** Allocated antenna time against the union of
that antenna's visibility, which is computed by a function the schedulers never
call; allocated contacts against the standalone interval-scheduling capacity of
each request; the objective against a ceiling computed from the requests alone
with no knowledge of conflicts. A bound derived elsewhere catches a mistake that
two implementations of the same expression would share.

**Negative controls.** Corruptions the checks must catch. Every control asserts
that it actually fired, because a control that does not fire proves nothing, and
every control is anchored on the *true* value of the quantity it corrupts. The
boundary is tested from both sides: a separation exactly equal to the turnaround
is legal and must NOT fire, one second inside it must. An offset that merely
reaches a bound does not cross it.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

from contactsched.benchmark import run_method  # noqa: E402
from contactsched.dataset import generate_instance  # noqa: E402
from contactsched.exact import FairnessPolicy, solve_exact, standalone_capacity  # noqa: E402
from contactsched.fairness import analyse  # noqa: E402
from contactsched.greedy import schedule_greedy  # noqa: E402
from contactsched.problem import (  # noqa: E402
    Candidate,
    ContactRequest,
    build_problem,
    check_feasible,
    shortfalls,
)
from contactsched.station import GroundStation, HorizonMask  # noqa: E402
from contactsched.visibility import ContactWindow, supply_seconds_by_station  # noqa: E402

FAILURES: list[str] = []
SEEDS = range(0, 120)
HORIZON_S = 43200.0
METHODS = ("greedy", "greedy-demand", "exact")


def main() -> None:
    out = Tee("validate_invariants_output.txt")
    out.rule("contactsched validate_invariants")
    for line in environment_lines():
        out(line)

    instances = [generate_instance(seed, horizon_s=HORIZON_S) for seed in SEEDS]
    usable = [i for i in instances if i.problem.candidates]

    out.rule("Family")
    out(f"seeds    : {SEEDS.start}..{SEEDS.stop - 1}")
    out(f"usable   : {len(usable)} of {len(instances)}")
    out(f"horizon  : {HORIZON_S:.0f} s")

    out.rule("1. Feasibility invariants, every method, every instance")
    out(f"{'method':<26} {'instances':>10} {'violations':>11} {'contacts':>10}")
    out("-" * 62)
    for method in METHODS:
        violations = 0
        contacts = 0
        for instance in usable:
            result = run_method(instance.problem, method)
            violations += len(result.violations)
            contacts += result.assignment.contact_count
            if result.violations:
                FAILURES.append(
                    f"{method} violated feasibility on seed {instance.seed}: "
                    f"{result.violations[0]}"
                )
        out(f"{method:<26} {len(usable):>10d} {violations:>11d} {contacts:>10d}")

    out.rule("2. Minimum-contact invariant under the fairness floors")
    feasible = 0
    short_requests = 0
    for instance in usable:
        fair = solve_exact(
            instance.problem, policy=FairnessPolicy(enforce_min_contacts=True)
        )
        if fair.status == "infeasible":
            continue
        feasible += 1
        for request_id, (_granted, contacts_short, _s) in shortfalls(
            instance.problem, fair.selected
        ).items():
            if contacts_short:
                short_requests += 1
                FAILURES.append(
                    f"seed {instance.seed}: {request_id} is short by {contacts_short} "
                    "contacts despite the floors being enforced"
                )
    out(f"instances with a feasible constrained schedule : {feasible}")
    out(f"requests still short in those schedules        : {short_requests}")
    out("")
    out("Zero is the only acceptable value in the second row. A feasible solve that")
    out("left a floor unmet would mean the floor constraints are not reaching HiGHS.")

    out.rule("3. Independently derived bounds")
    out(
        f"{'bound':<54} {'instances':>10} {'breaches':>9}"
    )
    out("-" * 76)

    breaches = 0
    for instance in usable:
        supply = supply_seconds_by_station(list(instance.problem.windows))
        assignment = solve_exact(instance.problem)
        for name, seconds in assignment.seconds_by_station(instance.problem).items():
            if seconds > supply.get(name, 0.0) + 1e-6:
                breaches += 1
                FAILURES.append(
                    f"seed {instance.seed}: antenna {name} allocated {seconds:.3f} s "
                    f"against a union supply of {supply.get(name, 0.0):.3f} s"
                )
    out(f"{'allocated antenna time <= union visibility':<54} {len(usable):>10d} {breaches:>9d}")

    breaches = 0
    for instance in usable:
        assignment = solve_exact(instance.problem)
        counts = assignment.contacts_by_request(instance.problem)
        for request in instance.problem.requests:
            capacity = standalone_capacity(instance.problem, request.request_id)
            if counts[request.request_id] > capacity:
                breaches += 1
                FAILURES.append(
                    f"seed {instance.seed}: {request.request_id} got "
                    f"{counts[request.request_id]} contacts against a standalone "
                    f"capacity of {capacity}"
                )
    out(f"{'contacts per request <= standalone capacity':<54} {len(usable):>10d} {breaches:>9d}")

    breaches = 0
    for instance in usable:
        ceiling = 0.0
        for request in instance.problem.requests:
            candidates = instance.problem.candidates_of(request.request_id)
            if candidates:
                ceiling += candidates[0].weight * request.max_contacts
        if solve_exact(instance.problem).objective > ceiling + 1e-9:
            breaches += 1
            FAILURES.append(f"seed {instance.seed}: objective exceeds the weight ceiling")
    out(f"{'objective <= per-request weight ceiling':<54} {len(usable):>10d} {breaches:>9d}")

    breaches = 0
    for instance in usable:
        report = analyse(instance.problem, schedule_greedy(instance.problem))
        demand = sum(r.requested_seconds for r in instance.problem.requests)
        with_demand = [o for o in report.outcomes if o.has_demand]
        lower = 1.0 / len(with_demand) if with_demand else 0.0
        if report.total_seconds_short > demand + 1e-9:
            breaches += 1
            FAILURES.append(f"seed {instance.seed}: shortfall exceeds declared demand")
        if with_demand and not lower - 1e-12 <= report.jain <= 1.0 + 1e-12:
            breaches += 1
            FAILURES.append(f"seed {instance.seed}: Jain index outside [1/n, 1]")
        if not 0.0 <= report.max_min <= 1.0 + 1e-12:
            breaches += 1
            FAILURES.append(f"seed {instance.seed}: max-min share outside [0, 1]")
    out(f"{'fairness metrics inside their declared ranges':<54} {len(usable):>10d} {breaches:>9d}")

    out.rule("4. Negative controls")
    out("One antenna with a 200 s turnaround and a single 2000 s window, so the 200 s")
    out("candidate grid gives ten placements of a 200 s contact and every corruption")
    out("below stays inside the window. That matters: a corruption that also moved a")
    out("contact out of its window would be caught by the containment check instead of")
    out("by the one under test.")
    out("")
    antenna = GroundStation("A1", 0.0, 0.0, 0.0, HorizonMask(floor_deg=5.0), 200.0)
    requests = (
        ContactRequest("R1", "M1", "SC-1", 1, 200.0, min_contacts=2, max_contacts=2),
    )
    windows = (ContactWindow("A1", "SC-1", "M1", 0.0, 2000.0, 60.0, 1000.0, 10.0),)

    def fresh():
        return build_problem((antenna,), requests, windows, slot_step_s=200.0)

    def at(problem, start_s: float) -> Candidate:
        return next(c for c in problem.candidates if abs(c.start_s - start_s) < 1e-9)

    def relocate(problem, template: Candidate, start_s: float) -> Candidate:
        moved = Candidate(
            index=template.index,
            request_id=template.request_id,
            mission=template.mission,
            spacecraft=template.spacecraft,
            station=template.station,
            start_s=start_s,
            end_s=start_s + template.duration_s,
            weight=template.weight,
            max_elevation_deg=template.max_elevation_deg,
        )
        corrupted = list(problem.candidates)
        corrupted[template.index] = moved
        problem.candidates = tuple(corrupted)
        return moved

    out(f"{'control':<54} {'fired':>7} {'expected':>9}")
    out("-" * 74)
    controls: list[tuple[str, bool, bool, str]] = []

    problem = fresh()
    first, second = at(problem, 0.0), at(problem, 1000.0)
    found = check_feasible(problem, (first.index, second.index))
    controls.append(("positive control: 800 s apart, 200 s turnaround", bool(found), False,
                     found[0] if found else ""))

    problem = fresh()
    first, second = at(problem, 0.0), at(problem, 400.0)
    found = check_feasible(problem, (first.index, second.index))
    controls.append(("separation EXACTLY one turnaround (200 s)", bool(found), False,
                     found[0] if found else ""))

    problem = fresh()
    first = at(problem, 0.0)
    moved = relocate(problem, at(problem, 400.0), 399.0)
    found = check_feasible(problem, (first.index, moved.index))
    controls.append(("separation one second inside the turnaround", bool(found), True,
                     found[0] if found else ""))

    problem = fresh()
    first = at(problem, 0.0)
    moved = relocate(problem, at(problem, 1000.0), 201.0)
    found = check_feasible(problem, (first.index, moved.index))
    controls.append(("overlap: second starts 1 s after the first ends", bool(found), True,
                     found[0] if found else ""))

    problem = fresh()
    moved = relocate(problem, at(problem, 1000.0), 5000.0)
    found = check_feasible(problem, (moved.index,))
    controls.append(("contact relocated outside every window", bool(found), True,
                     found[0] if found else ""))

    problem = fresh()
    first = at(problem, 0.0)
    found = check_feasible(problem, (first.index, first.index))
    controls.append(("the same candidate selected twice", bool(found), True,
                     found[0] if found else ""))

    problem = fresh()
    # Three MUTUALLY COMPATIBLE placements, 400 s apart against a 200 s
    # turnaround, so the only violation available is the ceiling. Picking the
    # first three grid candidates instead would have fired the exclusion check
    # and the control would not have tested what its label claims.
    indices = tuple(at(problem, start).index for start in (0.0, 400.0, 800.0))
    found = check_feasible(problem, indices)
    controls.append(("three compatible contacts against a ceiling of two", bool(found), True,
                     found[0] if found else ""))

    clash_antennas = (
        GroundStation("A1", 0.0, 0.0, 0.0, HorizonMask(floor_deg=5.0), 0.0),
        GroundStation("A2", 0.0, 0.0, 0.0, HorizonMask(floor_deg=5.0), 0.0),
    )
    clash_requests = (
        ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 1),
        ContactRequest("R2", "M1", "SC-1", 2, 100.0, 1, 1),
    )
    clash_windows = (
        ContactWindow("A1", "SC-1", "M1", 0.0, 100.0, 60.0, 50.0, 10.0),
        ContactWindow("A2", "SC-1", "M1", 0.0, 100.0, 60.0, 50.0, 10.0),
    )
    clash = build_problem(clash_antennas, clash_requests, clash_windows, 100.0)
    pair = tuple(
        c.index
        for c in clash.candidates
        if (c.request_id, c.station) in {("R1", "A1"), ("R2", "A2")}
    )
    found = check_feasible(clash, pair)
    controls.append(("one spacecraft on two antennas at once", bool(found), True,
                     found[0] if found else ""))

    for label, fired, expected, detail in controls:
        out(f"{label:<54} {str(fired):>7} {str(expected):>9}")
        if fired != expected:
            FAILURES.append(
                f"negative control '{label}' fired={fired}, expected {expected}"
            )
        if fired and detail:
            out(f"    -> {detail[:100]}")
    out("")
    out("The two rows expecting False are the ones that make the rest meaningful. A")
    out("checker that fires on everything is not a checker, and a control built on an")
    out("offset that merely reaches the turnaround would be one of those rows, not a")
    out("violation.")

    out.rule("5. Starvation is measured, never silently repaired")
    starved_requests = ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=2, max_contacts=2)
    starved_window = ContactWindow("A1", "SC-1", "M1", 0.0, 100.0, 60.0, 50.0, 10.0)
    starved_problem = build_problem((antenna,), (starved_requests,), (starved_window,), 100.0)
    starved_assignment = solve_exact(starved_problem)
    starved_violations = check_feasible(starved_problem, starved_assignment.selected)
    out(f"feasibility violations : {len(starved_violations)}")
    granted, contacts_short, seconds_short = shortfalls(
        starved_problem, starved_assignment.selected
    )["R1"]
    out(f"contacts granted       : {granted}")
    out(f"contacts short         : {contacts_short}")
    out(f"seconds short          : {seconds_short:.1f}")
    if contacts_short != 1:
        FAILURES.append("the starvation measurement did not report the expected shortfall")
    out("")
    out("A schedule that misses a declared minimum is FEASIBLE and SHORT. Treating the")
    out("miss as infeasibility would throw away the number this product exists to")
    out("report, which is why check_feasible and shortfalls are separate functions.")

    out.rule("6. Determinism")
    out("Every method re-run three times on the same instance must return the identical")
    out("selection, not merely the identical objective.")
    out("")
    out(f"{'method':<26} {'instances':>10} {'selection mismatches':>21}")
    out("-" * 62)
    for method in METHODS:
        mismatches = 0
        for instance in usable[:40]:
            first = run_method(instance.problem, method).assignment.selected
            for _ in range(2):
                if run_method(instance.problem, method).assignment.selected != first:
                    mismatches += 1
        out(f"{method:<26} {40:>10d} {mismatches:>21d}")
        if mismatches:
            FAILURES.append(f"{method} is not deterministic: {mismatches} mismatches")

    out.rule("Result")
    if FAILURES:
        out(f"FAILED {len(FAILURES)} check(s):")
        for failure in FAILURES[:20]:
            out(f"  - {failure}")
        if len(FAILURES) > 20:
            out(f"  ... and {len(FAILURES) - 20} more")
    else:
        out("All checks in this script passed.")
    out("")
    out(f"written to {out.save()}")
    if FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
