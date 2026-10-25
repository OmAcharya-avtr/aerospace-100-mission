"""Fairness metrics, their behaviour under priority inversion, and their price.

Four sections.

1. **Known answers.** Jain's index and the max-min share on allocations whose
   values can be worked out on paper, including the degenerate all-zero case
   where the index returns 1.0 by convention and the max-min share returns 0.0.
   The pairing is the point: the index alone would say an allocation of nothing
   is perfectly fair.

2. **One instance in full.** The complete accounting for a named instance under
   every method, so that a reader can see which request lost and by how much
   rather than only an aggregate.

3. **Behaviour under priority inversion.** The priority labels on an instance
   are reversed and everything is re-measured. A priority scheme that delivers
   what it promises should produce a different winner when the promise changes;
   a metric that cannot see the difference is not measuring the promise. Both
   the per-pair inversion count and the per-priority delivered satisfaction are
   reported before and after.

4. **The price of the fairness constraints.** Objective given up against
   equality bought, over the instances where the constrained program is feasible
   at all.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

from contactsched.assignment import Assignment  # noqa: E402
from contactsched.benchmark import fairness_cost, run_method  # noqa: E402
from contactsched.dataset import generate_instance  # noqa: E402
from contactsched.exact import solve_exact  # noqa: E402
from contactsched.fairness import analyse, jain_index, max_min_share  # noqa: E402
from contactsched.problem import ContactRequest, SchedulingProblem  # noqa: E402
from contactsched.report import format_fairness  # noqa: E402

FAILURES: list[str] = []
SEEDS = range(0, 160)
HORIZON_S = 43200.0


def reverse_priorities(problem: SchedulingProblem) -> SchedulingProblem:
    """A copy of ``problem`` with the priority ORDER reversed.

    The set of priority values is preserved and the mapping from request to
    priority is reversed, so the contention structure is untouched and only the
    declared promise changes. Candidate weights are rebuilt from the new
    priorities.
    """
    from contactsched.problem import build_problem

    priorities = sorted(r.priority for r in problem.requests)
    by_rank = sorted(problem.requests, key=lambda r: (r.priority, r.request_id))
    reversed_priorities = {
        request.request_id: priorities[len(priorities) - 1 - index]
        for index, request in enumerate(by_rank)
    }
    new_requests = tuple(
        ContactRequest(
            request_id=r.request_id,
            mission=r.mission,
            spacecraft=r.spacecraft,
            priority=reversed_priorities[r.request_id],
            duration_s=r.duration_s,
            min_contacts=r.min_contacts,
            max_contacts=r.max_contacts,
            stations=r.stations,
        )
        for r in problem.requests
    )
    return build_problem(
        problem.stations, new_requests, problem.windows, slot_step_s=problem.slot_step_s
    )


def main() -> None:
    out = Tee("validate_fairness_output.txt")
    out.rule("contactsched validate_fairness")
    for line in environment_lines():
        out(line)

    out.rule("1. Jain's index and the max-min share on hand-calculable allocations")
    out(f"{'allocation':<30} {'Jain (computed)':>16} {'Jain (by hand)':>15} {'max-min':>9}")
    out("-" * 74)
    cases = [
        ([1.0, 1.0, 1.0, 1.0], 1.0),
        ([0.5, 0.5], 1.0),
        ([1.0, 0.5], 0.9),
        ([1.0, 0.5, 0.5], 4.0 / 4.5),
        ([1.0, 0.0, 0.0, 0.0], 0.25),
        ([1.0, 0.0], 0.5),
        ([0.0, 0.0, 0.0], 1.0),
        ([5e-324], 1.0),
    ]
    for values, expected in cases:
        computed = jain_index(values)
        out(
            f"{str(values):<30} {computed:>16.12f} {expected:>15.12f} "
            f"{max_min_share(values):>9.6f}"
        )
        if abs(computed - expected) > 1e-12:
            FAILURES.append(f"Jain index of {values} is {computed}, expected {expected}")
    out("")
    out("The last two rows are the ones that matter for honesty. An all-zero")
    out("allocation scores 1.0 on Jain's index -- every request got the same, namely")
    out("nothing -- and 0.0 on the max-min share, which is the number that says what")
    out("happened. Both are always reported together for that reason. The single")
    out("denormal row is a pinned Hypothesis counterexample: the un-normalised index")
    out("expression raised ZeroDivisionError there, because the sum of squares")
    out("underflowed to zero while the sum did not.")

    out.rule("2. One instance in full: seed 7, twelve-hour horizon")
    instance = generate_instance(7, horizon_s=HORIZON_S)
    problem = instance.problem
    out(f"antennas {len(instance.stations)}, spacecraft {len(instance.fleet)}, "
        f"requests {len(problem.requests)}, windows {len(problem.windows)}, "
        f"candidates {len(problem.candidates)}")
    for method in ("greedy", "greedy-demand", "exact", "exact-fair"):
        result = run_method(problem, method)
        out("")
        out("-" * 78)
        out(f"method {method}: status {result.assignment.status}")
        out("-" * 78)
        if result.assignment.status == "infeasible":
            for note in result.assignment.notes:
                out(f"  {note}")
            continue
        out(format_fairness(result.report))

    out.rule("3. Behaviour under priority inversion")
    out("The priority labels on each instance are reversed: the request that was most")
    out("urgent becomes the least and vice versa, with the multiset of priority values")
    out("and the whole contention structure left untouched. Only the declared promise")
    out("changes.")
    out("")
    out("What a working metric must do: the set of requests that win should change,")
    out("and the per-priority delivered satisfaction should follow the new labels. The")
    out("per-pair inversion count need NOT fall -- it measures whether the delivered")
    out("order matches the declared order, and reversing the declaration can leave an")
    out("instance equally mismatched -- so the measurement below reports what happens")
    out("rather than asserting a direction.")
    out("")
    out(
        f"{'seed':>5} {'selection changed':>18} {'inv before':>11} {'inv after':>10} "
        f"{'Jain before':>12} {'Jain after':>11} {'p1 before':>10} {'p1 after':>9}"
    )
    out("-" * 92)
    changed = 0
    considered = 0
    inversions_before: list[int] = []
    inversions_after: list[int] = []
    monotone_before = 0
    monotone_after = 0
    for seed in list(SEEDS)[:24]:
        source = generate_instance(seed, horizon_s=HORIZON_S)
        if not source.problem.candidates:
            continue
        flipped = reverse_priorities(source.problem)
        before = analyse(source.problem, solve_exact(source.problem))
        after = analyse(flipped, solve_exact(flipped))
        considered += 1
        selection_changed = (
            solve_exact(source.problem).selected != solve_exact(flipped).selected
        )
        changed += int(selection_changed)
        inversions_before.append(before.inversion_count)
        inversions_after.append(after.inversion_count)
        monotone_before += int(before.priority_monotone)
        monotone_after += int(after.priority_monotone)
        p1_before = before.delivered_by_priority.get(1, float("nan"))
        p1_after = after.delivered_by_priority.get(1, float("nan"))
        out(
            f"{seed:>5d} {str(selection_changed):>18} {before.inversion_count:>11d} "
            f"{after.inversion_count:>10d} {before.jain:>12.6f} {after.jain:>11.6f} "
            f"{p1_before:>10.4f} {p1_after:>9.4f}"
        )
    out("")
    out(f"instances measured                        : {considered}")
    out(f"instances whose selection changed         : {changed}")
    out(
        f"mean inversions before / after            : "
        f"{np.mean(inversions_before):.4f} / {np.mean(inversions_after):.4f}"
    )
    out(
        f"instances where the aggregate promise held: "
        f"{monotone_before} before, {monotone_after} after"
    )
    if changed == 0:
        FAILURES.append(
            "reversing every priority changed no schedule, so the priority weights "
            "are not reaching the solver"
        )
    out("")
    out("A reversal that changed no schedule anywhere would mean the priority weights")
    out("never reach the solver, so this script fails if that happens. It is a")
    out("behavioural check on the whole chain and not a fairness claim.")

    out.rule("4. A constructed priority inversion, in full")
    out("An inversion that the declared weights cannot prevent, because it is not")
    out("about weights. R-HIGH is priority 1 and declares it needs three contacts;")
    out("its spacecraft has exactly one usable pass, so it can be served once.")
    out("R-LOW is priority 4 and declares it needs one; its pass is disjoint, so it")
    out("is served in full. Satisfaction is allocated over DECLARED need, so the")
    out("priority-1 request ends three times worse served than the priority-4 one,")
    out("and the inversion is real even though no slot was ever contested.")
    out("")
    from contactsched.problem import build_problem
    from contactsched.station import GroundStation, HorizonMask
    from contactsched.visibility import ContactWindow

    antenna = GroundStation("A1", 0.0, 0.0, 0.0, HorizonMask(floor_deg=5.0), 0.0)
    requests = (
        ContactRequest("R-HIGH", "M1", "SC-1", 1, 300.0, min_contacts=3, max_contacts=3),
        ContactRequest("R-LOW", "M2", "SC-2", 4, 300.0, min_contacts=1, max_contacts=1),
    )
    windows = (
        ContactWindow("A1", "SC-1", "M1", 0.0, 300.0, 60.0, 150.0, 10.0),
        ContactWindow("A1", "SC-2", "M2", 3000.0, 3300.0, 60.0, 3150.0, 10.0),
    )
    constructed = build_problem((antenna,), requests, windows, slot_step_s=300.0)
    assignment = solve_exact(constructed)
    report = analyse(constructed, assignment)
    out(f"optimum {assignment.objective:.4f}, contacts {assignment.contact_count}")
    out(format_fairness(report))
    out("")
    if report.inversion_count != 1:
        FAILURES.append(
            f"the constructed instance produced {report.inversion_count} inversions, "
            "expected exactly 1"
        )
    if abs(report.worst_inversion_gap - 2.0 / 3.0) > 1e-9:
        FAILURES.append(
            f"the constructed inversion gap is {report.worst_inversion_gap:.6f}, "
            "expected 0.666667"
        )
    out("The gap is 1.0 - 1/3 = 0.666667 exactly, and the aggregate promise fails:")
    out("delivered satisfaction RISES with the priority number. No scheduler could")
    out("have avoided it, which is the useful content of the measurement. An")
    out("operations team reading only the schedule would see both requests served and")
    out("would not see that the urgent one got a third of what it asked for.")
    out("")
    out("Contrast: a second instance where the two requests DO contest one antenna.")
    out("")
    contested_requests = (
        ContactRequest("R-HIGH", "M1", "SC-1", 1, 900.0, min_contacts=1, max_contacts=1),
        ContactRequest("R-LOW", "M2", "SC-2", 4, 300.0, min_contacts=3, max_contacts=3),
    )
    contested_windows = (
        ContactWindow("A1", "SC-1", "M1", 0.0, 900.0, 60.0, 450.0, 10.0),
        ContactWindow("A1", "SC-2", "M2", 0.0, 900.0, 60.0, 450.0, 10.0),
    )
    contested = build_problem((antenna,), contested_requests, contested_windows, 300.0)
    for mode in ("weighted_contacts", "weighted_seconds"):
        contested_assignment = solve_exact(contested, objective_mode=mode)
        contested_report = analyse(contested, contested_assignment)
        out(
            f"objective mode {mode}: optimum {contested_assignment.objective:.4f}, "
            f"contacts {contested_assignment.contact_count}, "
            f"inversions {contested_report.inversion_count}"
        )
    out("")
    out("Here the higher priority wins under both objective modes: three priority-4")
    out("contacts at weight 0.125 sum to 0.375 against a single priority-1 contact's")
    out("1.0, and in weighted seconds they are 0.125 * 900 = 112.5 against")
    out("1.0 * 900 = 900. The declared geometric weights do what they say when the")
    out("antenna is contested. The inversion in the first instance comes from the")
    out("geometry and from differing declared need, not from the weights, and that")
    out("distinction is the reason both the per-pair inversion count and the")
    out("per-priority delivered satisfaction are reported.")

    out.rule("5. The price of the declared fairness constraints")
    out("Measured only on instances where the constrained program is feasible, which")
    out("is a minority, and the count is reported so the selection bias is visible.")
    out("")
    out("The measured answer on this family is surprising and is published as measured:")
    out("the price is ZERO. Where the declared floors can be met at all, they are met")
    out("by an alternative optimal schedule rather than by giving up objective. The")
    out("reason is visible in the structure: the per-contact objective is indifferent")
    out("between two placements of the same request, so the solver has freedom to")
    out("satisfy a floor for free, and the instances that cost something are the ones")
    out("where the floors are infeasible and no schedule exists. A non-zero price needs")
    out("a specific structure, and one is constructed below rather than claimed.")
    out("")
    out(
        f"{'seed':>5} {'free obj':>10} {'fair obj':>10} {'loss':>8} {'loss frac':>10} "
        f"{'Jain gain':>10} {'max-min gain':>13} {'starved':>8}"
    )
    out("-" * 82)
    feasible = 0
    infeasible = 0
    losses: list[float] = []
    jain_gains: list[float] = []
    maxmin_gains: list[float] = []
    for seed in SEEDS:
        source = generate_instance(seed, horizon_s=HORIZON_S)
        if not source.problem.candidates:
            continue
        free = run_method(source.problem, "exact")
        fair = run_method(source.problem, "exact-fair")
        if fair.assignment.status == "infeasible":
            infeasible += 1
            continue
        feasible += 1
        price = fairness_cost(free, fair, "weighted_contacts")
        losses.append(price["objective_loss"])
        jain_gains.append(price["jain_gain"])
        maxmin_gains.append(price["max_min_gain"])
        out(
            f"{seed:>5d} {price['objective_unconstrained']:>10.4f} "
            f"{price['objective_constrained']:>10.4f} {price['objective_loss']:>8.4f} "
            f"{price['objective_loss_fraction']:>10.4f} {price['jain_gain']:>10.6f} "
            f"{price['max_min_gain']:>13.6f} "
            f"{int(price['starved_before'])}->{int(price['starved_after']):<4d}"
        )
    out("")
    out(f"instances feasible under the floors   : {feasible}")
    out(f"instances infeasible under the floors : {infeasible}")
    if losses:
        out(f"mean objective loss                   : {np.mean(losses):.6f}")
        out(f"worst objective loss                  : {np.max(losses):.6f}")
        out(f"mean Jain gain                        : {np.mean(jain_gains):.6f}")
        out(f"mean max-min gain                     : {np.mean(maxmin_gains):.6f}")
        if np.min(losses) < -1e-9:
            FAILURES.append("a fairness constraint INCREASED the objective, which is impossible")
    else:
        out("No instance in this family admits a schedule meeting every declared floor.")

    out.rule("5b. A constructed instance where the fairness floor DOES cost objective")
    out("Three 100 s slots on one antenna. SC-1 sees all three, SC-2 only the first")
    out("two. R1 is priority 1, needs one contact and is allowed three; R2 is")
    out("priority 2 and declares it needs two. The free optimum gives R1 all three")
    out("slots and starves R2 entirely; the constrained optimum gives R2 the two slots")
    out("it needs and leaves R1 one.")
    out("")
    priced_requests = (
        ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=1, max_contacts=3),
        ContactRequest("R2", "M2", "SC-2", 2, 100.0, min_contacts=2, max_contacts=2),
    )
    priced_windows = (
        ContactWindow("A1", "SC-1", "M1", 0.0, 100.0, 60.0, 50.0, 10.0),
        ContactWindow("A1", "SC-1", "M1", 400.0, 500.0, 60.0, 450.0, 10.0),
        ContactWindow("A1", "SC-1", "M1", 800.0, 900.0, 60.0, 850.0, 10.0),
        ContactWindow("A1", "SC-2", "M2", 0.0, 100.0, 60.0, 50.0, 10.0),
        ContactWindow("A1", "SC-2", "M2", 400.0, 500.0, 60.0, 450.0, 10.0),
    )
    priced = build_problem((antenna,), priced_requests, priced_windows, 100.0)
    free_priced = run_method(priced, "exact")
    fair_priced = run_method(priced, "exact-fair")
    price = fairness_cost(free_priced, fair_priced, "weighted_contacts")
    for key in (
        "objective_unconstrained",
        "objective_constrained",
        "objective_loss",
        "objective_loss_fraction",
        "jain_gain",
        "max_min_gain",
        "starved_before",
        "starved_after",
        "seconds_short_before",
        "seconds_short_after",
    ):
        out(f"  {key:<26} {price[key]:.6f}")
    out("")
    if abs(price["objective_loss"] - 1.0) > 1e-9:
        FAILURES.append(
            f"the constructed fairness price is {price['objective_loss']:.6f}, expected 1.0"
        )
    out("The constraint costs exactly one unit of objective, one third of the free")
    out("optimum, and buys a max-min share that rises from 0 to 1. That trade is the")
    out("number a planner needs and the one this product exists to print.")

    out.rule("6. The degenerate convention, on a real problem")
    empty = Assignment("nothing-scheduled", "heuristic", (), 0.0)
    degenerate = analyse(problem, empty)
    out(f"Jain index of the empty schedule    : {degenerate.jain:.6f}")
    out(f"max-min share of the empty schedule : {degenerate.max_min:.6f}")
    out(f"all_starved flag                    : {degenerate.all_starved}")
    out(f"total seconds short                 : {degenerate.total_seconds_short:.1f}")
    if degenerate.jain != 1.0 or degenerate.max_min != 0.0:
        FAILURES.append("the degenerate all-zero convention is not what is documented")
    out("")
    out("A reader who took the Jain index alone would call this schedule perfectly")
    out("fair. That is why contactsched.report.format_fairness prints the sentence")
    out("naming the convention whenever the flag is set.")

    out.rule("Result")
    if FAILURES:
        out(f"FAILED {len(FAILURES)} check(s):")
        for failure in FAILURES[:20]:
            out(f"  - {failure}")
    else:
        out("All checks in this script passed.")
    out("")
    out(f"written to {out.save()}")
    if FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
