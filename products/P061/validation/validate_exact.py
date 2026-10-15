"""The MILP optimum cross-checked against exhaustive enumeration.

Level 3 check 2. The MILP and the enumeration share nothing but the declared
problem: the MILP is a linear model solved by HiGHS, and the enumeration tries
every assignment of start step or not-placed and tests each one with the
resource *simulator* through the feasibility report. Agreement is therefore
evidence that the linear model and the simulator describe the same problem.

This check is what found the defect recorded in VALIDATION.md: an earlier draft
treated aggregate downlink volume as a hard constraint in the report but not in
the MILP, and the two disagreed by 1.5 priority units on a four-activity
instance.
"""

from __future__ import annotations

import time
from pathlib import Path

from _bootstrap import Tee, add_src_to_path, environment_lines

add_src_to_path()

from activityplan.exact import enumerate_optimum, solve_exact  # noqa: E402
from activityplan.feasibility import evaluate  # noqa: E402
from activityplan.instances import named_instance, sample_instance_spec  # noqa: E402
from activityplan.model import Problem  # noqa: E402
from activityplan.timeline import RasterisedProblem  # noqa: E402

OUTPUT = Path(__file__).with_name("validate_exact_output.txt")


def _with_drain(problem: Problem, *, require: bool) -> Problem:
    return Problem(
        activities=problem.activities,
        resources=problem.resources,
        horizon_s=problem.horizon_s,
        precedences=problem.precedences,
        exclusions=problem.exclusions,
        require_empty_recorder=require,
        name=problem.name + ("/drained" if require else ""),
    )


def _small_cases() -> list[tuple[str, RasterisedProblem]]:
    cases: list[tuple[str, RasterisedProblem]] = []
    tiny = named_instance("tiny-enumerable")
    cases.append(("tiny-enumerable", RasterisedProblem.build(tiny, 100.0)))
    cases.append(
        ("tiny-enumerable/drained", RasterisedProblem.build(_with_drain(tiny, require=True), 100.0))
    )
    cases.append(("tiny-enumerable/step200", RasterisedProblem.build(tiny, 200.0)))
    for seed in (101, 102, 103, 104, 105, 106):
        spec = sample_instance_spec(seed, n_activities=4, n_orbits=1, step_s=675.0)
        cases.append((spec.name, RasterisedProblem.build(spec.analytic_problem(), spec.step_s)))
    return cases


def main() -> int:
    say = Tee(OUTPUT)
    started = time.perf_counter()
    say("activityplan -- validate_exact.py")
    say("MILP optimum cross-checked against exhaustive enumeration on small instances")
    for line in environment_lines():
        say(line)

    say.rule("method")
    say("enumeration: every activity is assigned one admissible start step or not placed,")
    say("so the candidate count is prod_i (|starts_i| + 1). Each candidate is tested with")
    say("activityplan.feasibility.evaluate, which runs the resource SIMULATOR. The MILP")
    say("shares no code with that path beyond the rasterised arrays.")

    say.rule("results")
    say(
        f"{'instance':26s} {'T':>4s} {'A':>3s} {'candidates':>11s} {'enum':>7s} {'milp':>7s} "
        f"{'gap':>10s} {'nodes':>7s} {'enum s':>8s} {'milp s':>8s} {'agree':>6s}"
    )
    disagreements = 0
    checked = 0
    worst_difference = 0.0
    for label, rasterised in _small_cases():
        t0 = time.perf_counter()
        enum_plan, enum_value, candidates = enumerate_optimum(rasterised)
        enum_s = time.perf_counter() - t0
        result = solve_exact(rasterised, time_limit_s=120.0)
        difference = abs(result.value - enum_value)
        worst_difference = max(worst_difference, difference)
        agree = difference <= 1e-9
        disagreements += 0 if agree else 1
        checked += 1
        say(
            f"{label:26s} {rasterised.n_steps:4d} {rasterised.n_activities:3d} "
            f"{candidates:11d} {enum_value:7.2f} {result.value:7.2f} {result.gap:10.2e} "
            f"{result.node_count:7d} {enum_s:8.2f} {result.solve_s:8.3f} "
            f"{'yes' if agree else 'NO':>6s}"
        )
        if not agree:
            say(f"    enumeration plan : {dict(sorted(enum_plan.starts.items()))}")
            say(f"    milp plan        : {dict(sorted(result.plan.starts.items()))}")
        enum_report = evaluate(rasterised, enum_plan)
        milp_report = evaluate(rasterised, result.plan)
        if not enum_report.feasible or not milp_report.feasible:
            say("    FAIL: a returned plan is not feasible under the simulator")
            disagreements += 1

    say("")
    say(f"instances cross-checked          : {checked}")
    say(f"disagreements                    : {disagreements}")
    say(f"worst objective difference       : {worst_difference:.3e} priority units")
    say("criterion                        : difference <= 1e-9 on every instance")
    say(f"verdict                          : {'FAIL' if disagreements else 'PASS'}")

    say.rule("the defect this check found during the build")
    say("An earlier draft of activityplan.feasibility treated the aggregate downlink")
    say("volume -- total data produced against total declared contact capacity -- as a")
    say("hard constraint. The MILP never had that constraint. On the four-activity")
    say("instance below the MILP returned 6.5 priority units and the enumeration 5.0.")
    say("The MILP was right: data left on the recorder at the horizon is not an")
    say("infeasibility unless require_empty_recorder is declared. Both readings are")
    say("now reproduced, and the instance is pinned in tests/test_regressions.py.")
    say("")
    tiny = named_instance("tiny-enumerable")
    for require in (False, True):
        rasterised = RasterisedProblem.build(_with_drain(tiny, require=require), 100.0)
        _, enum_value, _ = enumerate_optimum(rasterised)
        result = solve_exact(rasterised, time_limit_s=120.0)
        say(
            f"  require_empty_recorder={str(require):5s} enumeration {enum_value:.2f} "
            f"milp {result.value:.2f}"
        )

    say.rule("summary")
    say(f"wall clock                       {time.perf_counter() - started:.1f} s")
    say(f"overall                          {'FAIL' if disagreements else 'PASS'}")
    say(f"raw output written to            {OUTPUT.name}")
    say.save()
    return 1 if disagreements else 0


if __name__ == "__main__":
    raise SystemExit(main())
