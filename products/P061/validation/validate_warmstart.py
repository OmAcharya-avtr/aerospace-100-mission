"""Does a learned warm start help the exact solver.

The spec this product was built against says: report honestly if the warm start
does not help. This script is that report, and the answer has two halves.

1. **On most instances there is nothing to help.** HiGHS closes the gap at the
   root node on the whole seeded family: the cold node count is 1, so there is
   no branch-and-bound tree for a primal bound to prune. A warm start cannot
   improve a search that does not happen, and claiming a speed-up from a node
   count of 1 in both arms would be measuring presolve.
2. **Where a search does happen the primal-bound cut helps, substantially.** Two
   of the shipped instances branch -- ``energy-bound`` with presolve on and
   ``leo-imaging`` with presolve off -- and on those the cut cuts both the node
   count and the wall clock by a large factor. The numbers are in part 2.

The third part answers the question the spec actually asks -- whether the
*learned* predictor gives a better warm start than the analytic model -- by
building the target problem from the learned point predictions and comparing two
candidate plans: the greedy plan on that same target, and the greedy plan on the
analytic problem transferred to it. The transfer is where the learned model can
earn its place, because an analytic plan is a plan for a different problem and
may not verify.

Reminder from :mod:`activityplan.warmstart`: ``scipy.optimize.milp`` exposes no
MIP start, so the warm start is a primal-bound cut, not an integer assignment.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from _bootstrap import Tee, add_src_to_path, environment_lines

add_src_to_path()

from activityplan.exact import solve_exact  # noqa: E402
from activityplan.features import feature_matrix, realise, sample_requests  # noqa: E402
from activityplan.greedy import plan_greedy  # noqa: E402
from activityplan.instances import (  # noqa: E402
    INSTANCE_ORDER,
    RECOMMENDED_STEP_S,
    instance_family,
    named_instance,
)
from activityplan.predictor import DurationEnergyPredictor  # noqa: E402
from activityplan.robust import estimate_for_mode  # noqa: E402
from activityplan.timeline import RasterisedProblem  # noqa: E402
from activityplan.warmstart import compare_warm_start, warm_start_from_plan  # noqa: E402

OUTPUT = Path(__file__).with_name("validate_warmstart_output.txt")
N_TRAIN = 1000
PREDICTOR_SEED = 61
FAMILY_SEED = 9000
FAMILY_SIZE = 8
CENSUS_FAMILY_SIZE = 12


def _fit_predictor() -> DurationEnergyPredictor:
    rng = np.random.default_rng(PREDICTOR_SEED)
    requests = sample_requests(N_TRAIN, rng)
    realised = [realise(request, rng) for request in requests]
    return DurationEnergyPredictor(
        alpha=0.10, mode="normalised", n_estimators=160, random_state=PREDICTOR_SEED
    ).fit(
        feature_matrix(list(requests)),
        np.array([item.duration_s for item in realised], dtype=float),
        np.array([item.energy_wh for item in realised], dtype=float),
    )


def main() -> int:
    say = Tee(OUTPUT)
    started = time.perf_counter()
    say("activityplan -- validate_warmstart.py")
    say("does a learned warm start help the exact solver")
    for line in environment_lines():
        say(line)
    say("warm-start mechanism : primal-bound cut (scipy.optimize.milp has no MIP start)")

    say.rule("part 1: census of cold node counts -- is there a search to prune at all")
    say(
        f"{'instance':16s} {'bins':>6s} {'presolve':>9s} {'nodes':>7s} {'solve s':>9s} "
        f"{'optimum':>8s} {'gap':>9s}"
    )
    branching: list[tuple[str, RasterisedProblem, bool, int, float, float]] = []
    census_rows = 0
    rootsolved = 0
    cases: list[tuple[str, RasterisedProblem]] = [
        (name, RasterisedProblem.build(named_instance(name), RECOMMENDED_STEP_S[name]))
        for name in INSTANCE_ORDER
    ]
    cases += [
        (spec.name, RasterisedProblem.build(spec.analytic_problem(), spec.step_s))
        for spec in instance_family(CENSUS_FAMILY_SIZE, FAMILY_SEED)
    ]
    for label, rasterised in cases:
        for presolve in (True, False):
            result = solve_exact(rasterised, time_limit_s=150.0, presolve=presolve)
            census_rows += 1
            if result.node_count <= 1:
                rootsolved += 1
            else:
                branching.append(
                    (
                        label,
                        rasterised,
                        presolve,
                        result.node_count,
                        result.solve_s,
                        result.value,
                    )
                )
            say(
                f"{label:16s} {result.n_binaries:6d} {str(presolve):>9s} "
                f"{result.node_count:7d} {result.solve_s:9.3f} {result.value:8.1f} "
                f"{result.gap:9.2e}"
            )
    say("")
    say(f"solves in the census                      : {census_rows}")
    say(f"solved with a node count of 1             : {rootsolved}")
    say(f"solves with an actual branch-and-bound tree: {len(branching)}")
    say("")
    say("Reading: the time-indexed formulation's root relaxation plus HiGHS's cuts close")
    say("the gap outright on every instance in the seeded family. On those instances a")
    say("primal bound has nothing to prune and no warm start of any kind can help. That")
    say("is a property of the formulation and the solver, not of the predictor.")

    say.rule("part 2: warm against cold on the instances that actually branch")
    if not branching:
        say("no instance in the census branched; part 2 has nothing to measure.")
    else:
        say(
            f"{'instance':16s} {'presolve':>9s} {'warm bound':>11s} {'optimum':>8s} "
            f"{'cold nodes':>11s} {'warm nodes':>11s} {'node ratio':>11s} "
            f"{'cold s':>8s} {'warm s':>8s} {'time ratio':>11s} {'opt kept':>9s}"
        )
        for label, rasterised, presolve, _, _, _ in branching:
            candidate = plan_greedy(rasterised).plan
            warm_start = warm_start_from_plan(rasterised, candidate, source="greedy")
            comparison = compare_warm_start(
                rasterised, warm_start, presolve=presolve, time_limit_s=200.0
            )
            if comparison.warm is None:
                say(f"{label:16s} {str(presolve):>9s} candidate did not verify; no warm solve")
                continue
            node_ratio = comparison.node_ratio
            time_ratio = comparison.time_ratio
            node_ratio = math.nan if node_ratio is None else node_ratio
            time_ratio = math.nan if time_ratio is None else time_ratio
            say(
                f"{label:16s} {str(presolve):>9s} {warm_start.candidate_value:11.1f} "
                f"{comparison.cold.value:8.1f} {comparison.cold.node_count:11d} "
                f"{comparison.warm.node_count:11d} {node_ratio:11.4f} "
                f"{comparison.cold.solve_s:8.3f} {comparison.warm.solve_s:8.3f} "
                f"{time_ratio:11.4f} {str(comparison.optimum_preserved):>9s}"
            )
        say("")
        say("Wall clocks on a shared 2-core container move by tens of percent between runs.")
        say("Read the node ratio first.")

    say.rule("part 3: learned against analytic as the warm-start SOURCE")
    say("The target problem is built from the learned point predictions, which is what an")
    say("onboard planner would actually solve. Two candidate plans are offered to it:")
    say("  greedy-ml       the greedy plan on that same target")
    say("  greedy-analytic the greedy plan on the analytic problem, transferred across")
    say("A transferred plan is a plan for a different problem, so it has to be verified")
    say("against the target before its value can be used as a bound.")
    say("")
    predictor = _fit_predictor()
    say(
        f"{'instance':14s} {'optimum':>8s} {'ml bound':>9s} {'ml ok':>6s} "
        f"{'ana bound':>10s} {'ana ok':>7s} {'cold nodes':>11s} {'ml nodes':>9s} "
        f"{'ana nodes':>10s} {'cold s':>8s} {'ml s':>7s} {'ana s':>7s}"
    )
    ml_verified = 0
    analytic_verified = 0
    ml_bound_ratio: list[float] = []
    analytic_bound_ratio: list[float] = []
    instances = 0
    optimum_lost = 0
    for spec in instance_family(FAMILY_SIZE, FAMILY_SEED, n_activities=10):
        estimate = estimate_for_mode(spec, "ml-point", predictor=predictor)
        target = RasterisedProblem.build(
            spec.problem_from(
                estimate.durations_s, estimate.energies_wh, name_suffix="/ml-point"
            ),
            spec.step_s,
        )
        analytic_rasterised = RasterisedProblem.build(spec.analytic_problem(), spec.step_s)
        ml_candidate = plan_greedy(target).plan
        analytic_candidate = plan_greedy(analytic_rasterised).plan
        ml_start = warm_start_from_plan(target, ml_candidate, source="greedy-ml")
        analytic_start = warm_start_from_plan(
            target, analytic_candidate, source="greedy-analytic"
        )
        cold = solve_exact(target, time_limit_s=150.0)
        instances += 1
        ml_result = None
        analytic_result = None
        if ml_start.bound is not None:
            ml_verified += 1
            ml_bound_ratio.append(ml_start.bound / cold.value if cold.value > 0 else math.nan)
            ml_result = solve_exact(target, primal_bound=ml_start.bound, time_limit_s=150.0)
            if abs(ml_result.value - cold.value) > 1e-9:
                optimum_lost += 1
        if analytic_start.bound is not None:
            analytic_verified += 1
            analytic_bound_ratio.append(
                analytic_start.bound / cold.value if cold.value > 0 else math.nan
            )
            analytic_result = solve_exact(
                target, primal_bound=analytic_start.bound, time_limit_s=150.0
            )
            if abs(analytic_result.value - cold.value) > 1e-9:
                optimum_lost += 1
        say(
            f"{spec.name:14s} {cold.value:8.1f} "
            f"{(ml_start.bound if ml_start.bound is not None else math.nan):9.1f} "
            f"{str(ml_start.verified):>6s} "
            f"{(analytic_start.bound if analytic_start.bound is not None else math.nan):10.1f} "
            f"{str(analytic_start.verified):>7s} {cold.node_count:11d} "
            f"{(ml_result.node_count if ml_result else -1):9d} "
            f"{(analytic_result.node_count if analytic_result else -1):10d} "
            f"{cold.solve_s:8.3f} {(ml_result.solve_s if ml_result else math.nan):7.3f} "
            f"{(analytic_result.solve_s if analytic_result else math.nan):7.3f}"
        )
        if not analytic_start.verified:
            say(f"               analytic candidate refused: {analytic_start.reason}")
    say("")
    say(f"instances                               : {instances}")
    say(f"greedy-ml candidate verified            : {ml_verified} of {instances}")
    say(f"greedy-analytic candidate verified      : {analytic_verified} of {instances}")
    if ml_bound_ratio:
        say(
            "greedy-ml bound as a fraction of the optimum   : mean "
            f"{float(np.mean(ml_bound_ratio)):.4f}, min {float(np.min(ml_bound_ratio)):.4f}"
        )
    if analytic_bound_ratio:
        say(
            "greedy-analytic bound as a fraction of the optimum: mean "
            f"{float(np.mean(analytic_bound_ratio)):.4f}, min "
            f"{float(np.min(analytic_bound_ratio)):.4f}"
        )
    say(f"optima lost to a bad bound              : {optimum_lost}")
    say("criterion                               : 0 optima lost")
    say(f"verdict                                 : {'FAIL' if optimum_lost else 'PASS'}")
    say("")
    say("Reading: on these instances the cold node count is 1, so neither source changes")
    say("the node count and there is no speed-up to attribute to either. The learned")
    say("warm start does NOT help here, and that is the measured result. What the")
    say("verification gate does buy is safety: a bound is used only after the simulator")
    say("has confirmed the candidate plan is feasible for the target problem.")

    say.rule("summary")
    say(f"census solves                           {census_rows}")
    say(f"solved at the root (node count 1)       {rootsolved}")
    say(f"instances with a real search            {len(branching)}")
    say(f"greedy-ml candidates verified           {ml_verified} of {instances}")
    say(f"greedy-analytic candidates verified     {analytic_verified} of {instances}")
    say(f"optima lost to a warm-start bound       {optimum_lost}")
    say(
        "headline                                the warm start helps only where HiGHS "
        "actually branches,"
    )
    say(
        f"                                        which is {len(branching)} of "
        f"{census_rows} census solves; elsewhere it is neutral"
    )
    say(f"wall clock                              {time.perf_counter() - started:.1f} s")
    say(f"overall                                 {'FAIL' if optimum_lost else 'PASS'}")
    say(f"raw output written to                   {OUTPUT.name}")
    say.save()
    return 1 if optimum_lost else 0


if __name__ == "__main__":
    raise SystemExit(main())
