"""Greedy against the exact optimum over a seeded instance family.

Level 3 check 3. Two things are measured:

1. The **optimality gap**: how far below the proven optimum the greedy
   baseline's objective lands, per instance and over the family, with the
   distribution shown rather than only the mean.
2. The **dominance lemma** in practice: every plan the MILP returns is handed to
   the resource simulator and must be feasible there. The MILP treats shunting
   and downlink as free continuous decisions; the simulator fixes them to the
   clamp and maximal-drain policies. If the lemma in
   :mod:`activityplan.simulate` were wrong, this is where it would show.

Also recorded: which constraint is named binding, and how often that is
``horizon`` rather than a resource, because the objective gives no reward for
finishing early and an optimal plan may therefore place an activity arbitrarily
late.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from _bootstrap import Tee, add_src_to_path, environment_lines

add_src_to_path()

from activityplan.exact import solve_exact  # noqa: E402
from activityplan.feasibility import evaluate  # noqa: E402
from activityplan.greedy import plan_greedy  # noqa: E402
from activityplan.instances import (  # noqa: E402
    INSTANCE_ORDER,
    RECOMMENDED_STEP_S,
    instance_family,
    named_instance,
)
from activityplan.timeline import RasterisedProblem  # noqa: E402

OUTPUT = Path(__file__).with_name("validate_gap_output.txt")
BASE_SEED = 9000
FAMILY_SIZE = 24


def main() -> int:
    say = Tee(OUTPUT)
    started = time.perf_counter()
    say("activityplan -- validate_gap.py")
    say("greedy against the exact optimum, and the dominance lemma, over a seeded family")
    for line in environment_lines():
        say(line)
    say(
        f"family           : {FAMILY_SIZE} instances, seeds {BASE_SEED}.."
        f"{BASE_SEED + FAMILY_SIZE - 1}"
    )
    say("sizing           : activityplan.instances.sample_instance_spec defaults")
    say("                   (14 activities, 2 orbits, 150 s step)")

    say.rule("part 1: the five shipped hand-declared instances")
    say(
        f"{'instance':18s} {'total':>7s} {'greedy':>7s} {'exact':>7s} {'ratio':>7s} "
        f"{'short %':>8s} {'nodes':>7s} {'greedy s':>9s} {'exact s':>8s} {'binding (exact)':s}"
    )
    for name in INSTANCE_ORDER:
        rasterised = RasterisedProblem.build(named_instance(name), RECOMMENDED_STEP_S[name])
        t0 = time.perf_counter()
        greedy = plan_greedy(rasterised)
        greedy_s = time.perf_counter() - t0
        exact = solve_exact(rasterised, time_limit_s=150.0)
        report = evaluate(rasterised, exact.plan)
        ratio = greedy.value / exact.value if exact.value > 0 else math.nan
        shortfall = 100.0 * (1.0 - ratio) if exact.value > 0 else math.nan
        say(
            f"{name:18s} {rasterised.problem.total_priority:7.1f} {greedy.value:7.1f} "
            f"{exact.value:7.1f} {ratio:7.4f} {shortfall:8.2f} {exact.node_count:7d} "
            f"{greedy_s:9.3f} {exact.solve_s:8.3f} {report.binding.name:s}"
        )

    say.rule("part 2: the seeded family")
    say(
        f"{'instance':14s} {'total':>7s} {'greedy':>7s} {'exact':>7s} {'ratio':>7s} "
        f"{'nodes':>7s} {'gap':>9s} {'greedy s':>9s} {'exact s':>8s} {'sim ok':>7s} "
        f"{'binding (exact)':s}"
    )
    ratios: list[float] = []
    greedy_values: list[float] = []
    exact_values: list[float] = []
    greedy_times: list[float] = []
    exact_times: list[float] = []
    binding_counts: Counter[str] = Counter()
    resource_counts: Counter[str] = Counter()
    not_proven = 0
    simulator_rejections = 0
    greedy_infeasible = 0
    exact_ties = 0

    for spec in instance_family(FAMILY_SIZE, BASE_SEED):
        rasterised = RasterisedProblem.build(spec.analytic_problem(), spec.step_s)
        t0 = time.perf_counter()
        greedy = plan_greedy(rasterised)
        greedy_s = time.perf_counter() - t0
        exact = solve_exact(rasterised, time_limit_s=150.0)
        if not exact.proven_optimal:
            not_proven += 1
        greedy_report = evaluate(rasterised, greedy.plan)
        exact_report = evaluate(rasterised, exact.plan)
        if not greedy_report.feasible:
            greedy_infeasible += 1
        if not exact_report.feasible:
            simulator_rejections += 1
        ratio = greedy.value / exact.value if exact.value > 0 else math.nan
        if exact.value > 0 and abs(greedy.value - exact.value) <= 1e-9:
            exact_ties += 1
        ratios.append(ratio)
        greedy_values.append(greedy.value)
        exact_values.append(exact.value)
        greedy_times.append(greedy_s)
        exact_times.append(exact.solve_s)
        binding_counts[exact_report.binding.name.split(":")[0]] += 1
        resource = exact_report.binding_resource
        if resource is not None:
            resource_counts[resource.name] += 1
        say(
            f"{spec.name:14s} {rasterised.problem.total_priority:7.1f} {greedy.value:7.1f} "
            f"{exact.value:7.1f} {ratio:7.4f} {exact.node_count:7d} {exact.gap:9.2e} "
            f"{greedy_s:9.3f} {exact.solve_s:8.3f} "
            f"{'yes' if exact_report.feasible else 'NO':>7s} {exact_report.binding.name:s}"
        )

    array = np.array(ratios, dtype=float)
    say("")
    say("optimality ratio greedy/exact over the family")
    say(f"  instances               : {array.size}")
    say(f"  mean                    : {float(np.mean(array)):.4f}")
    say(f"  median                  : {float(np.median(array)):.4f}")
    say(f"  minimum                 : {float(np.min(array)):.4f}")
    say(f"  maximum                 : {float(np.max(array)):.4f}")
    say(f"  standard deviation      : {float(np.std(array, ddof=1)):.4f}")
    say(f"  10th percentile         : {float(np.percentile(array, 10)):.4f}")
    say(f"  instances where greedy ties the optimum : {exact_ties} of {array.size}")
    pooled = math.fsum(greedy_values) / math.fsum(exact_values)
    say(f"  pooled ratio sum(greedy)/sum(exact)     : {pooled:.4f}")
    say(f"  pooled shortfall                        : {100.0 * (1.0 - pooled):.2f} %")
    say("")
    say("cost")
    say(f"  greedy wall clock, mean : {float(np.mean(greedy_times)):.3f} s")
    say(f"  exact wall clock, mean  : {float(np.mean(exact_times)):.3f} s")
    say(f"  exact wall clock, max   : {float(np.max(exact_times)):.3f} s")
    say(
        f"  exact/greedy time ratio : "
        f"{math.fsum(exact_times) / math.fsum(greedy_times):.2f}"
    )

    say("")
    say("dominance lemma: every MILP plan simulated")
    say(f"  MILP plans rejected by the simulator : {simulator_rejections} of {array.size}")
    say(f"  greedy plans rejected                : {greedy_infeasible} of {array.size}")
    say(f"  instances not proven optimal         : {not_proven} of {array.size}")
    say("  criterion                            : all three zero")
    failed = simulator_rejections or greedy_infeasible or not_proven
    say(f"  verdict                              : {'FAIL' if failed else 'PASS'}")

    say("")
    say("which constraint the exact optimum leaves binding")
    for name, count in binding_counts.most_common():
        say(f"  {name:30s} {count:3d} of {array.size}")
    say("")
    say("tightest RESOURCE constraint, reported separately because the objective gives no")
    say("reward for finishing early, so an optimal plan may place an activity arbitrarily")
    say("late and leave a one-step horizon slack that is tie-breaking, not a resource limit")
    for name, count in resource_counts.most_common():
        say(f"  {name:30s} {count:3d} of {array.size}")

    say.rule("summary")
    say(f"greedy/exact mean ratio          {float(np.mean(array)):.4f}")
    say(f"greedy/exact worst ratio         {float(np.min(array)):.4f}")
    say(f"pooled shortfall                 {100.0 * (1.0 - pooled):.2f} %")
    say(f"MILP plans rejected by simulator {simulator_rejections}")
    say(f"wall clock                       {time.perf_counter() - started:.1f} s")
    say(f"overall                          {'FAIL' if failed else 'PASS'}")
    say(f"raw output written to            {OUTPUT.name}")
    say.save()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
