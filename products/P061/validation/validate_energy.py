"""Energy conservation over the timeline, against a closed-form integral.

Level 3 check 1. Four things are measured, in increasing strength:

1. The closed-form solar integral
   (:meth:`activityplan.model.SolarProfile.energy_wh`) against adaptive
   numerical quadrature (``scipy.integrate.quad``), which is an independent
   implementation of the same integral.
2. The sum of the per-step closed-form integrals against the whole-horizon
   closed form. This is not a tautology: ``energy_wh`` accumulates whole orbits
   in closed form and handles the partial arcs at both ends separately, so the
   per-step decomposition exercises different branches of the same function.
3. The midpoint-sampled Riemann sum against the closed form, and its
   convergence order under step halving, to quantify what the package would
   lose by sampling the profile instead of integrating it.
4. The battery recursion's closure: ``(E_final - E_initial)`` against
   ``generated - consumed - shunted``, over every shipped instance and over a
   seeded family, under both planners.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad

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
from activityplan.model import SolarProfile  # noqa: E402
from activityplan.timeline import Plan, RasterisedProblem  # noqa: E402

OUTPUT = Path(__file__).with_name("validate_energy_output.txt")

PROFILES = (
    ("no eclipse", SolarProfile(100.0, 1000.0, 0.0, 0.0)),
    ("eclipse 0.34, phase 900 s", SolarProfile(165.0, 5400.0, 0.34, 900.0)),
    ("eclipse 0.36, phase 0", SolarProfile(118.0, 5400.0, 0.36, 0.0)),
    ("eclipse 0.5, negative phase", SolarProfile(50.0, 600.0, 0.5, -250.0)),
)


def main() -> int:
    say = Tee(OUTPUT)
    started = time.perf_counter()
    say("activityplan -- validate_energy.py")
    say("energy conservation over the timeline against a closed-form integral")
    for line in environment_lines():
        say(line)

    say.rule("check 1: closed-form solar integral against scipy.integrate.quad")
    say(
        "the closed form accumulates whole orbits analytically; quad is an independent"
    )
    say("adaptive quadrature of the same declared power profile.")
    say("")
    say(
        f"{'profile':30s} {'horizon/s':>10s} {'closed/Wh':>14s} {'quad/Wh':>14s} "
        f"{'rel diff':>12s}"
    )
    worst_quad = 0.0
    for label, profile in PROFILES:
        for horizon in (profile.orbit_period_s, 3.0 * profile.orbit_period_s):
            closed = profile.energy_wh(0.0, horizon)
            numerical = quad(profile.power_at, 0.0, horizon, limit=600)[0] / 3600.0
            rel = abs(closed - numerical) / max(abs(numerical), 1e-30)
            worst_quad = max(worst_quad, rel)
            say(
                f"{label:30s} {horizon:10.1f} {closed:14.9f} {numerical:14.9f} {rel:12.3e}"
            )
    say("")
    say(f"worst relative difference against quad : {worst_quad:.3e}")
    say("criterion                               : < 1e-8 (quad's own tolerance)")
    say(f"verdict                                 : {'PASS' if worst_quad < 1e-8 else 'FAIL'}")

    say.rule("check 2: per-step closed-form integrals sum to the whole-horizon closed form")
    say("exercises the orbit-accumulation branch against the partial-arc branch.")
    say("")
    say(
        f"{'instance':18s} {'steps':>6s} {'sum of steps/Wh':>18s} {'whole/Wh':>16s} "
        f"{'abs diff':>12s}"
    )
    worst_sum = 0.0
    for name in INSTANCE_ORDER:
        problem = named_instance(name)
        rasterised = RasterisedProblem.build(problem, RECOMMENDED_STEP_S[name])
        stepwise = math.fsum(rasterised.solar_wh.tolist())
        whole = problem.resources.solar.energy_wh(0.0, problem.horizon_s)
        diff = abs(stepwise - whole)
        worst_sum = max(worst_sum, diff)
        say(f"{name:18s} {rasterised.n_steps:6d} {stepwise:18.12f} {whole:16.12f} {diff:12.3e}")
    say("")
    say(f"worst absolute difference : {worst_sum:.3e} W*h")
    say("criterion                 : < 1e-9 W*h (floating-point accumulation only)")
    say(f"verdict                   : {'PASS' if worst_sum < 1e-9 else 'FAIL'}")

    say.rule("check 3: what midpoint sampling would cost, and its convergence order")
    say("the package integrates the solar profile in closed form per step. This check")
    say("measures the alternative. On a smooth profile the midpoint rule is second")
    say("order; across the eclipse kink it is not, which is the reason for the choice.")
    say("")
    for label, profile in (PROFILES[0], PROFILES[1]):
        horizon = profile.orbit_period_s * 2.0
        exact = profile.energy_wh(0.0, horizon)
        say(f"profile: {label}, horizon {horizon:.1f} s, exact {exact:.9f} W*h")
        say(f"  {'step/s':>8s} {'sampled/Wh':>16s} {'abs err/Wh':>14s} {'err ratio':>10s}")
        previous = None
        steps = [horizon / n for n in (36, 72, 144, 288)]
        for step in steps:
            grid_steps = int(round(horizon / step))
            mids = (np.arange(grid_steps) + 0.5) * step
            sampled = float(
                math.fsum((profile.power_at(float(t)) * step / 3600.0) for t in mids)
            )
            error = abs(sampled - exact)
            ratio = "-" if previous is None else f"{previous / max(error, 1e-300):10.3f}"
            say(f"  {step:8.2f} {sampled:16.9f} {error:14.3e} {ratio:>10s}")
            previous = error
        say("")
    say("the no-eclipse ratios are near 4, which is second order. The eclipse profile's")
    say("ratios are not, and are reported as measured.")

    say.rule("check 4: battery recursion closure under both planners")
    say("residual = (E_final - E_initial) - (effective generated - consumed - shunted)")
    say("")
    say(
        f"{'instance':18s} {'planner':8s} {'placed':>7s} {'gen/Wh':>11s} {'cons/Wh':>11s} "
        f"{'shunt/Wh':>11s} {'dE/Wh':>11s} {'residual':>12s}"
    )
    worst_residual = 0.0
    rows = 0
    for name in INSTANCE_ORDER:
        rasterised = RasterisedProblem.build(named_instance(name), RECOMMENDED_STEP_S[name])
        plans = {
            "empty": Plan({}, rasterised.grid),
            "greedy": plan_greedy(rasterised).plan,
            "exact": solve_exact(rasterised, time_limit_s=120.0).plan,
        }
        for planner, plan in plans.items():
            report = evaluate(rasterised, plan)
            audit = report.energy_audit()
            residual = audit["residual"]
            worst_residual = max(worst_residual, abs(residual))
            rows += 1
            say(
                f"{name:18s} {planner:8s} {plan.n_placed:7d} {audit['generated_steps']:11.5f} "
                f"{audit['consumed_steps']:11.5f} {audit['shunted']:11.5f} "
                f"{audit['battery_delta']:11.5f} {residual:12.3e}"
            )
    for spec in instance_family(6, 9000):
        rasterised = RasterisedProblem.build(spec.analytic_problem(), spec.step_s)
        plan = plan_greedy(rasterised).plan
        report = evaluate(rasterised, plan)
        audit = report.energy_audit()
        residual = audit["residual"]
        worst_residual = max(worst_residual, abs(residual))
        rows += 1
        say(
            f"{spec.name:18s} {'greedy':8s} {plan.n_placed:7d} {audit['generated_steps']:11.5f} "
            f"{audit['consumed_steps']:11.5f} {audit['shunted']:11.5f} "
            f"{audit['battery_delta']:11.5f} {residual:12.3e}"
        )
    say("")
    say(f"traces checked            : {rows}")
    say(f"worst absolute residual   : {worst_residual:.3e} W*h")
    say("criterion                 : < 1e-9 W*h")
    say(f"verdict                   : {'PASS' if worst_residual < 1e-9 else 'FAIL'}")

    say.rule("check 5: the fixed-load closed form against the per-step sum")
    say(
        f"{'instance':18s} {'fixed closed/Wh':>18s} {'fixed steps/Wh':>18s} {'abs diff':>12s}"
    )
    worst_fixed = 0.0
    for name in INSTANCE_ORDER:
        problem = named_instance(name)
        rasterised = RasterisedProblem.build(problem, RECOMMENDED_STEP_S[name])
        _, closed = problem.baseline_energy_balance_wh()
        stepwise = math.fsum(rasterised.fixed_wh.tolist())
        diff = abs(closed - stepwise)
        worst_fixed = max(worst_fixed, diff)
        say(f"{name:18s} {closed:18.12f} {stepwise:18.12f} {diff:12.3e}")
    say("")
    say(f"worst absolute difference : {worst_fixed:.3e} W*h")
    say("criterion                 : < 1e-9 W*h")
    say(f"verdict                   : {'PASS' if worst_fixed < 1e-9 else 'FAIL'}")

    elapsed = time.perf_counter() - started
    say.rule("summary")
    say(f"check 1 quad cross-check        worst rel diff {worst_quad:.3e}")
    say(f"check 2 per-step additivity     worst abs diff {worst_sum:.3e} W*h")
    say("check 3 midpoint sampling       reported, no pass criterion")
    say(
        f"check 4 battery closure         worst residual {worst_residual:.3e} W*h "
        f"over {rows} traces"
    )
    say(f"check 5 fixed-load additivity   worst abs diff {worst_fixed:.3e} W*h")
    say(f"wall clock                      {elapsed:.1f} s")
    failed = not (
        worst_quad < 1e-8 and worst_sum < 1e-9 and worst_residual < 1e-9 and worst_fixed < 1e-9
    )
    say(f"overall                         {'FAIL' if failed else 'PASS'}")
    say(f"raw output written to           {OUTPUT.name}")
    say.save()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
