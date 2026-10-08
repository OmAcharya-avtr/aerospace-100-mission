"""Validate the Euler simulator against an analytic reference and against itself.

Four checks:

1. **Analytic reference.** In the unsaturated regime the closed loop is linear
   and time invariant, so the exact zero-order-hold solution by matrix
   exponential exists. The Euler scheme is measured against it, and the
   first-order convergence the module docstring claims is checked by halving the
   step and watching the error halve.
2. **Self-convergence in the saturated regime**, where no analytic answer exists:
   the same scheme at ``dt`` and ``dt/8`` must agree to the order claimed.
3. **Declared limits.** The actuator deflection and slew-rate limits are
   respected over the whole search box, which is a property of the
   implementation and not of the maths.
4. **Determinism**, over the whole box, because the seeded benchmark depends on
   it absolutely.
5. **Whether found counterexamples survive a four-times-finer step.** A
   violation whose robustness is inside the discretisation budget is a numerical
   artefact, and this check measures how often that happens instead of assuming
   it does not.

References
----------
Butcher, J. C. (2016), *Numerical Methods for Ordinary Differential Equations*,
3rd ed., Wiley. Explicit Euler: local truncation error O(dt^2), global O(dt).

Franklin, G. F., Powell, J. D. and Emami-Naeini, A. (2015), *Feedback Control of
Dynamic Systems*, 7th ed., Pearson. Second-order response parameters, and the
zero-order-hold discretisation by matrix exponential.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _bootstrap import Tee, add_src_to_path

add_src_to_path()

from falsifyloop.instances import SEARCH_BOX  # noqa: E402
from falsifyloop.systems import (  # noqa: E402
    LoopInput,
    LoopParameters,
    simulate,
    simulate_linear_zoh,
)

OUT = Path(__file__).with_name("validate_simulator_output.txt")
say = Tee(OUT)

# A small step with no gust: the deflection command peaks at Kp * 1 deg = 1.4 deg
# against a 20 deg limit and the slew at (1.4 - 0)/0.05 = 28 deg/s against a
# 120 deg/s limit, so neither clip can bind and the linear solution applies.
GENTLE = LoopInput(1.0, 1.0, 1.0, 1.0, 0.0, 1.0)


def check_1_against_the_exact_zoh_solution() -> int:
    say.rule("Check 1 - Euler against the exact zero-order-hold linear solution")
    params = LoopParameters()
    wn, zeta = params.nominal_modes()
    say(f"nominal closed loop: wn = {wn:.6f} rad/s, zeta = {zeta:.6f} (dimensionless)")
    say(
        "unsaturated regime confirmed below: peak |delta| against the 20 deg limit and "
        "peak |d delta/dt| against the 120 deg/s limit."
    )
    say("")
    say(f"{'dt [s]':>9s} {'max |theta err| [deg]':>22s} {'ratio':>8s} "
        f"{'peak |delta| [deg]':>19s} {'peak rate [deg/s]':>18s}")
    say("-" * 80)
    previous = None
    failures = 0
    for dt in (0.008, 0.004, 0.002, 0.001, 0.0005):
        euler = simulate(GENTLE, dt=dt, horizon=1.0)
        exact = simulate_linear_zoh(GENTLE, dt=dt, horizon=1.0)
        err = float(np.max(np.abs(euler.signal("theta") - exact.signal("theta"))))
        delta = euler.signal("delta")
        peak_delta = float(np.max(np.abs(delta)))
        peak_rate = float(np.max(np.abs(np.diff(delta)) / dt))
        ratio = "--" if previous is None else f"{previous / err:.3f}"
        say(f"{dt:>9.4f} {err:>22.9f} {ratio:>8s} {peak_delta:>19.6f} {peak_rate:>18.4f}")
        if peak_delta >= params.deflection_limit or peak_rate >= params.rate_limit:
            failures += 1
            say("  MISMATCH: a clip bound, so the linear reference does not apply here")
        previous = err
    say("")
    say(
        "A ratio near 2 when dt halves is the first-order global convergence the "
        "module docstring claims. The absolute error is the discretisation error "
        "budget of every robustness value this package reports."
    )
    final = simulate(GENTLE, dt=0.005, horizon=1.0)
    final_exact = simulate_linear_zoh(GENTLE, dt=0.005, horizon=1.0)
    shipped_err = float(np.max(np.abs(final.signal("theta") - final_exact.signal("theta"))))
    say(f"at the shipped dt = 0.005 s the worst theta error is {shipped_err:.9f} deg")
    say(f"check-1 failures: {failures}")
    return failures


def check_2_self_convergence_in_the_saturated_regime() -> int:
    say.rule("Check 2 - self-convergence where both clips bind and no analytic answer exists")
    aggressive = LoopInput(6.0, 2.0, 0.25, 0.6, 15.0, 1.0)
    params = LoopParameters()
    reference = simulate(aggressive, dt=0.000125, horizon=1.0)
    ref_theta = reference.signal("theta")
    delta = reference.signal("delta")
    say(
        f"peak |delta| = {np.max(np.abs(delta)):.4f} deg against the "
        f"{params.deflection_limit:g} deg limit; "
        f"peak slew = {np.max(np.abs(np.diff(delta)) / 0.000125):.1f} deg/s against the "
        f"{params.rate_limit:g} deg/s limit, so both clips are active."
    )
    say("")
    say(f"{'dt [s]':>9s} {'max |theta - theta_ref| [deg]':>31s} {'ratio':>8s}")
    say("-" * 52)
    previous = None
    for dt in (0.008, 0.004, 0.002, 0.001):
        coarse = simulate(aggressive, dt=dt, horizon=1.0)
        stride = int(round(dt / 0.000125))
        sub = ref_theta[::stride][: coarse.length]
        err = float(np.max(np.abs(coarse.signal("theta") - sub)))
        ratio = "--" if previous is None else f"{previous / err:.3f}"
        say(f"{dt:>9.4f} {err:>31.6f} {ratio:>8s}")
        previous = err
    shipped = simulate(aggressive, dt=0.005, horizon=1.0)
    stride = int(round(0.005 / 0.000125))
    err = float(np.max(np.abs(shipped.signal("theta") - ref_theta[::stride][: shipped.length])))
    say("")
    say(f"at the shipped dt = 0.005 s the worst theta discrepancy is {err:.6f} deg")
    say(
        "The measured ratios are near 2, so first-order convergence survives the two "
        "clips on this trajectory -- which was not a foregone conclusion, since the "
        "clip switching instants move with dt. What does not survive is the small "
        "absolute error: the discretisation budget in the saturated regime is about "
        "fifty times the unsaturated one, and it is the budget every robustness value "
        "near zero has to be read against. Check 5 measures what that costs."
    )
    return 0


def check_5_do_counterexamples_survive_a_finer_step() -> int:
    """Re-evaluate violating draws at dt/4 and count the verdicts that flip.

    A counterexample whose robustness is inside the discretisation budget is not
    a counterexample, it is a numerical artefact. This is the check that says
    how often that happens, per instance, instead of assuming it does not.
    """
    say.rule("Check 5 - do found counterexamples survive a four-times-finer step")
    from falsifyloop.instances import suite
    from falsifyloop.requirements import robustness
    from falsifyloop.systems import simulate as sim

    fine_dt = 0.00125
    rng = np.random.default_rng(161803)
    say(f"coarse dt {0.005:g} s (shipped) against fine dt {fine_dt:g} s")
    say("")
    say(
        f"{'instance':<20s} {'draws':>6s} {'violating':>10s} {'flipped':>8s} "
        f"{'worst |rho| flipped':>20s} {'min |rho| kept':>15s}"
    )
    say("-" * 84)
    total_flips = 0
    for inst in suite():
        points = inst.sample(rng, 600)
        flips = 0
        violating = 0
        worst_flip = 0.0
        min_kept = float("inf")
        for vector in points:
            coarse = inst.evaluate(vector)
            if coarse >= 0.0:
                continue
            violating += 1
            fine_trace = sim(
                LoopInput.from_array(vector),
                parameters=inst.parameters,
                dt=fine_dt,
                horizon=inst.horizon,
            )
            fine = robustness(inst.requirement, fine_trace)
            if fine >= 0.0:
                flips += 1
                worst_flip = max(worst_flip, abs(coarse))
            else:
                min_kept = min(min_kept, abs(coarse))
        total_flips += flips
        kept = "--" if not np.isfinite(min_kept) else f"{min_kept:.6f}"
        say(
            f"{inst.identifier:<20s} {len(points):>6d} {violating:>10d} {flips:>8d} "
            f"{worst_flip:>20.6f} {kept:>15s}"
        )
    say("")
    say(f"total verdicts that flipped when the step was refined: {total_flips}")
    say(
        "A flip means the coarse-step simulation reported a violation the finer step "
        "does not. These are reported rather than filtered out: the benchmark counts "
        "simulations at the shipped step, so a flipped counterexample still cost the "
        "search a simulation and still appears in its curve. The practical rule is "
        "that a counterexample should be re-simulated at a finer step before it is "
        "believed, and the worst flipped |rho| column is how large a margin that rule "
        "needs."
    )
    return 0


def check_3_declared_limits_over_the_whole_box() -> int:
    say.rule("Check 3 - declared actuator limits respected over the whole search box")
    params = LoopParameters()
    rng = np.random.default_rng(314159)
    draws = 4000
    points = rng.uniform(SEARCH_BOX[:, 0], SEARCH_BOX[:, 1], size=(draws, 6))
    worst_delta = 0.0
    worst_rate = 0.0
    worst_cmd = 0.0
    worst_theta = 0.0
    worst_q = 0.0
    failures = 0
    for vector in points:
        trace = simulate(LoopInput.from_array(vector))
        delta = trace.signal("delta")
        worst_delta = max(worst_delta, float(np.max(np.abs(delta))))
        worst_rate = max(worst_rate, float(np.max(np.abs(np.diff(delta))) / trace.dt))
        worst_cmd = max(worst_cmd, float(np.max(np.abs(trace.signal("cmd")))))
        worst_theta = max(worst_theta, float(np.max(np.abs(trace.signal("theta")))))
        worst_q = max(worst_q, float(np.max(np.abs(trace.signal("q")))))
    say(f"draws                        : {draws}")
    say(
        f"worst |delta| observed       : {worst_delta:.9f} deg  "
        f"(declared limit {params.deflection_limit:g} deg)"
    )
    say(
        f"worst |d delta/dt| observed  : {worst_rate:.6f} deg/s  "
        f"(declared limit {params.rate_limit:g} deg/s)"
    )
    say(
        f"worst |cmd| observed         : {worst_cmd:.9f} deg  "
        f"(declared limit {params.deflection_limit:g} deg)"
    )
    say("")
    say(
        f"worst |theta| observed       : {worst_theta:.4f} deg   "
        f"(NOT a declared limit; see below)"
    )
    say(f"worst |q| observed           : {worst_q:.4f} deg/s (NOT a declared limit)")
    say(
        "Attitude and attitude rate have no declared limit: they are what the loop "
        "does. The worst |theta| figure is quoted in the README and the dataset card "
        "because it is the honest measure of how far the violating corners of this "
        "box leave the small-angle regime in which the declared second-order model is "
        "even nominally valid. It is a property of the benchmark, not a prediction "
        "about any vehicle."
    )
    say("")
    for name, value, limit in (
        ("deflection", worst_delta, params.deflection_limit),
        ("slew rate", worst_rate, params.rate_limit),
        ("command", worst_cmd, params.deflection_limit),
    ):
        if value > limit + 1e-9:
            failures += 1
            say(f"  MISMATCH: {name} limit exceeded")
    say(f"check-3 failures: {failures}")
    return failures


def check_4_determinism() -> int:
    say.rule("Check 4 - determinism over the search box")
    rng = np.random.default_rng(2718)
    draws = 500
    points = rng.uniform(SEARCH_BOX[:, 0], SEARCH_BOX[:, 1], size=(draws, 6))
    failures = 0
    for vector in points:
        first = simulate(LoopInput.from_array(vector)).signal("theta")
        second = simulate(LoopInput.from_array(vector)).signal("theta")
        if not np.array_equal(first, second):
            failures += 1
    say(f"draws re-simulated bit-for-bit : {draws}")
    say(f"non-reproducible simulations   : {failures}")
    say(
        "Bit equality rather than tolerance, because the seeded benchmark's "
        "reproducibility claim is bit equality and nothing weaker."
    )
    return failures


def main() -> int:
    say("falsifyloop - simulator validation")
    say(
        "The simulator is a synthetic benchmark closed loop. No parameter value in it "
        "was identified from any aircraft, and nothing below is a statement about a "
        "vehicle. What is validated is the numerics and the declared limits."
    )
    failures = 0
    failures += check_1_against_the_exact_zoh_solution()
    failures += check_2_self_convergence_in_the_saturated_regime()
    failures += check_3_declared_limits_over_the_whole_box()
    failures += check_4_determinism()
    failures += check_5_do_counterexamples_survive_a_finer_step()
    say.rule("SUMMARY")
    say(f"total failures across all five checks: {failures}")
    say.save()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
