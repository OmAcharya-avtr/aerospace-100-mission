"""Light-time iteration: convergence, residuals, and the exact closed-form check.

The iterated light-time solution is the reference the closed-form point-ahead
baseline is scored against, so it needs a reference of its own. For a
constant-velocity remote trajectory the light-time equation is a quadratic with
a closed-form root, so the iteration is checked against an exact answer rather
than against itself.

The convergence history is printed per iteration rather than summarised,
because the light-time iteration residual is one of the budget lines and a
reader has to be able to see what truncating the iteration actually costs.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, bootstrap, header

bootstrap()

from pointahead.angle import light_time_pair  # noqa: E402
from pointahead.geometry import C_LIGHT, norm  # noqa: E402
from pointahead.kepler import LinearTrajectory  # noqa: E402
from pointahead.lighttime import (  # noqa: E402
    light_time_residual,
    linear_light_time_closed_form,
    solve_light_time,
)
from pointahead.scenario import NAMED_SCENARIOS, scenario_family  # noqa: E402


def main() -> int:
    tee = Tee("validate_lighttime_output.txt")
    header(tee, "light-time solution: iterated against the exact quadratic root")

    tee.rule("1. the exact relativistic case, hand-calculable")
    tee("  Remote one light-second away along +x, moving at 0.6 c along +y, local at")
    tee("  the origin. |[c, 0.6 c tau, 0]| = c tau gives tau = 1 / sqrt(1 - 0.36)")
    tee("  = 1.25 s exactly, in both directions because d . v = 0.")
    tee("")
    trajectory = LinearTrajectory(
        np.array([C_LIGHT, 0.0, 0.0]), np.array([0.0, 0.6 * C_LIGHT, 0.0]), 0.0
    )
    for direction in ("transmit", "receive"):
        exact = linear_light_time_closed_form(trajectory, np.zeros(3), 0.0, direction)
        iterated = solve_light_time(trajectory, np.zeros(3), 0.0, direction, max_iter=300)
        tee(f"  {direction:<9s} closed form   : {exact:.15f} s")
        tee(f"  {direction:<9s} iterated      : {iterated.tau_s:.15f} s")
        tee(f"  {direction:<9s} iterations    : {iterated.iterations}")
        tee(f"  {direction:<9s} residual      : {iterated.residual_s:.6e} s")
        tee.check(
            f"{direction} light time is 1.25 s to 1e-14 relative",
            abs(exact - 1.25) / 1.25 < 1.0e-14,
            f"closed form {exact:.15f} s",
        )
    tee("")
    tee("  At beta = 0.6 the fixed-point map contracts by only about 0.6 per step, so")
    tee("  the 30-iteration default cap does not reach the default 1e-15 s tolerance.")
    default_cap = solve_light_time(trajectory, np.zeros(3), 0.0, "transmit")
    tee(f"  default cap converged              : {default_cap.converged}")
    tee(f"  default cap residual               : {default_cap.residual_s:.6e} s")
    tee.check(
        "the solver reports non-convergence rather than claiming it",
        not default_cap.converged,
    )

    tee.rule("2. iterated vs closed form over a seeded straight-line family")
    rng = np.random.default_rng(760076)
    worst = 0.0
    worst_iterations = 0
    samples = 400
    for _ in range(samples):
        position = np.array([8.0e6, 0.0, 0.0]) + rng.normal(size=3) * 2.0e6
        velocity = rng.normal(size=3) * 4.0e3
        straight = LinearTrajectory(position, velocity, 0.0)
        local = rng.normal(size=3) * 2.0e6
        for direction in ("transmit", "receive"):
            exact = linear_light_time_closed_form(straight, local, 0.0, direction)
            iterated = solve_light_time(straight, local, 0.0, direction)
            worst = max(worst, abs(iterated.tau_s - exact) / exact)
            worst_iterations = max(worst_iterations, iterated.iterations)
    tee(f"  samples (each direction)           : {samples}")
    tee(f"  worst relative difference          : {worst:.6e}")
    tee(f"  worst iteration count              : {worst_iterations}")
    tee.check(
        "iterated light time matches the closed form to 1e-14 relative",
        worst < 1.0e-14,
        f"worst {worst:.6e}",
    )

    tee.rule("3. convergence history on the three named geometries")
    tee("  The iterate sequence, the residual at each truncation, and the range")
    tee("  error the residual corresponds to.")
    for name, builder in NAMED_SCENARIOS.items():
        link = builder()
        tee("")
        tee(f"  {name}: range {link.range_m / 1.0e3:.6f} km, "
            f"nominal light time {link.nominal_light_time_s:.9f} s")
        for direction in ("receive", "transmit"):
            tee(f"    {direction}")
            tee(f"      {'cap':>4s} {'tau / s':>22s} {'residual / s':>14s} "
                f"{'range error / m':>16s}")
            for cap in range(5):
                solution = solve_light_time(
                    link.trajectories()[1],
                    link.local.position_m,
                    link.epoch_s,
                    direction,
                    max_iter=cap,
                )
                tee(
                    f"      {cap:>4d} {solution.tau_s:>22.15f} "
                    f"{solution.residual_s:>14.3e} {C_LIGHT * solution.residual_s:>16.3e}"
                )
        receive, transmit = light_time_pair(link)
        tee(f"    converged tau_rx                 : {receive.tau_s:.15f} s "
            f"({receive.iterations} iterations)")
        tee(f"    converged tau_tx                 : {transmit.tau_s:.15f} s "
            f"({transmit.iterations} iterations)")
        tee(f"    tau_tx - tau_rx                  : {transmit.tau_s - receive.tau_s:.6e} s")
        tee(f"    light-time range difference      : "
            f"{C_LIGHT * (transmit.tau_s - receive.tau_s):.6f} m")
        tee.check(f"{name} receive solution converged", receive.converged)
        tee.check(f"{name} transmit solution converged", transmit.converged)
        residual = light_time_residual(
            link.trajectories()[1], link.local.position_m, link.epoch_s, transmit.tau_s, "transmit"
        )
        tee.check(
            f"{name} transmit residual is below 1e-17 s at the solution",
            abs(residual) < 1.0e-17,
            f"{residual:.3e} s",
        )

    tee.rule("4. contraction factor against the remote terminal's radial speed over c")
    tee("  The fixed-point map is tau -> |r_remote(t +/- tau) - r_local| / c with")
    tee("  r_local HELD FIXED, because the point-ahead angle is defined at one event")
    tee("  at the local terminal. Its derivative is therefore +/- (v_remote . n) / c")
    tee("  and NOT (v_rel . n) / c: the contraction is driven by the remote")
    tee("  terminal's radial velocity in the frame the states are given in, not by")
    tee("  the relative radial velocity. Both candidates are printed, because the")
    tee("  first version of this check used the relative one and the measurement")
    tee("  disagreed with it by a factor of two on one geometry and by twelve orders")
    tee("  of magnitude on another.")
    tee("")
    tee(f"  {'scenario':<20s} {'|v_rem,r| / c':>15s} {'|v_rel,r| / c':>15s} "
        f"{'measured ratio':>15s}")
    tee(f"  {'-' * 20} {'-' * 15} {'-' * 15} {'-' * 15}")
    for name, builder in NAMED_SCENARIOS.items():
        link = builder()
        remote = link.trajectories()[1]
        remote_radial = abs(float(np.dot(link.remote.velocity_mps, link.line_of_sight)))
        relative_radial = abs(float(np.dot(link.relative_velocity_mps, link.line_of_sight)))
        residuals = [
            abs(
                solve_light_time(
                    remote, link.local.position_m, link.epoch_s, "transmit", max_iter=cap
                ).residual_s
            )
            for cap in (0, 1)
        ]
        measured = residuals[1] / residuals[0] if residuals[0] > 0.0 else float("nan")
        tee(
            f"  {name:<20s} {remote_radial / C_LIGHT:>15.6e} "
            f"{relative_radial / C_LIGHT:>15.6e} {measured:>15.6e}"
        )
        tee.check(
            f"{name} contraction factor matches |v_remote . n| / c to 1 per cent",
            abs(measured - remote_radial / C_LIGHT) / (remote_radial / C_LIGHT) < 1.0e-2,
            f"measured {measured:.6e}, predicted {remote_radial / C_LIGHT:.6e}",
        )

    tee.rule("5. the receive and transmit light times bracket the geometric one")
    tee("  tau_rx < range / c < tau_tx for a receding pair and the reverse for an")
    tee("  approaching one; the geometric light time always lies between them.")
    violations = 0
    checked = 0
    for link in scenario_family(760, 60):
        receive, transmit = light_time_pair(link)
        nominal = link.nominal_light_time_s
        low, high = sorted((receive.tau_s, transmit.tau_s))
        checked += 1
        if not low <= nominal <= high:
            violations += 1
    tee(f"  seeded crosslinks checked          : {checked}")
    tee(f"  violations                         : {violations}")
    tee.check("the geometric light time is always bracketed", violations == 0)

    tee.rule("6. the gap between the two light times")
    tee("  tau_tx - tau_rx = 2 (v_remote . n) tau / c to first order, for the same")
    tee("  reason as section 4. The leo_crossing and leo_trailing geometries share a")
    tee("  local terminal and a range and differ only in the remote plane, and their")
    tee("  remote radial velocities happen to be equal, so their light-time gaps are")
    tee("  equal too even though their RELATIVE radial velocities are 3736 m/s and")
    tee("  zero. That coincidence is what exposed the wrong predictor.")
    tee("")
    tee(f"  {'scenario':<20s} {'measured gap / s':>20s} {'predicted gap / s':>20s} "
        f"{'rel':>10s}")
    tee(f"  {'-' * 20} {'-' * 20} {'-' * 20} {'-' * 10}")
    for name, builder in NAMED_SCENARIOS.items():
        link = builder()
        receive, transmit = light_time_pair(link)
        remote_radial = float(np.dot(link.remote.velocity_mps, link.line_of_sight))
        predicted = 2.0 * remote_radial * link.nominal_light_time_s / C_LIGHT
        measured = transmit.tau_s - receive.tau_s
        relative = abs(measured - predicted) / abs(predicted)
        tee(f"  {name:<20s} {measured:>20.9e} {predicted:>20.9e} {relative:>10.3e}")
        tee.check(
            f"{name} light-time gap matches 2 (v_remote . n) tau / c to 1 per cent",
            relative < 1.0e-2,
            f"relative difference {relative:.3e}",
        )
    tee("")
    tee("  relative speeds, for context:")
    for name, builder in NAMED_SCENARIOS.items():
        link = builder()
        tee(f"    {name:<20s} |v_rel| = {norm(link.relative_velocity_mps):>10.3f} m/s")

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
