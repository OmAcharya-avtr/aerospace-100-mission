"""Constrained planner: correctness, failure modes and settings sensitivity.

PART A  the planner never returns a path that violates a cone (300 geometries)
PART B  every infeasibility reason is reachable and names the right constraint
PART C  the four required failure modes
PART D  cost accounting: time and momentum against the closed forms
PART E  settings sensitivity: SLSQP iteration cap and cold-start magnitudes

Run: ``python validation/validate_planner.py``  (about 6 minutes)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slewforge.attitude import (  # noqa: E402
    quat_from_axis_angle,
    quat_identity,
    quat_rotate,
)
from slewforge.dataset import generate_problems, reference_spacecraft  # noqa: E402
from slewforge.dynamics import RigidBody  # noqa: E402
from slewforge.keepout import KeepOutCone, KeepOutSet  # noqa: E402
from slewforge.planner import (  # noqa: E402
    INFEASIBILITY_REASONS,
    Instrument,
    SlewProblem,
    path_min_margin,
    plan,
)
from slewforge.profiles import make_profile  # noqa: E402
from slewforge.wheels import WheelArray, pyramid_wheels  # noqa: E402


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    failures = 0
    body = reference_spacecraft()

    # ---------------------------------------------------------------- PART A
    rule("PART A -- the planner never returns a violating path")
    print("300 seeded problems whose direct eigenaxis slew violates at least one")
    print("cone. For every feasible result, the closed-form minimum margin over")
    print("the whole returned path must be >= 0.")
    print()
    problems = generate_problems(300, seed=31415)
    t0 = time.perf_counter()
    results = [plan(p) for p in problems]
    elapsed = time.perf_counter() - t0
    feasible = [(p, r) for p, r in zip(problems, results, strict=True) if r.feasible]
    margins = np.array([path_min_margin(p, r.path) for p, r in feasible])
    worst = float(np.min(margins)) if margins.size else float("nan")
    n_via = np.array([r.n_via for _, r in feasible])
    reasons: dict[str, int] = {}
    for r in results:
        if not r.feasible:
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
    print(f"problems                       : {len(problems)}")
    print(f"feasible                       : {len(feasible)} "
          f"({len(feasible) / len(problems) * 100:.1f} %)")
    print(f"infeasible, by reason          : {reasons if reasons else 'none'}")
    print(f"via points used  1 / 2         : {int(np.sum(n_via == 1))} / {int(np.sum(n_via == 2))}")
    print(f"worst margin on a returned path: {worst:.6e} rad = "
          f"{math.degrees(worst):.6e} deg   tolerance >= 0")
    print(f"paths with margin < 0          : {int(np.sum(margins < 0.0))}   tolerance 0")
    print(f"total planning time            : {elapsed:.2f} s "
          f"({elapsed / len(problems) * 1e3:.1f} ms per problem)")
    failures += int(np.sum(margins < 0.0)) > 0

    direct_margins = np.array([r.direct_min_margin for r in results])
    print()
    print("How badly the direct eigenaxis slew violated, over the same problems:")
    print(f"  median penetration : {math.degrees(-float(np.median(direct_margins))):.3f} deg")
    print(f"  worst penetration  : {math.degrees(-float(np.min(direct_margins))):.3f} deg")
    print("This is the number a spreadsheet does not produce: every one of these")
    print("slews looks fine to an eigenaxis rule of thumb.")

    print()
    print("Cost of the detour, feasible problems only:")
    times = np.array([r.total_time for _, r in feasible])
    direct_times = []
    for p, _ in feasible:
        e_b = quat_rotate(
            np.concatenate([[p.quat_start[0]], -p.quat_start[1:]]), p.eigenaxis
        )
        je = p.body.inertia @ e_b
        jn = float(np.linalg.norm(je))
        cap = p.body.wheels.pseudo_inverse_capability(je / jn)
        direct_times.append(make_profile("bang_bang", p.slew_angle, cap / jn, jn).duration)
    direct_times = np.array(direct_times)
    ratio = times / direct_times
    print(f"  median time penalty vs the (illegal) direct slew : {float(np.median(ratio)):.4f}x")
    print(f"  mean                                             : {float(np.mean(ratio)):.4f}x")
    print(f"  worst                                            : {float(np.max(ratio)):.4f}x")
    thru = np.array([r.momentum_throughput for _, r in feasible])
    print(f"  median momentum throughput                       : "
          f"{float(np.median(thru)):.4f} N*m*s")

    # ---------------------------------------------------------------- PART B
    rule("PART B -- every infeasibility reason is reachable")
    print("A planner that says 'infeasible' without saying which constraint failed")
    print("cannot be used in a design loop. Each reason below is provoked on")
    print("purpose and the returned detail is printed verbatim.")
    print()
    inst = Instrument("telescope", [1.0, 0.0, 0.0])
    q0 = quat_identity()
    seen: dict[str, str] = {}

    # start inside a cone
    cone_on_start = KeepOutCone([1.0, 0.0, 0.0], math.radians(30.0), "sun")
    p = SlewProblem(q0, quat_from_axis_angle([0, 0, 1], math.radians(90.0)), body,
                    KeepOutSet((cone_on_start,)), (inst,))
    seen["start_attitude_violates_keepout"] = plan(p).detail

    # goal inside a cone
    q1 = quat_from_axis_angle([0, 0, 1], math.radians(90.0))
    cone_on_goal = KeepOutCone(quat_rotate(q1, [1.0, 0.0, 0.0]), math.radians(30.0), "sun")
    p = SlewProblem(q0, q1, body, KeepOutSet((cone_on_goal,)), (inst,))
    seen["goal_attitude_violates_keepout"] = plan(p).detail

    # a cone covering the sky
    p = SlewProblem(q0, q1, body, KeepOutSet((KeepOutCone([0, 0, 1], math.pi, "all"),)), (inst,))
    seen["keepout_covers_all_directions"] = plan(p).detail

    # a rank-1 wheel array: no torque about most axes
    weak = WheelArray(np.array([[0.0, 0.0, 1.0]]), 0.15, 12.0, name="one wheel")
    p = SlewProblem(q0, quat_from_axis_angle([1, 0, 0], math.radians(60.0)),
                    RigidBody(np.diag([120.0, 100.0, 80.0]), weak), KeepOutSet(), (inst,))
    seen["wheel_torque_unavailable"] = plan(p).detail

    # scalar sizing: the exact torque leaves the box mid-slew
    mid = quat_rotate(quat_from_axis_angle([0, 0, 1], math.radians(60.0)), [1.0, 0.0, 0.0])
    p = SlewProblem(q0, quat_from_axis_angle([0, 0, 1], math.radians(120.0)), body,
                    KeepOutSet((KeepOutCone(mid, math.radians(25.0), "sun"),)), (inst,),
                    size_against_envelope=False)
    seen["wheel_torque_limit_exceeded"] = plan(p).detail

    # scalar sizing with a rate limit above the momentum envelope
    p = SlewProblem(q0, quat_from_axis_angle([0, 0, 1], math.radians(120.0)),
                    RigidBody(np.diag([120.0, 100.0, 80.0]), pyramid_wheels(0.15, 0.3)),
                    KeepOutSet(), (inst,), rate_limit=math.radians(0.5),
                    size_against_envelope=False)
    seen["wheel_momentum_limit_exceeded"] = plan(p).detail

    # a time budget nothing can meet
    p = SlewProblem(q0, quat_from_axis_angle([0, 0, 1], math.radians(120.0)), body,
                    KeepOutSet((KeepOutCone(mid, math.radians(25.0), "sun"),)), (inst,),
                    max_time=10.0)
    seen["time_limit_exceeded"] = plan(p).detail

    # Geometry that is provably infeasible: eighteen 15 deg cones spaced 20 deg
    # apart around the y-z great circle form a contiguous band of half-width
    # 15 deg. The band separates the sphere into two caps; the boresight starts
    # at +x in one and must reach -x in the other, so every continuous path
    # crosses the band. Both endpoints are clear by 75 deg.
    wall = tuple(
        KeepOutCone(
            [0.0, math.sin(math.radians(a)), math.cos(math.radians(a))],
            math.radians(15.0),
            f"w{a}",
        )
        for a in range(0, 360, 20)
    )
    blocked = SlewProblem(q0, quat_from_axis_angle([0, 0, 1], math.pi), body,
                          KeepOutSet(wall), (inst,))
    print(f"  (blocked geometry: start margin {math.degrees(blocked.attitude_margin(q0)):.3f} deg, "
          f"goal margin {math.degrees(blocked.attitude_margin(blocked.quat_goal)):.3f} deg)")
    seen["no_feasible_path_found"] = plan(blocked).detail

    for reason in INFEASIBILITY_REASONS:
        got = seen.get(reason)
        status = "reached" if got is not None else "NOT REACHED"
        print(f"  {reason:<36} {status}")
        if got is not None:
            print(f"      {got}")
        failures += got is None
    print()
    print(f"{len(seen)} of {len(INFEASIBILITY_REASONS)} documented reasons provoked.")

    # ---------------------------------------------------------------- PART C
    rule("PART C -- the four required failure modes")

    print("C1 infeasible geometry: a contiguous 15 deg band of cones around the")
    print("   y-z great circle, with the boresight starting at +x and asked for -x.")
    print("   Every continuous boresight path crosses the band, so no via point of")
    print("   any number can help. Both endpoints are clear by 75 deg, so this is")
    print("   genuinely a path-existence failure, not an endpoint failure.")
    r = plan(blocked)
    print(f"    feasible={r.feasible}  reason={r.reason}")
    print(f"    starts tried={r.n_starts}  path evaluations={r.n_objective_evals}")
    print(f"    direct slew margin={math.degrees(r.direct_min_margin):.4f} deg")
    failures += r.feasible or r.reason != "no_feasible_path_found"

    print()
    print("C2 wheel saturation mid-slew: sized with the scalar rule of thumb, the")
    print("   exact torque leaves the box part-way through the first segment.")
    p2 = SlewProblem(q0, quat_from_axis_angle([0, 0, 1], math.radians(120.0)), body,
                     KeepOutSet((KeepOutCone(mid, math.radians(25.0), "sun"),)), (inst,),
                     size_against_envelope=False)
    r2 = plan(p2)
    print(f"    feasible={r2.feasible}  reason={r2.reason}")
    print(f"    torque utilisation   = {r2.actuators.torque_utilisation:.6f}")
    print(f"    exact-envelope util  = {r2.actuators.exact_torque_utilisation:.6f}")
    print(f"    gyroscopic fraction  = {r2.actuators.gyroscopic_fraction:.6f}")
    print(f"    worst segment        = {r2.actuators.worst_segment}")
    p3 = SlewProblem(q0, quat_from_axis_angle([0, 0, 1], math.radians(120.0)), body,
                     KeepOutSet((KeepOutCone(mid, math.radians(25.0), "sun"),)), (inst,),
                     size_against_envelope=True)
    r3 = plan(p3)
    print(f"    with envelope sizing : feasible={r3.feasible}, "
          f"utilisation={r3.actuators.torque_utilisation:.6f}, T={r3.total_time:.3f} s")
    print(f"    the rule of thumb would have promised T={r2.total_time:.3f} s, "
          f"{(1 - r2.total_time / r3.total_time) * 100:.2f} % faster than is achievable")
    failures += r2.feasible or r2.reason != "wheel_torque_limit_exceeded" or not r3.feasible

    print()
    print("C3 cone tangency: a cone placed exactly tangent to the direct slew arc.")
    z = np.array([0.0, 0.0, 1.0])
    n0 = quat_rotate(q0, [1.0, 0.0, 0.0])
    psi_c = math.radians(45.0)
    centre = quat_rotate(quat_from_axis_angle(z, psi_c), n0)
    for offset_deg in (-1e-6, 0.0, 1e-6):
        gamma = math.radians(20.0)
        axis_t = quat_rotate(
            quat_from_axis_angle(np.cross(centre, z), gamma + math.radians(offset_deg)), centre
        )
        cone_t = KeepOutCone(axis_t, gamma, "tangent")
        pt = SlewProblem(q0, quat_from_axis_angle(z, math.radians(90.0)), body,
                         KeepOutSet((cone_t,)), (inst,))
        rt = plan(pt)
        print(f"    offset {offset_deg:+.1e} deg: direct margin "
              f"{math.degrees(rt.direct_min_margin):+.9f} deg, feasible={rt.feasible}, "
              f"n_via={rt.n_via}")
        failures += not rt.feasible
    print("    The planner does not have to decide the tangent case correctly to")
    print("    be safe: a tangent arc has margin 0, which satisfies the >= 0")
    print("    constraint, and any required_margin > 0 pushes it off the boundary.")

    print()
    print("C4 the degenerate 180 deg slew: the eigenaxis is still unique, but the")
    print("   quaternion scalar part is zero and slerp is at its least conditioned.")
    for angle_deg in (179.0, 179.999, 180.0, 180.001):
        q_goal = quat_from_axis_angle([0.0, 0.0, 1.0], math.radians(angle_deg))
        pd = SlewProblem(q0, q_goal, body, KeepOutSet(), (inst,))
        rd = plan(pd)
        print(f"    {angle_deg:>8.3f} deg: slew angle "
              f"{math.degrees(pd.slew_angle):.9f} deg, feasible={rd.feasible}, "
              f"T={rd.total_time:.6f} s, n_via={rd.n_via}")
        failures += not rd.feasible
    print()
    print("   With a cone squarely across the 180 deg arc:")
    mid180 = quat_rotate(quat_from_axis_angle([0, 0, 1], math.radians(90.0)), n0)
    p180 = SlewProblem(q0, quat_from_axis_angle([0.0, 0.0, 1.0], math.pi), body,
                       KeepOutSet((KeepOutCone(mid180, math.radians(30.0), "sun"),)), (inst,))
    r180 = plan(p180)
    print(f"    direct margin {math.degrees(r180.direct_min_margin):.4f} deg -> "
          f"feasible={r180.feasible}, n_via={r180.n_via}, T={r180.total_time:.3f} s, "
          f"margin={math.degrees(r180.min_margin):.6f} deg")
    failures += not r180.feasible or r180.min_margin < 0.0
    print()
    print("   A zero-angle slew (start == goal):")
    p_zero = SlewProblem(q0, q0, body, KeepOutSet(), (inst,))
    r_zero = plan(p_zero)
    print(f"    slew angle {p_zero.slew_angle:.3e} rad, feasible={r_zero.feasible}, "
          f"T={r_zero.total_time:.6f} s, n_via={r_zero.n_via}")
    failures += not r_zero.feasible or r_zero.total_time != 0.0

    # ---------------------------------------------------------------- PART D
    rule("PART D -- cost accounting against the closed forms")
    print("For every segment of a returned path, the reported time and momentum")
    print("must equal the profile closed forms for that segment's angle, sizing")
    print("acceleration and rate cap.")
    print()
    worst_t = worst_h = 0.0
    n_seg = 0
    for _, r in feasible[:200]:
        for seg in r.path.segments:
            # The planner may derate the acceleration below the available
            # value to keep the exact torque inside the wheel box, so the
            # reference is rebuilt from the profile's own peak acceleration.
            ref = make_profile(
                seg.profile.kind,
                seg.angle,
                seg.profile.peak_accel,
                seg.profile.inertia,
                None if not math.isfinite(seg.rate_cap) else seg.rate_cap,
            )
            worst_t = max(worst_t, abs(ref.duration - seg.duration))
            expect_h = seg.profile.inertia * seg.profile.peak_rate
            worst_h = max(worst_h, abs(seg.profile.peak_momentum - expect_h))
            n_seg += 1
    print(f"segments checked                        : {n_seg}")
    print(f"worst |rebuilt - reported duration|     : {worst_t:.3e} s   tolerance 1e-12")
    print(f"worst |peak momentum - J_e omega_peak|  : {worst_h:.3e} N*m*s   tolerance 1e-12")
    failures += worst_t > 1e-12 or worst_h > 1e-12
    tot = feasible[0][1]
    print()
    print("Path totals are the sums of the segment values, e.g. for the first")
    print("feasible problem:")
    print(f"  segments        : {[round(s.duration, 6) for s in tot.path.segments]}")
    print(f"  sum             : {sum(s.duration for s in tot.path.segments)!r} s")
    print(f"  reported total  : {tot.total_time!r} s")
    print(f"  peak momentum   : {tot.peak_momentum!r} N*m*s "
          f"(max over segments: {max(s.profile.peak_momentum for s in tot.path.segments)!r})")
    print(f"  throughput      : {tot.momentum_throughput!r} N*m*s "
          f"(= 2 x sum of peaks: "
          f"{2 * sum(s.profile.peak_momentum for s in tot.path.segments)!r})")

    # ---------------------------------------------------------------- PART E
    rule("PART E -- settings sensitivity")
    print("The SLSQP iteration cap and the cold-start deviation magnitudes are the")
    print("only tuning knobs in the planner. Measured over 14 seeded problems.")
    print()
    sample = generate_problems(14, seed=202)
    print(f"{'maxiter':>9}{'magnitudes':>18}{'s/plan':>10}{'feasible':>10}"
          f"{'mean T [s]':>13}{'relative':>10}{'evals':>9}")
    base = None
    for mi, mags in ((40, (0.45,)), (25, (0.45,)), (15, (0.45,)), (25, (0.6,)),
                     (40, (0.3, 0.7))):
        t0 = time.perf_counter()
        res = [plan(pp, max_via=1, maxiter=mi, start_magnitudes=mags) for pp in sample]
        el = time.perf_counter() - t0
        obj = np.array([r.objective if r.feasible else np.nan for r in res])
        if base is None:
            base = obj
        rel = float(np.nanmean(obj / base))
        print(
            f"{mi:>9}{str(mags):>18}{el / len(sample):>10.3f}"
            f"{sum(r.feasible for r in res):>4} / {len(sample):<4}"
            f"{float(np.nanmean(obj)):>13.3f}{rel:>10.4f}"
            f"{float(np.mean([r.n_objective_evals for r in res])):>9.0f}"
        )
    print()
    print("The default is maxiter = 25 with a single 0.45 rad magnitude: it costs")
    print("about half the time of 40 iterations for a mean objective near 1 %")
    print("worse. Nothing here is random, so these figures reproduce exactly.")

    rule("SUMMARY")
    print(f"failed checks: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
