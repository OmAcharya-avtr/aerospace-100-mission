"""Eigenaxis rest-to-rest timing: closed forms, arithmetic, and simulation.

PART A  bang-bang time against hand arithmetic for a known inertia and torque
PART B  the three profiles' closed forms against numerical integration
PART C  the momentum identity  integral |tau| dt = 2 J_e omega_peak
PART D  open-loop simulation of a principal-axis slew reaches the goal attitude
PART E  the gyroscopic term the scalar sizing model drops, quantified

Run: ``python validation/validate_eigenaxis_time.py``
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slewforge.attitude import (  # noqa: E402
    quat_angle,
    quat_from_axis_angle,
    quat_identity,
)
from slewforge.dynamics import (  # noqa: E402
    RigidBody,
    eigenaxis_torque,
    propagate,
    simulate_profile,
)
from slewforge.profiles import bang_bang_profile, make_profile, smoothed_profile  # noqa: E402
from slewforge.wheels import pyramid_wheels  # noqa: E402


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    failures = 0

    # ---------------------------------------------------------------- PART A
    rule("PART A -- bang-bang rest-to-rest time against hand arithmetic")
    j_e = 100.0  # kg m^2
    tau = 0.15  # N*m
    delta = math.pi / 2  # rad (90 deg)
    alpha = tau / j_e
    print(f"effective inertia J_e            = {j_e} kg m^2")
    print(f"available torque  tau            = {tau} N*m")
    print(f"slew angle        Delta          = {delta!r} rad = 90 deg")
    print(f"peak acceleration alpha = tau/J_e= {alpha!r} rad/s^2")
    print()
    print("Wie (2008) Sec. 5.3: a rest-to-rest bang-bang slew accelerates at")
    print("alpha for T/2 then decelerates, so Delta = alpha (T/2)^2 and")
    print("T = 2 sqrt(Delta / alpha).")
    print()
    ratio = delta / alpha
    root = math.sqrt(ratio)
    t_hand = 2.0 * root
    print(f"  Delta / alpha                  = {ratio!r} s^2")
    print(f"  sqrt(Delta / alpha)            = {root!r} s")
    print(f"  T = 2 sqrt(Delta / alpha)      = {t_hand!r} s")
    prof = bang_bang_profile(delta, alpha, j_e)
    print(f"  library duration               = {prof.duration!r} s")
    diff = abs(prof.duration - t_hand)
    print(f"  |difference|                   = {diff:.3e} s   tolerance 1e-12")
    failures += diff > 1e-12

    w_hand = math.sqrt(delta * alpha)
    h_hand = j_e * w_hand
    print()
    print(f"  omega_peak = sqrt(Delta alpha) = {w_hand!r} rad/s = {math.degrees(w_hand):.10f} deg/s")
    print(f"  library peak_rate              = {prof.peak_rate!r} rad/s")
    print(f"  h_peak = J_e omega_peak        = {h_hand!r} N*m*s")
    print(f"  library peak_momentum          = {prof.peak_momentum!r} N*m*s")
    d2 = max(abs(prof.peak_rate - w_hand), abs(prof.peak_momentum - h_hand))
    print(f"  worst |difference|             = {d2:.3e}   tolerance 1e-12")
    failures += d2 > 1e-12

    print()
    print("Rate-limited (trapezoidal): T = Delta/omega_max + omega_max/alpha")
    w_lim = math.radians(1.5)
    t_tr_hand = delta / w_lim + w_lim / alpha
    tr = bang_bang_profile(delta, alpha, j_e, rate_limit=w_lim)
    print(f"  omega_max = 1.5 deg/s          = {w_lim!r} rad/s")
    print(f"  hand T                         = {t_tr_hand!r} s")
    print(f"  library T                      = {tr.duration!r} s  kind={tr.kind}")
    d3 = abs(tr.duration - t_tr_hand)
    print(f"  |difference|                   = {d3:.3e} s   tolerance 1e-12")
    failures += d3 > 1e-12

    print()
    print("Smoothed (sinusoidal acceleration): T = sqrt(2 pi Delta / alpha)")
    t_sm_hand = math.sqrt(2.0 * math.pi * delta / alpha)
    sm = smoothed_profile(delta, alpha, j_e)
    print(f"  hand T                         = {t_sm_hand!r} s")
    print(f"  library T                      = {sm.duration!r} s")
    print(f"  penalty vs bang-bang           = {sm.duration / prof.duration!r}")
    print(f"  sqrt(2 pi) / 2                 = {math.sqrt(2.0 * math.pi) / 2.0!r}")
    d4 = abs(sm.duration - t_sm_hand)
    print(f"  |difference|                   = {d4:.3e} s   tolerance 1e-12")
    failures += d4 > 1e-12

    # ---------------------------------------------------------------- PART B
    rule("PART B -- closed forms against numerical integration of the profile")
    print(f"{'profile':<14}{'T [s]':>14}{'int psi_dot dt':>20}{'Delta':>16}{'|err|':>12}")
    for kind in ("bang_bang", "trapezoidal", "smoothed"):
        p = make_profile(kind, delta, alpha, j_e, w_lim if kind == "trapezoidal" else None)
        swept, _ = quad(
            lambda t, p=p: float(p.rate_at(t)), 0.0, p.duration, limit=400, points=[p.duration / 2]
        )
        err = abs(swept - delta)
        print(f"{p.kind:<14}{p.duration:>14.9f}{swept:>20.12f}{delta:>16.12f}{err:>12.3e}")
        failures += err > 1e-8
    print("tolerance 1e-8 rad on the swept angle")

    print()
    print("Terminal rate is zero and the profile is monotone in angle:")
    for kind in ("bang_bang", "trapezoidal", "smoothed"):
        p = make_profile(kind, delta, alpha, j_e, w_lim if kind == "trapezoidal" else None)
        t = np.linspace(0.0, p.duration, 4001)
        ang = np.asarray(p.angle_at(t))
        rate = np.asarray(p.rate_at(t))
        mono = float(np.min(np.diff(ang)))
        print(
            f"  {p.kind:<12} psi(T)-Delta = {float(ang[-1]) - delta:+.3e} rad, "
            f"psi_dot(0) = {float(rate[0]):.3e}, psi_dot(T) = {float(rate[-1]):.3e}, "
            f"min d(psi) = {mono:+.3e}"
        )
        failures += abs(float(ang[-1]) - delta) > 1e-9 or mono < -1e-15

    # ---------------------------------------------------------------- PART C
    rule("PART C -- momentum identity: integral |tau| dt = 2 J_e omega_peak")
    print(f"{'profile':<14}{'int |J_e psi_ddot| dt':>24}{'2 J_e omega_peak':>20}{'|err|':>12}")
    for kind in ("bang_bang", "trapezoidal", "smoothed"):
        p = make_profile(kind, delta, alpha, j_e, w_lim if kind == "trapezoidal" else None)
        integral, _ = quad(
            lambda t, p=p: j_e * abs(float(p.accel_at(t))),
            0.0,
            p.duration,
            limit=500,
            points=[p.duration / 2],
        )
        err = abs(integral - p.momentum_throughput)
        print(f"{p.kind:<14}{integral:>24.12f}{p.momentum_throughput:>20.12f}{err:>12.3e}")
        failures += err > 1e-6
    print("tolerance 1e-6 N*m*s")

    # ---------------------------------------------------------------- PART D
    rule("PART D -- open-loop simulation of a principal-axis slew")
    body = RigidBody(np.diag([120.0, 100.0, 80.0]), pyramid_wheels(0.15, 12.0))
    axis = np.array([0.0, 0.0, 1.0])  # principal axis of the diagonal inertia
    j_axis = body.effective_inertia(axis)
    a_axis = 0.15 / j_axis
    p = bang_bang_profile(delta, a_axis, j_axis)
    q0 = quat_identity()
    q_goal = quat_from_axis_angle(axis, delta)
    print(f"inertia diag(120, 100, 80) kg m^2, eigenaxis {axis.tolist()} (principal)")
    print(f"J_e = {j_axis} kg m^2, alpha = {a_axis!r} rad/s^2, T = {p.duration!r} s")
    print()
    print()
    print("D1 -- smoothed profile (continuous torque): clean RK4 convergence")
    psm = smoothed_profile(delta, a_axis, j_axis)
    print(f"{'dt [s]':>10}{'attitude error [rad]':>24}{'error ratio':>16}")
    prev = None
    for dt in (0.4, 0.2, 0.1, 0.05):
        sim = simulate_profile(body, q0, axis, psm, dt)
        err = quat_angle(sim.quat[-1], q_goal)
        ratio = "" if prev is None or err <= 0.0 else f"{prev / err:.3f}"
        print(f"{dt:>10.3f}{err:>24.6e}{ratio:>16}")
        prev = err
    sim = simulate_profile(body, q0, axis, psm, 0.05)
    final = quat_angle(sim.quat[-1], q_goal)
    print(f"final attitude error at dt = 0.05 s: {final:.6e} rad "
          f"= {math.degrees(final) * 3600:.6f} arcsec   tolerance 1e-6 rad")
    failures += final > 1e-6

    print()
    print("D2 -- bang-bang: what phase splitting is worth")
    print("The commanded torque jumps at T/2. A fixed-step RK4 that steps across")
    print("that instant weights the pre-switch acceleration into a post-switch step")
    print("and drops to first order. simulate_profile() therefore integrates each")
    print("phase separately (SlewProfile.switch_times). The naive single-interval")
    print("integration is run here for contrast; nothing in the library uses it.")
    print()
    print(f"{'dt [s]':>10}{'split [rad]':>18}{'ratio':>9}{'single call [rad]':>20}{'ratio':>9}")
    prev_s = prev_n = None
    for dt in (0.4, 0.2, 0.1, 0.05):
        sim = simulate_profile(body, q0, axis, p, dt)
        err_s = quat_angle(sim.quat[-1], q_goal)
        je_b = body.inertia @ axis
        naive = propagate(
            body,
            q0,
            np.zeros(3),
            p.duration,
            dt,
            torque_fn=lambda t, p=p, je_b=je_b: float(p.accel_at(min(t, p.duration))) * je_b,
        )
        err_n = quat_angle(naive.quat[-1], q_goal)
        rs = "" if prev_s is None or err_s <= 0.0 else f"{prev_s / err_s:.2f}"
        rn = "" if prev_n is None or err_n <= 0.0 else f"{prev_n / err_n:.2f}"
        print(f"{dt:>10.3f}{err_s:>18.6e}{rs:>9}{err_n:>20.6e}{rn:>9}")
        prev_s, prev_n = err_s, err_n
    sim = simulate_profile(body, q0, axis, p, 0.05)
    final_bb = quat_angle(sim.quat[-1], q_goal)
    print(f"\nphase-split bang-bang error at dt = 0.05 s: {final_bb:.6e} rad"
          f"   tolerance 1e-9 rad")
    failures += final_bb > 1e-9
    print(f"peak PER-WHEEL momentum simulated  : {np.max(np.abs(sim.wheel_momentum)):.9f} N*m*s")
    print(f"analytic BODY momentum J_e omega   : {p.peak_momentum:.9f} N*m*s")
    print(f"ratio                              : "
          f"{p.peak_momentum / float(np.max(np.abs(sim.wheel_momentum))):.9f}")
    print("The four pyramid wheels share the body momentum, and for a +z slew the")
    print("minimum-norm allocation puts 1 / (4 cos 54.7356 deg) = 0.4330 of it on")
    print("each wheel, so 4.3416 / 1.8800 = 2.3094 = 4 cos(54.7356 deg).")
    print(f"any wheel saturation flagged       : {sim.any_saturation}")

    # ---------------------------------------------------------------- PART E
    rule("PART E -- the gyroscopic term the scalar sizing model drops")
    print("tau_exact = psi_ddot J e + psi_dot^2 (e x J e);  the scalar model keeps")
    print("only the first term with |J e| replaced by e^T J e. The second term")
    print("vanishes if and only if e is a principal axis.")
    print()
    print(
        f"{'eigenaxis':<26}{'J_e':>9}{'|J e|':>10}{'|tau|max/J_e alpha':>22}"
        f"{'gyro fraction':>16}"
    )
    for name, e in [
        ("z (principal)", np.array([0.0, 0.0, 1.0])),
        ("x (principal)", np.array([1.0, 0.0, 0.0])),
        ("(1,1,0)/sqrt2", np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)),
        ("(1,1,1)/sqrt3", np.ones(3) / math.sqrt(3.0)),
        ("(3,1,2)/|.|", np.array([3.0, 1.0, 2.0]) / math.sqrt(14.0)),
    ]:
        je = body.inertia @ e
        j_eff = float(e @ je)
        a = 0.15 / j_eff
        pr = bang_bang_profile(delta, a, j_eff)
        t = np.linspace(0.0, pr.duration, 2001)
        tau_v = eigenaxis_torque(
            body.inertia, e, np.asarray(pr.rate_at(t)), np.asarray(pr.accel_at(t))
        )
        gyro = np.cross(e, je)[None, :] * (np.asarray(pr.rate_at(t)) ** 2)[:, None]
        mag = np.linalg.norm(tau_v, axis=1)
        scalar_mag = j_eff * a
        frac = np.max(np.linalg.norm(gyro, axis=1) / np.maximum(mag, 1e-300))
        print(
            f"{name:<26}{j_eff:>9.3f}{float(np.linalg.norm(je)):>10.3f}"
            f"{float(np.max(mag)) / scalar_mag:>22.6f}{frac:>16.6f}"
        )
    print()
    print("A slew about (1,1,0)/sqrt2 of this spacecraft therefore needs more")
    print("torque than the rule of thumb budgets, at the middle of the slew,")
    print("exactly where the rate is highest.")

    rule("SUMMARY")
    print(f"failed checks: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
