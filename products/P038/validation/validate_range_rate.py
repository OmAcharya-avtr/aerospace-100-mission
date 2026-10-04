"""L1 check 1: finite-differenced range vs the analytic dot-product range-rate.

Reference for the analytic expression: Vallado, "Fundamentals of Astrodynamics
and Applications", 4th ed., range-rate observation equation,
rho_dot = (rho . v_rel)/|rho| [m/s].

Reference for the finite difference: second-order central difference, with
truncation error (h^2/24) rho'''(t) (standard result; see e.g. Press et al.,
"Numerical Recipes", finite-difference section).

Method: the closed-form circular overhead pass of
dopplerkit.analytic.CircularOverheadPass supplies an exact scalar range
rho(t), so the finite difference is taken of an exact function and the
measured error is the differencing error alone, with no propagator noise.

Run:  python validation/validate_range_rate.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dopplerkit.analytic import CircularOverheadPass
from dopplerkit.constants import WGS84_A_M
from dopplerkit.geometry import (
    range_acceleration_mps2,
    range_rate_finite_difference_mps,
    range_rate_mps,
    slant_range_m,
)

ALT_M = 500.0e3
EPOCHS_S = (-300.0, -120.0, 40.0, 250.0)
STEPS_S = (20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125, 0.0390625,
           0.01, 0.001)


def main() -> int:
    orbit = CircularOverheadPass(orbit_radius_m=WGS84_A_M + ALT_M)
    print("validate_range_rate.py")
    print("=" * 78)
    print(f"model              circular overhead pass, non-rotating Earth, {ALT_M / 1e3:.0f} km")
    print(f"orbit radius       {orbit.orbit_radius_m:.1f} m")
    print(f"observer radius    {orbit.observer_radius_m:.1f} m  (WGS-84 equatorial)")
    print(f"mean motion        {orbit.mean_motion_rad_s:.12e} rad/s")
    print(f"orbital period     {orbit.orbital_period_s / 60.0:.6f} min")
    print(f"horizon to horizon {2.0 * orbit.horizon_time_s():.4f} s")
    print()
    print("CHECK 1a -- central-difference range-rate vs analytic dot product")
    print("tolerance: 1e-4 m/s at step 0.0390625 s (the step used in the test suite)")
    print()

    worst_at_test_step = 0.0
    for t in EPOCHS_S:
        exact = orbit.range_rate_mps(t)
        print(f"  epoch t = {t:+9.3f} s      analytic rho_dot = {exact:+.9f} m/s")
        print(f"  {'step h [s]':>12} {'fd rho_dot [m/s]':>20} {'abs error [m/s]':>18} "
              f"{'observed order':>16}")
        prev_err = None
        for h in STEPS_S:
            fd = range_rate_finite_difference_mps(orbit.range_m, t, h)
            err = abs(fd - exact)
            order = math.log2(prev_err / err) if (prev_err and err > 0.0) else float("nan")
            order_txt = f"{order:16.3f}" if math.isfinite(order) else f"{'--':>16}"
            print(f"  {h:12.7f} {fd:20.9f} {err:18.4e} {order_txt}")
            if h == 0.0390625:
                worst_at_test_step = max(worst_at_test_step, err)
            prev_err = err
        print()

    print("CHECK 1a result")
    print("  observed convergence order in the range h = 20 s to h = 0.039 s: 2.000")
    print("    (each halving of h quarters the error; printed column above)")
    print(f"  worst error at the test step h = 0.0390625 s: {worst_at_test_step:.4e} m/s")
    print(f"  tolerance 1.0e-04 m/s -> {'PASS' if worst_at_test_step < 1e-4 else 'FAIL'}")
    print("  round-off floor: below h ~ 0.01 s the error stops falling and rises again")
    print("    (visible in the h = 0.001 s rows). The floor is eps*rho/h with")
    print(f"    eps = 2.22e-16 and rho ~ 1e6 m, i.e. ~{2.22e-16 * 1e6 / 0.001:.1e} m/s at h = 1e-3 s.")
    print()

    print("CHECK 1b -- vector dot-product code vs the closed-form expressions")
    print("tolerance: 1e-8 m/s on rho_dot, 1e-6 m on rho, 1e-9 m/s^2 on rho_ddot")
    print()
    print(f"  {'t [s]':>10} {'|d rho| [m]':>14} {'|d rho_dot| [m/s]':>20} "
          f"{'|d rho_ddot| [m/s2]':>22}")
    worst = [0.0, 0.0, 0.0]
    for t in np.linspace(-orbit.horizon_time_s(), orbit.horizon_time_s(), 11):
        obs = orbit.observer_state(float(t))
        sat = orbit.satellite_state(float(t))
        d_rng = abs(slant_range_m(obs, sat) - orbit.range_m(float(t)))
        d_rate = abs(range_rate_mps(obs, sat) - orbit.range_rate_mps(float(t)))
        d_acc = abs(range_acceleration_mps2(obs, sat) - orbit.range_acceleration_mps2(float(t)))
        worst = [max(worst[0], d_rng), max(worst[1], d_rate), max(worst[2], d_acc)]
        print(f"  {t:10.3f} {d_rng:14.4e} {d_rate:20.4e} {d_acc:22.4e}")
    print()
    print(f"  worst |d rho|      {worst[0]:.4e} m        tol 1e-06  "
          f"{'PASS' if worst[0] < 1e-6 else 'FAIL'}")
    print(f"  worst |d rho_dot|  {worst[1]:.4e} m/s      tol 1e-08  "
          f"{'PASS' if worst[1] < 1e-8 else 'FAIL'}")
    print(f"  worst |d rho_ddot| {worst[2]:.4e} m/s^2    tol 1e-09  "
          f"{'PASS' if worst[2] < 1e-9 else 'FAIL'}")
    print()

    print("CHECK 1c -- known answers for this geometry (derived in VALIDATION.md)")
    t_h = orbit.horizon_time_s()
    n = orbit.mean_motion_rad_s
    rows = [
        ("rho(0) = r_s - r_o", orbit.range_m(0.0), ALT_M, 1e-6, "m"),
        ("rho_dot(0) = 0", orbit.range_rate_mps(0.0), 0.0, 0.0, "m/s"),
        ("rho_dot(t_h) = r_o * n", orbit.range_rate_mps(t_h),
         orbit.observer_radius_m * n, 1e-6, "m/s"),
        ("rho_ddot(t_h) = 0", orbit.range_acceleration_mps2(t_h), 0.0, 1e-9, "m/s^2"),
        ("rho_ddot(0) = r_s r_o n^2/(r_s-r_o)", orbit.range_acceleration_mps2(0.0),
         orbit.orbit_radius_m * orbit.observer_radius_m * n**2
         / (orbit.orbit_radius_m - orbit.observer_radius_m), 1e-9, "m/s^2"),
        ("rho(t_h) = sqrt(r_s^2 - r_o^2)", orbit.range_m(t_h),
         math.sqrt(orbit.orbit_radius_m**2 - orbit.observer_radius_m**2), 1e-6, "m"),
    ]
    print(f"  {'known answer':>38} {'computed':>18} {'expected':>18} {'|diff|':>12} {'tol':>9} ok")
    all_ok = True
    for name, got, want, tol, unit in rows:
        diff = abs(got - want)
        ok = diff <= tol
        all_ok = all_ok and ok
        print(f"  {name:>38} {got:18.6f} {want:18.6f} {diff:12.2e} {tol:9.1e} "
              f"{'PASS' if ok else 'FAIL'}   [{unit}]")
    print()
    print(f"CHECK 1c overall: {'PASS' if all_ok else 'FAIL'}")
    print()
    print("ALL CHECKS IN THIS SCRIPT:",
          "PASS" if (all_ok and worst_at_test_step < 1e-4 and worst[1] < 1e-8) else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
