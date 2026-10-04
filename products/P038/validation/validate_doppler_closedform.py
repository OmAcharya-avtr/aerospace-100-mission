"""L1 checks 2 and 3: closed-form Doppler, and the zero crossing at closest approach.

Check 2 reference: the classical (non-relativistic) Doppler relation
f_rx = f_tx (1 - rho_dot/c), giving Delta_f = -f_c rho_dot / c [Hz], first
order in v/c. Geometry from Vallado, "Fundamentals of Astrodynamics and
Applications", 4th ed., two-body circular orbit.

Closed form under test, for a satellite passing through the zenith of an
inertially fixed observer (derivation in VALIDATION.md):

    Delta_f(t) = -f_c r_s r_o n sin(n t) / (c sqrt(r_s^2 + r_o^2 - 2 r_s r_o cos n t))

Check 3: the Doppler zero crossing must coincide with the time of closest
approach. The two are located by independent numerical methods -- Brent root
finding on the range-rate, and bounded Brent minimisation of the range -- so
that agreement is evidence rather than a tautology. The result is also
compared against the sampling step of a coarse profile, which is the
"to within the integration step" form the specification asks for.

Run:  python validation/validate_doppler_closedform.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dopplerkit.analytic import CircularOverheadPass
from dopplerkit.constants import C_M_S, WGS84_A_M
from dopplerkit.doppler import one_way_doppler_hz
from dopplerkit.passprofile import (
    compute_profile,
    doppler_zero_crossing_s,
    time_of_closest_approach_s,
)

ALT_M = 500.0e3
CARRIERS_HZ = (401.0e6, 2.2e9, 8.4e9, 26.0e9)


def closed_form_doppler_hz(orbit: CircularOverheadPass, t: float, carrier_hz: float) -> float:
    """The closed form written out independently of the library code."""
    n = orbit.mean_motion_rad_s
    r_s = orbit.orbit_radius_m
    r_o = orbit.observer_radius_m
    rng = np.sqrt(r_s**2 + r_o**2 - 2.0 * r_s * r_o * np.cos(n * t))
    return float(-carrier_hz * r_s * r_o * n * np.sin(n * t) / (C_M_S * rng))


def main() -> int:
    orbit = CircularOverheadPass(orbit_radius_m=WGS84_A_M + ALT_M)
    horizon = orbit.horizon_time_s()
    print("validate_doppler_closedform.py")
    print("=" * 78)
    print(f"model              circular overhead pass, non-rotating Earth, {ALT_M / 1e3:.0f} km")
    print(f"mean motion        {orbit.mean_motion_rad_s:.12e} rad/s")
    print(f"orbital speed      {orbit.orbital_speed_mps:.6f} m/s")
    print(f"horizon time       {horizon:.6f} s either side of the zenith")
    print("convention         Doppler positive (up-shift) when approaching;")
    print("                   Delta_f = -f_c rho_dot / c, rho_dot positive when receding")
    print()

    print("CHECK 2 -- library Doppler vs the closed form, four carriers")
    print("tolerance: 1e-9 relative")
    print()
    worst_rel = 0.0
    for carrier in CARRIERS_HZ:
        print(f"  carrier {carrier / 1e9:9.4f} GHz")
        print(f"  {'t [s]':>10} {'library [Hz]':>18} {'closed form [Hz]':>18} {'rel diff':>12}")
        for t in np.linspace(-horizon, horizon, 9):
            lib = orbit.one_way_doppler_hz(float(t), carrier)
            ref = closed_form_doppler_hz(orbit, float(t), carrier)
            denom = max(abs(ref), 1.0)
            rel = abs(lib - ref) / denom
            worst_rel = max(worst_rel, rel)
            print(f"  {t:10.3f} {lib:18.6f} {ref:18.6f} {rel:12.2e}")
        print()
    print(f"  worst relative difference {worst_rel:.3e}   tol 1.0e-09  "
          f"{'PASS' if worst_rel < 1e-9 else 'FAIL'}")
    print()

    print("CHECK 2b -- peak Doppler against the hand expression -f_c r_o n / c")
    print("  (at the geometric horizon rho_dot collapses to exactly r_o * n; see VALIDATION.md)")
    print()
    print(f"  {'carrier [GHz]':>14} {'computed [Hz]':>18} {'hand [Hz]':>18} {'|diff| [Hz]':>14}")
    peak_ok = True
    for carrier in CARRIERS_HZ:
        got = orbit.one_way_doppler_hz(-horizon, carrier)
        hand = carrier * orbit.observer_radius_m * orbit.mean_motion_rad_s / C_M_S
        diff = abs(got - hand)
        peak_ok = peak_ok and diff < 1e-6 * max(abs(hand), 1.0)
        print(f"  {carrier / 1e9:14.4f} {got:18.6f} {hand:18.6f} {diff:14.2e}")
    print(f"  tolerance 1e-6 relative -> {'PASS' if peak_ok else 'FAIL'}")
    print()

    print("CHECK 2c -- Doppler rate at closest approach against the hand expression")
    print("  d(Delta_f)/dt at TCA = -f_c r_s r_o n^2 / (c (r_s - r_o))")
    print()
    n = orbit.mean_motion_rad_s
    rho_ddot_tca = (orbit.orbit_radius_m * orbit.observer_radius_m * n**2
                    / (orbit.orbit_radius_m - orbit.observer_radius_m))
    rate_ok = True
    print(f"  {'carrier [GHz]':>14} {'computed [Hz/s]':>18} {'hand [Hz/s]':>18} {'|diff|':>12}")
    for carrier in CARRIERS_HZ:
        got = orbit.max_doppler_rate_hz_per_s(carrier)
        hand = -carrier * rho_ddot_tca / C_M_S
        diff = abs(got - hand)
        rate_ok = rate_ok and diff < 1e-9 * abs(hand)
        print(f"  {carrier / 1e9:14.4f} {got:18.6f} {hand:18.6f} {diff:12.2e}")
    print(f"  rho_ddot at TCA = {rho_ddot_tca:.9f} m/s^2")
    print(f"  tolerance 1e-9 relative -> {'PASS' if rate_ok else 'FAIL'}")
    print()

    print("CHECK 3 -- Doppler zero crossing vs time of closest approach")
    print("two independent methods: brentq on rho_dot, bounded Brent minimisation of rho")
    print()
    tca = time_of_closest_approach_s(
        orbit.observer_state, orbit.satellite_state, -horizon, horizon, xatol_s=1e-6
    )
    zero = doppler_zero_crossing_s(
        orbit.observer_state, orbit.satellite_state, -horizon, horizon, xtol_s=1e-12
    )
    print(f"  time of closest approach (minimise rho)   {tca:+.9f} s")
    print(f"  Doppler zero crossing (root of rho_dot)   {zero:+.9f} s")
    print(f"  |difference|                              {abs(tca - zero):.3e} s")
    print("  minimiser xatol                           1.0e-06 s")
    print(f"  tolerance: the minimiser's own xatol -> "
          f"{'PASS' if abs(tca - zero) <= 1e-6 else 'FAIL'}")
    print()
    print("  Same comparison against a sampled profile's integration step:")
    print(f"  {'step [s]':>10} {'nearest-sample crossing [s]':>30} {'|diff| vs TCA [s]':>20} ok")
    step_ok = True
    for step in (30.0, 10.0, 1.0, 0.1):
        times = np.arange(-horizon, horizon + 0.5 * step, step)
        profile = compute_profile(
            orbit.observer_state, orbit.satellite_state, times, 2.2e9
        )
        idx = int(np.argmin(np.abs(profile.doppler_hz)))
        diff = abs(float(times[idx]) - tca)
        ok = diff <= step
        step_ok = step_ok and ok
        print(f"  {step:10.2f} {float(times[idx]):30.6f} {diff:20.6f} "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"  tolerance: within one integration step -> {'PASS' if step_ok else 'FAIL'}")
    print()

    print("CHECK 3b -- the zero crossing is carrier-independent")
    print("  (the carrier cancels in Delta_f = 0, so the crossing is a geometric instant)")
    crossings = []
    for carrier in CARRIERS_HZ:
        times = np.linspace(-horizon, horizon, 20001)
        doppler = np.array([one_way_doppler_hz(float(r), carrier)
                            for r in orbit.range_rate_mps(times)])
        crossings.append(float(times[int(np.argmin(np.abs(doppler)))]))
    print(f"  crossings at the four carriers: {crossings}")
    carrier_ok = max(crossings) - min(crossings) == 0.0
    print(f"  spread {max(crossings) - min(crossings):.3e} s, expected exactly 0 -> "
          f"{'PASS' if carrier_ok else 'FAIL'}")
    print()

    overall = (worst_rel < 1e-9 and peak_ok and rate_ok
               and abs(tca - zero) <= 1e-6 and step_ok and carrier_ok)
    print("ALL CHECKS IN THIS SCRIPT:", "PASS" if overall else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
