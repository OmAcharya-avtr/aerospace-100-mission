"""V3 - Does detumble time scale with gain the way the first-order model says?

Model (see ``detumblesim/analytic.py`` for the derivation)
---------------------------------------------------------
For an isotropic inertia and the ideal, unsaturated B-dot law, the body rate
obeys the linear time-varying system ``j omega_dot = -k (|B|^2 I - B B^T)
omega``.  Orbit-averaging gives a constant damping matrix ``D = k (<|B|^2> I -
<B B^T>)`` with modal time constants ``tau_i = j / lambda_i`` and

    t_detumble = tau ln(omega_0 / omega_target)                        (4)

Two predictions follow and are tested separately:

C1  ``t_detumble`` is proportional to ``1 / k``.  On log-log axes the slope of
    ``t`` against ``k`` must be -1.  Pass criterion, fixed before the run:
    fitted slope within 0.05 of -1.
C2  the measured time lies inside the analytic modal bracket
    ``[t(fastest mode), t(slowest mode)]``, which brackets the true answer for
    a rate whose direction is not known in advance.

Both predictions assume **no saturation**, so the dipole limit is set to a
deliberately unphysical 50 A m^2 for this check and the measured saturation
fraction is printed to prove the assumption held.  The gain range is chosen so
that every run lasts at least two orbital periods, because orbit-averaging is
what the model is; where that fails, it is reported.
"""

from __future__ import annotations

import time

import numpy as np

from _support import Tee  # noqa: E402

from detumblesim.analytic import (  # noqa: E402
    detumble_time_first_order,
    geometry_factors,
    orbit_field_moments,
)
from detumblesim.orbit import CircularOrbit  # noqa: E402
from detumblesim.policies import FixedGainPolicy  # noqa: E402
from detumblesim.simulate import DetumbleConfig, simulate_detumble  # noqa: E402
from detumblesim.spacecraft import Magnetorquer, inertia_from_diagonal  # noqa: E402

J_SCALAR = 0.05
UNSATURATED_DIPOLE_AM2 = 50.0
OMEGA0_DEG_S = np.array([6.0, -5.0, 5.7])
TARGET_DEG_S = 1.0
SLOPE_TOL = 0.05


def sweep(orbit, gains, moments, out, label):
    inertia = inertia_from_diagonal(J_SCALAR, J_SCALAR, J_SCALAR)
    w0 = np.radians(OMEGA0_DEG_S)
    rate0 = float(np.linalg.norm(w0))
    target = np.radians(TARGET_DEG_S)
    out(f"  {label}: altitude {orbit.altitude_km:.0f} km, inclination "
        f"{orbit.inclination_deg:.1f} deg, period {orbit.period_s:.1f} s")
    out(f"  |omega_0| = {np.degrees(rate0):.4f} deg/s -> target "
        f"{TARGET_DEG_S:.1f} deg/s, ln ratio = {np.log(rate0 / target):.6f}")
    out(f"  RMS field over 10 orbits = {moments.rms_b_t * 1e6:.4f} uT, "
        f"geometry factors = {np.array2string(geometry_factors(moments), precision=5)}")
    out("")
    out(f"  {'gain k':>11} {'t_sim [s]':>11} {'orbits':>7} {'sat[%]':>7} "
        f"{'t_iso [s]':>11} {'t_fast [s]':>11} {'t_slow [s]':>11} {'in bracket':>11}")
    times, in_bracket, n_orbits = [], [], []
    for k in gains:
        cfg = DetumbleConfig(
            inertia=inertia,
            orbit=orbit,
            magnetorquer=Magnetorquer.isotropic(UNSATURATED_DIPOLE_AM2),
            omega0_rad_s=w0,
            duration_s=250000.0,
            control_dt_s=4.0,
            substeps=1,
            target_rate_rad_s=target,
            stop_when_detumbled=True,
        )
        r = simulate_detumble(cfg, FixedGainPolicy(float(k)))
        t_iso = detumble_time_first_order(J_SCALAR, k, moments, rate0, target, "isotropic")
        t_fast = detumble_time_first_order(J_SCALAR, k, moments, rate0, target, "fastest")
        t_slow = detumble_time_first_order(J_SCALAR, k, moments, rate0, target, "slowest")
        ok = bool(r.detumbled and t_fast <= r.detumble_time_s <= t_slow)
        times.append(r.detumble_time_s)
        in_bracket.append(ok)
        n_orbits.append(r.detumble_time_s / orbit.period_s)
        t_txt = f"{r.detumble_time_s:11.1f}" if r.detumbled else f"{'not reached':>11}"
        o_txt = (
            f"{r.detumble_time_s / orbit.period_s:7.2f}" if r.detumbled else f"{'-':>7}"
        )
        out(f"  {k:11.4e} {t_txt} {o_txt} "
            f"{100 * r.saturated_fraction:7.3f} {t_iso:11.1f} {t_fast:11.1f} "
            f"{t_slow:11.1f} {str(ok):>11}")
    return np.array(times), np.array(in_bracket), np.array(n_orbits)


def main() -> None:
    t_start = time.perf_counter()
    with Tee(__file__) as out:
        out("V3  Detumble time against B-dot gain")
        out("=" * 78)
        out("")
        out(f"inertia (isotropic) j = {J_SCALAR} kg m^2;  dipole limit "
            f"{UNSATURATED_DIPOLE_AM2:.0f} A m^2 (saturation deliberately disabled)")
        out("integrator: RK4, control step 4 s, 1 substep")
        out("")

        orbit = CircularOrbit(altitude_km=500.0, inclination_deg=97.4)
        moments = orbit_field_moments(orbit, 4000, 10.0 * orbit.period_s)
        gains = np.geomspace(3.0e3, 4.8e4, 8)
        out("C1/C2  Sun-synchronous orbit")
        times, ok, orbits = sweep(orbit, gains, moments, out, "orbit")
        slope, intercept = np.polyfit(np.log(gains), np.log(times), 1)
        resid = np.log(times) - (slope * np.log(gains) + intercept)
        out("")
        out(f"  log-log fit: slope = {slope:.6f}, predicted -1")
        out(f"  |slope + 1|                        = {abs(slope + 1.0):.6f}")
        out(f"  RMS residual of the power-law fit  = {np.sqrt(np.mean(resid**2)):.6f} "
            "(natural log units)")
        out(f"  product k * t (should be constant) = "
            f"{np.array2string(gains * times, precision=4)}")
        out(f"  spread of k*t about its mean       = "
            f"{100 * np.std(gains * times) / np.mean(gains * times):.3f} %")
        c1 = abs(slope + 1.0) <= SLOPE_TOL
        c2 = bool(np.all(ok))
        out(f"  shortest run                       = {orbits.min():.2f} orbits")
        out(f"  C1 slope within {SLOPE_TOL} of -1        : "
            f"{'PASS' if c1 else 'FAIL'}")
        out(f"  C2 all points inside the bracket   : {'PASS' if c2 else 'FAIL'} "
            f"({int(ok.sum())} of {ok.size})")
        out("")

        out("C3  Where the orbit-averaged model breaks down")
        out("  Extending the sweep to gains whose detumble time is a fraction of an")
        out("  orbit removes the separation of timescales the averaging assumes.")
        fast_gains = np.geomspace(6.4e4, 1.0e6, 5)
        t2, ok2, orb2 = sweep(orbit, fast_gains, moments, out, "orbit")
        out("")
        out(f"  points inside the analytic bracket  = {int(ok2.sum())} of {ok2.size}")
        out(f"  detumble durations                  = "
            f"{np.array2string(orb2, precision=3)} orbits")
        all_g = np.concatenate([gains, fast_gains])
        all_t = np.concatenate([times, t2])
        s_all = np.polyfit(np.log(all_g), np.log(all_t), 1)[0]
        out(f"  log-log slope over the WHOLE range  = {s_all:.6f} "
            f"(vs {slope:.6f} over the multi-orbit range only)")
        out("  Reported as measured: the 1/k law is a multi-orbit-average result and")
        out("  visibly degrades once the detumble takes less than about one orbit.")
        out("")

        out("C4  Same test on a near-equatorial orbit (weak controllability)")
        eq = CircularOrbit(altitude_km=500.0, inclination_deg=5.0)
        eq_mom = orbit_field_moments(eq, 4000, 10.0 * eq.period_s)
        t3, ok3, orb3 = sweep(eq, gains, eq_mom, out, "orbit")
        good = np.isfinite(t3)
        s3 = float(np.polyfit(np.log(gains[good]), np.log(t3[good]), 1)[0])
        out("")
        out(f"  runs that never reached the target within 250000 s = "
            f"{int((~good).sum())} of {good.size} (the smallest gains)")
        out(f"  log-log slope over the runs that finished = {s3:.6f}")
        out(f"  points inside the analytic bracket = {int(ok3.sum())} of {ok3.size}")
        out(f"  detumble times are {np.mean(t3[good] / times[good]):.3f}x the "
            "sun-synchronous case at the same gains,")
        out(f"  consistent with the smallest geometry factor falling from "
            f"{geometry_factors(moments)[0]:.4f} to {geometry_factors(eq_mom)[0]:.4f}.")
        out("")
        out(f"OVERALL V3: C1 {'PASS' if c1 else 'FAIL'}, C2 {'PASS' if c2 else 'FAIL'}. "
            "C3 and C4 are measurements, not pass/fail checks.")
        out(f"wall time {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
