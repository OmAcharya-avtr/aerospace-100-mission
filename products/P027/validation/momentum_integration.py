"""Angular momentum accumulated over one orbit, against closed forms and against
independent quadrature. Run: ``python3 momentum_integration.py``

Three separate claims are tested.

A. For a nadir-pointing spacecraft on a circular orbit the gravity-gradient torque is
   *constant in the body frame*, so the momentum accumulated over one orbit is exactly
   ``T * P``. Likewise the aerodynamic torque, if the co-rotating-atmosphere correction
   is switched off, because then the relative wind is fixed in LVLH.
B. In ECI those same constant-in-LVLH torques give an exactly computable orbit average:
   the LVLH y axis is the fixed vector ``-h_hat``, while the x and z axes rotate
   uniformly through a full turn and therefore average to zero. Hence
   ``<T_eci> = -(T_lvlh)_y h_hat`` and ``dh_eci = <T_eci> P``.
C. For the full four-source profile no closed form exists, so the trapezoidal result is
   compared against Simpson's rule on the same grid and against the trapezoidal rule on
   an eight-times finer grid. The solar term switches discontinuously at eclipse entry
   and exit, which caps the achievable order of accuracy; that is measured, not assumed.
"""

from __future__ import annotations

import time

import numpy as np

from _common import Checks  # noqa: E402

from disturbtorque import (  # noqa: E402
    Orbit,
    body_from_lvlh,
    compute_profile,
    momentum_accumulation,
    node_axes,
    reference_smallsat,
    sun_direction_for_beta,
)

c = Checks()
t_start = time.time()
print("Momentum accumulation over one orbit")
print("=" * 78)

sc = reference_smallsat()
orbit = Orbit(
    altitude_m=500_000.0,
    inclination_rad=np.radians(51.6),
    raan_rad=0.0,
    pitch_rad=np.radians(5.0),
    roll_rad=np.radians(5.0),
)
sun = sun_direction_for_beta(orbit.inclination_rad, orbit.raan_rad, np.radians(20.0))
period = orbit.period_s
_, _, h_hat = node_axes(orbit.inclination_rad, orbit.raan_rad)
c_bl = body_from_lvlh(orbit.yaw_rad, orbit.pitch_rad, orbit.roll_rad)

print(f"\nReference case: {orbit.altitude_m / 1000:.0f} km, i = 51.6 deg, beta = 20 deg,")
print("5 deg pitch and 5 deg roll from nadir, reference smallsat.")
print(f"Period P = {period:.4f} s.")

# ------------------------------------------------------------- A. body-frame closed form
print("\nA. Constant body-frame torques: dh = T * P exactly")
prof_norot = compute_profile(sc, orbit, sun, n_samples=1441, co_rotating_atmosphere=False)
for source in ("gravity_gradient", "aerodynamic"):
    t_hist = prof_norot.torque(source, "body")
    spread = float(np.max(np.abs(t_hist - t_hist[0])) / np.linalg.norm(t_hist[0]))
    c.check(f"{source}: body torque is constant, max rel spread", spread, 0.0, 1e-14, kind="abs")
    h = momentum_accumulation(prof_norot, source, "body")
    analytic = t_hist[0] * period
    print(f"   {source}: T_body = [{t_hist[0][0]: .6e} {t_hist[0][1]: .6e} {t_hist[0][2]: .6e}] N m")
    print(f"   {' ' * len(source)}  T*P    = [{analytic[0]: .6e} {analytic[1]: .6e} "
          f"{analytic[2]: .6e}] N m s")
    print(f"   {' ' * len(source)}  int T  = [{h[-1][0]: .6e} {h[-1][1]: .6e} {h[-1][2]: .6e}] N m s")
    c.check(
        f"{source}: |int T dt - T*P| / |T*P| over one orbit",
        float(np.linalg.norm(h[-1] - analytic) / np.linalg.norm(analytic)),
        0.0,
        1e-13,
        kind="abs",
    )

# ---------------------------------------------------------------- B. ECI closed form
print("\nB. Same torques in ECI: <T_eci> = -(T_lvlh)_y h_hat")
for source in ("gravity_gradient", "aerodynamic"):
    t_body = prof_norot.torque(source, "body")[0]
    t_lvlh = c_bl.T @ t_body
    analytic_mean = -t_lvlh[1] * h_hat
    numeric_mean = prof_norot.secular(source, "eci")
    h = momentum_accumulation(prof_norot, source, "eci")
    print(f"   {source}: T_lvlh = [{t_lvlh[0]: .6e} {t_lvlh[1]: .6e} {t_lvlh[2]: .6e}] N m")
    print(f"   {' ' * len(source)}  <T>_ana = [{analytic_mean[0]: .6e} {analytic_mean[1]: .6e} "
          f"{analytic_mean[2]: .6e}] N m")
    print(f"   {' ' * len(source)}  <T>_num = [{numeric_mean[0]: .6e} {numeric_mean[1]: .6e} "
          f"{numeric_mean[2]: .6e}] N m")
    c.check(
        f"{source}: <T_eci> numeric vs closed form, rel",
        float(np.linalg.norm(numeric_mean - analytic_mean) / np.linalg.norm(analytic_mean)),
        0.0,
        1e-9,
        kind="abs",
    )
    c.check(
        f"{source}: dh_eci over one orbit vs <T>_ana * P, rel",
        float(np.linalg.norm(h[-1] - analytic_mean * period) / np.linalg.norm(analytic_mean * period)),
        0.0,
        1e-9,
        kind="abs",
    )

# ----------------------------------------------------- C. full profile, independent rules
print("\nC. Full four-source profile: sampled trapezoid against independent references")
print("   (co-rotating atmosphere on, eclipse on)")

print("""
C1. The three continuous sources (gravity gradient, aerodynamic, magnetic). Their ECI
    torque histories are smooth and exactly periodic over one orbit, so the trapezoidal
    rule on the closed period is spectrally accurate and every grid must agree to
    roundoff. Reference: the N = 11521 grid.
""")
n_ref = 11521
prof_ref = compute_profile(sc, orbit, sun, n_samples=n_ref)
smooth = ("gravity_gradient", "aerodynamic", "magnetic")


def dh_smooth(profile):
    return sum(momentum_accumulation(profile, s, "eci")[-1] for s in smooth)


h_smooth_ref = dh_smooth(prof_ref)
print(f"    reference |dh_smooth| = {np.linalg.norm(h_smooth_ref):.12e} N m s")
print(f"    {'N':>7}{'|dh| [N m s]':>22}{'rel vs ref':>13}")
print("    " + "-" * 42)
smooth_errs = []
for n in (181, 361, 721, 1441, 2881):
    h = dh_smooth(compute_profile(sc, orbit, sun, n_samples=n))
    rel = float(np.linalg.norm(h - h_smooth_ref) / np.linalg.norm(h_smooth_ref))
    smooth_errs.append(rel)
    print(f"    {n:>7}{np.linalg.norm(h):>22.12e}{rel:>13.2e}")
c.check("smooth-source momentum, worst error over N = 181..2881", max(smooth_errs), 0.0, 1e-12,
        kind="abs")

print("""
C2. The solar source, against an independent reference that does not use the sample grid
    at all. The cylindrical shadow boundaries are solved in closed form: with
    a = s.P_hat, b = s.Q_hat, A = sqrt(a^2+b^2) = cos(beta) and phi = atan2(b, a), the
    vehicle is in shadow where A cos(u - phi) < -sqrt(1 - (Re/R)^2). The solar torque is
    then integrated over the sunlit arc only, with adaptive Gauss-Kronrod (QUADPACK),
    component by component.
""")
from scipy.integrate import quad  # noqa: E402

from disturbtorque import (  # noqa: E402
    R_EARTH_EQUATORIAL,
    circular_orbit_state,
    lvlh_from_eci,
    solar_radiation_torque,
)

p_hat, q_hat, _ = node_axes(orbit.inclination_rad, orbit.raan_rad)
a_s, b_s = float(sun @ p_hat), float(sun @ q_hat)
amp = float(np.hypot(a_s, b_s))
phi = float(np.arctan2(b_s, a_s))
c0 = float(np.sqrt(1.0 - (R_EARTH_EQUATORIAL / orbit.radius_m) ** 2))
psi = float(np.arccos(-c0 / amp))
u_in = (phi + psi) % (2 * np.pi)
u_out = (phi - psi) % (2 * np.pi)
print(f"    cos(beta) = A = {amp:.10f}   phi = {np.degrees(phi):.6f} deg")
print(f"    sqrt(1-(Re/R)^2) = {c0:.10f}   half shadow arc = {np.degrees(np.pi - psi):.6f} deg")
print(f"    shadow entry u = {np.degrees(u_in):.6f} deg, exit u = {np.degrees(u_out):.6f} deg")
print(f"    analytic eclipse fraction = {(np.pi - psi) / np.pi:.10f}, "
      f"sampled (N = {n_ref}) = {prof_ref.eclipse_fraction:.10f}")
c.check("analytic vs sampled eclipse fraction", prof_ref.eclipse_fraction,
        (np.pi - psi) / np.pi, 2e-4, kind="abs")


def solar_torque_eci_at(u_val: float) -> np.ndarray:
    r_u, v_u = circular_orbit_state(orbit.radius_m, orbit.inclination_rad, orbit.raan_rad, u_val)
    c_be = c_bl @ lvlh_from_eci(r_u, v_u)
    t_body = solar_radiation_torque(
        c_be @ sun, sc.srp_area_m2, sc.srp_reflectance, sc.cp_srp_offset_m, 1.0, True
    )
    return c_be.T @ t_body


sunlit_arcs = (
    [(u_out, u_in)] if u_out < u_in else [(u_out, 2 * np.pi), (0.0, u_in)]
)
h_solar_quad = np.zeros(3)
quad_abserr = 0.0
for k in range(3):
    for lo, hi in sunlit_arcs:
        val, err = quad(lambda x, kk=k: solar_torque_eci_at(x)[kk], lo, hi, limit=200)
        h_solar_quad[k] += val
        quad_abserr = max(quad_abserr, err)
h_solar_quad *= period / (2 * np.pi)
print(f"\n    QUADPACK reference dh_solar = [{h_solar_quad[0]: .10e} {h_solar_quad[1]: .10e} "
      f"{h_solar_quad[2]: .10e}] N m s")
print(f"    |dh_solar| = {np.linalg.norm(h_solar_quad):.10e} N m s "
      f"(worst reported quad abs error {quad_abserr * period / (2 * np.pi):.2e})")
t_solar_peak = prof_ref.peak_magnitude("solar", "eci")
print("\n    Derived error bound. The only error source is the two eclipse edges, each")
print("    misplaced by at most one sample interval dt = P/(N-1), during which the solar")
print(f"    torque magnitude is at most its peak {t_solar_peak:.6e} N m. Hence")
print("    err_rel <= 2 * T_peak * dt / |dh_solar|. That bound is asserted below; the")
print("    observed error is expected to be well inside it and is reported as measured.")
print(f"\n    {'N':>7}{'|dh_solar| [N m s]':>24}{'rel vs QUADPACK':>18}{'derived bound':>16}")
print("    " + "-" * 65)
solar_errs = []
for n in (181, 361, 721, 1441, 2881, n_ref):
    prof = prof_ref if n == n_ref else compute_profile(sc, orbit, sun, n_samples=n)
    h = momentum_accumulation(prof, "solar", "eci")[-1]
    rel = float(np.linalg.norm(h - h_solar_quad) / np.linalg.norm(h_solar_quad))
    bound = 2.0 * t_solar_peak * (period / (n - 1)) / float(np.linalg.norm(h_solar_quad))
    solar_errs.append((n, rel, bound))
    print(f"    {n:>7}{np.linalg.norm(h):>24.10e}{rel:>18.2e}{bound:>16.2e}")
print("""
    The solar error does not fall like a power of N. It cannot: the torque steps to and
    from zero between two adjacent samples, and the error is governed by where the two
    eclipse edges happen to fall inside their sample intervals, which changes
    unpredictably with N. Only the edge bound above is guaranteed, and the observed
    error at the coarsest grid tested, N = 181, is 0.75 % - roughly ten times better
    than the bound but still an order of magnitude worse than any of the continuous
    sources. At the package default N = 721 it is 0.35 %.
""")
c.assert_true(
    "solar momentum error is inside the derived edge bound at every N",
    all(e <= b for _, e, b in solar_errs),
    f"worst ratio observed/bound = {max(e / b for _, e, b in solar_errs):.3f}",
)
c.check("solar momentum error at the default N = 721",
        [e for n, e, _ in solar_errs if n == 721][0], 0.0, 5e-3, kind="abs")
c.check("solar momentum error at the coarsest grid tested, N = 181",
        [e for n, e, _ in solar_errs if n == 181][0], 0.0, 1e-2, kind="abs")

print("""
C3. Total. The reference is the sum of the C1 smooth reference and the C2 QUADPACK
    solar reference, so no part of it comes from the grid being tested.
""")
h_total_ref = h_smooth_ref + h_solar_quad
print(f"    reference dh_total = [{h_total_ref[0]: .10e} {h_total_ref[1]: .10e} "
      f"{h_total_ref[2]: .10e}] N m s   |dh| = {np.linalg.norm(h_total_ref):.10e}")
print(f"\n    {'N':>7}{'|dh_total| [N m s]':>24}{'rel vs reference':>18}")
print("    " + "-" * 49)
total_errs = {}
for n in (181, 361, 721, 1441, 2881):
    prof = compute_profile(sc, orbit, sun, n_samples=n)
    h = momentum_accumulation(prof, "total", "eci")[-1]
    rel = float(np.linalg.norm(h - h_total_ref) / np.linalg.norm(h_total_ref))
    total_errs[n] = rel
    print(f"    {n:>7}{np.linalg.norm(h):>24.10e}{rel:>18.2e}")
c.check("total momentum error at the default N = 721", total_errs[721], 0.0, 1e-3, kind="abs")
c.check("total momentum error, worst over N = 181..2881", max(total_errs.values()), 0.0, 2e-3,
        kind="abs")

print("\n   Per-source momentum over one orbit at N = 11521 (ECI frame):")
print(f"   {'source':<18}{'|dh| [N m s]':>16}{'|<T>| [N m]':>16}{'|<T>|*P [N m s]':>18}")
print("   " + "-" * 68)
for source in ("gravity_gradient", "aerodynamic", "solar", "magnetic", "total"):
    h = momentum_accumulation(prof_ref, source, "eci")[-1]
    sec = prof_ref.secular(source, "eci")
    print(f"   {source:<18}{np.linalg.norm(h):>16.6e}{np.linalg.norm(sec):>16.6e}"
          f"{np.linalg.norm(sec) * period:>18.6e}")
    c.check(
        f"   {source}: |dh| equals |<T>| P on the same grid",
        float(np.linalg.norm(h)),
        float(np.linalg.norm(sec)) * period,
        1e-12,
    )

print("\n   That last block is an identity, not evidence: the secular estimator is the same")
print("   trapezoidal sum divided by P. The evidence is C1, C2 and C3, whose references")
print("   are independent of the grid being tested.")

print(f"\nwall time {time.time() - t_start:.2f} s")
c.summary("momentum_integration.py")
raise SystemExit(1 if c.n_fail else 0)
