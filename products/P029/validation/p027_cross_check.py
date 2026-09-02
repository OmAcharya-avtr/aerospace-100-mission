"""Independent cross-check of momentum accumulation against P027 ``disturbtorque``.

Run: ``python3 p027_cross_check.py``

What is being checked, and why it means anything
-----------------------------------------------
P027 ``disturbtorque`` publishes the angular momentum a reference smallsat accumulates
over one orbit from four environmental disturbance torques. This package implements the
same four torques, the same exponential atmosphere table, the same centred-dipole field
and the same orbit geometry **from the cited physics, in its own code**, and integrates
them with a **different numerical method**: Gauss-Legendre quadrature in argument of
latitude, with the solar term integrated only over sunlit arcs whose limits come from a
closed form, where P027's own package function uses a uniform-grid trapezoidal rule.

No module of P027 is imported here and none was copied. What is reproduced is the
*inputs*: the reference vehicle, the orbit, the beta angle, the constants and the density
and field models. That is the definition of the environment, and reproducing it is the
whole point of a cross-check.

P027's published numbers, quoted below, are taken from its committed raw output file
``products/P027/validation/momentum_integration_output.txt``. Where P027 printed only
six or seven significant figures the tolerance here is set to match that printing, not
tighter; where it printed a full-precision reference (its QUADPACK solar vector, its
smooth-source reference and its total reference vector) the tolerance is tight.
"""

from __future__ import annotations

import time

import numpy as np
from _common import Checks  # noqa: E402

from momentummgr import (  # noqa: E402
    SOURCES,
    body_dcm_from_lvlh,
    eclipse_boundaries,
    eclipse_fraction,
    momentum_history_eci,
    momentum_per_orbit_eci,
    node_axes,
    reference_orbit,
    reference_smallsat,
    sun_direction_for_beta,
    sweep_orbit,
)

c = Checks()
t_start = time.time()
print("Independent cross-check against P027 disturbtorque")
print("=" * 90)

BETA_DEG = 20.0
sc = reference_smallsat()
orbit = reference_orbit(500.0)
sun = sun_direction_for_beta(orbit.inclination_rad, orbit.raan_rad, np.radians(BETA_DEG))
period = orbit.period_s

print(f"""
Environment (identical inputs, independent code)
  vehicle    inertia diag(4, 8, 10) kg m^2, drag 0.6 m^2 Cd 2.2, sunlit 1.2 m^2 q 0.6,
             both centres of pressure at (0.02, 0.02, 0.05) m, residual dipole
             (0.05, 0.05, 0.10) A m^2
  orbit      circular 500 km, i = 51.6 deg, RAAN 0, nadir with 5 deg pitch and 5 deg roll
  Sun        beta = {BETA_DEG:.0f} deg, in-plane phase 0
  models     Vallado exponential density table, centred non-tilted dipole
             B0 Re^3 = 7.96e15 T m^3, solar constant 1361 W m^-2, cylindrical umbra
  period     {period:.4f} s   (P027 published 5676.9780 s)
""")
c.check("orbital period against P027", period, 5676.9780, 1e-7)

# ------------------------------------------------------------------ 1. orbit geometry
print("\n1. Orbit and eclipse geometry against P027's closed-form values")
p_hat, q_hat, _ = node_axes(orbit.inclination_rad, orbit.raan_rad)
amp = float(np.hypot(sun @ p_hat, sun @ q_hat))
u_in, u_out = eclipse_boundaries(orbit.radius_m, orbit.inclination_rad, orbit.raan_rad, sun)
frac = eclipse_fraction(orbit.radius_m, orbit.inclination_rad, orbit.raan_rad, sun)
print(f"   A = cos(beta)              {amp:.10f}   P027 0.9396926208")
print(f"   shadow entry u [deg]       {np.degrees(u_in):.6f}    P027 113.473596")
print(f"   shadow exit  u [deg]       {np.degrees(u_out):.6f}    P027 246.526404")
print(f"   eclipse fraction           {frac:.10f}   P027 0.3695911346")
c.check("cos(beta)", amp, 0.9396926208, 1e-10)
c.check("shadow entry [deg]", float(np.degrees(u_in)), 113.473596, 1e-6, kind="abs")
c.check("shadow exit [deg]", float(np.degrees(u_out)), 246.526404, 1e-6, kind="abs")
c.check("eclipse fraction", frac, 0.3695911346, 1e-10)

# ---------------------------------------------------- 2. constant body-frame torques
print("\n2. Constant body-frame torques (P027 section A, co-rotating atmosphere off)")
sweep_norot = sweep_orbit(sc, orbit, sun, n_samples=1441, co_rotating_atmosphere=False)
published_body = {
    "gravity_gradient": np.array([6.332938e-07, 1.907139e-06, -1.112353e-07]),
    "aerodynamic": np.array([-3.615203e-08, -1.281033e-06, 5.268739e-07]),
}
published_tp = {
    "gravity_gradient": np.array([3.595195e-03, 1.082678e-02, -6.314806e-04]),
    "aerodynamic": np.array([-2.052343e-04, -7.272394e-03, 2.991051e-03]),
}
for source, target in published_body.items():
    mine = sweep_norot.torque(source, "body")
    spread = float(np.max(np.abs(mine - mine[0])) / np.linalg.norm(mine[0]))
    c.check(f"{source}: body torque constant over the orbit (rel spread)", spread, 0.0, 1e-14,
            kind="abs")
    print(f"   {source:<17} mine  [{mine[0][0]: .6e} {mine[0][1]: .6e} {mine[0][2]: .6e}] N m")
    print(f"   {'':<17} P027  [{target[0]: .6e} {target[1]: .6e} {target[2]: .6e}] N m")
    c.check(f"{source}: body torque vs P027 (7 printed figures)",
            float(np.linalg.norm(mine[0] - target)), 0.0,
            2e-6 * float(np.linalg.norm(target)), kind="abs")
    c.check(f"{source}: T*P vs P027 (7 printed figures)",
            float(np.linalg.norm(mine[0] * period - published_tp[source])), 0.0,
            2e-6 * float(np.linalg.norm(published_tp[source])), kind="abs")

# ------------------------------------------------------ 3. momentum per orbit in ECI
print("\n3. Momentum accumulated over one orbit in ECI (P027 section C and its table)")
print("   Mine: Gauss-Legendre, 96 nodes, solar split at the closed-form eclipse edges.")
print("   P027: its own reference values, which are a trapezoid on N = 11521 for the")
print("   three smooth sources and adaptive QUADPACK for the solar term.\n")
# P027's own per-source table at N = 11521 is a *sampled trapezoid*. For the three
# continuous sources that rule is spectrally accurate on a closed period and agrees with
# any reference to roundoff, so those rows are a fair target. The solar row is not: the
# solar torque steps to and from zero at the eclipse edges, and P027 states in its own
# output that its N = 11521 solar value differs from its QUADPACK reference by 1.69e-04
# relative. The right target for the solar and total rows is therefore P027's published
# reference, not its sampled row, and both are shown below.
published_table = {
    "gravity_gradient": 1.084062e-02,
    "aerodynamic": 6.911776e-03,
    "solar": 4.111292e-04,
    "magnetic": 2.236218e-03,
    "total": 4.354188e-03,
}
print(f"   {'source':<18}{'mine |dh| [N m s]':>22}{'P027 N=11521 [N m s]':>24}{'rel diff':>12}")
print("   " + "-" * 76)
mine_dh = {}
for source in (*SOURCES, "total"):
    dh = momentum_per_orbit_eci(sc, orbit, sun, source)
    mine_dh[source] = dh
    mag = float(np.linalg.norm(dh))
    ref = published_table[source]
    print(f"   {source:<18}{mag:>22.10e}{ref:>24.6e}{abs(mag - ref) / ref:>12.2e}")
for source in ("gravity_gradient", "aerodynamic", "magnetic"):
    c.check(
        f"{source}: |dh| vs P027's sampled table (7 printed figures)",
        float(np.linalg.norm(mine_dh[source])), published_table[source], 2e-6,
    )

print("""
   The solar and total rows differ from that table by 1.9e-05 and 3.9e-06 relative. That
   is not a disagreement between the two implementations: it is P027's *sampled* solar
   term against a reference. P027 publishes both, and the comparison below is against its
   reference.

   Full-precision vector comparisons, against P027's own reference values.""")
quadpack_solar = np.array([-1.2065216122e-04, -1.2863853089e-04, 3.7137042242e-04])
total_ref = np.array([-2.4484285744e-03, 2.8372528957e-03, -2.2167544874e-03])
smooth_ref_mag = 4.573122077267e-03
mine_solar = mine_dh["solar"]
mine_total = mine_dh["total"]
mine_smooth = sum(
    (mine_dh[s] for s in ("gravity_gradient", "aerodynamic", "magnetic")), start=np.zeros(3)
)
print(f"   solar mine  [{mine_solar[0]: .10e} {mine_solar[1]: .10e} {mine_solar[2]: .10e}]")
print(f"   solar P027  [{quadpack_solar[0]: .10e} {quadpack_solar[1]: .10e} "
      f"{quadpack_solar[2]: .10e}]  (QUADPACK)")
c.check(
    "solar dh vector vs P027 QUADPACK reference, relative",
    float(np.linalg.norm(mine_solar - quadpack_solar) / np.linalg.norm(quadpack_solar)),
    0.0, 5e-10, kind="abs",
)
print(f"   total mine  [{mine_total[0]: .10e} {mine_total[1]: .10e} {mine_total[2]: .10e}]")
print(f"   total P027  [{total_ref[0]: .10e} {total_ref[1]: .10e} {total_ref[2]: .10e}]")
c.check(
    "total dh vector vs P027 reference, relative",
    float(np.linalg.norm(mine_total - total_ref) / np.linalg.norm(total_ref)),
    0.0, 5e-10, kind="abs",
)
c.check("smooth-source |dh| vs P027 reference", float(np.linalg.norm(mine_smooth)),
        smooth_ref_mag, 5e-12)

sampled_solar_gap = abs(published_table["solar"] - float(np.linalg.norm(quadpack_solar)))
mine_solar_gap = abs(float(np.linalg.norm(mine_solar)) - float(np.linalg.norm(quadpack_solar)))
ref_mag = float(np.linalg.norm(quadpack_solar))
print("\n   Distance of each code's solar |dh| from P027's own QUADPACK reference:")
print(f"     P027 sampled trapezoid, N = 11521   {sampled_solar_gap:.6e} N m s "
      f"(rel {sampled_solar_gap / ref_mag:.2e})")
print(f"     this package, Gauss-Legendre        {mine_solar_gap:.6e} N m s "
      f"(rel {mine_solar_gap / ref_mag:.2e})")
print("   P027's own output reports 1.69e-04 relative for that row and derives an")
print("   eclipse-edge bound of 1.20e-03 for it, so the gap is the documented")
print("   quadrature error of a uniform grid and nothing else.\n")
c.assert_true(
    "this package's solar dh is closer to P027's reference than P027's own sampled row",
    mine_solar_gap < sampled_solar_gap,
    f"{mine_solar_gap:.3e} < {sampled_solar_gap:.3e} N m s",
)

# -------------------------------------------------- 4. quadrature convergence, mine
print("\n4. Convergence of this package's own quadrature (so the agreement above is not")
print("   an accident of one node count). Reference: 384 nodes.")
ref384 = {s: momentum_per_orbit_eci(sc, orbit, sun, s, n_nodes=384) for s in (*SOURCES, "total")}
print(f"   {'nodes':>7}{'total |dh|':>22}{'rel vs 384':>13}{'solar rel vs 384':>19}")
print("   " + "-" * 61)
worst = 0.0
for n in (8, 16, 32, 64, 96, 192):
    tot = momentum_per_orbit_eci(sc, orbit, sun, "total", n_nodes=n)
    sol = momentum_per_orbit_eci(sc, orbit, sun, "solar", n_nodes=n)
    rel_t = float(np.linalg.norm(tot - ref384["total"]) / np.linalg.norm(ref384["total"]))
    rel_s = float(np.linalg.norm(sol - ref384["solar"]) / np.linalg.norm(ref384["solar"]))
    if n >= 16:
        worst = max(worst, rel_t, rel_s)
    print(f"   {n:>7}{np.linalg.norm(tot):>22.12e}{rel_t:>13.2e}{rel_s:>19.2e}")
c.check("worst quadrature self-convergence for n_nodes >= 16", worst, 0.0, 1e-12, kind="abs")

# ------------------------------------------- 5. sampled trapezoid, the practical grid
print("\n5. The sampled trapezoid this package also provides, against its own")
print("   Gauss-Legendre reference. This is the discretisation error a time-stepped")
print("   scheduler inherits, and for the solar term it does not fall as a power of N,")
print("   because the torque steps to and from zero at the eclipse edges.")
print(f"\n   {'N':>7}{'total rel err':>16}{'solar rel err':>16}{'smooth rel err':>17}")
print("   " + "-" * 56)
for n in (181, 361, 721, 1441, 2881):
    sw = sweep_orbit(sc, orbit, sun, n_samples=n)
    tot = momentum_history_eci(sw, "total")[-1]
    sol = momentum_history_eci(sw, "solar")[-1]
    smo = sum(
        (momentum_history_eci(sw, s)[-1] for s in ("gravity_gradient", "aerodynamic", "magnetic")),
        start=np.zeros(3),
    )
    rel_t = float(np.linalg.norm(tot - mine_total) / np.linalg.norm(mine_total))
    rel_s = float(np.linalg.norm(sol - mine_solar) / np.linalg.norm(mine_solar))
    rel_m = float(np.linalg.norm(smo - mine_smooth) / np.linalg.norm(mine_smooth))
    print(f"   {n:>7}{rel_t:>16.2e}{rel_s:>16.2e}{rel_m:>17.2e}")
    if n == 721:
        c.check("sampled trapezoid at the default N = 721, total", rel_t, 0.0, 1e-3, kind="abs")
        c.check("sampled trapezoid at the default N = 721, smooth sources", rel_m, 0.0, 1e-12,
                kind="abs")

# ------------------------------------------- 6. wheel equation reproduces the integral
print("\n6. Internal consistency: integrating the body-frame wheel equation")
print("   dh_w/dt = T_dist - omega x (I omega + h_w) over one orbit must give the same")
print("   inertial momentum change as the ECI integral, because")
print("   H_eci = C_be^T (I omega + h_w) and dH_eci/dt = T_dist. This ties the scheduler's")
print("   dynamics to the cross-checked accumulation; the two are otherwise separate code.")
n_step = 200_000
u = np.linspace(0.0, 2.0 * np.pi, n_step + 1)
sw = sweep_orbit(sc, orbit, sun, n_samples=n_step + 1)
dt = period / n_step
omega = orbit.body_rate_body_rad_s
i_omega = sc.inertia @ omega
t_body = sw.torque("total", "body")
h_w = np.zeros(3)
for i in range(n_step):
    k1 = t_body[i] - np.cross(omega, i_omega + h_w)
    k2 = t_body[i + 1] - np.cross(omega, i_omega + h_w + dt * k1)
    h_w = h_w + 0.5 * dt * (k1 + k2)
c_bl = body_dcm_from_lvlh(orbit.yaw_rad, orbit.pitch_rad, orbit.roll_rad)
c_be_0, c_be_p = sw.c_be[0], sw.c_be[-1]
dh_from_wheels = c_be_p.T @ (i_omega + h_w) - c_be_0.T @ i_omega
print(f"\n   wheel-equation dh_eci  [{dh_from_wheels[0]: .10e} {dh_from_wheels[1]: .10e} "
      f"{dh_from_wheels[2]: .10e}] N m s")
print(f"   quadrature     dh_eci  [{mine_total[0]: .10e} {mine_total[1]: .10e} "
      f"{mine_total[2]: .10e}] N m s")
c.check(
    "wheel-equation momentum change vs ECI quadrature, relative",
    float(np.linalg.norm(dh_from_wheels - mine_total) / np.linalg.norm(mine_total)),
    0.0, 2e-3, kind="abs",
)
print("   (the tolerance is 2e-3: the trapezoidal ODE integration inherits the same")
print("    eclipse-edge error as the sampled quadrature in section 5, and the C_bl")
print("    matrix is used only through the sweep, so this is a consistency check of the")
print("    dynamics, not a second measurement of the accumulation.)")
print(f"   unused reference: |C_bl| = {np.linalg.norm(c_bl):.6f}")

print(f"\nwall time {time.time() - t_start:.2f} s")
c.summary("p027_cross_check.py")
raise SystemExit(1 if c.n_fail else 0)
