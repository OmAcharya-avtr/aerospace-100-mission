"""Orbit geometry, Sun model, eclipse and the orbit-averaged geomagnetic field, each
against a closed form or an externally checkable astronomical fact.

Run: ``python3 orbit_geometry.py``
"""

from __future__ import annotations

import numpy as np

from _common import Checks  # noqa: E402

from disturbtorque import (  # noqa: E402
    EARTH_DIPOLE_MOMENT,
    MU_EARTH,
    R_EARTH_EQUATORIAL,
    beta_angle,
    circular_orbit_state,
    dipole_field_eci,
    eclipse_fraction_cylindrical,
    julian_date,
    lvlh_from_eci,
    mean_dipole_field_over_orbit,
    node_axes,
    orbital_period,
    sun_direction_for_beta,
    sun_distance_au,
    sun_unit_vector_eci,
)

c = Checks()
print("Orbit geometry, Sun model, eclipse and orbit-averaged field")
print("=" * 78)

# ----------------------------------------------------------------- 1. orbit basics
print("\n1. Circular orbit kinematics at 500 km (R = 6378137 + 500000 = 6878137 m)")
r_orbit = R_EARTH_EQUATORIAL + 500_000.0
speed = np.sqrt(MU_EARTH / r_orbit)
period = orbital_period(r_orbit)
print(f"""
   v = sqrt(mu/R) = sqrt(3.986004418e14 / 6878137) = sqrt({MU_EARTH / r_orbit:.7e})
                  = {speed:.6f} m s^-1
   T = 2 pi sqrt(R^3/mu) = {period:.4f} s = {period / 60:.4f} min
""")
c.check("period from 2 pi R / v [s]", period, 2 * np.pi * r_orbit / speed, 1e-14)

inc, raan = np.radians(51.6), np.radians(30.0)
u = np.linspace(0.0, 2 * np.pi, 2001)
r, v = circular_orbit_state(r_orbit, inc, raan, u)
c.check("|r| constant, max rel deviation", float(np.max(np.abs(np.linalg.norm(r, axis=1) / r_orbit - 1))), 0.0, 1e-14, kind="abs")
c.check("|v| constant, max rel deviation", float(np.max(np.abs(np.linalg.norm(v, axis=1) / speed - 1))), 0.0, 1e-14, kind="abs")
c.check("r . v = 0, max |cos angle|", float(np.max(np.abs(np.sum(r * v, axis=1))) / (r_orbit * speed)), 0.0, 1e-15, kind="abs")
p_hat, q_hat, h_hat = node_axes(inc, raan)
c.check("max declination [deg] equals inclination", float(np.degrees(np.max(np.arcsin(r[:, 2] / r_orbit)))), 51.6, 1e-6, kind="abs")
c.check("h_hat = P_hat x Q_hat, max component error", float(np.max(np.abs(np.cross(p_hat, q_hat) - h_hat))), 0.0, 1e-15, kind="abs")

c_lvlh = np.array([lvlh_from_eci(r[i], v[i]) for i in range(0, 2001, 100)])
ortho = np.max(np.abs(c_lvlh @ np.transpose(c_lvlh, (0, 2, 1)) - np.eye(3)))
c.check("LVLH DCM orthonormality, max |C C^T - I|", float(ortho), 0.0, 1e-14, kind="abs")
dets = np.linalg.det(c_lvlh)
c.check("LVLH DCM determinant, max |det - 1|", float(np.max(np.abs(dets - 1.0))), 0.0, 1e-14, kind="abs")
nadir_err = np.max(np.abs(np.einsum("nij,nj->ni", c_lvlh, r[::100] / r_orbit) - np.array([0, 0, -1.0])))
c.check("LVLH maps r_hat to (0, 0, -1), max error", float(nadir_err), 0.0, 1e-14, kind="abs")

# --------------------------------------------------------------------- 2. Sun model
print("\n2. Low-precision Sun model against externally checkable astronomical facts")
print("""
   Facts used (no page reference, none needed - these are definitional or annual):
   * Earth perihelion occurs in the first days of January at about 0.9833 AU and
     aphelion in early July at about 1.0167 AU.
   * The Sun's geocentric declination reaches +/- the obliquity of the ecliptic,
     about 23.44 deg, at the solstices, and passes through zero at the equinoxes.
   * The Earth-Sun unit vector has unit length by construction.
""")
jd_j2000 = julian_date(2000, 1, 1, 12, 0, 0.0)
c.check("julian_date(2000-01-01 12:00 UTC) [d]", jd_j2000, 2451545.0, 1e-12, kind="abs")

jds = np.array([julian_date(2026, 1, 1) + d for d in range(0, 366)])
dists = np.array([sun_distance_au(j) for j in jds])
decs = np.degrees(np.array([np.arcsin(sun_unit_vector_eci(j)[2]) for j in jds]))
norms = np.array([np.linalg.norm(sun_unit_vector_eci(j)) for j in jds])
i_min, i_max = int(np.argmin(dists)), int(np.argmax(dists))
print(f"   over calendar year 2026, daily samples (JD {jds[0]:.1f} to {jds[-1]:.1f}):")
print(f"     perihelion  day-of-year {i_min + 1:3d}  distance {dists[i_min]:.6f} AU")
print(f"     aphelion    day-of-year {i_max + 1:3d}  distance {dists[i_max]:.6f} AU")
print(f"     declination range  {decs.min():+.4f} deg to {decs.max():+.4f} deg")
c.check("minimum Earth-Sun distance [AU]", float(dists.min()), 0.9833, 5e-4, kind="abs")
c.check("maximum Earth-Sun distance [AU]", float(dists.max()), 1.0167, 5e-4, kind="abs")
c.check("maximum solar declination [deg]", float(decs.max()), 23.44, 0.05, kind="abs")
c.check("minimum solar declination [deg]", float(decs.min()), -23.44, 0.05, kind="abs")
c.check("Sun unit vector norm, max |1 - |s||", float(np.max(np.abs(norms - 1.0))), 0.0, 1e-15, kind="abs")
c.assert_true("perihelion falls in the first week of January", 1 <= i_min + 1 <= 7, f"(day {i_min + 1})")
c.assert_true("aphelion falls in the first week of July", 183 <= i_max + 1 <= 189, f"(day {i_max + 1})")

print("\n   sun_direction_for_beta reproduces the requested beta angle exactly:")
worst_beta = 0.0
for b_deg in (-80.0, -45.0, 0.0, 12.5, 45.0, 80.0):
    s = sun_direction_for_beta(inc, raan, np.radians(b_deg), phase_rad=0.7)
    got = np.degrees(beta_angle(s, inc, raan))
    worst_beta = max(worst_beta, abs(got - b_deg))
    print(f"     requested {b_deg:+7.2f} deg -> recovered {got:+13.9f} deg")
c.check("beta round trip, worst error [deg]", worst_beta, 0.0, 1e-10, kind="abs")

# ------------------------------------------------------------------------ 3. eclipse
print("\n3. Cylindrical eclipse fraction against its closed form")
print("""
   For a circular orbit of radius R about a sphere of radius Re, the umbra spans the
   argument-of-latitude interval where the Sun-line distance is below Re. That gives

       f_ecl(beta) = (1/pi) arccos( sqrt(R^2 - Re^2) / (R cos beta) ),

   which at beta = 0 reduces to (1/pi) arcsin(Re/R), and which is zero above the
   critical beta where the orbit never enters the shadow cylinder,
   cos(beta_crit) = sqrt(1 - (Re/R)^2).
""")
ratio = R_EARTH_EQUATORIAL / r_orbit
beta_crit = np.degrees(np.arccos(np.sqrt(1.0 - ratio**2)))
print(f"   Re/R = {ratio:.8f}   f_ecl(0) = arcsin(Re/R)/pi = {np.arcsin(ratio) / np.pi:.8f}")
print(f"   beta_crit = {beta_crit:.4f} deg")
for b_deg in (0.0, 20.0, 40.0, 60.0, 67.0):
    s = sun_direction_for_beta(inc, raan, np.radians(b_deg))
    num = eclipse_fraction_cylindrical(r_orbit, inc, raan, s, n_samples=200000)
    arg = np.sqrt(r_orbit**2 - R_EARTH_EQUATORIAL**2) / (r_orbit * np.cos(np.radians(b_deg)))
    ana = float(np.arccos(np.clip(arg, -1.0, 1.0)) / np.pi)
    c.check(f"eclipse fraction at beta = {b_deg:.0f} deg", num, ana, 2e-5, kind="abs")
s_high = sun_direction_for_beta(inc, raan, np.radians(beta_crit + 1.0))
c.check(
    "eclipse fraction 1 deg above beta_crit",
    eclipse_fraction_cylindrical(r_orbit, inc, raan, s_high, n_samples=200000),
    0.0,
    1e-12,
    kind="abs",
)

# -------------------------------------------------- 4. orbit-averaged magnetic field
print("\n4. Orbit-averaged dipole field: closed form vs numerical quadrature")
print("""
   With r_hat(u) = cos(u) P_hat + sin(u) Q_hat and sin(dec) = sin(i) sin(u),

       <B> = (k/R^3) [ z_hat - 3 <sin(i) sin(u) r_hat(u)> ]
           = (k/R^3) [ z_hat - (3/2) sin(i) Q_hat ]

   because <sin u cos u> = 0 and <sin^2 u> = 1/2 over a revolution. The numerical
   column is a trapezoidal average of the sampled field over one orbit. The tolerance
   below is 1e-10 relative, set by the accumulated float64 roundoff of a 100001-term
   trapezoidal sum, N * eps = 1e5 * 2.2e-16 = 2.2e-11, not by any modelling argument.
""")
for inc_deg in (0.0, 28.5, 51.6, 90.0, 97.8):
    ii = np.radians(inc_deg)
    uu = np.linspace(0.0, 2 * np.pi, 100001)
    rr, _ = circular_orbit_state(r_orbit, ii, raan, uu)
    num = np.trapezoid(dipole_field_eci(rr), uu, axis=0) / (2 * np.pi)
    ana = mean_dipole_field_over_orbit(r_orbit, ii, raan)
    rel = float(np.max(np.abs(num - ana)) / np.linalg.norm(ana))
    print(f"   i = {inc_deg:5.1f} deg  <B>_ana = [{ana[0]: .6e} {ana[1]: .6e} {ana[2]: .6e}] T")
    c.check(f"  numerical vs closed form at i = {inc_deg:.1f} deg", rel, 0.0, 1e-10, kind="abs")

b_500 = EARTH_DIPOLE_MOMENT / r_orbit**3
print(f"\n   Equatorial field magnitude at 500 km: k/R^3 = {b_500:.6e} T = {b_500 * 1e9:.0f} nT")
print(f"   Polar field magnitude at 500 km:      2k/R^3 = {2 * b_500:.6e} T = {2 * b_500 * 1e9:.0f} nT")

c.summary("orbit_geometry.py")
raise SystemExit(1 if c.n_fail else 0)
