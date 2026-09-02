"""Every expression in the package against arithmetic done by hand.

Run: ``python3 hand_calculations.py``

Each block states the numbers, the arithmetic and the expected result before calling the
code. Nothing here uses another part of the package to check a part of the package: the
targets are worked out from the printed formula, on paper, with the constants written out.
"""

from __future__ import annotations

import time

import numpy as np
from _common import Checks  # noqa: E402

from momentummgr import (  # noqa: E402
    EARTH_REDUCED_DIPOLE,
    MU_EARTH,
    SRP_PRESSURE_1AU,
    STANDARD_GRAVITY,
    aerodynamic_torque,
    density,
    dipole_field_eci,
    gravity_gradient_torque,
    gravity_gradient_worst_case,
    orbital_period,
    pyramid_four,
    residual_dipole_torque,
    srp_torque,
    thruster_dump,
)

c = Checks()
t0 = time.time()
print("Hand calculations")
print("=" * 90)

# ------------------------------------------------------------------- constants
print("""
0. Derived constants.
   P_1AU = Phi / c = 1361 / 299792458 = 4.5398073...e-06 N m^-2
   B0(surface, equator) = 7.96e15 / 6371200^3
       6371200^3 = 2.5859...e20 m^3, so B0 = 3.0778635...e-05 T
""")
c.check("solar radiation pressure at 1 AU", SRP_PRESSURE_1AU, 1361.0 / 299792458.0, 1e-15)
c.check("SRP numeric value", SRP_PRESSURE_1AU, 4.5398073e-06, 1e-8)
c.check("equatorial surface field from the reduced moment",
        EARTH_REDUCED_DIPOLE / 6371200.0**3, 3.0778635e-05, 1e-7)

# --------------------------------------------------------------- orbital period
print("""
1. Orbital period at 500 km. R = 6378137 + 500000 = 6878137 m.
   R^3 = 3.2539619...e20 m^3;  R^3 / mu = 3.25396e20 / 3.986004418e14 = 816355.3... s^2
   sqrt = 903.5238... s;  T = 2 pi * 903.5238 = 5676.9780... s
""")
r500 = 6378137.0 + 500000.0
period_hand = 2.0 * np.pi * np.sqrt(r500**3 / MU_EARTH)
print(f"   R^3 = {r500**3:.6e} m^3, T = {period_hand:.6f} s")
c.check("orbital period at 500 km", orbital_period(r500), period_hand, 1e-15)
c.check("orbital period against the value P027 publishes", orbital_period(r500), 5676.9780, 1e-7)

# ------------------------------------------------------------- gravity gradient
print("""
2. Gravity gradient, I = diag(4, 8, 10) kg m^2, nadir 30 deg off the body z axis in the
   y-z plane: u = (0, sin 30, cos 30) = (0, 0.5, 0.8660254038).
   I u = (0, 4.0, 8.660254038)
   u x (I u) = (0.5*8.660254038 - 0.8660254038*4.0, 0, 0) = (0.8660254038, 0, 0)
   3 mu / R^3 = 3*3.986004418e14 / 3.2539619e20 = 3.6749088...e-06 s^-2
   T = 3.6749088e-06 * 0.8660254038 = 3.1825644...e-06 N m about body x, zero elsewhere.
   Cross-check on the same number from the planar closed form
   T = (3 mu / 2R^3)(Izz - Iyy) sin 2 theta = 3.6749088e-06/2 * 2 * sin 60 deg. Same.
""")
u_nadir = np.array([0.0, 0.5, np.sqrt(3.0) / 2.0])
three_n2 = 3.0 * MU_EARTH / r500**3
hand_gg = np.array([three_n2 * (0.5 * (10.0 * np.sqrt(3.0) / 2.0) - (np.sqrt(3.0) / 2.0) * 4.0),
                    0.0, 0.0])
got = gravity_gradient_torque(np.diag([4.0, 8.0, 10.0]), u_nadir, r500)
print(f"   3 mu / R^3 = {three_n2:.9e} s^-2")
print(f"   hand  [{hand_gg[0]: .9e} {hand_gg[1]: .9e} {hand_gg[2]: .9e}] N m")
print(f"   code  [{got[0]: .9e} {got[1]: .9e} {got[2]: .9e}] N m")
c.check("gravity-gradient torque, x", float(got[0]), float(hand_gg[0]), 1e-14)
c.check("gravity-gradient torque, magnitude", float(np.linalg.norm(got)), 3.1825644e-06, 1e-7)
c.check("gravity-gradient torque, y and z are zero",
        float(abs(got[1]) + abs(got[2])), 0.0, 1e-22, kind="abs")
planar = 0.5 * three_n2 * (10.0 - 8.0) * np.sin(np.radians(60.0))
c.check("planar closed form gives the same number", float(got[0]), float(planar), 1e-14)
print("""
   Worst case: T_max = (3 mu / 2 R^3)|Izz - Iyy| = 3.6749088e-06 / 2 * 2 = 3.6749088e-06
   N m, attained at 45 deg. The code's worst-case helper must return exactly that.
""")
c.check("worst-case gravity gradient", gravity_gradient_worst_case(8.0, 10.0, r500),
        float(three_n2 * 0.5 * 2.0), 1e-15)

# --------------------------------------------------------------------- density
print("""
3. Exponential density at 500 km is a table base altitude, so no exponential is involved:
   rho(500 km) = 6.967e-13 kg m^-3 exactly. At 550 km,
   rho = 6.967e-13 * exp(-50/63.822) = 6.967e-13 * 0.456654... = 3.18150...e-13.
""")
print(f"   rho(500 km) = {float(density(500e3)):.6e} kg m^-3")
print(f"   rho(550 km) = {float(density(550e3)):.6e} kg m^-3")
c.check("density at 500 km", float(density(500e3)), 6.967e-13, 1e-15)
c.check("density at 550 km", float(density(550e3)), 6.967e-13 * np.exp(-50.0 / 63.822), 1e-15)

# ----------------------------------------------------------------- aerodynamic
print("""
4. Aerodynamic torque. rho = 6.967e-13 kg m^-3, v_rel = (7600, 0, 0) m s^-1, Cd = 2.2,
   A = 0.6 m^2, cp offset (0, 0, 0.05) m.
   0.5 * rho * Cd * A = 0.5 * 6.967e-13 * 2.2 * 0.6 = 4.598220e-13
   |F| = 4.598220e-13 * 7600^2 = 4.598220e-13 * 5.776e7 = 2.6559319...e-05 N, along -x
   T = (0, 0, 0.05) x (-2.6559319e-05, 0, 0) = (0, -1.3279659...e-06, 0) N m
""")
half_rho_cd_a = 0.5 * 6.967e-13 * 2.2 * 0.6
f_mag = half_rho_cd_a * 7600.0**2
hand_aero = np.array([0.0, -0.05 * f_mag, 0.0])
got = aerodynamic_torque(6.967e-13, [7600.0, 0.0, 0.0], 2.2, 0.6, [0.0, 0.0, 0.05])
print(f"   |F| = {f_mag:.9e} N")
print(f"   hand  [{hand_aero[0]: .9e} {hand_aero[1]: .9e} {hand_aero[2]: .9e}] N m")
print(f"   code  [{got[0]: .9e} {got[1]: .9e} {got[2]: .9e}] N m")
c.check("aerodynamic torque, y", float(got[1]), float(hand_aero[1]), 1e-14)
c.check("aerodynamic torque, y numeric", float(got[1]), -1.3279659e-06, 1e-7)

# ------------------------------------------------------------------------ SRP
print("""
5. Solar radiation pressure torque. P = 4.5398073e-06 N m^-2, A = 1.2 m^2, q = 0.6,
   Sun along body +z, cp offset (0.02, 0, 0) m.
   |F| = 4.5398073e-06 * 1.2 * 1.6 = 8.7164301...e-06 N, along -z
   T = (0.02, 0, 0) x (0, 0, -8.7164301e-06) = (0, 1.7432860...e-07, 0) N m
""")
f_srp = SRP_PRESSURE_1AU * 1.2 * 1.6
hand_srp = np.array([0.0, 0.02 * f_srp, 0.0])
got = srp_torque([0.0, 0.0, 1.0], 1.2, 0.6, [0.02, 0.0, 0.0])
print(f"   |F| = {f_srp:.9e} N")
print(f"   hand  [{hand_srp[0]: .9e} {hand_srp[1]: .9e} {hand_srp[2]: .9e}] N m")
print(f"   code  [{got[0]: .9e} {got[1]: .9e} {got[2]: .9e}] N m")
c.check("SRP torque, y", float(got[1]), float(hand_srp[1]), 1e-15)
c.check("SRP torque, y numeric", float(got[1]), 1.7432860e-07, 1e-7)
c.check("SRP torque is zero in eclipse",
        float(np.linalg.norm(srp_torque([0.0, 0.0, 1.0], 1.2, 0.6, [0.02, 0.0, 0.0],
                                        illuminated=False))), 0.0, 1e-30, kind="abs")

# ------------------------------------------------------------- residual dipole
print("""
6. Residual dipole torque. m = (0.05, 0.05, 0.10) A m^2, B = (0, 0, 3.0e-05) T.
   T = m x B = (0.05*3e-05 - 0.10*0, 0.10*0 - 0.05*3e-05, 0.05*0 - 0.05*0)
             = (1.5e-06, -1.5e-06, 0) N m
""")
got = residual_dipole_torque([0.05, 0.05, 0.10], [0.0, 0.0, 3.0e-5])
print(f"   code  [{got[0]: .9e} {got[1]: .9e} {got[2]: .9e}] N m")
c.check("dipole torque, x", float(got[0]), 1.5e-06, 1e-15)
c.check("dipole torque, y", float(got[1]), -1.5e-06, 1e-15)
c.check("dipole torque, z", float(got[2]), 0.0, 1e-30, kind="abs")

# ------------------------------------------------------------- dipole field
print("""
7. Dipole field. m_hat = -z_hat, so at r = (7e6, 0, 0) m the dot product m_hat . r_hat is
   zero and B = (k / r^3) z_hat with k / r^3 = 7.96e15 / 3.43e20 = 2.320699...e-05 T.
   Over the north pole r = (0, 0, 7e6): m_hat . r_hat = -1, so
   B = (k/r^3)(3(-1)(0,0,1) - (0,0,-1)) = (k/r^3)(0, 0, -2), magnitude twice the
   equatorial value and pointing into the Earth.
""")
b_eq = dipole_field_eci([7.0e6, 0.0, 0.0])
b_pole = dipole_field_eci([0.0, 0.0, 7.0e6])
print(f"   equator  [{b_eq[0]: .6e} {b_eq[1]: .6e} {b_eq[2]: .6e}] T")
print(f"   pole     [{b_pole[0]: .6e} {b_pole[1]: .6e} {b_pole[2]: .6e}] T")
c.check("equatorial field, z", float(b_eq[2]), EARTH_REDUCED_DIPOLE / 7.0e6**3, 1e-15)
c.check("equatorial field numeric", float(b_eq[2]), 2.320700e-05, 1e-6)
c.check("polar field is minus twice the equatorial", float(b_pole[2]),
        -2.0 * EARTH_REDUCED_DIPOLE / 7.0e6**3, 1e-15)

# ---------------------------------------------------------------- wheel array
print("""
8. Isotropic four-wheel pyramid. Half-angle b = arctan sqrt 2, so sin b = sqrt(2/3) =
   0.8164965809 and cos b = 1/sqrt 3 = 0.5773502692.
   A A^T = diag(2 sin^2 b, 2 sin^2 b, 4 cos^2 b) = diag(4/3, 4/3, 4/3): isotropic.
   A^+ = A^T (A A^T)^-1 = (3/4) A^T, so the largest row norm of A^+ is 3/4 and the
   guaranteed body envelope is h_max / (3/4) = (4/3) h_max = 0.0666666... N m s for
   0.05 N m s wheels.
   For a request along body z the minimum-norm solution puts (3/4) cos b h on every wheel:
   0.75 * 0.5773502692 * 0.02 = 8.660254...e-03 N m s each.
""")
w = pyramid_four()
aat = w.distribution_matrix @ w.distribution_matrix.T
print(f"   A A^T diagonal  {np.diag(aat).round(12).tolist()}")
print(f"   envelope        {w.guaranteed_body_envelope_nms:.10f} N m s")
alloc = w.allocate([0.0, 0.0, 0.02], avoid_zero_speed=False)
print(f"   min-norm wheels {alloc.wheel_momentum_nms.round(10).tolist()} N m s")
c.check("A A^T is 4/3 on the diagonal", float(np.diag(aat).min()), 4.0 / 3.0, 1e-14)
c.check("A A^T is diagonal", float(np.abs(aat - np.diag(np.diag(aat))).max()), 0.0, 1e-16,
        kind="abs")
c.check("guaranteed body envelope", w.guaranteed_body_envelope_nms, 4.0 / 3.0 * 0.05, 1e-14)
c.check("min-norm wheel momentum for a z request", float(alloc.wheel_momentum_nms[0]),
        0.75 / np.sqrt(3.0) * 0.02, 1e-14)
c.check("min-norm wheel momentum numeric", float(alloc.wheel_momentum_nms[0]), 8.660254e-03, 1e-6)

# ------------------------------------------------------------------ thrusters
print("""
9. Thruster dump. dh = 0.05 N m s, moment arm 0.5 m, thruster couple (two jets),
   Isp 220 s, efficiency 1.
   I = 2 * 0.05 / 0.5 = 0.2 N s
   m_p = 0.2 / (220 * 9.80665) = 0.2 / 2157.463 = 9.270147...e-05 kg
""")
d = thruster_dump(0.05, 0.5, 220.0)
print(f"   impulse {d.impulse_ns:.10f} N s, propellant {d.propellant_kg:.10e} kg")
c.check("thruster impulse", d.impulse_ns, 0.2, 1e-15)
c.check("thruster propellant", d.propellant_kg, 0.2 / (220.0 * STANDARD_GRAVITY), 1e-15)
c.check("thruster propellant numeric", d.propellant_kg, 9.270147e-05, 1e-6)
single = thruster_dump(0.05, 0.5, 220.0, couple=False)
c.check("a single jet costs half the propellant of a couple",
        single.propellant_kg, 0.5 * d.propellant_kg, 1e-15)

print(f"\nwall time {time.time() - t0:.2f} s")
c.summary("hand_calculations.py")
raise SystemExit(1 if c.n_fail else 0)
