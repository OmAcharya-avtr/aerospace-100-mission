"""Each torque expression evaluated for a geometry with a closed form, against hand
arithmetic written out in full below. Run: ``python3 hand_calculations.py``."""

from __future__ import annotations

import numpy as np

from _common import Checks  # noqa: E402

from disturbtorque import (  # noqa: E402
    EARTH_DIPOLE_MOMENT,
    MU_EARTH,
    R_EARTH_MEAN,
    SRP_PRESSURE_1AU,
    aerodynamic_torque,
    dipole_field_eci,
    dipole_field_magnitude,
    gravity_gradient_max_magnitude,
    gravity_gradient_planar,
    gravity_gradient_torque,
    magnetic_torque,
    solar_radiation_torque,
)

c = Checks()
np.set_printoptions(precision=12)

print("disturbtorque 0.1.0 - hand-checked closed forms")
print("=" * 78)
print(f"mu    = {MU_EARTH:.10e} m^3 s^-2   (WGS-84)")
print(f"P_srp = {SRP_PRESSURE_1AU:.10e} N m^-2  = 1361 / 299792458")
print(f"k_dip = {EARTH_DIPOLE_MOMENT:.10e} T m^3")
print()

# ---------------------------------------------------------------- 1. gravity gradient
print("1. Gravity gradient, I = diag(10, 20, 30) kg m^2, R = 7.000000e6 m, 45 deg off nadir")
print("""
   T = (3 mu / R^3) u_hat x (I u_hat),  u_hat = (0, sin45, cos45) = (0, 0.70710678, 0.70710678)

   R^3            = 3.43e20 m^3
   3 mu / R^3     = 3 * 3.986004418e14 / 3.43e20 = 1.1958013254e15 / 3.43e20
                  = 3.4863012402e-06 s^-2
   I u_hat        = (0, 20*0.70710678, 30*0.70710678) = (0, 14.14213562, 21.21320344)
   u_hat x I u    = ( 0.70710678*21.21320344 - 0.70710678*14.14213562, 0, 0 )
                  = ( 0.70710678 * 7.07106781, 0, 0 ) = (5.0, 0, 0) kg m^2
   T              = 3.4863012402e-06 * 5.0 = 1.7431506201e-05 N m about +x
""")
inertia = np.diag([10.0, 20.0, 30.0])
radius = 7.0e6
u45 = np.array([0.0, np.sqrt(0.5), np.sqrt(0.5)])
t_gg = gravity_gradient_torque(inertia, u45, radius)
hand_coeff = 3.0 * MU_EARTH / radius**3
c.check("3 mu / R^3 [s^-2]", hand_coeff, 3.4863012402e-06, 1e-9)
c.check("T_gg,x at 45 deg [N m]", float(t_gg[0]), 1.7431506201e-05, 1e-9)
c.check("T_gg,y [N m]", float(t_gg[1]), 0.0, 1e-20, kind="abs")
c.check("T_gg,z [N m]", float(t_gg[2]), 0.0, 1e-20, kind="abs")

print("\n   Analytic maximum: T = (3 mu / 2R^3) |Iz - Iy| sin(2 theta), max at theta = 45 deg")
print("   (3 mu / 2R^3) * 10 = 1.7431506201e-06 * 10 = 1.7431506201e-05 N m")
t_max = gravity_gradient_max_magnitude(20.0, 30.0, radius)
c.check("gravity_gradient_max_magnitude [N m]", t_max, 1.7431506201e-05, 1e-9)
c.check("full tensor form at 45 deg equals the analytic max", float(t_gg[0]), t_max, 1e-14)

theta = np.linspace(0.0, np.pi / 2, 90001)
planar = gravity_gradient_planar(20.0, 30.0, theta, radius)
i_arg = int(np.argmax(np.abs(planar)))
print(f"\n   Numerical argmax over 90001 samples in [0, 90] deg: "
      f"theta = {np.degrees(theta[i_arg]):.6f} deg")
c.check("argmax theta [deg]", float(np.degrees(theta[i_arg])), 45.0, 1e-4, kind="abs")
c.check("max over sweep [N m]", float(np.max(np.abs(planar))), t_max, 1e-12)

# full-tensor sweep of the same rotation, to prove the planar form is not a separate model
u_sweep = np.stack([np.zeros_like(theta), np.sin(theta), np.cos(theta)], axis=1)
tensor_sweep = np.array(
    [gravity_gradient_torque(inertia, u_sweep[i], radius)[0] for i in range(0, len(theta), 500)]
)
planar_sub = planar[::500]
# tolerance is 1e-14 of the torque scale (1.743e-05 N m), i.e. ~50 ulp of float64
c.check(
    "tensor form vs planar form, max diff / T_max",
    float(np.max(np.abs(tensor_sweep - planar_sub)) / t_max),
    0.0,
    1e-14,
    kind="abs",
)
c.assert_true(
    "torque vanishes when nadir is a principal axis",
    float(np.linalg.norm(gravity_gradient_torque(inertia, [0, 0, 1], radius))) == 0.0,
)

# ------------------------------------------------------------------- 2. aerodynamic
print("\n2. Aerodynamic, rho = 1e-12 kg m^-3, v = (7500, 0, 0) m s^-1, Cd = 2.2, A = 1.5 m^2,")
print("   cp offset = (0, 0, 0.1) m")
print("""
   F = -0.5 rho Cd A |v| v
   0.5 * 2.2      = 1.1
   1.1 * 1.5      = 1.65
   1.65 * 1e-12   = 1.65e-12
   |v|^2          = 5.625e7 m^2 s^-2
   |F|            = 1.65e-12 * 5.625e7 = 9.28125e-05 N, along -x
   T = r x F      = (0,0,0.1) x (-9.28125e-05, 0, 0) = (0, -9.28125e-06, 0) N m
""")
t_aero = aerodynamic_torque(1e-12, np.array([7500.0, 0.0, 0.0]), 2.2, 1.5, np.array([0, 0, 0.1]))
c.check("T_aero,y [N m]", float(t_aero[1]), -9.28125e-06, 1e-12)
c.check("T_aero,x [N m]", float(t_aero[0]), 0.0, 1e-20, kind="abs")
c.check("T_aero,z [N m]", float(t_aero[2]), 0.0, 1e-20, kind="abs")
c.assert_true(
    "torque vanishes when the cp offset is parallel to the flow",
    float(
        np.linalg.norm(
            aerodynamic_torque(1e-12, [7500.0, 0, 0], 2.2, 1.5, [0.1, 0, 0])
        )
    )
    == 0.0,
)

# --------------------------------------------------------------------------- 3. SRP
p_hand = 1361.0 / 299792458.0
print("\n3. Solar radiation pressure, sun along +z_body, A = 2.0 m^2, q = 0.6,")
print("   cp offset = (0.3, 0, 0) m, d = 1 AU")
print(f"""
   P   = Phi / c = 1361 / 299792458 = {p_hand:.10e} N m^-2
   |F| = P A (1+q) = {p_hand:.10e} * 2.0 * 1.6 = {p_hand * 3.2:.10e} N, along -z
   T   = (0.3,0,0) x (0,0,-|F|) = (0, +0.3*|F|, 0) = (0, {0.3 * p_hand * 3.2:.10e}, 0) N m
""")
t_srp = solar_radiation_torque([0.0, 0.0, 1.0], 2.0, 0.6, [0.3, 0.0, 0.0])
c.check("P_srp at 1 AU [N m^-2]", SRP_PRESSURE_1AU, p_hand, 1e-15)
c.check("T_srp,y [N m]", float(t_srp[1]), 0.3 * p_hand * 3.2, 1e-12)
c.check("T_srp,x [N m]", float(t_srp[0]), 0.0, 1e-20, kind="abs")
c.check(
    "1/d^2 scaling at d = 2 AU",
    float(solar_radiation_torque([0, 0, 1.0], 2.0, 0.6, [0.3, 0, 0], distance_au=2.0)[1]),
    0.25 * 0.3 * p_hand * 3.2,
    1e-12,
)
c.assert_true(
    "torque is zero in eclipse",
    float(
        np.linalg.norm(solar_radiation_torque([0, 0, 1.0], 2.0, 0.6, [0.3, 0, 0], illuminated=False))
    )
    == 0.0,
)

# ---------------------------------------------------------------------- 4. magnetic
print("\n4. Residual magnetic dipole, m = (0.1, 0, 0) A m^2, B = (0, 3e-5, 0) T")
print("""
   T = m x B = (0.1, 0, 0) x (0, 3e-05, 0) = (0, 0, 0.1 * 3e-05) = (0, 0, 3e-06) N m
   1 A m^2 * 1 T = 1 N m exactly, so no unit conversion appears.
""")
t_mag = magnetic_torque([0.1, 0.0, 0.0], [0.0, 3e-5, 0.0])
c.check("T_mag,z [N m]", float(t_mag[2]), 3e-06, 1e-14)
c.check("T_mag,x [N m]", float(t_mag[0]), 0.0, 1e-20, kind="abs")
c.assert_true(
    "torque vanishes for m parallel to B",
    float(np.linalg.norm(magnetic_torque([0.1, 0, 0], [3e-5, 0, 0]))) == 0.0,
)

print("\n   Dipole field at the mean Earth radius:")
print(f"""
   k / Re^3 = {EARTH_DIPOLE_MOMENT:.4e} / {R_EARTH_MEAN:.7e}^3
            = {EARTH_DIPOLE_MOMENT / R_EARTH_MEAN**3:.10e} T   (equatorial, northward)
   at the pole the magnitude is twice that: {2 * EARTH_DIPOLE_MOMENT / R_EARTH_MEAN**3:.10e} T
""")
b_eq = dipole_field_eci([R_EARTH_MEAN, 0.0, 0.0])
b_pole = dipole_field_eci([0.0, 0.0, R_EARTH_MEAN])
k_over_re3 = EARTH_DIPOLE_MOMENT / R_EARTH_MEAN**3
c.check("|B| equator surface [T]", float(np.linalg.norm(b_eq)), k_over_re3, 1e-14)
c.check("B_z equator surface [T] (northward)", float(b_eq[2]), k_over_re3, 1e-14)
c.check("|B| north pole surface [T]", float(np.linalg.norm(b_pole)), 2 * k_over_re3, 1e-14)
c.check("B_z north pole surface [T] (into the Earth)", float(b_pole[2]), -2 * k_over_re3, 1e-14)
dec = np.linspace(-np.pi / 2, np.pi / 2, 401)
r_vec = R_EARTH_MEAN * np.stack([np.cos(dec), np.zeros_like(dec), np.sin(dec)], axis=1)
c.check(
    "vector vs closed-form |B| over declination, max rel diff",
    float(
        np.max(
            np.abs(
                np.linalg.norm(dipole_field_eci(r_vec), axis=1)
                / dipole_field_magnitude(R_EARTH_MEAN, dec)
                - 1.0
            )
        )
    ),
    0.0,
    1e-14,
    kind="abs",
)

c.summary("hand_calculations.py")
raise SystemExit(1 if c.n_fail else 0)
