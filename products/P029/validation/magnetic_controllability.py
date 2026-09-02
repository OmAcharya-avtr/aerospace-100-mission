"""The magnetic desaturation controllability constraint, demonstrated and quantified.

Run: ``python3 magnetic_controllability.py``

A magnetorquer can only produce ``T = m x B``, so at any instant the achievable torque
lies in the plane perpendicular to the field and the momentum component along ``B`` is
untouchable. This script does not assert that; it measures it, five ways:

1. the rank and singular values of the cross-product operator, which is where the
   constraint comes from;
2. the exact orthogonality of every commanded torque to the field;
3. how much of a fixed momentum direction is uncontrollable at each point of a real orbit,
   over a range of inclinations;
4. the time-averaged controllability Gramian and its eigenvalues, which say how much
   longer the worst direction takes to dump than the best;
5. a closed-loop dump in a frozen field, where the parallel component provably never
   decays, against the same dump in the real rotating field, where it does.
"""

from __future__ import annotations

import time

import numpy as np
from _common import Checks  # noqa: E402

from momentummgr import (  # noqa: E402
    averaged_controllability,
    circular_state,
    dipole_field_eci,
    magnetic_dump_command,
    reference_orbit,
    uncontrollable_fraction,
)

c = Checks()
t0 = time.time()
print("Magnetic desaturation: the direction you cannot dump")
print("=" * 90)

# ------------------------------------------------------------ 1. rank of [B x]
print("""
1. The constraint is the rank of the cross-product operator.
   For any B the matrix [B x] has singular values (|B|, |B|, 0): two directions of torque
   authority and one null direction, along B itself. This is exact linear algebra, not a
   modelling choice, so no control law of any kind can escape it.
""")
rng = np.random.default_rng(20260902)
worst_sigma3 = 0.0
worst_sigma_ratio = 0.0
for _ in range(200):
    b = rng.normal(size=3) * 1e-5
    skew = np.array([[0.0, -b[2], b[1]], [b[2], 0.0, -b[0]], [-b[1], b[0], 0.0]])
    sv = np.linalg.svd(skew, compute_uv=False)
    worst_sigma3 = max(worst_sigma3, float(sv[2] / np.linalg.norm(b)))
    worst_sigma_ratio = max(worst_sigma_ratio, abs(float(sv[0] / sv[1]) - 1.0))
    worst_sigma_ratio = max(worst_sigma_ratio, abs(float(sv[0] / np.linalg.norm(b)) - 1.0))
print(f"   worst |sigma_3| / |B| over 200 random fields   {worst_sigma3:.3e}")
print(f"   worst deviation of sigma_1, sigma_2 from |B|   {worst_sigma_ratio:.3e}")
c.check("third singular value of [B x] is zero", worst_sigma3, 0.0, 1e-15, kind="abs")
c.check("first two singular values of [B x] equal |B|", worst_sigma_ratio, 0.0, 1e-14, kind="abs")

# ------------------------------------------- 2. every commanded torque is perpendicular
print("""
2. Every torque the dumping law can command is exactly perpendicular to B, and the
   commanded dipole removes exactly the perpendicular part of the momentum:
   T = m x B = -k [h - (h.B_hat) B_hat] before the dipole limit bites.
""")
worst_perp = 0.0
worst_law = 0.0
for _ in range(200):
    h = rng.normal(size=3) * 0.02
    b = rng.normal(size=3) * 1e-5
    cmd = magnetic_dump_command(h, b, gain=1.0)
    b_hat = b / np.linalg.norm(b)
    worst_perp = max(worst_perp, abs(float(cmd.torque_nm @ b_hat)) / float(
        np.linalg.norm(cmd.torque_nm)))
    expected = -(h - float(h @ b_hat) * b_hat)
    worst_law = max(
        worst_law,
        float(np.linalg.norm(cmd.torque_nm - expected) / np.linalg.norm(expected)),
    )
print(f"   worst |T . B_hat| / |T| over 200 random cases   {worst_perp:.3e}")
print(f"   worst deviation from -k h_perp                  {worst_law:.3e}")
c.check("commanded torque is perpendicular to B", worst_perp, 0.0, 1e-14, kind="abs")
c.check("cross-product law removes exactly h_perp", worst_law, 0.0, 1e-14, kind="abs")

print("""
   Hand check. B = (0, 0, 3.0e-05) T, h = (0.01, 0, 0) N m s, gain k = 1 s^-1.
   B x h = (0*0 - 3e-05*0, 3e-05*0.01 - 0*0, 0) = (0, 3e-07, 0)
   m = -(k/|B|^2)(B x h) = -(1/9e-10)(0, 3e-07, 0) = (0, -333.3333... , 0) A m^2
   T = m x B = (-333.333... * 3e-05, 0, 0) = (-0.01, 0, 0) N m = -k h, as it must be
   because h is already perpendicular to B.
""")
cmd = magnetic_dump_command([0.01, 0.0, 0.0], [0.0, 0.0, 3.0e-5], gain=1.0)
c.check("hand check: m_y", float(cmd.dipole_am2[1]), -0.01 / 3.0e-5, 1e-14)
c.check("hand check: T_x", float(cmd.torque_nm[0]), -0.01, 1e-14)
c.check("hand check: uncontrollable fraction is zero", cmd.uncontrollable_fraction, 0.0, 1e-15,
        kind="abs")
cmd_par = magnetic_dump_command([0.0, 0.0, 0.01], [0.0, 0.0, 3.0e-5], gain=1.0)
print(f"   momentum along B: |m| = {np.linalg.norm(cmd_par.dipole_am2):.3e} A m^2, "
      f"|T| = {np.linalg.norm(cmd_par.torque_nm):.3e} N m, "
      f"uncontrollable fraction = {cmd_par.uncontrollable_fraction:.6f}")
c.check("momentum along B produces zero dipole", float(np.linalg.norm(cmd_par.dipole_am2)),
        0.0, 1e-18, kind="abs")
c.check("momentum along B is 100 % uncontrollable", cmd_par.uncontrollable_fraction, 1.0, 1e-15)

# ------------------------------------- 3. uncontrollable fraction along a real orbit
print("""
3. How bad is it on a real orbit? For a momentum fixed along ECI z (the worst case for a
   dipole field, whose direction is dominated by z), the uncontrollable fraction is
   sampled at 3601 points of one orbit, at 500 km, for a range of inclinations.
""")
n = 3601
u = np.linspace(0.0, 2.0 * np.pi, n)
h_dir = np.array([0.0, 0.0, 1.0])
print(f"   {'i [deg]':>9}{'mean frac':>12}{'max frac':>11}{'orbit above 0.5':>18}"
      f"{'orbit above 0.9':>18}")
print("   " + "-" * 68)
rows = []
for inc_deg in (0.0, 28.5, 51.6, 70.0, 90.0, 98.0):
    orbit = reference_orbit(500.0)
    orbit = type(orbit)(
        altitude_m=orbit.altitude_m, inclination_rad=np.radians(inc_deg), raan_rad=0.0,
        pitch_rad=orbit.pitch_rad, roll_rad=orbit.roll_rad,
    )
    r, _ = circular_state(orbit.radius_m, orbit.inclination_rad, orbit.raan_rad, u)
    b = dipole_field_eci(r)
    frac = uncontrollable_fraction(np.repeat(h_dir[None, :], n, axis=0), b)
    rows.append((inc_deg, float(frac.mean()), float(frac.max()),
                 float(np.mean(frac > 0.5)), float(np.mean(frac > 0.9))))
    print(f"   {inc_deg:>9.1f}{frac.mean():>12.4f}{frac.max():>11.4f}"
          f"{np.mean(frac > 0.5):>18.4f}{np.mean(frac > 0.9):>18.4f}")
print("""
   Read the first row. At zero inclination the field is exactly along -z everywhere on
   the orbit, so a momentum along z is 100 % uncontrollable for 100 % of the orbit: an
   equatorial spacecraft cannot magnetically dump momentum along the orbit normal at all.
   That is not a defect of the law; it is the geometry, and it is why equatorial missions
   carry thrusters for that axis.
""")
c.check("equatorial orbit: momentum along z is fully uncontrollable everywhere",
        rows[0][1], 1.0, 1e-12)
best = min(rows[1:], key=lambda r: r[1])
c.assert_true(
    "the mean uncontrollable fraction is NOT monotone in inclination",
    not all(rows[i][1] > rows[i + 1][1] for i in range(len(rows) - 1)),
    "means: " + ", ".join(f"{r[1]:.3f}" for r in rows),
)
print(f"""
   Finding, reported because it is a trap. The mean uncontrollable fraction for a
   z-directed momentum is **not** monotone in inclination. It falls from 1.000 at 0 deg
   to a minimum of {best[1]:.4f} at {best[0]:.1f} deg and then rises again to
   {rows[-2][1]:.4f} at 90 deg. A near-polar orbit is *worse* than a mid-inclination one
   for dumping momentum along the Earth's rotation axis, because near the poles the
   dipole field is itself close to the z axis. Anyone who assumes "more inclination is
   more magnetic authority" will size a polar mission wrongly.
""")
c.assert_true(
    "mid-inclination beats polar for a z-directed momentum",
    best[1] < rows[-2][1],
    f"{best[1]:.4f} at {best[0]:.1f} deg vs {rows[-2][1]:.4f} at 90 deg",
)

# ---------------------------------------------- 4. averaged controllability Gramian
print("""
4. The time-averaged Gramian G = <I - B_hat B_hat^T> over one orbit. Its trace is exactly
   2 for any field history. Its eigenvalues are the fraction of the orbit for which each
   principal direction is dumpable; the ratio of the largest to the smallest is how much
   longer the worst direction takes than the best, for the same gain.
""")
print(f"   {'i [deg]':>9}{'trace':>10}{'eig_min':>11}{'eig_mid':>11}{'eig_max':>11}"
      f"{'max/min':>11}")
print("   " + "-" * 63)
grams = []
for inc_deg in (0.0, 28.5, 51.6, 70.0, 90.0, 98.0):
    orbit = reference_orbit(500.0)
    orbit = type(orbit)(
        altitude_m=orbit.altitude_m, inclination_rad=np.radians(inc_deg), raan_rad=0.0,
    )
    r, _ = circular_state(orbit.radius_m, orbit.inclination_rad, orbit.raan_rad, u)
    b = dipole_field_eci(r)
    t = u / (2.0 * np.pi) * orbit.period_s
    gram, eig, _ = averaged_controllability(b, t)
    grams.append((inc_deg, float(gram.trace()), eig))
    ratio = float(eig[2] / eig[0]) if eig[0] > 1e-12 else float("inf")
    print(f"   {inc_deg:>9.1f}{gram.trace():>10.6f}{eig[0]:>11.6f}{eig[1]:>11.6f}"
          f"{eig[2]:>11.6f}{ratio:>11.3f}")
worst_trace = max(abs(t - 2.0) for _, t, _ in grams)
c.check("Gramian trace is exactly 2 at every inclination", worst_trace, 0.0, 1e-12, kind="abs")
c.check("equatorial orbit: smallest Gramian eigenvalue is zero", float(grams[0][2][0]), 0.0,
        1e-12, kind="abs")
c.assert_true(
    "every non-equatorial inclination tested has all three eigenvalues > 0",
    all(float(e[0]) > 1e-3 for _, _, e in grams[1:]),
    "smallest eigenvalues: " + ", ".join(f"{float(e[0]):.4f}" for _, _, e in grams[1:]),
)
print("""
   The 51.6 deg row is the case used everywhere else in this package: the worst direction
   is dumpable for 60.5 % of the orbit and the best for 77.8 %, a ratio of 1.286. So the
   averaged controllability is comfortable at that inclination, and the constraint shows
   up as a scheduling opportunity (dump when sin(theta) is large) rather than as a
   blocked axis. At zero inclination one axis is blocked outright.
""")

# ------------------------------------- 5. closed loop, frozen field vs rotating field
print("""
5. Closed-loop demonstration. Start with 0.02 N m s along the instantaneous field
   direction and run the cross-product law for one orbit, with no disturbance torque and
   no dipole limit, twice: once with the field frozen at its initial value, once with the
   real field along the orbit. Gain 5e-4 s^-1.
""")
orbit = reference_orbit(500.0)
r, _ = circular_state(orbit.radius_m, orbit.inclination_rad, orbit.raan_rad, u)
b_hist = dipole_field_eci(r)
gain = 5.0e-4
dt = orbit.period_s / (n - 1)
for label, frozen in (("frozen field", True), ("real rotating field", False)):
    b0 = b_hist[0]
    h = 0.02 * b0 / np.linalg.norm(b0)
    h0 = h.copy()
    parallel0 = float(h @ (b0 / np.linalg.norm(b0)))
    for i in range(n - 1):
        b = b0 if frozen else b_hist[i]
        cmd = magnetic_dump_command(h, b, gain=gain)
        h = h + dt * cmd.torque_nm
    parallel = float(h @ (b0 / np.linalg.norm(b0)))
    print(f"   {label:<22} |h| {np.linalg.norm(h0):.6f} -> {np.linalg.norm(h):.6f} N m s, "
          f"component along initial B {parallel0:.6f} -> {parallel:.6f} N m s")
    if frozen:
        c.check("frozen field: momentum along B is unchanged after a full orbit",
                parallel, parallel0, 1e-12)
        c.check("frozen field: |h| is unchanged after a full orbit",
                float(np.linalg.norm(h)), 0.02, 1e-12)
    else:
        ratio = float(np.linalg.norm(h)) / 0.02
        c.assert_true(
            "rotating field: the same momentum is reduced by at least a factor 3",
            ratio < 1.0 / 3.0,
            f"|h| {np.linalg.norm(h):.6e} N m s from 2.0e-02, ratio {ratio:.4f}",
        )
        print(f"   measured reduction over one orbit: factor {1.0 / ratio:.3f} "
              f"({100.0 * (1.0 - ratio):.1f} % of the initial momentum removed) at "
              f"gain 5e-4 s^-1")
print("""
   That pair is the whole point. Magnetic desaturation is not instantaneously
   controllable in three axes and never will be; it is controllable on average because
   the field direction sweeps. Any schedule that tries to dump a specific direction
   inside a short window is fighting the geometry, and any budget that assumes three-axis
   authority at every instant is wrong.
""")

print(f"wall time {time.time() - t0:.2f} s")
c.summary("magnetic_controllability.py")
raise SystemExit(1 if c.n_fail else 0)
