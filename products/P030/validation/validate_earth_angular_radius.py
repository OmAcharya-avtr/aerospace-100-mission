"""Earth angular radius against the analytic expression and an independent route.

Checks
------
1. ``keepout.earth_angular_radius(h)`` against ``arcsin(R_E / (R_E + h))``
   written out directly, at a table of altitudes from the surface to beyond
   geostationary.
2. The same values against a route that never uses the tangency identity: the
   angle subtended at the observer by a point on the sphere's surface is
   maximised numerically over the surface. The maximum of that angle is the
   apparent angular radius, by definition; ``arcsin(R/d)`` is the closed form
   of it, and this check confirms the closed form rather than assuming it.
3. Limiting behaviour: 90 deg at the surface, monotone decrease with altitude,
   and the small-angle limit ``alpha -> R_E / (R_E + h)``.
4. The limb-referenced cone half-angle built by ``body_exclusion_cone`` equals
   ``arcsin(R_E / (R_E + h)) + instrument angle``.

Run from products/P030/:  python validation/validate_earth_angular_radius.py
"""

import pathlib
import sys

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from keepout import (  # noqa: E402
    EARTH_RADIUS_M,
    body_exclusion_cone,
    earth_angular_radius,
)

ALTITUDES_M = [
    0.0, 1.0e3, 10.0e3, 100.0e3, 200.0e3, 300.0e3, 400.0e3, 500.0e3, 550.0e3,
    700.0e3, 800.0e3, 1000.0e3, 2000.0e3, 5000.0e3, 10000.0e3, 20200.0e3,
    35786.0e3, 100000.0e3, 384400.0e3,
]


def subtended_angle(psi: float, d: float, r: float) -> float:
    """Angle at the observer between the body centre and a surface point.

    Observer at distance ``d`` on the ``+x`` axis; the surface point is at polar
    angle ``psi`` from the observer direction, ``P = r (cos psi, sin psi)``.
    No tangency assumption is made anywhere.
    """
    px, py = r * np.cos(psi), r * np.sin(psi)
    vx, vy = px - d, py
    to_centre = np.array([-d, 0.0])
    v = np.array([vx, vy])
    cos_t = float(np.dot(v, to_centre) / (np.linalg.norm(v) * np.linalg.norm(to_centre)))
    return float(np.arccos(np.clip(cos_t, -1.0, 1.0)))


def numeric_angular_radius(d: float, r: float) -> float:
    """Maximise the subtended angle over the visible surface, numerically."""
    res = minimize_scalar(
        lambda psi: -subtended_angle(psi, d, r),
        bounds=(1e-9, np.pi - 1e-9),
        method="bounded",
        options={"xatol": 1e-13},
    )
    return float(-res.fun)


def main() -> None:
    print("=" * 96)
    print("KeepOut validation: Earth angular radius, arcsin(R_E / (R_E + h))")
    print("=" * 96)
    print(f"R_E = {EARTH_RADIUS_M:.1f} m (WGS-84 equatorial, NIMA TR8350.2 3rd ed.)")
    print()

    print("1 & 2. Library vs the analytic expression and vs numerical maximisation")
    header = (f"{'h [km]':>10} {'library [deg]':>16} {'arcsin(R/(R+h)) [deg]':>23} "
              f"{'|diff| [rad]':>13} {'numeric max [deg]':>19} {'|diff| [rad]':>13}")
    print(header)
    worst_analytic = 0.0
    worst_numeric = 0.0
    for h in ALTITUDES_M:
        lib = float(earth_angular_radius(h))
        ana = float(np.arcsin(EARTH_RADIUS_M / (EARTH_RADIUS_M + h)))
        num = numeric_angular_radius(EARTH_RADIUS_M + h, EARTH_RADIUS_M)
        d1, d2 = abs(lib - ana), abs(lib - num)
        worst_analytic = max(worst_analytic, d1)
        worst_numeric = max(worst_numeric, d2)
        print(f"{h / 1e3:10.1f} {np.degrees(lib):16.10f} {np.degrees(ana):23.10f} "
              f"{d1:13.3e} {np.degrees(num):19.10f} {d2:13.3e}")
    print(f"   worst |library - analytic| = {worst_analytic:.3e} rad   tolerance 0 (bit-exact)"
          f"   {'PASS' if worst_analytic == 0.0 else 'FAILED'}")
    print(f"   worst |library - numeric maximisation| = {worst_numeric:.3e} rad"
          f"   tolerance 1e-9   {'PASS' if worst_numeric < 1e-9 else 'FAILED'}")
    print()

    print("3. Limits and monotonicity")
    surface = float(earth_angular_radius(0.0))
    print(f"   h = 0            : {np.degrees(surface):.12f} deg "
          f"(exact 90 deg, |diff| = {abs(surface - np.pi / 2):.3e} rad)")
    h_fine = np.linspace(0.0, 5.0e7, 20001)
    alpha = np.asarray(earth_angular_radius(h_fine))
    monotone = bool(np.all(np.diff(alpha) < 0.0))
    print(f"   strictly decreasing over h = 0 to 5e7 m, 20001 samples: {monotone}")
    print(f"{'h [km]':>12} {'alpha [rad]':>16} {'R/(R+h) [rad]':>16} {'relative diff':>15}")
    for h in (1e9, 1e10, 1e11, 1e12):
        a = float(earth_angular_radius(h))
        small = EARTH_RADIUS_M / (EARTH_RADIUS_M + h)
        print(f"{h / 1e3:12.3e} {a:16.9e} {small:16.9e} {abs(a - small) / small:15.3e}")
    limit_ok = abs(surface - np.pi / 2) < 1e-15 and monotone
    print(f"   {'PASS' if limit_ok else 'FAILED'}")
    print()

    print("4. Limb-referenced cone half-angle = arcsin(R_E/(R_E+h)) + instrument angle")
    print(f"{'h [km]':>10} {'instrument [deg]':>18} {'cone half-angle [deg]':>23} {'|diff| [rad]':>13}")
    worst_cone = 0.0
    for h, instr_deg in ((400e3, 10.0), (550e3, 10.0), (700e3, 20.0), (35786e3, 5.0)):
        rho = float(earth_angular_radius(h))
        cone = body_exclusion_cone(
            "earth", [0.0, 0.0, -1.0], rho, np.radians(instr_deg), "limb"
        )
        expected = rho + np.radians(instr_deg)
        diff = abs(cone.half_angle - expected)
        worst_cone = max(worst_cone, diff)
        print(f"{h / 1e3:10.1f} {instr_deg:18.1f} {cone.half_angle_deg:23.10f} {diff:13.3e}")
    print(f"   worst |diff| = {worst_cone:.3e} rad   tolerance 1e-15   "
          f"{'PASS' if worst_cone < 1e-15 else 'FAILED'}")
    print()

    passed = (
        worst_analytic == 0.0 and worst_numeric < 1e-9 and limit_ok and worst_cone < 1e-15
    )
    print("=" * 96)
    print(f"OVERALL: {'PASS' if passed else 'FAILED'}")
    print("=" * 96)


if __name__ == "__main__":
    main()
