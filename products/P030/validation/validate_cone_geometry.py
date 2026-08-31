"""Cone-intersection geometry against closed-form spherical geometry.

Checks
------
1. Cap solid angle ``2 pi (1 - cos a)`` against adaptive numerical quadrature of
   ``int_0^a 2 pi sin(theta) d(theta)``.
2. The two-cap lens formula against three exactly-known special cases:
   coincident caps, disjoint caps, and two hemispheres whose intersection is a
   lune of area ``2 (pi - d)``.
3. The two-cap lens formula (Gauss-Bonnet) against the band quadrature in
   ``keepout.regions`` (per-ring azimuth arcs plus Gauss-Legendre) over 300
   random configurations. Two entirely different algorithms.
4. Convergence of the band quadrature with node count.
5. Both of the above against Monte Carlo integration with a binomial error bar.

Run from products/P030/:  python validation/validate_cone_geometry.py
"""

import pathlib
import sys

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from keepout import (  # noqa: E402
    ExclusionCone,
    KeepOutSet,
    allowed_solid_angle,
    allowed_solid_angle_monte_carlo,
    angular_separation,
    cap_intersection_solid_angle,
    cap_solid_angle,
    unit,
)

FULL = 4.0 * np.pi


def check_cap_solid_angle() -> float:
    print("1. Cap solid angle vs adaptive quadrature of int 2 pi sin(theta) d(theta)")
    print(f"{'alpha [deg]':>12} {'closed form [sr]':>20} {'quadrature [sr]':>20} {'|diff|':>12}")
    worst = 0.0
    for a_deg in (0.5, 1.0, 5.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 179.0, 180.0):
        a = np.radians(a_deg)
        closed = cap_solid_angle(a)
        num, _ = quad(lambda t: 2.0 * np.pi * np.sin(t), 0.0, a, epsabs=1e-13, epsrel=1e-13)
        diff = abs(closed - num)
        worst = max(worst, diff)
        print(f"{a_deg:12.1f} {closed:20.15f} {num:20.15f} {diff:12.3e}")
    print(f"   worst |diff| = {worst:.3e} sr   tolerance 1e-12   "
          f"{'PASS' if worst < 1e-12 else 'FAILED'}\n")
    return worst


def check_special_cases() -> float:
    print("2. Two-cap lens formula vs exactly known special cases")
    worst = 0.0
    rows = []
    for r_deg in (10.0, 45.0, 80.0):
        r = np.radians(r_deg)
        rows.append(
            (f"coincident caps r={r_deg:g} deg -> cap area",
             cap_intersection_solid_angle(r, r, 0.0), float(cap_solid_angle(r)))
        )
    for r1_deg, r2_deg, d_deg in ((10.0, 15.0, 30.0), (20.0, 20.0, 41.0)):
        rows.append(
            (f"disjoint r1={r1_deg:g} r2={r2_deg:g} d={d_deg:g} deg -> 0",
             cap_intersection_solid_angle(
                 np.radians(r1_deg), np.radians(r2_deg), np.radians(d_deg)), 0.0)
        )
    for d_deg in (5.0, 30.0, 90.0, 150.0, 175.0):
        d = np.radians(d_deg)
        rows.append(
            (f"two hemispheres d={d_deg:g} deg -> lune 2(pi - d)",
             cap_intersection_solid_angle(np.pi / 2, np.pi / 2, d), 2.0 * (np.pi - d))
        )
    for r1_deg, r2_deg, d_deg in ((170.0, 170.0, 30.0), (175.0, 100.0, 120.0)):
        r1, r2, d = np.radians([r1_deg, r2_deg, d_deg])
        rows.append(
            (f"complements disjoint r1={r1_deg:g} r2={r2_deg:g} d={d_deg:g} -> A1+A2-4pi",
             cap_intersection_solid_angle(r1, r2, d),
             float(cap_solid_angle(r1) + cap_solid_angle(r2) - FULL))
        )
    print(f"{'case':>62} {'library [sr]':>18} {'exact [sr]':>18} {'|diff|':>11}")
    for label, got, exact in rows:
        diff = abs(got - exact)
        worst = max(worst, diff)
        print(f"{label:>62} {got:18.14f} {exact:18.14f} {diff:11.3e}")
    print(f"   worst |diff| = {worst:.3e} sr   tolerance 1e-13   "
          f"{'PASS' if worst < 1e-13 else 'FAILED'}\n")
    return worst


def check_against_band_quadrature(n_cases: int = 300, seed: int = 20260831) -> float:
    print(f"3. Lens formula (Gauss-Bonnet) vs band quadrature, {n_cases} random configurations")
    rng = np.random.default_rng(seed)
    worst = 0.0
    worst_case = None
    for _ in range(n_cases):
        a1 = unit(rng.normal(size=3))
        a2 = unit(rng.normal(size=3))
        r1 = rng.uniform(0.05, np.pi - 0.05)
        r2 = rng.uniform(0.05, np.pi - 0.05)
        d = angular_separation(a1, a2)
        closed_allowed = (
            FULL - cap_solid_angle(r1) - cap_solid_angle(r2)
            + cap_intersection_solid_angle(r1, r2, d)
        )
        ks = KeepOutSet((ExclusionCone(a1, r1, "a"), ExclusionCone(a2, r2, "b")))
        quad_allowed = allowed_solid_angle(ks).solid_angle
        diff = abs(closed_allowed - quad_allowed)
        if diff > worst:
            worst = diff
            worst_case = (np.degrees(r1), np.degrees(r2), np.degrees(d),
                          closed_allowed, quad_allowed)
    r1d, r2d, dd, closed_allowed, quad_allowed = worst_case
    print(f"   worst case: r1 = {r1d:.4f} deg, r2 = {r2d:.4f} deg, separation = {dd:.4f} deg")
    print(f"   closed form  = {closed_allowed:.15f} sr")
    print(f"   band quad    = {quad_allowed:.15f} sr")
    print(f"   worst |diff| = {worst:.3e} sr   tolerance 1e-10   "
          f"{'PASS' if worst < 1e-10 else 'FAILED'}\n")
    return worst


def check_quadrature_convergence(n_cases: int = 300, seed: int = 20260831) -> None:
    print("4. Band-quadrature convergence with node count (same 300 configurations)")
    print(f"{'nodes_per_band':>16} {'worst |diff| [sr]':>20}")
    for n in (12, 24, 48, 64, 96, 128):
        rng = np.random.default_rng(seed)
        worst = 0.0
        for _ in range(n_cases):
            a1 = unit(rng.normal(size=3))
            a2 = unit(rng.normal(size=3))
            r1 = rng.uniform(0.05, np.pi - 0.05)
            r2 = rng.uniform(0.05, np.pi - 0.05)
            d = angular_separation(a1, a2)
            closed = (
                FULL - cap_solid_angle(r1) - cap_solid_angle(r2)
                + cap_intersection_solid_angle(r1, r2, d)
            )
            ks = KeepOutSet((ExclusionCone(a1, r1, "a"), ExclusionCone(a2, r2, "b")))
            worst = max(worst, abs(closed - allowed_solid_angle(ks, n).solid_angle))
        print(f"{n:16d} {worst:20.3e}")
    print()


def check_monte_carlo() -> bool:
    print("5. Monte Carlo cross-check (independent sampling, binomial error bar)")
    configs = [
        ("two caps, 30/25 deg, 40 deg apart",
         KeepOutSet((ExclusionCone([1, 0, 0], np.radians(30.0), "a"),
                     ExclusionCone([np.cos(np.radians(40.0)), np.sin(np.radians(40.0)), 0.0],
                                   np.radians(25.0), "b")))),
        ("three caps, 50/45/40 deg, mutually overlapping",
         KeepOutSet((ExclusionCone([1, 0, 0], np.radians(50.0), "a"),
                     ExclusionCone([0.6, 0.8, 0.0], np.radians(45.0), "b"),
                     ExclusionCone([0.6, 0.3, 0.74], np.radians(40.0), "c")))),
        ("Sun 45 deg + Earth 77 deg + Moon 15 deg, LEO-like",
         KeepOutSet((ExclusionCone([1, 0, 0], np.radians(45.0), "sun"),
                     ExclusionCone([-0.2, -0.9, 0.3], np.radians(77.0), "earth"),
                     ExclusionCone([0.1, 0.3, 0.95], np.radians(15.0), "moon")))),
    ]
    ok = True
    for label, ks in configs:
        q = allowed_solid_angle(ks).solid_angle
        mc = allowed_solid_angle_monte_carlo(ks, 2_000_000, seed=4242)
        z = abs(q - mc.solid_angle) / mc.standard_error
        ok &= z < 4.0
        print(f"   {label}")
        print(f"     band quadrature : {q:.9f} sr  ({q / FULL * 100:.5f} % of the sky)")
        print(f"     Monte Carlo     : {mc.solid_angle:.9f} +/- {mc.standard_error:.9f} sr "
              f"(n = {mc.n_samples})")
        print(f"     discrepancy     : {abs(q - mc.solid_angle):.3e} sr = {z:.2f} sigma")
    print(f"   all within 4 sigma: {'PASS' if ok else 'FAILED'}\n")
    return ok


def main() -> None:
    print("=" * 78)
    print("KeepOut validation: cone geometry vs closed-form spherical geometry")
    print("=" * 78)
    print()
    w1 = check_cap_solid_angle()
    w2 = check_special_cases()
    w3 = check_against_band_quadrature()
    check_quadrature_convergence()
    ok5 = check_monte_carlo()
    passed = (w1 < 1e-12) and (w2 < 1e-13) and (w3 < 1e-10) and ok5
    print("=" * 78)
    print(f"OVERALL: {'PASS' if passed else 'FAILED'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
