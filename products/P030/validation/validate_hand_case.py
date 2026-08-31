"""The hand-computed two-overlapping-cone case, reproduced step by step.

Geometry, chosen so that every angle can be read off by inspection:

    cone A: axis along +x,                        half-angle 30 deg
    cone B: axis at 40 deg from +x in the x-y plane, half-angle 25 deg

Every test boresight lies in the x-y plane, so its angle to axis A is its own
azimuth and its angle to axis B is |azimuth - 40 deg|. Cone A therefore covers
azimuths (-30, +30), cone B covers (+15, +65), the overlap is (+15, +30) and
the union is (-30, +65).

Because |30 - 25| = 5 deg < 40 deg < 55 deg = 30 + 25, the caps properly
overlap and neither contains the other, which is the case the lens formula in
``cap_intersection_solid_angle`` is written for.

The intersection area is worked through below with every intermediate value
printed, then compared with the library.

Run from products/P030/:  python validation/validate_hand_case.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from keepout import (  # noqa: E402
    ExclusionCone,
    KeepOutSet,
    allowed_fraction,
    allowed_solid_angle,
    allowed_solid_angle_monte_carlo,
    cap_intersection_solid_angle,
    cap_solid_angle,
    cap_union_solid_angle,
    spherical_to_unit,
)

R1_DEG, R2_DEG, SEP_DEG = 30.0, 25.0, 40.0


def in_plane(azimuth_deg: float) -> np.ndarray:
    return spherical_to_unit(np.radians(azimuth_deg), 0.0)


def main() -> None:
    r1, r2, d = np.radians([R1_DEG, R2_DEG, SEP_DEG])
    print("=" * 88)
    print("KeepOut validation: hand-computed two-overlapping-cone case")
    print("=" * 88)
    print(f"cone A: axis +x, half-angle {R1_DEG:g} deg")
    print(f"cone B: axis at {SEP_DEG:g} deg from +x in the x-y plane, "
          f"half-angle {R2_DEG:g} deg")
    print()

    print("1. The hand arithmetic, printed term by term")
    c1, c2, cd = np.cos(r1), np.cos(r2), np.cos(d)
    s1, s2, sd = np.sin(r1), np.sin(r2), np.sin(d)
    print(f"   cos r1 = {float(c1)!r}    sin r1 = {float(s1)!r}")
    print(f"   cos r2 = {float(c2)!r}    sin r2 = {float(s2)!r}")
    print(f"   cos d  = {float(cd)!r}    sin d  = {float(sd)!r}")
    print()
    n1, dn1 = c2 - c1 * cd, s1 * sd
    print("   cos a1 = (cos r2 - cos r1 cos d) / (sin r1 sin d)")
    print(f"          cos r1 cos d = {float(c1 * cd)!r}")
    print(f"          numerator    = {float(c2)!r} - {float(c1 * cd)!r} = {float(n1)!r}")
    print(f"          denominator  = {float(s1)!r} * {float(sd)!r} = {float(dn1)!r}")
    ca1 = n1 / dn1
    a1 = 2.0 * np.arctan(
        np.sqrt(np.sin(0.5 * (r1 + r2 + d) - r1) * np.sin(0.5 * (r1 + r2 + d) - d)
                / (np.sin(0.5 * (r1 + r2 + d)) * np.sin(0.5 * (r1 + r2 + d) - r2)))
    )
    print(f"          cos a1 = {float(ca1)!r}  ->  a1 = {float(np.arccos(ca1))!r} rad "
          f"= {float(np.degrees(np.arccos(ca1)))!r} deg")
    print(f"          half-angle form (used by the library): a1 = {float(a1)!r} rad "
          f"(|diff| = {abs(a1 - np.arccos(ca1)):.3e})")
    print()
    n2, dn2 = c1 - c2 * cd, s2 * sd
    ca2 = n2 / dn2
    print("   cos a2 = (cos r1 - cos r2 cos d) / (sin r2 sin d)")
    print(f"          cos r2 cos d = {float(c2 * cd)!r}")
    print(f"          numerator    = {float(c1)!r} - {float(c2 * cd)!r} = {float(n2)!r}")
    print(f"          denominator  = {float(s2)!r} * {float(sd)!r} = {float(dn2)!r}")
    a2 = np.arccos(ca2)
    print(f"          cos a2 = {float(ca2)!r}  ->  a2 = {float(a2)!r} rad = {float(np.degrees(a2))!r} deg")
    print()
    ng, dng = cd - c1 * c2, s1 * s2
    cg = ng / dng
    g = np.arccos(cg)
    print("   cos g  = (cos d - cos r1 cos r2) / (sin r1 sin r2)")
    print(f"          cos r1 cos r2 = {float(c1 * c2)!r}")
    print(f"          numerator     = {float(cd)!r} - {float(c1 * c2)!r} = {float(ng)!r}")
    print(f"          denominator   = {float(s1)!r} * {float(s2)!r} = {float(dng)!r}")
    print(f"          cos g  = {float(cg)!r}  ->  g  = {float(g)!r} rad = {float(np.degrees(g))!r} deg")
    print()
    t0, t1, t2 = 2.0 * (np.pi - g), 2.0 * a1 * c1, 2.0 * a2 * c2
    hand = t0 - t1 - t2
    print("   A = 2 (pi - g) - 2 a1 cos r1 - 2 a2 cos r2")
    print(f"          2 (pi - g)  = {float(t0)!r}")
    print(f"          2 a1 cos r1 = {float(t1)!r}")
    print(f"          2 a2 cos r2 = {float(t2)!r}")
    print(f"          A           = {float(hand)!r} sr")
    print()

    lib = cap_intersection_solid_angle(r1, r2, d)
    print("2. Library vs the hand arithmetic")
    print(f"   hand    A_intersection = {float(hand)!r} sr")
    print(f"   library A_intersection = {float(lib)!r} sr")
    print(f"   |diff| = {abs(lib - hand):.3e} sr   tolerance 1e-15   "
          f"{'PASS' if abs(lib - hand) < 1e-15 else 'FAILED'}")
    print()

    a_cap1 = float(cap_solid_angle(r1))
    a_cap2 = float(cap_solid_angle(r2))
    union_hand = a_cap1 + a_cap2 - hand
    union_lib = cap_union_solid_angle(r1, r2, d)
    print("3. Union and allowed sky")
    print(f"   A1 = 2 pi (1 - cos 30 deg) = {float(a_cap1)!r} sr")
    print(f"   A2 = 2 pi (1 - cos 25 deg) = {float(a_cap2)!r} sr")
    print(f"   union (hand)    = {float(a_cap1)!r} + {float(a_cap2)!r} - {float(hand)!r} = {float(union_hand)!r} sr")
    print(f"   union (library) = {float(union_lib)!r} sr   |diff| = {abs(union_lib - union_hand):.3e}")
    allowed_hand = 4.0 * np.pi - union_hand
    print(f"   allowed (hand)  = 4 pi - union = {float(allowed_hand)!r} sr")

    ks = KeepOutSet((ExclusionCone(in_plane(0.0), r1, "A"),
                     ExclusionCone(in_plane(SEP_DEG), r2, "B")))
    quad = allowed_solid_angle(ks)
    print(f"   allowed (band quadrature) = {float(quad.solid_angle)!r} sr  "
          f"({quad.n_samples} nodes)")
    print(f"   |diff| = {abs(quad.solid_angle - allowed_hand):.3e} sr   tolerance 1e-10   "
          f"{'PASS' if abs(quad.solid_angle - allowed_hand) < 1e-10 else 'FAILED'}")
    frac = allowed_fraction(ks)
    print(f"   allowed fraction of the sky = {float(frac)!r} "
          f"({frac * 100:.6f} %)")
    mc = allowed_solid_angle_monte_carlo(ks, 2_000_000, seed=99)
    z = abs(mc.solid_angle - allowed_hand) / mc.standard_error
    print(f"   Monte Carlo (n = {mc.n_samples}) = {mc.solid_angle:.9f} "
          f"+/- {mc.standard_error:.9f} sr -> {z:.2f} sigma from the hand value")
    print()

    print("4. Violation verdicts at hand-checkable boresights (all in the x-y plane)")
    cases = [
        (0.0, "0 < 30 inside A; |40-0| = 40 > 25 outside B", {"A"}),
        (5.0, "5 < 30 inside A; |40-5| = 35 > 25 outside B", {"A"}),
        (20.0, "20 < 30 inside A; |40-20| = 20 < 25 inside B", {"A", "B"}),
        (25.0, "25 < 30 inside A; |40-25| = 15 < 25 inside B", {"A", "B"}),
        (35.0, "35 > 30 outside A; |40-35| = 5 < 25 inside B", {"B"}),
        (64.0, "64 > 30 outside A; |40-64| = 24 < 25 inside B", {"B"}),
        (66.0, "66 > 30 outside A; |40-66| = 26 > 25 outside B", set()),
        (70.0, "70 > 30 outside A; |40-70| = 30 > 25 outside B", set()),
        (-29.0, "29 < 30 inside A; |40+29| = 69 > 25 outside B", {"A"}),
        (-31.0, "31 > 30 outside A; |40+31| = 71 > 25 outside B", set()),
        (180.0, "opposite the union", set()),
    ]
    ok = True
    print(f"{'azimuth [deg]':>14} {'hand reasoning':>52} {'expected':>10} {'library':>10} {'':>6}")
    for az, why, expected in cases:
        got = set(ks.violations(in_plane(az)))
        agree = got == expected
        ok &= agree
        e = "{" + ",".join(sorted(expected)) + "}" if expected else "{}"
        gtxt = "{" + ",".join(sorted(got)) + "}" if got else "{}"
        print(f"{az:14.1f} {why:>52} {e:>10} {gtxt:>10} {'ok' if agree else 'MISMATCH':>6}")
    print(f"   {'PASS' if ok else 'FAILED'}")
    print()

    print("5. Deepest-violation ordering at azimuth 25 deg")
    m = ks.margins(in_plane(25.0))
    print(f"   margin to A = 25 - 30 = -5 deg  -> library {np.degrees(m[0]):+.10f} deg")
    print(f"   margin to B = 15 - 25 = -10 deg -> library {np.degrees(m[1]):+.10f} deg")
    order = ks.violations(in_plane(25.0))
    print(f"   worst-first order: expected ('B', 'A'), library {order}")
    order_ok = order == ("B", "A")
    margin_ok = (abs(np.degrees(m[0]) + 5.0) < 1e-12) and (abs(np.degrees(m[1]) + 10.0) < 1e-12)
    print(f"   {'PASS' if order_ok and margin_ok else 'FAILED'}")
    print()

    passed = (
        abs(lib - hand) < 1e-15
        and abs(quad.solid_angle - allowed_hand) < 1e-10
        and ok
        and order_ok
        and margin_ok
        and z < 4.0
    )
    print("=" * 88)
    print(f"OVERALL: {'PASS' if passed else 'FAILED'}")
    print("=" * 88)


if __name__ == "__main__":
    main()
