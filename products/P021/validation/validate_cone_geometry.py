"""Keep-out cone geometry: the closed-form arc test against everything else.

PART A  a hand-computed case, every intermediate printed
PART B  closed-form minimum margin vs dense sampling, 4000 random geometries
PART C  violation-interval endpoints are roots of the margin function
PART D  the degenerate cases: boresight on the eigenaxis, exact tangency,
        a cone that contains the whole arc, a cone the arc never reaches
PART E  rotation invariance of margins and verdicts, 20 000 random rotations
PART F  the sampling trap: how narrow a violation a sampled planner misses

Run: ``python validation/validate_cone_geometry.py``
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slewforge.attitude import rotate_about_axis  # noqa: E402
from slewforge.keepout import KeepOutCone, KeepOutSet  # noqa: E402


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def random_geometry(rng: np.random.Generator):
    e = rng.normal(size=3)
    e /= np.linalg.norm(e)
    n0 = rng.normal(size=3)
    n0 /= np.linalg.norm(n0)
    c = rng.normal(size=3)
    c /= np.linalg.norm(c)
    gamma = float(rng.uniform(math.radians(2.0), math.radians(80.0)))
    sweep = float(rng.uniform(math.radians(5.0), math.pi))
    return n0, e, KeepOutCone(c, gamma, "cone"), sweep


def main() -> int:
    failures = 0
    rng = np.random.default_rng(20260902)

    # ---------------------------------------------------------------- PART A
    rule("PART A -- hand-computed case")
    print("Eigenaxis e = z. Boresight at psi = 0 is n0 = (sin 60, 0, cos 60),")
    print("so the boresight traces a small circle of angular radius 60 deg about z.")
    print("Cone axis c = x, half-angle gamma = 40 deg. Sweep 120 deg.")
    print()
    e = np.array([0.0, 0.0, 1.0])
    n0 = np.array([math.sin(math.radians(60.0)), 0.0, math.cos(math.radians(60.0))])
    cone = KeepOutCone(np.array([1.0, 0.0, 0.0]), math.radians(40.0), "hand")
    sweep = math.radians(120.0)

    a, b, c = cone.arc_coefficients(n0, e)
    print("f(psi) = n(psi) . c = A cos psi + B sin psi + C with")
    print("  A = n0.c - (e.n0)(e.c),  B = (e x n0).c,  C = (e.n0)(e.c)")
    print(f"  e.n0  = cos 60 deg      = {float(np.dot(e, n0))!r}")
    print(f"  e.c   = z . x           = {float(np.dot(e, cone.axis))!r}")
    print(f"  n0.c  = sin 60 deg      = {float(np.dot(n0, cone.axis))!r}")
    print(f"  A     = sqrt(3)/2       = {a!r}   (hand: {math.sqrt(3) / 2!r})")
    print(f"  B                       = {b!r}")
    print(f"  C                       = {c!r}")
    err_a = abs(a - math.sqrt(3) / 2)
    print(f"  |A - sqrt(3)/2|         = {err_a:.3e}   tolerance 1e-15")
    failures += err_a > 1e-15 or abs(b) > 1e-15 or abs(c) > 1e-15
    print()
    print("So f(psi) = (sqrt(3)/2) cos psi, maximised at psi = 0 with value")
    print("sqrt(3)/2 = cos 30 deg. The boresight is therefore 30 deg from the cone")
    print("axis at the start, and the cone is 40 deg, so the margin is exactly")
    print("-10 deg and the arc starts 10 deg inside the cone.")
    m = cone.min_margin_on_arc(n0, e, sweep)
    print(f"  library min margin      = {math.degrees(m)!r} deg")
    print("  hand                    = -10 deg")
    print(f"  |difference|            = {abs(math.degrees(m) + 10.0):.3e} deg   tolerance 1e-13")
    failures += abs(math.degrees(m) + 10.0) > 1e-13
    print()
    print("Violation ends where f = cos gamma, i.e. cos psi = cos 40 / cos 30:")
    ratio = math.cos(math.radians(40.0)) / math.cos(math.radians(30.0))
    psi_exit_hand = math.acos(ratio)
    print(f"  cos 40 / cos 30         = {ratio!r}")
    print(f"  psi_exit = arccos(.)    = {psi_exit_hand!r} rad = "
          f"{math.degrees(psi_exit_hand)!r} deg")
    iv = cone.violation_intervals(n0, e, sweep)
    print(f"  library intervals [deg] = {[(math.degrees(x), math.degrees(y)) for x, y in iv]}")
    err_iv = abs(iv[0][1] - psi_exit_hand)
    print(f"  |difference|            = {err_iv:.3e} rad   tolerance 1e-14")
    failures += len(iv) != 1 or abs(iv[0][0]) > 1e-15 or err_iv > 1e-14

    v = KeepOutSet((cone,)).arc_violations(n0, e, sweep, "telescope")
    print(f"  reported depth          = {math.degrees(v[0].depth)!r} deg (hand: 10 deg)")
    failures += abs(math.degrees(v[0].depth) - 10.0) > 1e-13

    # ---------------------------------------------------------------- PART B
    rule("PART B -- closed form vs dense sampling, 4000 random geometries")
    print("The sampled reference evaluates the margin at 20 001 points per arc.")
    print("The closed form must never be larger than the sampled minimum (that")
    print("would be a missed violation) and must agree to the sampling resolution.")
    print()
    worst_diff = 0.0
    worst_over = 0.0
    n_trials = 4000
    for _ in range(n_trials):
        n0, e, cone, sweep = random_geometry(rng)
        exact = cone.min_margin_on_arc(n0, e, sweep)
        psi = np.linspace(0.0, sweep, 20001)
        pts = rotate_about_axis(n0, e, psi)
        sampled = float(np.min(cone.margin(pts)))
        worst_diff = max(worst_diff, abs(exact - sampled))
        worst_over = max(worst_over, exact - sampled)
    print(f"worst |closed form - sampled|   : {worst_diff:.6e} rad   tolerance 1e-6")
    print(f"worst (closed form - sampled)   : {worst_over:.6e} rad "
          f"(positive means the closed form was optimistic)")
    print("The sampled minimum can only be >= the true minimum, so a small positive")
    print("value here is the sampling error, not a defect in the closed form.")
    failures += worst_diff > 1e-6

    print()
    print("Verdict agreement (violates / does not) over the same 4000 geometries:")
    rng2 = np.random.default_rng(20260902)
    disagree = 0
    for _ in range(n_trials):
        n0, e, cone, sweep = random_geometry(rng2)
        exact_v = cone.min_margin_on_arc(n0, e, sweep) < 0.0
        psi = np.linspace(0.0, sweep, 20001)
        sampled_v = bool(np.any(cone.margin(rotate_about_axis(n0, e, psi)) < 0.0))
        # A disagreement is only meaningful away from tangency.
        if exact_v != sampled_v and abs(cone.min_margin_on_arc(n0, e, sweep)) > 1e-5:
            disagree += 1
    print(f"disagreements away from tangency (|margin| > 1e-5 rad): {disagree} / {n_trials}")
    failures += disagree > 0

    # ---------------------------------------------------------------- PART C
    rule("PART C -- interval endpoints are roots of the margin function")
    print("Every interior endpoint returned by violation_intervals() must sit at a")
    print("zero of the margin. Checked directly and refined with Brent's method.")
    print()
    rng3 = np.random.default_rng(11)
    worst_root = 0.0
    worst_brent = 0.0
    n_checked = 0
    for _ in range(3000):
        n0, e, cone, sweep = random_geometry(rng3)
        for lo, hi in cone.violation_intervals(n0, e, sweep):
            for edge in (lo, hi):
                if edge <= 1e-12 or edge >= sweep - 1e-12:
                    continue  # clipped by the arc ends, not a root
                n_checked += 1
                margin_at = float(cone.margin(rotate_about_axis(n0, e, edge)))
                worst_root = max(worst_root, abs(margin_at))

                def f(x, n0=n0, e=e, cone=cone):
                    return float(cone.margin(rotate_about_axis(n0, e, x)))

                lo_b, hi_b = edge - 1e-3, edge + 1e-3
                if f(lo_b) * f(hi_b) < 0.0:
                    root = brentq(f, lo_b, hi_b, xtol=1e-14)
                    worst_brent = max(worst_brent, abs(root - edge))
    print(f"interior endpoints checked            : {n_checked}")
    print(f"worst |margin| at an endpoint         : {worst_root:.6e} rad   tolerance 1e-12")
    print(f"worst |closed form - Brent root|      : {worst_brent:.6e} rad   tolerance 1e-9")
    failures += worst_root > 1e-12 or worst_brent > 1e-9

    # ---------------------------------------------------------------- PART D
    rule("PART D -- degenerate geometries")
    z = np.array([0.0, 0.0, 1.0])
    print("D1 boresight parallel to the eigenaxis: the small circle is a point.")
    cone_d = KeepOutCone(np.array([0.0, 0.0, 1.0]), math.radians(20.0), "d")
    m = cone_d.min_margin_on_arc(z, z, math.radians(180.0))
    print(f"   margin over a 180 deg sweep = {math.degrees(m):.12f} deg (hand: -20)")
    failures += abs(math.degrees(m) + 20.0) > 1e-12

    print()
    print("D2 exact tangency: cone axis on the eigenaxis, half-angle equal to the")
    print("   small-circle radius, so the boresight rides the boundary for the")
    print("   whole slew.")
    n_t = np.array([math.sin(math.radians(60.0)), 0.0, math.cos(math.radians(60.0))])
    cone_t = KeepOutCone(z, math.radians(60.0), "tangent")
    m = cone_t.min_margin_on_arc(n_t, z, math.radians(300.0))
    iv = cone_t.violation_intervals(n_t, z, math.radians(300.0))
    print(f"   margin        = {m:.3e} rad (hand: exactly 0)")
    print(f"   intervals     = {iv} (hand: none -- the boundary is allowed)")
    failures += abs(m) > 1e-15 or len(iv) != 0

    print()
    print("D3 cone containing the whole arc:")
    cone_c = KeepOutCone(z, math.radians(80.0), "big")
    m = cone_c.min_margin_on_arc(n_t, z, math.radians(300.0))
    iv = cone_c.violation_intervals(n_t, z, math.radians(300.0))
    print(f"   margin        = {math.degrees(m):.12f} deg (hand: 60 - 80 = -20)")
    print(f"   intervals     = {[(round(math.degrees(x), 9), round(math.degrees(y), 9)) for x, y in iv]}")
    failures += abs(math.degrees(m) + 20.0) > 1e-12 or len(iv) != 1

    print()
    print("D4 cone the arc never reaches:")
    cone_f = KeepOutCone(-z, math.radians(20.0), "far")
    m = cone_f.min_margin_on_arc(n_t, z, math.radians(300.0))
    iv = cone_f.violation_intervals(n_t, z, math.radians(300.0))
    print(f"   margin        = {math.degrees(m):.12f} deg (hand: 120 - 20 = 100)")
    print(f"   intervals     = {iv}")
    failures += abs(math.degrees(m) - 100.0) > 1e-12 or len(iv) != 0

    print()
    print("D5 zero sweep: the arc is a point, so the arc test reduces to the")
    print("   point test.")
    n0, e, cone, _ = random_geometry(rng)
    m_arc = cone.min_margin_on_arc(n0, e, 0.0)
    m_pt = float(cone.margin(n0))
    print(f"   arc margin {m_arc!r} vs point margin {m_pt!r}, diff "
          f"{abs(m_arc - m_pt):.3e}   tolerance 1e-15")
    failures += abs(m_arc - m_pt) > 1e-15

    # ---------------------------------------------------------------- PART E
    rule("PART E -- rotation invariance")
    print("Rotating the boresight, the eigenaxis and the cone axis together must")
    print("leave the margin unchanged. 20 000 Haar-uniform rotations.")
    n0, e, cone, sweep = random_geometry(np.random.default_rng(5))
    base = cone.min_margin_on_arc(n0, e, sweep)
    rng4 = np.random.default_rng(99)
    worst = 0.0
    flips = 0
    for _ in range(20000):
        q = rng4.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, zc = q
        r = np.array(
            [
                [1 - 2 * (y * y + zc * zc), 2 * (x * y - w * zc), 2 * (x * zc + w * y)],
                [2 * (x * y + w * zc), 1 - 2 * (x * x + zc * zc), 2 * (y * zc - w * x)],
                [2 * (x * zc - w * y), 2 * (y * zc + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        m = cone.rotated(r).min_margin_on_arc(r @ n0, r @ e, sweep)
        worst = max(worst, abs(m - base))
        flips += (m < 0.0) != (base < 0.0)
    print(f"base margin                     : {base!r} rad")
    print(f"worst |change| over 20 000       : {worst:.6e} rad   tolerance 1e-14")
    print(f"verdict flips                    : {flips}")
    failures += worst > 1e-14 or flips > 0

    # ---------------------------------------------------------------- PART F
    rule("PART F -- the sampling trap")
    print("A planner that checks the path at N samples misses any violation")
    print("narrower than the sample spacing. Constructed case: a 90 deg sweep")
    print("whose boresight clips a cone for a stretch of the widths below.")
    print()
    print("The cone is centred at psi = 0.3712345 of the sweep, deliberately not on")
    print("a grid point of any of the sample counts below.")
    print()
    print(f"{'violation width [deg]':>22}{'N=51':>10}{'N=201':>10}{'N=1001':>10}{'closed form':>14}")
    z = np.array([0.0, 0.0, 1.0])
    n_f = np.array([math.sin(math.radians(50.0)), 0.0, math.cos(math.radians(50.0))])
    sweep = math.radians(90.0)
    for width_deg in (12.0, 4.0, 1.0, 0.25, 0.05):
        # Place a cone centred on the arc, sized so the violating stretch has
        # the requested width.
        psi_c = sweep * 0.3712345
        centre = rotate_about_axis(n_f, z, psi_c)
        half = math.radians(width_deg) / 2.0
        edge = rotate_about_axis(n_f, z, psi_c + half)
        gamma = float(
            np.arctan2(float(np.linalg.norm(np.cross(edge, centre))), float(np.dot(edge, centre)))
        )
        cone_n = KeepOutCone(centre, gamma, "narrow")
        row = [f"{width_deg:>22.2f}"]
        for n in (51, 201, 1001):
            psi = np.linspace(0.0, sweep, n)
            hit = bool(np.any(cone_n.margin(rotate_about_axis(n_f, z, psi)) < 0.0))
            row.append(f"{'caught' if hit else 'MISSED':>10}")
        exact_hit = cone_n.min_margin_on_arc(n_f, z, sweep) < 0.0
        row.append(f"{'caught' if exact_hit else 'MISSED':>14}")
        print("".join(row))
        failures += not exact_hit
    print()
    print("The closed form catches every one. This is the reason the planner's")
    print("constraint is analytic rather than sampled.")

    rule("SUMMARY")
    print(f"failed checks: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
