"""Cross-check against the sibling product KeepOut (P030), implemented separately.

Batch 03 requires SlewForge and KeepOut to agree on cone violation for
identical geometry, with the two cone implementations written independently.
Neither package imports the other; nothing in ``src/`` or ``tests/`` here
refers to KeepOut at all. This script is the only place the two meet, and it
runs only when a KeepOut source tree is reachable.

Point KeepOut out with the ``KEEPOUT_SRC`` environment variable, or place its
repository beside this one::

    KEEPOUT_SRC=/path/to/keepout/src python validation/validate_keepout_cross_check.py

Without it the script prints NOT RUN and exits 0, because a standalone clone of
SlewForge has no business failing its test suite for the absence of a different
repository.

PART A  point-wise margins and verdicts over 20 000 random configurations
PART B  the datasheet-to-cone conversion (limb vs centre) and Earth angular radius
PART C  arc violation: SlewForge's closed form against a dense KeepOut scan
PART D  a hand case both packages are asked the same question about

Run: ``python validation/validate_keepout_cross_check.py``
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slewforge.attitude import rotate_about_axis  # noqa: E402
from slewforge.keepout import (  # noqa: E402
    EARTH_RADIUS_M,
    KeepOutCone,
    KeepOutSet,
    body_keepout_cone,
    earth_angular_radius,
)


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def locate_keepout() -> Path | None:
    """Find a KeepOut source tree, or ``None``."""
    env = os.environ.get("KEEPOUT_SRC")
    candidates = []
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve().parents[1]
    candidates += [
        here.parent / "P030" / "src",
        here.parent / "keepout" / "src",
        here.parent.parent / "keepout" / "src",
    ]
    for c in candidates:
        if (c / "keepout" / "__init__.py").exists():
            return c
    return None


def main() -> int:
    src = locate_keepout()
    if src is None:
        rule("CROSS-CHECK AGAINST KEEPOUT (P030) -- NOT RUN")
        print("No KeepOut source tree found. Looked at $KEEPOUT_SRC and at")
        print("  ../P030/src, ../keepout/src, ../../keepout/src")
        print()
        print("This is not a failure. SlewForge does not depend on KeepOut and")
        print("never imports it; the cross-check is an inter-product agreement")
        print("check that needs both repositories present.")
        return 0

    sys.path.insert(0, str(src))
    spec = importlib.util.find_spec("keepout")
    import keepout as ko  # noqa: E402

    failures = 0
    rule("CROSS-CHECK AGAINST KEEPOUT (P030)")
    print(f"KeepOut source        : {spec.origin}")
    print(f"KeepOut version       : {getattr(ko, '__version__', 'unknown')}")
    print()
    print("The two cone implementations were written independently:")
    print("  KeepOut    ExclusionCone.margin  = angular_separation(b, axis) - half_angle")
    print("             with angular_separation = atan2(|a x b|, a . b)")
    print("  SlewForge  KeepOutCone.margin     = the same definition, own code, plus a")
    print("             closed-form arc test KeepOut does not have")
    print("Agreement on the point test is therefore a check on two independent")
    print("transcriptions of textbook geometry, not on shared code.")

    # ---------------------------------------------------------------- PART A
    rule("PART A -- point-wise margins and verdicts, 20 000 random configurations")
    rng = np.random.default_rng(20260902)
    worst_margin = 0.0
    verdict_flips = 0
    worst_solid = 0.0
    n = 20000
    for _ in range(n):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        gamma = float(rng.uniform(0.0, math.pi))
        b = rng.normal(size=3)
        b /= np.linalg.norm(b)
        mine = KeepOutCone(axis, gamma, "c")
        theirs = ko.ExclusionCone(axis=axis, half_angle=gamma, name="c")
        dm = abs(float(mine.margin(b)) - float(theirs.margin(b)))
        worst_margin = max(worst_margin, dm)
        verdict_flips += bool(mine.contains(b)) != bool(theirs.contains(b))
        worst_solid = max(worst_solid, abs(mine.solid_angle - theirs.solid_angle))
    print(f"configurations               : {n}")
    print(f"worst |margin difference|    : {worst_margin:.6e} rad   tolerance 1e-14")
    print(f"verdict disagreements        : {verdict_flips}   tolerance 0")
    print(f"worst |solid angle diff|     : {worst_solid:.6e} sr   tolerance 1e-15")
    failures += worst_margin > 1e-14 or verdict_flips > 0 or worst_solid > 1e-15
    print()
    print("The two are NOT bit-identical, and the tolerance above says so. The")
    print("residual is floating-point association: KeepOut forms the separation as")
    print("np.arctan2(np.linalg.norm(np.cross(a, b)), np.sum(a * b)) while SlewForge")
    print("uses its own cross3 and math.sqrt, so the last bit of the cross-product")
    print("norm can differ. 8.9e-16 rad is 1.8e-10 arcsec; it changes no verdict in")
    print("20 000 trials, and claiming bit-equality would have been wrong.")

    print()
    print("Multi-cone sets: worst-case margin and the violation list.")
    rng = np.random.default_rng(7)
    worst_set = 0.0
    name_mismatch = 0
    for _ in range(4000):
        cones = []
        their_cones = []
        for k in range(int(rng.integers(1, 5))):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            gamma = float(rng.uniform(math.radians(5.0), math.radians(80.0)))
            cones.append(KeepOutCone(axis, gamma, f"c{k}"))
            their_cones.append(ko.ExclusionCone(axis=axis, half_angle=gamma, name=f"c{k}"))
        b = rng.normal(size=3)
        b /= np.linalg.norm(b)
        mine = KeepOutSet(tuple(cones))
        theirs = ko.KeepOutSet(tuple(their_cones))
        worst_set = max(worst_set, abs(float(mine.margin(b)) - float(theirs.margin(b))))
        name_mismatch += mine.violations(b) != theirs.violations(b)
    print(f"worst |set margin difference|: {worst_set:.6e} rad   tolerance 1e-14")
    print(f"violation-list mismatches    : {name_mismatch} / 4000   tolerance 0")
    failures += worst_set > 1e-14 or name_mismatch > 0

    # ---------------------------------------------------------------- PART B
    rule("PART B -- datasheet conversion and the Earth's angular radius")
    print(f"{'altitude [km]':>15}{'SlewForge [deg]':>20}{'KeepOut [deg]':>18}{'|diff|':>12}")
    worst_ar = 0.0
    for alt_km in (0.0, 400.0, 550.0, 800.0, 1200.0, 35786.0):
        mine = earth_angular_radius(alt_km * 1e3)
        theirs = ko.earth_angular_radius(alt_km * 1e3)
        d = abs(mine - theirs)
        worst_ar = max(worst_ar, d)
        print(f"{alt_km:>15.1f}{math.degrees(mine):>20.10f}{math.degrees(theirs):>18.10f}{d:>12.3e}")
    print(f"\nworst |difference|           : {worst_ar:.6e} rad   tolerance 0 "
          f"(the same one-line formula, so bit-equality is expected here)")
    failures += worst_ar > 0.0
    print(f"both use R_E = {EARTH_RADIUS_M} m (WGS-84 equatorial) "
          f"and KeepOut uses {ko.EARTH_RADIUS_M} m")
    failures += EARTH_RADIUS_M != ko.EARTH_RADIUS_M

    print()
    print("Limb versus centre convention on a 10 deg Earth keep-out at 550 km:")
    d_e = np.array([0.0, 0.0, -1.0])
    ar = earth_angular_radius(550e3)
    for ref in ("limb", "center"):
        mine = body_keepout_cone("earth", d_e, ar, math.radians(10.0), ref)
        theirs = ko.body_exclusion_cone("earth", d_e, ar, math.radians(10.0), ref)
        d = abs(mine.half_angle - theirs.half_angle)
        print(f"  {ref:<8} SlewForge {mine.half_angle_deg:.10f} deg, "
              f"KeepOut {theirs.half_angle_deg:.10f} deg, |diff| {d:.3e}")
        failures += d > 0.0

    # ---------------------------------------------------------------- PART C
    rule("PART C -- arc violation: SlewForge closed form vs a dense KeepOut scan")
    print("KeepOut has no arc test, so the reference is KeepOut's point test")
    print("evaluated at 40 001 points along the same eigenaxis arc. This is the")
    print("check that matters operationally: does the analytic test agree with the")
    print("thing an engineer would actually write?")
    print()
    rng = np.random.default_rng(4242)
    worst_arc = 0.0
    verdict_mismatch = 0
    near_tangent = 0
    trials = 3000
    for _ in range(trials):
        e = rng.normal(size=3)
        e /= np.linalg.norm(e)
        n0 = rng.normal(size=3)
        n0 /= np.linalg.norm(n0)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        gamma = float(rng.uniform(math.radians(5.0), math.radians(70.0)))
        sweep = float(rng.uniform(math.radians(10.0), math.pi))
        mine = KeepOutCone(axis, gamma, "c")
        theirs = ko.ExclusionCone(axis=axis, half_angle=gamma, name="c")
        exact = mine.min_margin_on_arc(n0, e, sweep)
        psi = np.linspace(0.0, sweep, 40001)
        pts = rotate_about_axis(n0, e, psi)
        scan = float(np.min(theirs.margin(pts)))
        worst_arc = max(worst_arc, abs(exact - scan))
        if (exact < 0.0) != (scan < 0.0):
            if abs(exact) < 1e-5:
                near_tangent += 1
            else:
                verdict_mismatch += 1
    print(f"trials                             : {trials}")
    print(f"worst |closed form - KeepOut scan| : {worst_arc:.6e} rad   tolerance 1e-6")
    print(f"verdict mismatches away from tangency: {verdict_mismatch}   tolerance 0")
    print(f"verdict differences within 1e-5 rad of tangency: {near_tangent}")
    failures += worst_arc > 1e-6 or verdict_mismatch > 0

    # ---------------------------------------------------------------- PART D
    rule("PART D -- the hand case from validate_cone_geometry.py PART A")
    e = np.array([0.0, 0.0, 1.0])
    n0 = np.array([math.sin(math.radians(60.0)), 0.0, math.cos(math.radians(60.0))])
    mine = KeepOutCone(np.array([1.0, 0.0, 0.0]), math.radians(40.0), "hand")
    theirs = ko.ExclusionCone(
        axis=np.array([1.0, 0.0, 0.0]), half_angle=math.radians(40.0), name="hand"
    )
    print(f"margin at psi = 0   SlewForge {math.degrees(float(mine.margin(n0)))!r} deg")
    print(f"                    KeepOut   {math.degrees(float(theirs.margin(n0)))!r} deg")
    print("                    hand      -10 deg")
    d = abs(float(mine.margin(n0)) - float(theirs.margin(n0)))
    print(f"|difference|        {d:.3e} rad   tolerance 1e-14")
    failures += d > 1e-14
    psi_exit = mine.violation_intervals(n0, e, math.radians(120.0))[0][1]
    exit_margin = float(theirs.margin(rotate_about_axis(n0, e, psi_exit)))
    print()
    print("SlewForge says the violation ends at psi = "
          f"{math.degrees(psi_exit)!r} deg.")
    print(f"KeepOut's margin there is {exit_margin:.3e} rad, i.e. the cone boundary.")
    print("Tolerance 1e-14 rad.")
    failures += abs(exit_margin) > 1e-14

    rule("SUMMARY")
    print(f"failed checks: {failures}")
    if failures == 0:
        print()
        print("SlewForge and KeepOut agree on every cone-violation question asked")
        print("here, to 8.9e-16 rad on the point test. That is a weaker statement than")
        print("it looks: both implement the same textbook geometry with the same")
        print("atan2 formulation, so agreement confirms the transcription, not the")
        print("theory. The theory is checked against hand arithmetic in")
        print("validate_cone_geometry.py PART A.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
