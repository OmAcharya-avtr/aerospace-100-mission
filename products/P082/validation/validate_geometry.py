"""Geometry: equations (1) to (3), against closed forms and an independent route.

Four claims are measured here, each against mathematics the implementation does
not use: the haversine separation against the spherical law of cosines, the
closed forms for a quarter meridian and an equatorial quadrant, the Pythagorean
identity of the along/cross decomposition, and the symmetry of that
decomposition -- which was a real defect in this product's first draft and is
why ``geometry.midpoint_bearing_deg`` exists.
"""

from __future__ import annotations

# See the module docstring of validation/_bootstrap.py for why this statement is
# here and not in _bootstrap: CPython writes a module's bytecode cache BEFORE
# executing that module, so _bootstrap cannot set the flag on its own behalf.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from _bootstrap import Tee, bootstrap, environment_lines  # noqa: E402

ROOT = bootstrap()

from sitediverse.geometry import (  # noqa: E402
    ADVECTION_AXIS_DEG,
    EARTH_RADIUS_KM,
    SITES,
    along_cross_km,
    along_cross_matrices,
    bearing_matrix,
    great_circle_km,
    initial_bearing_deg,
    midpoint_bearing_matrix,
    separation_matrix,
)
from sitediverse.report import format_table, number, scientific  # noqa: E402


def closed_forms(out: Tee) -> None:
    """Part 1: equation (1) against cases with an exact closed form."""
    out.rule("1. equation (1) against closed forms")
    quarter = np.pi * EARTH_RADIUS_KM / 2.0
    cases = (
        ("pole to equator on a meridian", (90.0, 0.0, 0.0, 0.0), quarter),
        ("ninety degrees along the equator", (0.0, 0.0, 0.0, 90.0), quarter),
        ("antipodes", (10.0, 20.0, -10.0, -160.0), 2.0 * quarter),
        (
            "one degree of latitude",
            (45.0, 7.0, 46.0, 7.0),
            np.pi * EARTH_RADIUS_KM / 180.0,
        ),
    )
    rows = []
    worst = 0.0
    worst_benign = 0.0
    for label, coordinates, expected in cases:
        got = float(great_circle_km(*coordinates))
        relative = abs(got - expected) / expected
        rows.append(
            [label, number(got, 6), number(expected, 6), scientific(relative)]
        )
        worst = max(worst, relative)
        if label != "antipodes":
            worst_benign = max(worst_benign, relative)
    out(format_table(["case", "measured km", "closed form km", "relative"], rows))
    out("")
    out.check(
        "equation (1) reproduces every closed form away from the antipodal limit",
        worst_benign < 1e-13,
        f"worst relative difference {worst_benign:.6e} against a tolerance of 1e-13",
    )
    out("")
    out("The antipodal case is the exception and it is a property of the haversine")
    out("form, not a defect. As the inner quantity approaches 1 the derivative of")
    out("arcsin(sqrt(a)) diverges, so a relative perturbation eps in a becomes one of")
    out("order sqrt(eps) in the angle. The declared network's largest separation is")
    out(f"{separation_matrix()[np.triu_indices(len(SITES), 1)].max():.1f} km, well")
    out("inside half the circumference, so the regime is never entered in practice.")
    out("")
    antipodal = abs(
        float(great_circle_km(10.0, 20.0, -10.0, -160.0)) - 2.0 * quarter
    ) / (2.0 * quarter)
    out(f"  antipodal relative error : {antipodal:.6e}")
    out(f"  sqrt(machine epsilon)    : {np.sqrt(np.finfo(float).eps):.6e}")
    out(f"  ratio                    : {antipodal / np.sqrt(np.finfo(float).eps):.6f}")
    out.check(
        "the antipodal error is of the order the analysis predicts and no worse",
        antipodal < 10.0 * np.sqrt(np.finfo(float).eps),
        f"{antipodal:.6e} against 10 sqrt(eps) = "
        f"{10.0 * np.sqrt(np.finfo(float).eps):.6e}",
    )


def independent_route(out: Tee) -> None:
    """Part 2: equation (1) against the spherical law of cosines.

    An independently derived form of the same quantity. It is algebraically
    identical and numerically different -- it loses precision for small
    separations, which is why equation (1) uses haversine -- so the agreement
    is bounded by the law-of-cosines form's own error at these separations.
    """
    out.rule("2. equation (1) against the spherical law of cosines")
    rng = np.random.default_rng(4242)
    count = 4000
    lat_a = rng.uniform(-80.0, 80.0, count)
    lon_a = rng.uniform(-180.0, 180.0, count)
    lat_b = rng.uniform(-80.0, 80.0, count)
    lon_b = rng.uniform(-180.0, 180.0, count)
    haversine = great_circle_km(lat_a, lon_a, lat_b, lon_b)
    phi_a, phi_b = np.radians(lat_a), np.radians(lat_b)
    cosine = np.clip(
        np.sin(phi_a) * np.sin(phi_b)
        + np.cos(phi_a) * np.cos(phi_b) * np.cos(np.radians(lon_b - lon_a)),
        -1.0,
        1.0,
    )
    law = EARTH_RADIUS_KM * np.arccos(cosine)
    worst = float(np.max(np.abs(haversine - law)))
    out(f"  random coordinate pairs  : {count}")
    out(f"  worst absolute difference: {worst:.6e} km")
    relative = float(np.max(np.abs(haversine - law) / np.maximum(law, 1.0)))
    out(f"  worst relative difference: {relative:.6e}")
    out.check(
        "equation (1) agrees with an independently derived spherical form",
        worst < 1e-5,
        f"worst absolute difference {worst:.6e} km against a tolerance of 1e-5 km",
    )


def decomposition(out: Tee) -> None:
    """Part 3: the identity and the symmetry of equation (3)."""
    out.rule("3. equation (3): the identity, and the symmetry defect it fixed")
    separation = separation_matrix()
    along, cross = along_cross_matrices()
    rows, cols = np.triu_indices(len(SITES), 1)
    identity = np.abs(
        along[rows, cols] ** 2 + cross[rows, cols] ** 2 - separation[rows, cols] ** 2
    ) / separation[rows, cols] ** 2
    out(f"  pairs                    : {rows.size}")
    out(f"  worst relative residual  : {float(identity.max()):.6e}")
    out.check(
        "d_along^2 + d_cross^2 == d^2 to relative rounding over every pair",
        float(identity.max()) < 1e-14,
        f"worst relative residual {float(identity.max()):.6e} against 1e-14",
    )
    out("")
    out("The symmetry of equation (3) was a real defect in this product's first")
    out("draft, found by its own property test. Built on the INITIAL bearing the two")
    out("matrices were asymmetric, so a pair's decomposition depended on which")
    out("station had the lower index. Both routes are measured below.")
    out("")
    old_along, old_cross = along_cross_km(separation, bearing_matrix())
    old_worst = max(
        float(np.max(np.abs(old_along - old_along.T))),
        float(np.max(np.abs(old_cross - old_cross.T))),
    )
    new_worst = max(
        float(np.max(np.abs(along - along.T))),
        float(np.max(np.abs(cross - cross.T))),
    )
    out(
        format_table(
            ["azimuth used", "worst asymmetry km", "verdict"],
            [
                ["initial bearing (the defect)", f"{old_worst:.4f}", "ASYMMETRIC"],
                ["arc midpoint (current)", scientific(new_worst), "symmetric"],
            ],
        )
    )
    out("")
    out.check(
        "the current decomposition is symmetric in the pair",
        new_worst < 1e-8,
        f"worst asymmetry {new_worst:.6e} km against a tolerance of 1e-8 km",
    )
    out.check(
        "the initial-bearing form really was asymmetric, so the fix was needed",
        old_worst > 100.0,
        f"worst asymmetry {old_worst:.4f} km on the initial-bearing form",
    )
    midpoint = midpoint_bearing_matrix()
    gap = np.abs(((midpoint[rows, cols] - midpoint[cols, rows]) % 360.0) - 180.0)
    out.check(
        "the midpoint azimuth from i to j and from j to i differ by 180 degrees",
        float(gap.max()) < 1e-9,
        f"worst deviation {float(gap.max()):.6e} degrees",
    )


def axis_cases(out: Tee) -> None:
    """Part 4: equation (3) at the two orientations with an exact answer."""
    out.rule("4. equation (3) at the axis and at its normal")
    rows = []
    worst = 0.0
    for separation in (100.0, 1000.0, 7000.0):
        on_axis = along_cross_km(separation, ADVECTION_AXIS_DEG)
        normal = along_cross_km(separation, ADVECTION_AXIS_DEG - 90.0)
        rows.append(
            [
                number(separation, 1),
                number(float(on_axis[0]), 9),
                scientific(float(on_axis[1])),
                scientific(float(normal[0])),
                number(float(normal[1]), 9),
            ]
        )
        worst = max(
            worst,
            abs(float(on_axis[0]) - separation),
            abs(float(on_axis[1])),
            abs(float(normal[0])),
            abs(float(normal[1]) - separation),
        )
    out(
        format_table(
            ["separation km", "along on axis", "cross on axis", "along normal",
             "cross normal"],
            rows,
        )
    )
    out("")
    out.check(
        "a baseline on the axis is entirely along it, and normal to it entirely across",
        worst < 1e-9,
        f"worst absolute deviation {worst:.6e} km",
    )


def network(out: Tee) -> None:
    """Part 5: the declared network, printed for the record."""
    out.rule("5. the declared synthetic network")
    out("DECLARED SYNTHETIC COORDINATES. Not the coordinates of any real optical")
    out("ground station, and no station's climatology is used anywhere in this")
    out("package. See DATASET_CARD.md.")
    out("")
    separation = separation_matrix()
    along, cross = along_cross_matrices()
    rows, cols = np.triu_indices(len(SITES), 1)
    values = separation[rows, cols]
    out(f"  stations                 : {len(SITES)} in 5 declared groups of 4")
    out(f"  pairs                    : {rows.size}")
    out(f"  separation range km      : {values.min():.1f} to {values.max():.1f}")
    out(f"  median separation km     : {float(np.median(values)):.1f}")
    out(f"  pairs under 300 km       : {int(np.count_nonzero(values < 300.0))}")
    out(f"  pairs over 1500 km       : {int(np.count_nonzero(values > 1500.0))}")
    out(f"  along range km           : {along[rows, cols].min():.1f} to "
        f"{along[rows, cols].max():.1f}")
    out(f"  cross range km           : {cross[rows, cols].min():.1f} to "
        f"{cross[rows, cols].max():.1f}")
    out("")
    order = np.argsort(values)[:10]
    out(
        format_table(
            ["pair", "separation km", "midpoint azimuth deg", "along km", "cross km"],
            [
                [
                    f"{SITES[rows[k]].name}-{SITES[cols[k]].name}",
                    f"{separation[rows[k], cols[k]]:.1f}",
                    f"{midpoint_bearing_matrix()[rows[k], cols[k]]:.1f}",
                    f"{along[rows[k], cols[k]]:.1f}",
                    f"{cross[rows[k], cols[k]]:.1f}",
                ]
                for k in order
            ],
        )
    )
    out("")
    out.check(
        "the network spans separations from under 100 km to over 7000 km",
        values.min() < 100.0 and values.max() > 7000.0,
        f"{values.min():.1f} km to {values.max():.1f} km",
    )
    out.check(
        "the near pairs point in a range of directions, so anisotropy is observable",
        float(np.ptp(along[rows, cols][values < 300.0])) > 50.0,
        f"along-axis component of the near pairs spans "
        f"{float(np.ptp(along[rows, cols][values < 300.0])):.1f} km",
    )
    out.check(
        "no pair of the network is near antipodal, so no midpoint azimuth is nan",
        not bool(np.isnan(along).any()),
        "0 nan entries in the along matrix",
    )
    _ = initial_bearing_deg


def main() -> int:
    out = Tee("validate_geometry_output.txt")
    out.rule("sitediverse -- geometry: separation, azimuth and the axis decomposition")
    for line in environment_lines():
        out(line)
    closed_forms(out)
    independent_route(out)
    decomposition(out)
    axis_cases(out)
    network(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
