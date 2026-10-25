"""Coordinate transforms, against independently derived values where one exists.

``astropy`` is not installed in the environment these numbers were measured in,
so nothing here is cross-checked against it and no such claim is made anywhere
in this repository. What is checked:

- Greenwich mean sidereal time at J2000.0 against the published constant
  280.46061837 degrees, and its advance over one solar day against the
  published 0.9856473 degree excess over a full turn.
- Every rotation matrix for orthogonality and unit determinant.
- Bowring's closed-form geodetic height against an independent fixed-point
  iteration over a latitude and altitude sweep, with the departure reported as
  a function of height rather than quoted from the paper.
- The radial / in-track / cross-track basis for orthonormality, handedness, and
  that the in-track axis is the one the secular growth actually appears on.
- The low-precision Sun and Moon series against independently known facts
  about those orbits: the solar distance at J2000.0 must be near perihelion,
  and the lunar distance must stay inside its published perigee-apogee range.
"""

from __future__ import annotations

import sys

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

import numpy as np  # noqa: E402

from orbitdrift import frames  # noqa: E402
from orbitdrift.constants import WGS84  # noqa: E402

J2000 = 2451545.0


def _iterative_height(r: np.ndarray) -> float:
    a = WGS84.radius_km
    f = WGS84.flattening
    assert f is not None
    e2 = f * (2.0 - f)
    x, y, z = r
    p = float(np.hypot(x, y))
    latitude = float(np.arctan2(z, p))
    height = 0.0
    for _ in range(300):
        n_curv = a / np.sqrt(1.0 - e2 * np.sin(latitude) ** 2)
        if abs(np.cos(latitude)) > abs(np.sin(latitude)):
            height = p / np.cos(latitude) - n_curv
        else:
            height = z / np.sin(latitude) - n_curv * (1.0 - e2)
        latitude = float(np.arctan2(z, p * (1.0 - e2 * n_curv / (n_curv + height))))
    return height


def main() -> int:
    out = Tee("validate_frames_output.txt")
    out.rule("orbitdrift -- coordinate transforms")
    for line in environment_lines():
        out(line)
    out("")
    out("  No transform here is cross-checked against astropy. It is not installed")
    out("  in this environment and no claim of such a check appears anywhere in")
    out("  this repository. Every reference below is either a published constant or")
    out("  an independent algorithm implemented in this script.")

    out.rule("Greenwich mean sidereal time")
    measured = float(np.rad2deg(frames.gmst_rad(J2000)))
    out(f"  GMST at J2000.0, measured     : {measured:.9f} deg")
    out("  published value               : 280.460618370 deg")
    out(f"  difference                    : {abs(measured - 280.46061837):.3e} deg")
    out.check(
        "GMST at J2000.0 matches the published constant to 1e-6 deg",
        abs(measured - 280.46061837) < 1e-6,
        f"{abs(measured - 280.46061837):.3e} deg",
    )
    advance = (float(np.rad2deg(frames.gmst_rad(J2000 + 1.0))) - measured) % 360.0
    out(f"  advance over one solar day    : 360 + {advance:.7f} deg")
    out("  published sidereal excess     : 0.9856473 deg")
    out.check(
        "the sidereal excess over a solar day matches to 1e-5 deg",
        abs(advance - 0.9856473) < 1e-5,
        f"{abs(advance - 0.9856473):.3e} deg",
    )

    out.rule("rotation matrices")
    worst_orthogonality = 0.0
    worst_determinant = 0.0
    for angle in np.linspace(0.0, 2.0 * np.pi, 37):
        matrix = frames.rotation_z(float(angle))
        worst_orthogonality = max(
            worst_orthogonality, float(np.max(np.abs(matrix @ matrix.T - np.eye(3))))
        )
        worst_determinant = max(
            worst_determinant, abs(float(np.linalg.det(matrix)) - 1.0)
        )
    out(f"  worst |R R^T - I|             : {worst_orthogonality:.3e}")
    out(f"  worst |det R - 1|             : {worst_determinant:.3e}")
    out.check(
        "every rotation is orthogonal with unit determinant to 1e-15",
        worst_orthogonality < 1e-15 and worst_determinant < 1e-15,
        f"{worst_orthogonality:.3e}, {worst_determinant:.3e}",
    )

    out.rule("TEME to ECEF round trip and invariants")
    rng = np.random.default_rng(31)
    worst_round_trip = 0.0
    worst_radius = 0.0
    worst_z = 0.0
    for _ in range(500):
        r = rng.normal(0.0, 8000.0, 3)
        jd = float(rng.uniform(J2000, J2000 + 10000.0))
        ecef = frames.teme_to_ecef(r, jd)
        back = frames.rotation_z(-float(frames.gmst_rad(jd))) @ ecef
        worst_round_trip = max(worst_round_trip, float(np.max(np.abs(back - r))))
        worst_radius = max(
            worst_radius,
            abs(float(np.linalg.norm(ecef) - np.linalg.norm(r))) / float(np.linalg.norm(r)),
        )
        worst_z = max(worst_z, abs(float(ecef[2] - r[2])))
    out(f"  worst round-trip closure      : {worst_round_trip:.3e} km")
    out(f"  worst relative radius change  : {worst_radius:.3e}")
    out(f"  worst z change                : {worst_z:.3e} km")
    out.check(
        "round trip closes to 1e-9 km and the rotation preserves radius and z",
        worst_round_trip < 1e-9 and worst_radius < 1e-14 and worst_z == 0.0,
        f"{worst_round_trip:.3e} km, {worst_radius:.3e}, {worst_z:.3e} km",
    )

    out.rule("geodetic height: Bowring's closed form against a fixed-point iteration")
    out("  Two different algorithms for the same quantity. Bowring's is a single")
    out("  pass built for points near the surface, so its departure grows with")
    out("  height; the growth is measured here rather than quoted.")
    out("  altitude km   worst |dh| over 91 latitudes   worst |dh| mm")
    a = WGS84.radius_km
    f = WGS84.flattening
    assert f is not None
    e2 = f * (2.0 - f)
    departures = {}
    for altitude in (0.0, 200.0, 400.0, 800.0, 1200.0, 1700.0, 3000.0):
        worst = 0.0
        for latitude_deg in np.linspace(-90.0, 90.0, 91):
            latitude = float(np.deg2rad(latitude_deg))
            n_curv = a / np.sqrt(1.0 - e2 * np.sin(latitude) ** 2)
            r = np.array(
                [
                    (n_curv + altitude) * np.cos(latitude),
                    0.0,
                    (n_curv * (1.0 - e2) + altitude) * np.sin(latitude),
                ]
            )
            worst = max(worst, abs(float(frames.geodetic_altitude_km(r)) - _iterative_height(r)))
        departures[altitude] = worst
        out(f"  {altitude:11.1f}   {worst:29.6e}   {worst * 1e6:13.3f}")
    out.check(
        "Bowring's height agrees to within 20 mm up to 1700 km",
        max(value for key, value in departures.items() if key <= 1700.0) < 2e-5,
        f"worst {max(value for key, value in departures.items() if key <= 1700.0) * 1e6:.3f} mm",
    )
    out.check(
        "the departure grows monotonically with height, as the single-pass form must",
        all(
            departures[low] <= departures[high]
            for low, high in zip(
                sorted(departures)[:-1], sorted(departures)[1:], strict=True
            )
        ),
        ", ".join(f"{departures[key] * 1e6:.3f} mm" for key in sorted(departures)),
    )

    out.rule("radial / in-track / cross-track basis")
    worst_orthonormality = 0.0
    worst_handedness = 0.0
    worst_isometry = 0.0
    for _ in range(2000):
        r = rng.normal(0.0, 7000.0, 3)
        v = rng.normal(0.0, 7.0, 3)
        if float(np.linalg.norm(np.cross(r, v))) < 1.0:
            continue
        basis = frames.ric_basis(r, v)
        worst_orthonormality = max(
            worst_orthonormality, float(np.max(np.abs(basis @ basis.T - np.eye(3))))
        )
        worst_handedness = max(worst_handedness, abs(float(np.linalg.det(basis)) - 1.0))
        delta = rng.normal(0.0, 10.0, 3)
        projected = frames.to_ric(delta, r, v)
        worst_isometry = max(
            worst_isometry,
            abs(float(np.linalg.norm(projected) - np.linalg.norm(delta)))
            / float(np.linalg.norm(delta)),
        )
    out(f"  worst |B B^T - I|             : {worst_orthonormality:.3e}")
    out(f"  worst |det B - 1|             : {worst_handedness:.3e}")
    out(f"  worst relative length change  : {worst_isometry:.3e}")
    out.check(
        "the RIC basis is orthonormal, right-handed and an isometry to 1e-12",
        worst_orthonormality < 1e-12
        and worst_handedness < 1e-12
        and worst_isometry < 1e-12,
        f"{worst_orthonormality:.3e}, {worst_handedness:.3e}, {worst_isometry:.3e}",
    )

    out.rule("the in-track axis is the one the secular growth appears on")
    out("  A real element set's divergence is decomposed; the in-track share must")
    out("  dominate, because a mean-motion error grows along track and the radial")
    out("  and cross-track separations are periodic. If the basis rows were")
    out("  permuted, this is the check that would catch it.")
    table = __import__("_bootstrap").load_dataset()
    radial = table.column("drift_radial_km")
    in_track = table.column("drift_in_track_km")
    cross = table.column("drift_cross_track_km")
    total = table.column("drift_km")
    share = in_track / total
    out(f"  rows                          : {share.size}")
    out(f"  median in-track share         : {float(np.median(share)):.6f}")
    out(f"  5th percentile in-track share : {float(np.percentile(share, 5)):.6f}")
    out(f"  median radial share           : {float(np.median(radial / total)):.6f}")
    out(f"  median cross-track share      : {float(np.median(cross / total)):.6f}")
    out.check(
        "the in-track component is the largest in at least 99 per cent of rows",
        float(np.mean((in_track > radial) & (in_track > cross))) > 0.99,
        f"{float(np.mean((in_track > radial) & (in_track > cross))):.6f}",
    )
    quadrature = np.sqrt(radial**2 + in_track**2 + cross**2)
    out.check(
        "the three components add in quadrature to the total",
        float(np.max(np.abs(quadrature - total) / total)) < 1e-10,
        f"worst relative departure {float(np.max(np.abs(quadrature - total) / total)):.3e}",
    )

    out.rule("the low-precision Sun and Moon series, against known facts")
    distance = float(np.linalg.norm(frames.sun_position_teme_km(J2000)))
    au = 149597870.7
    out(f"  solar distance at J2000.0     : {distance:.1f} km = {distance / au:.6f} AU")
    out("  Earth is near perihelion in the first week of January, so this must be")
    out("  about 1.7 per cent below one astronomical unit.")
    out.check(
        "the solar distance at J2000.0 is between 0.980 and 0.986 AU",
        0.980 * au < distance < 0.986 * au,
        f"{distance / au:.6f} AU",
    )
    jd = J2000 + np.linspace(0.0, 2000.0, 4001)
    lunar = np.linalg.norm(frames.moon_position_teme_km(jd), axis=-1)
    out(
        f"  lunar distance over 2000 days : {float(lunar.min()):.0f} to "
        f"{float(lunar.max()):.0f} km"
    )
    out("  published perigee-apogee range: about 356400 to 406700 km")
    out.check(
        "the lunar distance stays inside 352000 to 410000 km",
        float(lunar.min()) > 352000.0 and float(lunar.max()) < 410000.0,
        f"{float(lunar.min()):.0f} to {float(lunar.max()):.0f} km",
    )
    first = frames.moon_position_teme_km(J2000)
    second = frames.moon_position_teme_km(J2000 + 1.0)
    cosine = float(
        np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
    )
    daily = float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))
    out(f"  lunar angular motion per day  : {daily:.4f} deg")
    out("  published mean sidereal rate  : 13.176 deg/day")
    out.check(
        "the lunar daily motion is between 11 and 15 degrees",
        11.0 < daily < 15.0,
        f"{daily:.4f} deg/day",
    )

    out.rule("negative control")
    out("  A deliberately permuted RIC basis must break the in-track dominance")
    out("  check above, otherwise that check is not testing the axis order.")
    permuted_in_track = radial
    permuted_radial = in_track
    fraction = float(
        np.mean((permuted_in_track > permuted_radial) & (permuted_in_track > cross))
    )
    out(f"  in-track dominance with the radial and in-track axes swapped: {fraction:.6f}")
    out.check(
        "swapping the radial and in-track axes destroys the dominance (control fires)",
        fraction < 0.05,
        f"{fraction:.6f} against the 0.99 threshold the real check uses",
    )

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())
