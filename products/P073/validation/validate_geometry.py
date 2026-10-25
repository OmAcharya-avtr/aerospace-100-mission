"""Sidereal time, the two geocentric elevation derivations, and a derived bound.

Three checks, of three different kinds.

1. **Against an installed independent implementation.** The IAU-1982 GMST
   polynomial implemented in :func:`contactsched.geometry.gmst_rad` is measured
   against ``sgp4.propagation.gstime`` over sixty years. ``astropy`` is not
   installed in this environment and nothing here is compared against it.

2. **Two implementations agree.** The geocentric elevation computed from vectors
   and from three scalar lengths by the law of cosines. This is the weaker class
   of check and it is labelled as such: two implementations of one quantity can
   share a mistake.

3. **Against an independently derived bound.** The geodetic elevation against
   the geocentric elevation of the same slant vector, bounded by the closed-form
   maximum angle between the WGS-84 ellipsoid normal and the position vector.
   The bound comes from the ellipsoid and shares no expression with either
   elevation computation, which is what makes it able to catch a mistake both of
   them make. It is also the check that falsified a hard-coded constant during
   this build.

The ill-conditioning of the law-of-cosines form near the horizon is *measured*
here rather than asserted, because the magnitude is the useful output: it tells
a reader which form to use where.
"""

from __future__ import annotations

import math

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

from contactsched.geometry import (  # noqa: E402
    JD_J2000,
    MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG,
    WGS84_A_KM,
    WGS84_B_KM,
    WGS84_E2,
    elevation_geocentric_norms_deg,
    elevation_geocentric_vector_deg,
    geodetic_to_ecef_km,
    gmst_rad,
    look_angles_deg,
    max_geodetic_geocentric_deviation_deg,
    sez_matrix,
)

FAILURES: list[str] = []


def main() -> None:
    out = Tee("validate_geometry_output.txt")
    out.rule("contactsched validate_geometry")
    for line in environment_lines():
        out(line)

    out.rule("1. WGS-84 constants against their closed forms")
    out(f"{'quantity':<42} {'value':>22} {'reference':>22}")
    out("-" * 90)
    flattening = 1.0 / 298.257223563
    rows = [
        ("semi-major axis a (km)", WGS84_A_KM, 6378.137),
        ("semi-minor axis b = a(1-f) (km)", WGS84_B_KM, WGS84_A_KM * (1.0 - flattening)),
        ("first eccentricity squared e2", WGS84_E2, 2.0 * flattening - flattening**2),
        (
            "max normal-to-radial deviation (deg)",
            MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG,
            max_geodetic_geocentric_deviation_deg(),
        ),
    ]
    for name, value, reference in rows:
        out(f"{name:<42} {value:>22.12f} {reference:>22.12f}")
        if abs(value - reference) > 1e-10 * max(1.0, abs(reference)):
            FAILURES.append(f"constant {name} disagrees with its closed form")

    scan_best = 0.0
    scan_at = 0.0
    for index in range(0, 900001):
        phi = math.radians(index / 10000.0)
        psi = math.atan((1.0 - WGS84_E2) * math.tan(phi)) if index < 900000 else phi
        deviation = math.degrees(phi - psi)
        if deviation > scan_best:
            scan_best, scan_at = deviation, index / 10000.0
    out("")
    out(
        "brute-force scan of the deviation over latitude at 1e-4 deg resolution: "
        f"maximum {scan_best:.14f} deg at {scan_at:.4f} deg"
    )
    out(
        f"closed form minus scan                                       : "
        f"{max_geodetic_geocentric_deviation_deg() - scan_best:+.3e} deg"
    )
    if abs(max_geodetic_geocentric_deviation_deg() - scan_best) > 1e-11:
        FAILURES.append("the closed-form deviation does not match the brute-force scan")

    out.rule("2. GMST against sgp4.propagation.gstime (an installed implementation)")
    from sgp4.propagation import gstime

    offsets = [-21915.0, -7305.0, -365.25, -1.0, 0.0, 0.5, 1.0, 365.25, 7305.0, 21915.0]
    out(f"{'JD - J2000 (days)':>20} {'this package (rad)':>22} {'sgp4 (rad)':>22} {'|diff|':>12}")
    out("-" * 80)
    worst = 0.0
    for offset in offsets:
        jd = JD_J2000 + offset
        mine = gmst_rad(jd)
        theirs = gstime(jd)
        difference = abs(mine - theirs)
        difference = min(difference, 2.0 * math.pi - difference)
        worst = max(worst, difference)
        out(f"{offset:>20.2f} {mine:>22.15f} {theirs:>22.15f} {difference:>12.3e}")
    out("")
    out(f"worst absolute disagreement over 120 years of span: {worst:.3e} rad")
    out(f"equivalent Earth rotation                        : {math.degrees(worst):.3e} deg")
    out("tolerance                                        : 1.000e-08 rad")
    if worst > 1e-8:
        FAILURES.append(f"GMST disagrees with sgp4.gstime by {worst:.3e} rad")

    out.rule("3. Two geocentric elevation derivations (two implementations, one quantity)")
    out("Law of cosines on three lengths against the vector dot product.")
    out("")
    rng = np.random.default_rng(20261025)
    station = geodetic_to_ecef_km(55.0, -3.0, 0.1)
    radius = float(np.linalg.norm(station))
    offsets_km = rng.normal(scale=1800.0, size=(40000, 3))
    satellite = station[None, :] + offsets_km
    keep = np.linalg.norm(satellite, axis=1) > radius + 50.0
    satellite = satellite[keep]
    slant = np.linalg.norm(satellite - station[None, :], axis=1)
    vector = elevation_geocentric_vector_deg(satellite, station)
    norms = elevation_geocentric_norms_deg(
        np.linalg.norm(satellite, axis=1), radius, slant
    )
    difference = np.abs(vector - norms)
    out(f"{'|elevation| band (deg)':<26} {'n':>7} {'max |diff| (deg)':>18} {'median':>14}")
    out("-" * 70)
    bands = [(0.0, 0.1), (0.1, 1.0), (1.0, 5.0), (5.0, 20.0), (20.0, 60.0), (60.0, 90.0)]
    for low, high in bands:
        mask = (np.abs(vector) >= low) & (np.abs(vector) < high)
        if not mask.any():
            continue
        out(
            f"[{low:>6.2f}, {high:>6.2f})          {int(mask.sum()):>7d} "
            f"{float(difference[mask].max()):>18.3e} "
            f"{float(np.median(difference[mask])):>14.3e}"
        )
    out("")
    out("What the table actually shows, which is NOT what was expected:")
    out("")
    out("The law-of-cosines numerator r_sat**2 - r_station**2 - range**2 is a")
    out("difference of nearly equal large numbers near the horizon, so the expectation")
    out("was that the disagreement would grow as the elevation fell. It does the")
    out("opposite. The cancellation leaves an absolute error in sin(elevation) of order")
    out("eps * r_station / (2 * range), which is 3.9e-16 at these ranges and is almost")
    out("flat in elevation; the conversion through arcsin then multiplies it by")
    out("1 / cos(elevation), which is 1 at the horizon and 57 at 89 degrees. The")
    out("amplification near the zenith dominates the cancellation near the horizon, and")
    out("the worst disagreement in this sample is in the highest band, not the lowest.")
    out("")
    out("This is recorded rather than corrected because the prose written before the")
    out("measurement said the reverse, and the measurement is what is published.")
    out("")
    out("Scaling of the disagreement with slant range, at fixed elevation near 45 deg:")
    out("")
    out(f"{'slant range (km)':>18} {'eps*R/(2*range) (rad)':>23} {'measured max (deg)':>20}")
    out("-" * 64)
    site = geodetic_to_ecef_km(45.0, 0.0, 0.0)
    site_radius = float(np.linalg.norm(site))
    unit = site / site_radius
    tangent_dir = np.cross(unit, np.array([0.0, 0.0, 1.0]))
    tangent_dir = tangent_dir / np.linalg.norm(tangent_dir)
    direction = (unit + tangent_dir) / math.sqrt(2.0)
    for span in (100.0, 400.0, 1600.0, 6400.0, 25600.0):
        probe = (site[None, :] + span * direction[None, :])
        probe = probe + rng.normal(scale=1e-3, size=(2000, 3))
        probe_slant = np.linalg.norm(probe - site[None, :], axis=1)
        a = elevation_geocentric_vector_deg(probe, site)
        b = elevation_geocentric_norms_deg(
            np.linalg.norm(probe, axis=1), site_radius, probe_slant
        )
        predicted = 2.22e-16 * site_radius / (2.0 * span)
        out(
            f"{span:>18.1f} {predicted:>23.3e} "
            f"{float(np.max(np.abs(a - b))):>20.3e}"
        )
    out("")
    out("The tolerance asserted is on the band where the scheduler reads the angle:")
    high_band = np.abs(vector) > 10.0
    worst_high = float(difference[high_band].max())
    out(f"  above 10 deg elevation: worst {worst_high:.3e} deg, tolerance 1.000e-06 deg")
    if worst_high > 1e-6:
        FAILURES.append("the two geocentric derivations disagree above 10 deg elevation")
    worst_all = float(difference.max())
    out(f"  over the whole sample : worst {worst_all:.3e} deg, tolerance 1.000e-06 deg")
    if worst_all > 1e-6:
        FAILURES.append("the two geocentric derivations disagree over the whole sample")

    out.rule("4. Geodetic against geocentric elevation, against an INDEPENDENT BOUND")
    out("The bound is the closed-form maximum angle between the WGS-84 ellipsoid")
    out("normal and the position vector. It is derived from the ellipsoid and shares no")
    out("expression with either elevation computation, so it can falsify a mistake that")
    out("both of them make. It did: a hard-coded 0.19242218 deg stood in geometry.py")
    out("until this check failed by 1.0e-06 deg.")
    out("")
    out(f"{'station latitude (deg)':>22} {'n':>7} {'max |geodetic - geocentric|':>29}")
    out("-" * 62)
    bound = MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG
    worst_bound = 0.0
    for latitude in (-80.0, -60.0, -45.1, -20.0, 0.0, 20.0, 45.1, 60.0, 80.0):
        site = geodetic_to_ecef_km(latitude, 25.0, 0.0)
        sample = site[None, :] + rng.normal(scale=2000.0, size=(4000, 3))
        _az, geodetic, _r = look_angles_deg(sample, site, sez_matrix(latitude, 25.0))
        geocentric = elevation_geocentric_vector_deg(sample, site)
        gap = float(np.max(np.abs(geodetic - geocentric)))
        worst_bound = max(worst_bound, gap)
        out(f"{latitude:>22.1f} {4000:>7d} {gap:>29.9f}")
    out("")
    out(f"worst over all latitudes : {worst_bound:.9f} deg")
    out(f"independently derived bound: {bound:.9f} deg")
    out(f"margin                   : {bound - worst_bound:+.3e} deg")
    if worst_bound > bound + 1e-9:
        FAILURES.append("the geodetic-geocentric difference exceeds its derived bound")

    out.rule("5. Exact configurations")
    out(f"{'configuration':<48} {'computed':>14} {'exact':>10}")
    out("-" * 76)
    equator = geodetic_to_ecef_km(0.0, 0.0, 0.0)
    overhead = (equator * 1.1)[None, :]
    _az, el, _r = look_angles_deg(overhead, equator, sez_matrix(0.0, 0.0))
    out(f"{'radial point above an equatorial station (deg)':<48} {el[0]:>14.9f} {90.0:>10.1f}")
    if abs(el[0] - 90.0) > 1e-9:
        FAILURES.append("a radial point is not at 90 deg elevation")
    tangent = (equator + np.array([0.0, 500.0, 0.0]))[None, :]
    az, el, rngs = look_angles_deg(tangent, equator, sez_matrix(0.0, 0.0))
    out(f"{'due-east tangential point, elevation (deg)':<48} {el[0]:>14.9f} {0.0:>10.1f}")
    out(f"{'due-east tangential point, azimuth (deg)':<48} {az[0]:>14.9f} {90.0:>10.1f}")
    out(f"{'due-east tangential point, range (km)':<48} {rngs[0]:>14.9f} {500.0:>10.1f}")
    if abs(el[0]) > 1e-12 or abs(az[0] - 90.0) > 1e-12:
        FAILURES.append("the tangential configuration is wrong")

    out.rule("Result")
    if FAILURES:
        out(f"FAILED {len(FAILURES)} check(s):")
        for failure in FAILURES:
            out(f"  - {failure}")
    else:
        out("All checks in this script passed.")
    out("")
    out(f"written to {out.save()}")
    if FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
