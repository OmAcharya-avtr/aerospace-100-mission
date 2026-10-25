"""Visibility windows against an independent geometric computation.

The specification for this product asks for the visibility windows to be
cross-checked against an independent geometric computation. ``astropy`` is not
installed in this environment, so the independent computation is one written
here, in this file, from a different formulation: the spacecraft and the station
are reduced to geocentric latitude, longitude and radius, the central angle
between them is obtained from the spherical law of cosines, and the elevation
follows from

    ``tan(elevation) = (cos(gamma) - r_station / r_sat) / sin(gamma)``

which involves no Cartesian dot product and no south-east-zenith rotation. That
is a different route to the same angle, and it is described as a second
derivation written in this package, not as a cross-check against a third-party
library.

Three further checks:

* the window *boundaries* found by coarse scan plus bisection against a dense
  one-second scan of the same margin function, which checks the root search
  rather than the geometry;
* every window duration against the independently derived closed-form upper
  bound in :func:`contactsched.geometry.max_pass_duration_s`;
* the union supply per antenna against the naive sum, to show that overlapping
  visibility is not double counted.
"""

from __future__ import annotations

import math

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

from contactsched.ephemeris import Scenario, default_fleet  # noqa: E402
from contactsched.geometry import (  # noqa: E402
    elevation_geocentric_vector_deg,
    max_pass_duration_s,
)
from contactsched.station import default_network  # noqa: E402
from contactsched.visibility import (  # noqa: E402
    all_windows,
    compute_windows,
    mask_margin_deg,
    supply_seconds_by_station,
)

FAILURES: list[str] = []


def independent_geocentric_elevation_deg(
    satellite_ecef_km: np.ndarray, station_ecef_km: np.ndarray
) -> np.ndarray:
    """Geocentric elevation from spherical trigonometry on latitude and longitude.

    An independent derivation, written for this check only. The station and the
    spacecraft are expressed as geocentric latitude, longitude and radius; the
    central angle between them comes from the spherical law of cosines; the
    elevation comes from the plane triangle at the station. Nothing here is a
    dot product in the Earth-fixed frame and nothing uses a rotation matrix.
    """
    satellite = np.atleast_2d(np.asarray(satellite_ecef_km, dtype=float))
    station = np.asarray(station_ecef_km, dtype=float)
    r_sat = np.linalg.norm(satellite, axis=1)
    r_station = float(np.linalg.norm(station))
    lat_sat = np.arcsin(satellite[:, 2] / r_sat)
    lon_sat = np.arctan2(satellite[:, 1], satellite[:, 0])
    lat_station = math.asin(station[2] / r_station)
    lon_station = math.atan2(station[1], station[0])
    cos_gamma = np.clip(
        math.sin(lat_station) * np.sin(lat_sat)
        + math.cos(lat_station) * np.cos(lat_sat) * np.cos(lon_sat - lon_station),
        -1.0,
        1.0,
    )
    gamma = np.arccos(cos_gamma)
    sin_gamma = np.sin(gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        elevation = np.arctan2(cos_gamma - r_station / r_sat, sin_gamma)
    return np.degrees(np.where(sin_gamma > 0.0, elevation, np.pi / 2.0))


def main() -> None:
    out = Tee("validate_visibility_output.txt")
    out.rule("contactsched validate_visibility")
    for line in environment_lines():
        out(line)

    scenario = Scenario.from_calendar(2026, 4, 30, 0, 0, 0.0, duration_s=86400.0)
    network = default_network()
    fleet = default_fleet()
    windows = all_windows(scenario, network, fleet, step_s=30.0, tolerance_s=0.25)

    out.rule("Scenario")
    out("epoch                 : 2026-04-30T00:00:00 (UTC, see the UT1 caveat below)")
    out(f"horizon               : {scenario.duration_s:.0f} s")
    out(f"antennas              : {len(network)}")
    out(f"spacecraft            : {len(fleet)}")
    out(f"windows found         : {len(windows)}")
    out(f"total visible seconds : {sum(w.duration_s for w in windows):.3f}")
    out(f"mask-limited windows  : {sum(1 for w in windows if w.mask_limited)}")
    out(f"clipped at a horizon  : {sum(1 for w in windows if w.clipped_start or w.clipped_end)}")
    out("")
    out("UT1 against UTC is not modelled. The difference is below 0.9 s, which rotates")
    out("the Earth by up to 0.0037 deg and displaces a station by up to roughly 0.4 km.")
    out("That is why boundaries here are reported to a quarter of a second of SEARCH")
    out("resolution and not claimed as accurate to that level against a real sky.")

    out.rule("1. Elevation against an INDEPENDENT geometric derivation")
    out("Spherical law of cosines on geocentric latitude and longitude, against the")
    out("Cartesian dot product. The quantity compared is the GEOCENTRIC elevation,")
    out("because the spherical derivation has no access to the ellipsoid normal.")
    out("")
    out(f"{'antenna':<14} {'spacecraft':<12} {'n':>7} {'max |diff| (deg)':>18} {'median':>12}")
    out("-" * 70)
    worst = 0.0
    total_samples = 0
    for station in network:
        for spacecraft in fleet:
            times = np.linspace(0.0, scenario.duration_s, 2401)
            positions = spacecraft.ecef_km(scenario, times)
            mine = elevation_geocentric_vector_deg(positions, station.ecef_km)
            theirs = independent_geocentric_elevation_deg(positions, station.ecef_km)
            difference = np.abs(mine - theirs)
            worst = max(worst, float(difference.max()))
            total_samples += len(times)
            out(
                f"{station.name:<14} {spacecraft.name:<12} {len(times):>7d} "
                f"{float(difference.max()):>18.3e} {float(np.median(difference)):>12.3e}"
            )
    out("")
    out(f"samples compared : {total_samples}")
    out(f"worst difference : {worst:.3e} deg")
    out("tolerance        : 1.000e-09 deg")
    if worst > 1e-9:
        FAILURES.append(
            f"the independent geometric elevation disagrees by {worst:.3e} deg"
        )

    out.rule("2. Window boundaries against a dense one-second scan")
    out("The coarse scan plus bisection is checked against a brute-force scan of the")
    out("same margin function at one-second resolution. This checks the root search,")
    out("not the geometry: both use the same elevation and the same mask.")
    out("")
    out(f"{'antenna':<14} {'spacecraft':<12} {'windows':>8} {'dense':>7} {'max |dt| (s)':>13}")
    out("-" * 62)
    worst_boundary = 0.0
    mismatched = 0
    for station in network[:2]:
        for spacecraft in fleet[:3]:
            refined = compute_windows(
                scenario, station, spacecraft, step_s=30.0, tolerance_s=0.25
            )
            dense_times = np.arange(0.0, scenario.duration_s + 1.0, 1.0)
            margin = mask_margin_deg(scenario, station, spacecraft, dense_times)
            above = margin >= 0.0
            edges = np.flatnonzero(above[:-1] != above[1:])
            dense_boundaries = [0.5 * (dense_times[i] + dense_times[i + 1]) for i in edges]
            refined_boundaries: list[float] = []
            for window in refined:
                if not window.clipped_start:
                    refined_boundaries.append(window.rise_s)
                if not window.clipped_end:
                    refined_boundaries.append(window.set_s)
            if len(refined_boundaries) != len(dense_boundaries):
                mismatched += 1
                out(
                    f"{station.name:<14} {spacecraft.name:<12} "
                    f"{len(refined_boundaries):>8d} {len(dense_boundaries):>7d} "
                    f"{'COUNT MISMATCH':>13}"
                )
                continue
            differences = [
                abs(a - b)
                for a, b in zip(sorted(refined_boundaries), sorted(dense_boundaries), strict=True)
            ]
            worst_boundary = max(worst_boundary, max(differences, default=0.0))
            out(
                f"{station.name:<14} {spacecraft.name:<12} "
                f"{len(refined_boundaries):>8d} {len(dense_boundaries):>7d} "
                f"{max(differences, default=0.0):>13.4f}"
            )
    out("")
    out(f"boundary count mismatches : {mismatched}")
    out(f"worst boundary difference : {worst_boundary:.4f} s")
    out("tolerance                 : 0.7500 s (half the dense grid plus the bisection)")
    if mismatched:
        FAILURES.append(f"{mismatched} station-spacecraft pairs disagree on window count")
    if worst_boundary > 0.75:
        FAILURES.append(f"a window boundary moved by {worst_boundary:.4f} s against a dense scan")

    out.rule("3. Window durations against the INDEPENDENTLY DERIVED upper bound")
    out("The bound is closed-form spherical geometry on the TLE's own mean motion and")
    out("shares no expression with the root search. It is deliberately loose; the point")
    out("is that it cannot be exceeded, not that it is tight.")
    out("")
    out(
        f"{'antenna':<14} {'spacecraft':<12} {'longest (s)':>12} {'bound (s)':>11} "
        f"{'ratio':>7}"
    )
    out("-" * 62)
    exceeded = 0
    for station in network:
        for spacecraft in fleet:
            interior = [
                w
                for w in windows
                if w.station == station.name
                and w.spacecraft == spacecraft.name
                and not w.clipped_start
                and not w.clipped_end
            ]
            if not interior:
                continue
            longest = max(w.duration_s for w in interior)
            bound = max_pass_duration_s(
                spacecraft.elements.semi_major_axis_km,
                spacecraft.elements.eccentricity,
                station.geocentric_radius_km,
                station.mask.floor_deg,
            )
            out(
                f"{station.name:<14} {spacecraft.name:<12} {longest:>12.2f} "
                f"{bound:>11.2f} {longest / bound:>7.3f}"
            )
            if longest > bound:
                exceeded += 1
    out("")
    out(f"windows exceeding the bound : {exceeded}")
    if exceeded:
        FAILURES.append(f"{exceeded} station-spacecraft pairs exceed the derived duration bound")

    out.rule("4. Union supply against the naive sum per antenna")
    supply = supply_seconds_by_station(windows)
    out(f"{'antenna':<14} {'sum of windows (s)':>19} {'union (s)':>12} {'overlap (s)':>12}")
    out("-" * 62)
    for name in sorted(supply):
        raw = sum(w.duration_s for w in windows if w.station == name)
        out(f"{name:<14} {raw:>19.2f} {supply[name]:>12.2f} {raw - supply[name]:>12.2f}")
        if supply[name] > raw + 1e-6:
            FAILURES.append(f"the union supply for {name} exceeds the sum of its windows")
    out("")
    out("A positive overlap column is the point: two spacecraft visible at once supply")
    out("the antenna only once, and the contention ratio in contactsched.contention")
    out("divides by the union for that reason.")

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
