"""Pass geometry: provenance, an official known answer, and two cross-checks.

Five measurements, in the order a sceptic would ask for them.

1. **Provenance of the element set.** The two committed TLE lines are
   compared character for character against the official SGP4 verification
   file ``SGP4-VER.TLE`` installed by the `sgp4` package, which originates in
   Vallado, Crawford, Hujsak & Kelso (2006), "Revisiting Spacetrack Report
   #3", AIAA 2006-6753. The element set is real and published; this script is
   what makes that claim checkable rather than asserted.

2. **Known answer against the official verification output.** The propagation
   is compared against ``tcppver.out``, also installed by the package and also
   from that paper: the reference TEME position and velocity of catalogue
   object 06251 at a sequence of times since epoch. This is the strongest
   available check on the propagator, because the reference numbers were
   produced by the authors' own implementation and not by ours.

3. **GMST against the library's own implementation.** Our IAU-1982 polynomial
   against ``sgp4.propagation.gstime``. Two implementations of the same
   published polynomial, compared to tolerance.

4. **Slant range against an independent closed form.** The SGP4 chain against
   the spherical-Earth circular-orbit expression from Maral, Bousquet & Sun
   (2020) chapter 2. The two must disagree, and by a knowable amount: the
   closed form omits the orbit's eccentricity and the Earth's oblateness. The
   measured disagreement is reported rather than asserted to be small.

5. **The pass search's own margins.** The shortest window found against the
   coarse scan step, and the elevation sensitivity of the rate model at the
   mask, which is what bounds the cost of the TEME-to-Earth-fixed
   approximation.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from sgp4.api import Satrec  # noqa: E402
from sgp4.propagation import gstime  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from passyield.dataset import STATIONS, load_passes  # noqa: E402
from passyield.geometry import (  # noqa: E402
    TLE_LINE1,
    TLE_LINE2,
    campaign_epoch,
    find_passes,
    gmst_rad,
    satellite,
    slant_range_spherical_km,
    tle_source_path,
    topocentric_profile,
)
from passyield.link import DEFAULT_BUDGET, rate_mbit_s  # noqa: E402

#: Scan step of the committed campaign, seconds.
COARSE_STEP_S = 5.0

#: Elevation perturbation used to measure the rate gradient, degrees.
GRADIENT_PROBE_DEG = 1.0e-3


def main() -> int:
    """Run the five geometry measurements."""
    out = _bootstrap.Tee("validate_geometry_output.txt")
    out("Pass geometry validation -- P099 PassYield")
    out("")
    out(f"fast path (PASSYIELD_FAST=1): {_bootstrap.FAST}")
    out("")
    out("SGP4 is a MEAN-ELEMENTS model. A two-line element set is a set of Brouwer")
    out("mean elements fitted inside the analytical theory SGP4 inverts, so the")
    out("elements are meaningful only when fed to SGP4 itself. Output frame is TEME")
    out("of date, not J2000 and not ITRF.")
    out("Source: Vallado, Crawford, Hujsak & Kelso (2006), Revisiting Spacetrack")
    out("Report #3, AIAA 2006-6753.")

    satrec = satellite()

    out.rule("1. PROVENANCE -- the committed element set against SGP4-VER.TLE")
    source = tle_source_path()
    out(f"installed verification file: {source.name}")
    present = source.is_file()
    out.check("the sgp4 package installs its verification element-set file", present)
    if present:
        text = source.read_text()
        out(f"committed line 1: {TLE_LINE1}")
        out(f"committed line 2: {TLE_LINE2}")
        out.check("committed line 1 appears verbatim in the installed file", TLE_LINE1 in text)
        out.check("committed line 2 appears verbatim in the installed file", TLE_LINE2 in text)
        out("")
        out("catalogue object 06251, DELTA 1 DEB, epoch 2006-06-25. Inclination")
        out(f"{satrec.inclo * 180.0 / np.pi:.4f} deg, eccentricity {satrec.ecco:.7f},")
        out(f"mean motion {satrec.no_kozai * 1440.0 / (2.0 * np.pi):.8f} rev/day.")
        out("A moderate-drag low-Earth case, which is the regime a contact schedule")
        out("is actually written for.")
    else:
        out.unavailable("element-set provenance", "the installed package has no SGP4-VER.TLE")

    out.rule("2. KNOWN ANSWER -- propagation against the official tcppver.out")
    reference = source.parent / "tcppver.out"
    if not reference.is_file():
        out.unavailable("official verification output", "the package installs no tcppver.out")
    else:
        lines = reference.read_text().splitlines()
        start = next(
            index for index, line in enumerate(lines) if line.startswith("6251 xx")
        )
        rows: list[tuple[float, list[float]]] = []
        for line in lines[start + 1 :]:
            fields = line.split()
            if len(fields) < 7 or not fields[0].replace(".", "").replace("-", "").isdigit():
                break
            rows.append((float(fields[0]), [float(value) for value in fields[1:7]]))
        out(f"reference rows for catalogue 06251: {len(rows)}")
        out("")
        out(f"{'tsince, min':>13}{'|dr|, km':>14}{'|dv|, km/s':>14}")
        worst_position = 0.0
        worst_velocity = 0.0
        for tsince, expected in rows:
            error, position, velocity = satrec.sgp4_tsince(tsince)
            if error != 0:
                out(f"{tsince:>13.2f}  sgp4 error code {error}")
                continue
            position_error = float(
                np.linalg.norm(np.asarray(position) - np.asarray(expected[:3]))
            )
            velocity_error = float(
                np.linalg.norm(np.asarray(velocity) - np.asarray(expected[3:6]))
            )
            worst_position = max(worst_position, position_error)
            worst_velocity = max(worst_velocity, velocity_error)
            out(f"{tsince:>13.2f}{position_error:>14.2e}{velocity_error:>14.2e}")
        out("")
        out(f"worst position disagreement, km  : {worst_position:.3e}")
        out(f"worst velocity disagreement, km/s: {worst_velocity:.3e}")
        out("")
        out("the reference numbers are printed to 8 decimal places of a kilometre, so")
        out("1e-7 km is the file's own resolution and is the tolerance used here.")
        out.check(
            "every reference row agrees to the verification file's own resolution",
            worst_position < 1.0e-7 and worst_velocity < 1.0e-7,
            f"worst {worst_position:.3e} km and {worst_velocity:.3e} km/s",
        )

    out.rule("3. GMST -- our IAU-1982 polynomial against the library's own")
    out("two implementations of the same published polynomial. Compared to")
    out("tolerance, never bit-exactly: the two accumulate the Horner terms in a")
    out("different order.")
    out("")
    out(f"{'julian date':>14}{'ours, rad':>16}{'sgp4, rad':>16}{'difference':>14}")
    worst_gmst = 0.0
    for julian in (2451545.0, 2453911.5, 2455000.25, 2460000.75, 2470000.0):
        ours = float(gmst_rad(julian))
        theirs = float(gstime(julian))
        worst_gmst = max(worst_gmst, abs(ours - theirs))
        out(f"{julian:>14.2f}{ours:>16.10f}{theirs:>16.10f}{abs(ours - theirs):>14.2e}")
    out("")
    out(f"worst disagreement, rad: {worst_gmst:.3e}")
    out.check(
        "GMST agrees with the library implementation to 1e-9 rad",
        worst_gmst < 1.0e-9,
        f"worst {worst_gmst:.3e} rad",
    )

    out.rule("4. SLANT RANGE -- SGP4 chain against an independent closed form")
    out("Maral, Bousquet & Sun (2020), Satellite Communications Systems, 6th ed.,")
    out("chapter 2: d = R[sqrt(((R+h)/R)^2 - cos^2(el)) - sin(el)], for a SPHERICAL")
    out("Earth and a CIRCULAR orbit. The committed element set has eccentricity")
    out(f"{satrec.ecco:.7f} and the Earth is oblate, so the two MUST disagree. The")
    out("point of the check is that the disagreement is the size those two omissions")
    out("predict and not the size of a coding error.")
    out("")
    record = load_passes()
    station_names = {station.name: station for station in STATIONS}
    altitudes: list[float] = []
    disagreements: list[float] = []
    out(f"{'station':>9}{'max el, deg':>13}{'sgp4 range':>12}{'closed form':>13}{'ratio':>9}")
    for row in record[:40]:
        station = station_names[row.station]
        from passyield.dataset import window_of

        window = window_of(row)
        elevation, _azimuth, slant = topocentric_profile(
            satrec, station, window, np.asarray([0.5 * window.duration_s])
        )
        peak_elevation = float(elevation[0])
        measured = float(slant[0])
        # The altitude implied by the SGP4 state at that instant, so the
        # closed form is given the altitude it needs rather than a guess.
        error, position, _velocity = satrec.sgp4(
            window.rise_jd, window.rise_fr + 0.5 * window.duration_s / 86400.0
        )
        altitude = float(np.linalg.norm(position)) - 6378.137
        altitudes.append(altitude)
        closed = float(slant_range_spherical_km(peak_elevation, altitude))
        disagreements.append(abs(measured - closed) / measured)
        if len(disagreements) <= 10:
            out(
                f"{row.station:>9}{peak_elevation:>13.3f}{measured:>12.2f}"
                f"{closed:>13.2f}{closed / measured:>9.5f}"
            )
    out("")
    out(f"passes compared                     : {len(disagreements)}")
    out(f"implied altitude range, km          : {min(altitudes):.1f} to {max(altitudes):.1f}")
    out(f"median relative disagreement        : {float(np.median(disagreements)):.4e}")
    out(f"worst relative disagreement         : {float(np.max(disagreements)):.4e}")
    out("")
    out("the closed form is NOT used by the pass-finding path. It is an independent")
    out("expression, and a disagreement of a few per cent is what a spherical Earth")
    out("and a circular orbit cost. A disagreement of order one would be a coding")
    out("error; a disagreement of 1e-15 would mean the two were not independent.")
    out.check(
        "the closed form tracks the SGP4 chain to within 10 per cent",
        float(np.max(disagreements)) < 0.10,
        f"worst {float(np.max(disagreements)):.4e}",
    )
    out.control(
        "the closed form is not secretly the same expression",
        float(np.median(disagreements)) > 1.0e-9,
        must_fire=True,
        detail=f"median disagreement {float(np.median(disagreements)):.3e}",
    )

    out.rule("5. SCAN MARGINS AND THE COST OF THE FRAME APPROXIMATION")
    epoch_jd, epoch_fr = campaign_epoch(satrec)
    span = 2.0 if _bootstrap.FAST else 10.0
    shortest = None
    counts: dict[str, int] = {}
    for station in STATIONS:
        windows = find_passes(
            satrec, station, epoch_jd, epoch_fr, span, coarse_step_s=COARSE_STEP_S
        )
        counts[station.name] = len(windows)
        for window in windows:
            if shortest is None or window.duration_s < shortest:
                shortest = window.duration_s
    out(f"scan span, days            : {span!r}")
    out(f"coarse step, s             : {COARSE_STEP_S!r}")
    out(f"windows found per station  : {counts}")
    out(f"shortest window found, s   : {shortest:.3f}")
    out(f"shortest / coarse step     : {shortest / COARSE_STEP_S:.2f}")
    out("")
    out("a coarse scan CANNOT find a pass shorter than its step. The committed")
    out("campaign discards every window below a declared 60 s bookable threshold,")
    out("which is twelve coarse steps, so the hazard does not reach the record. It")
    out("is stated rather than hidden.")
    out.check(
        "the shortest window found spans at least four coarse steps",
        shortest / COARSE_STEP_S >= 4.0,
        f"{shortest / COARSE_STEP_S:.2f} steps",
    )

    out("")
    out("elevation sensitivity of the rate model at each station's mask, which is")
    out("what bounds the cost of neglecting polar motion, the equation of the")
    out("equinoxes and UT1-UTC in the TEME-to-Earth-fixed rotation. Those three are")
    out("together worth under 1e-5 degrees of elevation (see passyield.geometry).")
    out("")
    out(f"{'station':>9}{'mask, deg':>11}{'dR/del, Mbit/s per deg':>24}{'cost of 1e-5 deg':>18}")
    worst_cost = 0.0
    for station in STATIONS:
        reference_range = float(slant_range_spherical_km(station.mask_deg, 400.0))
        low = float(
            rate_mbit_s(
                station.mask_deg, reference_range,
                station.gain_over_temperature_db, station.zenith_attenuation_db,
            )
        )
        high = float(
            rate_mbit_s(
                station.mask_deg + GRADIENT_PROBE_DEG, reference_range,
                station.gain_over_temperature_db, station.zenith_attenuation_db,
            )
        )
        gradient = (high - low) / GRADIENT_PROBE_DEG
        cost = gradient * 1.0e-5
        worst_cost = max(worst_cost, abs(cost))
        out(f"{station.name:>9}{station.mask_deg:>11.1f}{gradient:>24.5f}{cost:>18.3e}")
    out("")
    out(f"worst rate error from the frame approximation, Mbit/s: {worst_cost:.3e}")
    out(f"as a fraction of the rate ceiling {DEFAULT_BUDGET.max_rate_mbit_s!r} Mbit/s: "
        f"{worst_cost / DEFAULT_BUDGET.max_rate_mbit_s:.3e}")
    out.check(
        "the frame approximation costs under 1e-6 of the rate ceiling",
        worst_cost / DEFAULT_BUDGET.max_rate_mbit_s < 1.0e-6,
        f"{worst_cost / DEFAULT_BUDGET.max_rate_mbit_s:.3e}",
    )
    assert isinstance(satrec, Satrec)
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
