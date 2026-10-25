"""Reproduce the published SGP4 verification vectors and report the disagreement.

The files are read out of the installed ``sgp4`` distribution as package data.
They are the verification set published with Vallado, Crawford, Hujsak and
Kelso (2006), "Revisiting Spacetrack Report #3", AIAA 2006-6753, together with
the ``testcpp.cpp`` reference program; the distribution carries them unchanged.
The parser in :mod:`orbitdrift.vectors` is this package's own and does not
import ``sgp4.tests``.
"""

from __future__ import annotations

import sys

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

import numpy as np  # noqa: E402

from orbitdrift.vectors import (  # noqa: E402
    EXPECTED_FILE,
    TLE_FILE,
    compare_to_published,
    published_cases,
    sgp4_distribution_version,
)


def main() -> int:
    out = Tee("validate_sgp4_vectors_output.txt")
    out.rule("orbitdrift -- SGP4 reproduced against its own published test vectors")
    for line in environment_lines():
        out(line)

    out.rule("source of the vectors")
    out(f"  distribution                  : sgp4 {sgp4_distribution_version()} (installed)")
    out(f"  element sets                  : {TLE_FILE} (package data inside that distribution)")
    out(f"  expected states               : {EXPECTED_FILE} (package data, 700 lines)")
    out("  published with                 : Vallado, Crawford, Hujsak and Kelso (2006),")
    out("                                   'Revisiting Spacetrack Report #3',")
    out("                                   AIAA 2006-6753, and its testcpp.cpp program")
    out("  parser                         : orbitdrift.vectors, this package's own;")
    out("                                   sgp4.tests is NOT imported")
    out("  printing precision of the file : 1e-8 km in position, 1e-9 km/s in velocity,")
    out("                                   so no comparison can be tighter than about")
    out("                                   5e-9 km and 5e-10 km/s from rounding alone")

    cases = published_cases()
    results = [compare_to_published(case) for case in cases]

    out.rule("per-case disagreement")
    out("  satnum   rows  skipped      max |dr| km    max |dv| km/s")
    for result in results:
        out(
            f"  {result.satnum:6d} {result.rows_compared:6d} "
            f"{result.rows_skipped_error_code:8d} "
            f"{result.max_position_error_km:16.6e} {result.max_velocity_error_km_s:16.6e}"
        )

    compared = sum(result.rows_compared for result in results)
    skipped = sum(result.rows_skipped_error_code for result in results)
    worst_position = max(result.max_position_error_km for result in results)
    worst_velocity = max(result.max_velocity_error_km_s for result in results)
    positions = np.array([result.max_position_error_km for result in results])

    out.rule("summary")
    out(f"  cases                         : {len(cases)}")
    out(f"  rows compared                 : {compared}")
    out(f"  rows skipped on an error code : {skipped}")
    out(f"  worst |dr|                    : {worst_position:.6e} km")
    out(f"  worst |dv|                    : {worst_velocity:.6e} km/s")
    out(f"  median per-case worst |dr|    : {float(np.median(positions)):.6e} km")

    out.rule("checks")
    out.check("33 cases parsed", len(cases) == 33, f"{len(cases)}")
    out.check(
        "667 published data rows accounted for",
        compared + skipped == 667,
        f"{compared} compared + {skipped} skipped",
    )
    out.check(
        "worst position disagreement below 1e-6 km",
        worst_position < 1e-6,
        f"{worst_position:.6e} km",
    )
    out.check(
        "worst velocity disagreement below 1e-8 km/s",
        worst_velocity < 1e-8,
        f"{worst_velocity:.6e} km/s",
    )

    out.rule("skipped rows, named")
    for result in results:
        if result.rows_skipped_error_code:
            out(
                f"  catalogue {result.satnum}: {result.rows_skipped_error_code} row(s), "
                f"SGP4 error code(s) {result.error_codes}"
            )
    out("  tcppver.out handles a satellite whose propagation fails at zero minutes by")
    out("  repeating the previous satellite's last data line, a quirk of the reference")
    out("  program. Comparing that row would report a spurious disagreement of")
    out("  thousands of kilometres, so it is skipped and counted here rather than")
    out("  dropped silently.")

    out.rule("negative control")
    case = cases[0]
    broken_line2 = case.line2[:8] + " 44.2682" + case.line2[16:]
    from orbitdrift.vectors import PublishedCase

    broken = PublishedCase(
        satnum=case.satnum,
        line1=case.line1,
        line2=broken_line2,
        schedule_min=case.schedule_min,
        times_min=case.times_min,
        positions_km=case.positions_km,
        velocities_km_s=case.velocities_km_s,
    )
    control = compare_to_published(broken)
    out("  catalogue 5 with its inclination edited from 34.2682 to 44.2682 degrees:")
    out(f"    rows compared               : {control.rows_compared}")
    out(f"    worst |dr|                  : {control.max_position_error_km:.6f} km")
    out.check(
        "the comparison can fail (negative control fires)",
        control.max_position_error_km > 100.0,
        f"{control.max_position_error_km:.3f} km against a 1e-6 km pass threshold",
    )

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())
