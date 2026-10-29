"""The low-elevation regime, exhibited rather than extrapolated into.

Each closed form in this package stops working somewhere, and this script shows
where and how, by evaluating each one past the point where it stops working and
printing what it returns. Nothing here is a model of good behaviour. The point
is that a reader who takes any of these formulae below the elevations tabulated
here is not getting a degraded answer, they are getting an answer with the
wrong sign or the wrong order of magnitude, and the library's job is to refuse
instead.

Three failure modes are distinguished, because they are not the same thing:

* **An analytic floor.** Equation (13) has no solution below
  ``90 - arcsin(1/n)``. There is nothing to extrapolate into: the equation is
  not satisfiable. This floor is computed in closed form.
* **A truncation inversion.** Equation (14) is a two-term asymptotic series,
  and below the elevation where its ``tan^3`` term exceeds its ``tan`` term the
  truncation has inverted. The series keeps returning a number. The number
  changes sign.
* **A fitted-range departure.** Equations (15) and (16) are fits that remain
  finite, smooth and plausible all the way to the horizon, and depart from the
  layered integral by amounts that this script measures. This is the dangerous
  mode, because nothing about the output announces it.
"""

from __future__ import annotations

import math

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines, published_state

ROOT = bootstrap()

from refractpath import format_table  # noqa: E402
from refractpath.models import (  # noqa: E402
    bennett_1982,
    oriani_breakdown_elevation_deg,
    oriani_series,
    oriani_series_unguarded,
    oriani_truncation_ratio,
    plane_parallel,
    plane_parallel_floor_deg,
    saemundsson_1986,
)
from refractpath.raytrace import raytrace_refraction  # noqa: E402

LOW_ELEVATIONS = (10.0, 7.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.75, 1.5, 1.25, 1.0, 0.5, 0.25, 0.0)


def main() -> int:
    out = Tee("validate_low_elevation_failure_output.txt")
    state = published_state()
    out.rule("refractpath -- the low-elevation regime, exhibited")
    for line in environment_lines():
        out(line)
    out("")
    plane_floor = plane_parallel_floor_deg(state)
    oriani_floor = oriani_breakdown_elevation_deg(state)
    out(
        format_table(
            ["limit", "value (deg)", "how it is obtained"],
            [
                [
                    "equation (13) analytic floor",
                    f"{plane_floor:.6f}",
                    "closed form, 90 - arcsin(1/n); the equation has no solution below it",
                ],
                [
                    "equation (14a) breakdown",
                    f"{oriani_floor:.6f}",
                    "closed form from the series coefficients; the truncation inverts",
                ],
                [
                    "equations (15) and (16)",
                    "none",
                    (
                        "both remain finite and smooth to the horizon; the departure is "
                        "measured below"
                    ),
                ],
                [
                    "equation (19) ray trace",
                    "none numerically",
                    (
                        "finite at the horizon; its validity is limited by the profile, "
                        "not by the integrator"
                    ),
                ],
            ],
        )
    )

    out.rule("1. what each form returns below its own limit, arcsec")
    rows = []
    for elevation in LOW_ELEVATIONS:
        plane = plane_parallel(elevation, state)
        guarded = oriani_series(elevation, state)
        raw = oriani_series_unguarded(elevation, state)
        rows.append(
            [
                f"{elevation:6.2f}",
                "undefined" if math.isnan(plane) else f"{plane:14.3f}",
                "undefined" if math.isnan(guarded) else f"{guarded:14.3f}",
                "undefined" if math.isnan(raw) else f"{raw:16.3f}",
                f"{bennett_1982(elevation, state):12.3f}",
                f"{saemundsson_1986(elevation, state):12.3f}",
                f"{raytrace_refraction(elevation, state):12.3f}",
            ]
        )
    out(
        format_table(
            [
                "elevation",
                "eq (13)",
                "eq (14) guarded",
                "eq (14) unguarded",
                "eq (15)",
                "eq (16)",
                "eq (19)",
            ],
            rows,
        )
    )
    out("")
    out("Read the unguarded column. Equation (14) does not fade out; it reverses. At")
    out("1.5 degrees it returns a smaller refraction than at 45 degrees, and below")
    out("about 1.4 degrees it returns a NEGATIVE refraction, which would move a mount")
    out("the wrong way. Equation (13) diverges upward instead, reaching more than")
    out("three times the correct value before its analytic floor cuts it off.")
    raw_at_1 = oriani_series_unguarded(1.0, state)
    out.check(
        "the unguarded Oriani series returns a negative refraction below 1.4 degrees",
        raw_at_1 < 0.0,
        f"{raw_at_1:.3f} arcsec at 1 degree, against a correct "
        f"{raytrace_refraction(1.0, state):.3f} arcsec",
    )
    out.check(
        "the guarded Oriani series refuses wherever the unguarded one has inverted",
        all(
            math.isnan(oriani_series(elevation, state))
            for elevation in LOW_ELEVATIONS
            if elevation < oriani_floor
        ),
        f"guard elevation {oriani_floor:.6f} deg, from equation (14a)",
    )
    plane_at_2 = plane_parallel(2.0, state)
    out.check(
        "equation (13) exceeds the layered integral by more than 75 per cent at "
        "2 degrees, before its own floor",
        plane_at_2 / raytrace_refraction(2.0, state) > 1.75,
        f"{plane_at_2:.3f} against {raytrace_refraction(2.0, state):.3f} arcsec, a "
        f"factor of {plane_at_2 / raytrace_refraction(2.0, state):.3f}",
    )

    out.rule("2. the truncation ratio of equation (14a) against elevation")
    rows = []
    for elevation in (45.0, 20.0, 15.0, 10.0, 5.0, 3.0, 2.5, 2.0, 1.9507, 1.9, 1.5, 1.0):
        ratio = oriani_truncation_ratio(elevation, state)
        rows.append(
            [
                f"{elevation:8.4f}",
                f"{ratio:12.6f}",
                "correction" if ratio < 0.1 else ("marginal" if ratio < 1.0 else "INVERTED"),
            ]
        )
    out(format_table(["elevation (deg)", "ratio", "regime"], rows))
    out("")
    out("The ratio at the package's declared 15 degree floor for equation (14) is in")
    out("the table above. The declared floor is conservative by this measure, and the")
    out("measured agreement with the ray trace at 15 degrees is in")
    out("validation/validate_model_grid_output.txt.")

    out.rule("3. the dangerous mode: equations (15) and (16) look fine all the way down")
    rows = []
    for elevation in LOW_ELEVATIONS:
        reference = raytrace_refraction(elevation, state)
        bennett = bennett_1982(elevation, state)
        saemundsson = saemundsson_1986(elevation, state)
        rows.append(
            [
                f"{elevation:6.2f}",
                f"{reference:10.3f}",
                f"{bennett - reference:+9.3f}",
                f"{(bennett - reference) / reference:+9.5f}",
                f"{saemundsson - reference:+9.3f}",
                f"{(saemundsson - reference) / reference:+9.5f}",
            ]
        )
    out(
        format_table(
            [
                "elevation",
                "eq (19)",
                "eq (15) error",
                "fraction",
                "eq (16) error",
                "fraction",
            ],
            rows,
        )
    )
    out("")
    out("Both stay within a few per cent of the layered integral all the way to the")
    out("horizon and both remain monotone and smooth. There is nothing in their")
    out("output to tell a caller that the layered integral they are being compared")
    out("against is itself uncertain by five to twenty arcminutes at these elevations,")
    out("which is measured in validation/validate_profile_sensitivity_output.txt. That")
    out("is the reason this package returns a refusal rather than a number.")
    worst_fraction = max(
        abs(bennett_1982(elevation, state) - raytrace_refraction(elevation, state))
        / raytrace_refraction(elevation, state)
        for elevation in LOW_ELEVATIONS
    )
    out.check(
        "equation (15) stays within 4 per cent of the layered integral down to the "
        "horizon, which is why its failure is not visible in its output",
        worst_fraction < 0.04,
        f"worst {worst_fraction:.5f} relative over the elevations tabulated",
    )

    out.rule("4. how many methods remain defined, against elevation")
    rows = []
    for elevation in LOW_ELEVATIONS:
        defined = [
            name
            for name, value in (
                ("eq (13)", plane_parallel(elevation, state)),
                ("eq (14)", oriani_series(elevation, state)),
                ("eq (15)", bennett_1982(elevation, state)),
                ("eq (16)", saemundsson_1986(elevation, state)),
                ("eq (19)", raytrace_refraction(elevation, state)),
            )
            if math.isfinite(value)
        ]
        rows.append([f"{elevation:6.2f}", str(len(defined)), ", ".join(defined)])
    out(format_table(["elevation", "defined", "which"], rows))
    out("")
    out("Below about 1.36 degrees only three of the five methods are defined, and two")
    out("of the three are monochromatic fits with no wavelength and no humidity. The")
    out("ensemble that remains is not an ensemble of independent physics.")
    assert np.isfinite(worst_fraction)

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
