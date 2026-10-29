"""Controls, each anchored on a truth and each asserted to fire or not to fire.

A control that does not fire is not evidence. Every control below is anchored
on a quantity that is known independently of the output being tested -- an
analytic floor, a closed-form breakdown elevation, a bracket derived from
different mathematics, or an injected error whose size is set by the
conditioning of the thing it is injected into -- and every one is checked in
the direction it is supposed to go. Controls that must NOT fire are asserted
not to fire, which is the half that is usually missing.

Nine controls must fire and five must not.
"""

from __future__ import annotations

import math

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines, published_state

ROOT = bootstrap()

from refractpath import AtmosphericState, Observer, format_table  # noqa: E402
from refractpath.atmosphere import build_profile  # noqa: E402
from refractpath.disagreement import default_ensemble  # noqa: E402
from refractpath.models import (  # noqa: E402
    oriani_breakdown_elevation_deg,
    oriani_series,
    plane_parallel,
    plane_parallel_floor_deg,
)
from refractpath.raytrace import raytrace_refraction  # noqa: E402
from refractpath.refractivity import birch_downs_1994  # noqa: E402
from refractpath.verdict import compute_verdict, profile_envelope_arcsec  # noqa: E402

ARCSEC_PER_RADIAN = 180.0 * 3600.0 / math.pi


def independent_bracket_arcsec(
    elevation_deg: float, state: object, top_m: float = 90_000.0
) -> tuple[float, float]:
    """Bracket on equation (19) from monotonicity and the mean value theorem."""
    observer = Observer()
    radius_0 = observer.radius_m
    radius_top = radius_0 + top_m
    profile = build_profile(state, np.array([0.0, top_m]), top_m=top_m)
    refractivity = birch_downs_1994(
        state.wavelength_um,
        profile.pressure_pa,
        profile.temperature_k - 273.15,
        profile.vapour_pressure_pa,
    )
    n_0 = 1.0 + float(refractivity[0])
    n_top = 1.0 + float(refractivity[1])
    measure = math.log(n_0) - math.log(n_top)
    sin_z = math.sin(math.radians(90.0 - elevation_deg))
    tan_0 = sin_z / math.sqrt(1.0 - sin_z**2)
    sin_top = n_0 * radius_0 * sin_z / (n_top * radius_top)
    tan_top = sin_top / math.sqrt(1.0 - sin_top**2)
    return tan_top * measure * ARCSEC_PER_RADIAN, tan_0 * measure * ARCSEC_PER_RADIAN


def main() -> int:
    out = Tee("validate_controls_output.txt")
    state = published_state()
    out.rule("refractpath -- controls, anchored on truths and asserted in both directions")
    for line in environment_lines():
        out(line)

    fired: list[tuple[str, bool, str]] = []
    quiet: list[tuple[str, bool, str]] = []

    out.rule("controls that MUST fire")

    # C1. The analytic floor of equation (13) is anchored on the refractivity,
    # which is computed independently of the model being tested.
    floor = plane_parallel_floor_deg(state)
    c1 = math.isnan(plane_parallel(floor - 1e-6, state)) and math.isfinite(
        plane_parallel(floor + 1e-6, state)
    )
    fired.append(
        (
            "C1 equation (13) refuses below its analytic floor and answers above it",
            c1,
            f"floor {floor:.6f} deg from 90 - arcsin(1/n); probed at +/- 1e-6 deg",
        )
    )

    # C2. The breakdown elevation of equation (14a) is a closed form in the
    # series coefficients, not a property of the series output.
    breakdown = oriani_breakdown_elevation_deg(state)
    c2 = math.isnan(oriani_series(breakdown - 1e-4, state)) and math.isfinite(
        oriani_series(breakdown + 1e-4, state)
    )
    fired.append(
        (
            "C2 equation (14) refuses below its computed breakdown and answers above it",
            c2,
            f"breakdown {breakdown:.6f} deg from equation (14a); probed at +/- 1e-4 deg",
        )
    )

    # C3 to C5. Injected errors anchored on the measured value and scaled by
    # the CONDITIONING of the independent bracket, not by an absolute constant.
    for elevation in (45.0, 20.0, 10.0):
        lower, upper = independent_bracket_arcsec(elevation, state)
        measured = raytrace_refraction(elevation, state, None, "birch_downs_1994")
        width = (upper - lower) / measured
        high = measured * (1.0 + 2.5 * width)
        low = measured * (1.0 - 2.5 * width)
        caught = not (lower <= high <= upper) and not (lower <= low <= upper)
        fired.append(
            (
                (
                    f"C{3 + (45.0, 20.0, 10.0).index(elevation)} the independent bracket "
                    f"rejects a {2.5 * width:.1%} error at {elevation:g} degrees"
                ),
                caught,
                (
                    f"bracket [{lower:.4f}, {upper:.4f}], measured {measured:.4f}, "
                    f"injected {low:.4f} and {high:.4f}; the injection is 2.5 times the "
                    f"bracket's own relative width of {width:.4%}"
                ),
            )
        )

    # C6. The bracket rejects a model the package ships, which proves the
    # bracket discriminates and is not merely wide.
    _, upper45 = independent_bracket_arcsec(45.0, state)
    plane45 = plane_parallel(45.0, state, None, "birch_downs_1994")
    fired.append(
        (
            "C6 the independent bracket rejects the exact plane-parallel answer",
            plane45 > upper45,
            (
                f"plane-parallel {plane45:.5f} arcsec exceeds the upper bound "
                f"{upper45:.5f} arcsec by {plane45 - upper45:+.5f}"
            ),
        )
    )

    # C7. The verdict refuses at an elevation below the measured floor.
    verdict_low = compute_verdict(2.0, state, 10.0)
    fired.append(
        (
            "C7 the verdict refuses at 2 degrees against a 10 arcsec tolerance",
            verdict_low.verdict == "refuse",
            (
                f"verdict {verdict_low.verdict}, disagreement "
                f"{verdict_low.disagreement_arcsec:.3f} arcsec, floor "
                f"{verdict_low.validity_floor_deg:.4f} deg"
            ),
        )
    )

    # C8. The envelope gate alone refuses, with the cross-model spread removed
    # by running a single-member ensemble. The anchor is the measured envelope.
    single = tuple(
        member for member in default_ensemble() if member.name == "raytrace"
    )
    envelope_3deg, _ = profile_envelope_arcsec(3.0, state)
    verdict_envelope = compute_verdict(3.0, state, envelope_3deg * 0.5, None, single)
    fired.append(
        (
            "C8 the profile envelope alone refuses when the cross-model spread is zero",
            verdict_envelope.verdict == "refuse"
            and verdict_envelope.disagreement_arcsec == 0.0,
            (
                f"single-member ensemble, spread {verdict_envelope.disagreement_arcsec:.3f} "
                f"arcsec, envelope {envelope_3deg:.3f} arcsec against a tolerance of "
                f"{envelope_3deg * 0.5:.3f} arcsec"
            ),
        )
    )

    # C9. A refractivity set to zero must give exactly zero refraction. The
    # anchor is the vacuum, which is a truth and not an output.
    vacuum_like = AtmosphericState(
        pressure_hpa=1e-6, temperature_c=10.0, wavelength_um=0.55
    )
    vacuum_refraction = raytrace_refraction(5.0, vacuum_like)
    fired.append(
        (
            "C9 a near-vacuum atmosphere gives a refraction below 1e-3 arcsec",
            abs(vacuum_refraction) < 1e-3,
            (
                f"{vacuum_refraction:.3e} arcsec at 1e-6 hPa, against "
                f"{raytrace_refraction(5.0, state):.3f} arcsec at "
                f"{state.pressure_hpa:g} hPa"
            ),
        )
    )

    for name, passed, detail in fired:
        out.check(name, passed, detail)

    out.rule("controls that MUST NOT fire")

    # N1. A correct ray trace must stay inside the independent bracket.
    inside = all(
        independent_bracket_arcsec(elevation, state)[0]
        <= raytrace_refraction(elevation, state, None, "birch_downs_1994")
        <= independent_bracket_arcsec(elevation, state)[1]
        for elevation in (60.0, 45.0, 30.0, 20.0, 10.0, 5.0, 2.0)
    )
    quiet.append(
        (
            "N1 the bracket does not reject the ray trace at any of 7 elevations",
            inside,
            "60, 45, 30, 20, 10, 5 and 2 degrees",
        )
    )

    # N2. The verdict must not refuse where the methods agree.
    verdict_high = compute_verdict(80.0, state, 10.0)
    quiet.append(
        (
            "N2 the verdict does not refuse at 80 degrees against a 10 arcsec tolerance",
            verdict_high.verdict == "usable",
            f"verdict {verdict_high.verdict}, disagreement "
            f"{verdict_high.disagreement_arcsec:.4f} arcsec, no reasons recorded"
            if not verdict_high.reasons
            else f"verdict {verdict_high.verdict}",
        )
    )

    # N3. A physically attainable surface inversion must not refuse at high
    # elevation, where the profile cannot reach.
    inverted = AtmosphericState(
        pressure_hpa=1010.0,
        temperature_c=10.0,
        wavelength_um=0.55,
        surface_lapse_rate_k_per_m=0.020,
        surface_layer_top_m=200.0,
    )
    verdict_inverted = compute_verdict(80.0, inverted, 10.0)
    quiet.append(
        (
            "N3 a 200 m +20 K/km surface inversion does not refuse at 80 degrees",
            verdict_inverted.verdict == "usable",
            (
                f"verdict {verdict_inverted.verdict}, envelope "
                f"{verdict_inverted.profile_envelope_arcsec:.5f} arcsec"
            ),
        )
    )

    # N4. The guarded Oriani series must not refuse above its breakdown.
    above_breakdown = all(
        math.isfinite(oriani_series(elevation, state))
        for elevation in (2.0, 3.0, 5.0, 15.0, 45.0, 89.0)
    )
    quiet.append(
        (
            "N4 the equation (14a) guard does not refuse above the breakdown elevation",
            above_breakdown,
            f"breakdown {breakdown:.6f} deg; probed at 2, 3, 5, 15, 45 and 89 degrees",
        )
    )

    # N5. A wavelength change inside every published range must not change the
    # verdict label. Anchored on the published ranges, not on the output.
    labels = {
        compute_verdict(
            80.0,
            AtmosphericState(pressure_hpa=1010.0, temperature_c=10.0, wavelength_um=w),
            10.0,
        ).verdict
        for w in (0.40, 0.50, 0.55, 0.633)
    }
    quiet.append(
        (
            "N5 the verdict label is unchanged across 400 to 633 nm at 80 degrees",
            labels == {"usable"},
            (
                f"labels {sorted(labels)}; all four wavelengths are inside every "
                "formulation's published range"
            ),
        )
    )

    for name, passed, detail in quiet:
        out.check(name, passed, detail)

    out.rule("summary")
    out(
        format_table(
            ["group", "count", "fired as required"],
            [
                [
                    "must fire",
                    str(len(fired)),
                    f"{sum(1 for _, passed, _ in fired if passed)} of {len(fired)}",
                ],
                [
                    "must not fire",
                    str(len(quiet)),
                    f"{sum(1 for _, passed, _ in quiet if passed)} of {len(quiet)}",
                ],
            ],
        )
    )
    out("")
    out("Every control above is anchored on a quantity computed independently of the")
    out("output it tests: a closed-form floor, a bracket from different mathematics,")
    out("the vacuum, or an injected error scaled by the conditioning of the bound it")
    out("is injected into. No control's criterion is a seed.")

    out.rule("the one external check that does not pass")
    out("validation/validate_published_references.py reports the Bennett (1982)")
    out("accuracy claim as FAILED against this package's ray trace below about 19")
    out("degrees of apparent elevation. That is not a control; it is a cross-check")
    out("against published material, and it is listed here so that a reader of this")
    out("file does not conclude that everything in this product passes.")

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
