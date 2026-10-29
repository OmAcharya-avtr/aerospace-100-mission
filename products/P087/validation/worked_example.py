"""The worked example quoted in README.md, run so that the README cannot drift.

Every line of output this script prints appears verbatim in README.md under
"A worked example". If the API changes, this script changes, and the README
text is replaced with what this script actually printed.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines

ROOT = bootstrap()

from refractpath import (  # noqa: E402
    AtmosphericState,
    Observer,
    format_table,
    format_verdict,
    refraction,
)


def main() -> int:
    out = Tee("worked_example_output.txt")
    out.rule("refractpath -- the worked example from README.md")
    for line in environment_lines():
        out(line)

    out.rule("the example, run")

    # ---------------------------------------------------------------- example
    state = AtmosphericState(
        pressure_hpa=1004.0,
        temperature_c=3.5,
        relative_humidity=0.62,
        wavelength_um=0.55,
    )
    observer = Observer(height_m=1180.0, latitude_deg=34.2)

    for elevation in (60.0, 25.0, 12.0, 6.0):
        result = refraction(elevation, state, observer, tolerance_arcsec=10.0)
        out("")
        out(format_verdict(result.verdict))
        out(
            f"  reference (ray trace)     : "
            f"{result.reference_refraction_arcsec:.3f} arcsec"
        )
        out(f"  true elevation            : {result.true_elevation_deg:.5f} deg")
        out(
            format_table(
                ["method", "refraction (arcsec)", "minus reference"],
                [
                    [
                        name,
                        f"{value:10.3f}",
                        f"{value - result.reference_refraction_arcsec:+9.3f}",
                    ]
                    for name, value in sorted(
                        result.methods_arcsec.items(), key=lambda item: item[1]
                    )
                ]
                + [[name, "undefined", "-"] for name in result.undefined_methods],
            )
        )
    # ------------------------------------------------------------ end example

    out.rule("what the four calls say, in one table")
    rows = []
    for elevation in (60.0, 25.0, 12.0, 6.0):
        result = refraction(elevation, state, observer, tolerance_arcsec=10.0)
        rows.append(
            [
                f"{elevation:6.1f}",
                result.verdict.verdict,
                f"{result.reference_refraction_arcsec:9.3f}",
                f"{result.disagreement_arcsec:9.3f}",
                f"{result.profile_envelope_arcsec:9.3f}",
                f"{result.verdict.validity_floor_deg:7.3f}",
            ]
        )
    out(
        format_table(
            [
                "elevation",
                "verdict",
                "reference (arcsec)",
                "eq (22)",
                "eq (23)",
                "floor (deg)",
            ],
            rows,
        )
    )
    out("")
    out("The declared tolerance is 10 arcsec throughout. The floor column is the same")
    out("number in every row because it is a property of the atmosphere and the")
    out("tolerance, not of the elevation asked for; the verdict column is what")
    out("changes.")

    out.check(
        "the four calls return the verdicts quoted in README.md",
        [
            refraction(h, state, observer, tolerance_arcsec=10.0).verdict.verdict
            for h in (60.0, 25.0, 12.0, 6.0)
        ]
        == ["usable", "degraded", "degraded", "refuse"],
        "usable, degraded, degraded, refuse",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
