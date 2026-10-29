"""Equation (23): how much of the answer is an input the caller did not supply.

The library is given one pressure, one temperature and one humidity, at one
point. Everything above the observer is invented by equation (2). This script
measures what that invention is worth, by sweeping the surface lapse rate, the
depth of a surface inversion, the observer height and the vapour scale height,
and reporting the result in arcseconds beside the cross-model disagreement of
equation (22) at the same elevation.

The headline comparison is the last table: at every elevation, which is larger,
the disagreement between five independent published methods, or the range the
answer takes under three physically attainable temperature profiles that the
observer's own measurements cannot distinguish.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines, published_state

ROOT = bootstrap()

from refractpath import AtmosphericState, Observer, format_table  # noqa: E402
from refractpath.disagreement import ensemble_at  # noqa: E402
from refractpath.raytrace import raytrace_refraction  # noqa: E402
from refractpath.verdict import (  # noqa: E402
    PROFILE_ENVELOPE_LAPSE_RATES,
    profile_envelope_arcsec,
)

ELEVATIONS = (90.0, 60.0, 45.0, 30.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.0)
LAPSE_RATES_K_PER_KM = (-9.8, -8.0, -6.5, -4.0, -2.0, 0.0, 5.0, 10.0, 20.0, 30.0)
INVERSION_DEPTHS_M = (25.0, 50.0, 100.0, 200.0, 400.0, 800.0)
OBSERVER_HEIGHTS_M = (0.0, 500.0, 1500.0, 3000.0, 5000.0)
VAPOUR_SCALE_HEIGHTS_M = (1000.0, 1500.0, 2000.0, 3000.0, 4000.0)


def main() -> int:
    out = Tee("validate_profile_sensitivity_output.txt")
    base = published_state()
    out.rule("refractpath -- equation (23), the unmeasured profile above the observer")
    for line in environment_lines():
        out(line)
    out("")
    out(
        f"base state: {base.pressure_hpa:g} hPa, {base.temperature_c:g} C, dry, "
        f"{base.wavelength_um:g} um"
    )
    out(
        "declared envelope members: "
        + ", ".join(
            f"{label} {rate * 1000.0:+.1f} K/km" for label, rate in PROFILE_ENVELOPE_LAPSE_RATES
        )
    )

    out.rule("1. surface lapse rate applied to the whole troposphere")
    rows = []
    for rate in LAPSE_RATES_K_PER_KM:
        state = AtmosphericState(
            pressure_hpa=base.pressure_hpa,
            temperature_c=base.temperature_c,
            wavelength_um=base.wavelength_um,
            surface_lapse_rate_k_per_m=rate / 1000.0,
        )
        rows.append(
            [
                f"{rate:+6.1f}",
                *[f"{raytrace_refraction(h, state):10.4f}" for h in (45.0, 10.0, 5.0, 1.0, 0.0)],
            ]
        )
    out(
        format_table(
            ["lapse (K/km)", "45 deg", "10 deg", "5 deg", "1 deg", "0 deg"],
            rows,
        )
    )
    by_elevation: dict[float, list[float]] = {}
    for elevation in (45.0, 10.0, 5.0, 1.0, 0.0):
        by_elevation[elevation] = [
            raytrace_refraction(
                elevation,
                AtmosphericState(
                    pressure_hpa=base.pressure_hpa,
                    temperature_c=base.temperature_c,
                    wavelength_um=base.wavelength_um,
                    surface_lapse_rate_k_per_m=rate / 1000.0,
                ),
            )
            for rate in LAPSE_RATES_K_PER_KM
        ]
    out("")
    out(
        format_table(
            ["elevation (deg)", "range over the swept lapse rates (arcsec)", "as arcmin"],
            [
                [
                    f"{elevation:g}",
                    f"{max(values) - min(values):10.4f}",
                    f"{(max(values) - min(values)) / 60.0:8.4f}",
                ]
                for elevation, values in by_elevation.items()
            ],
        )
    )
    declared_at_45, _ = profile_envelope_arcsec(45.0, base)
    full_at_45 = max(by_elevation[45.0]) - min(by_elevation[45.0])
    out("")
    out(
        f"range over the three DECLARED envelope members at 45 deg : "
        f"{declared_at_45:.5f} arcsec"
    )
    out(
        f"range over the full swept set at 45 deg                  : "
        f"{full_at_45:.5f} arcsec"
    )
    out(
        "The full swept set includes a +30 K/km gradient applied to the WHOLE 11 km"
    )
    out("troposphere, which is not a physical atmosphere; it is swept to show where the")
    out("sensitivity comes from, and the declared envelope of equation (23) uses only")
    out("the three attainable members.")
    out.check(
        "the three declared envelope members are worth less than 0.01 arcsec at 45 degrees",
        declared_at_45 < 0.01,
        f"range {declared_at_45:.5f} arcsec; the full swept set, including the "
        f"unphysical whole-troposphere {LAPSE_RATES_K_PER_KM[-1]:+g} K/km case, spans "
        f"{full_at_45:.5f} arcsec",
    )
    out.check(
        "the lapse rate is worth more than 1000 arcsec at zero elevation",
        max(by_elevation[0.0]) - min(by_elevation[0.0]) > 1000.0,
        f"range {max(by_elevation[0.0]) - min(by_elevation[0.0]):.2f} arcsec = "
        f"{(max(by_elevation[0.0]) - min(by_elevation[0.0])) / 60.0:.3f} arcmin; this is "
        "the single largest term in horizon refraction and the library cannot "
        "determine it from its inputs",
    )

    out.rule("2. a shallow surface inversion of +20 K/km, by depth")
    rows = []
    for depth in INVERSION_DEPTHS_M:
        state = AtmosphericState(
            pressure_hpa=base.pressure_hpa,
            temperature_c=base.temperature_c,
            wavelength_um=base.wavelength_um,
            surface_lapse_rate_k_per_m=0.020,
            surface_layer_top_m=depth,
        )
        rows.append(
            [
                f"{depth:6.0f}",
                *[f"{raytrace_refraction(h, state):10.4f}" for h in (45.0, 5.0, 1.0, 0.0)],
            ]
        )
    baseline = [raytrace_refraction(h, base) for h in (45.0, 5.0, 1.0, 0.0)]
    rows.append(["none", *[f"{value:10.4f}" for value in baseline]])
    out(format_table(["inversion depth (m)", "45 deg", "5 deg", "1 deg", "0 deg"], rows))
    out("")
    out("A 25 m inversion is invisible at 45 degrees and worth tens of arcseconds at")
    out("the horizon. Nothing in the inputs this library accepts constrains either the")
    out("depth or the strength, which is why both appear in the undeterminable list")
    out("returned by every verdict.")
    shallow = raytrace_refraction(
        0.0,
        AtmosphericState(
            pressure_hpa=base.pressure_hpa,
            temperature_c=base.temperature_c,
            wavelength_um=base.wavelength_um,
            surface_lapse_rate_k_per_m=0.020,
            surface_layer_top_m=25.0,
        ),
    )
    out.check(
        "a 25 m surface inversion moves the horizon refraction by more than 10 arcsec",
        abs(shallow - baseline[3]) > 10.0,
        f"{shallow - baseline[3]:+.3f} arcsec against the US Standard Atmosphere 1976 "
        "profile",
    )

    out.rule("3. observer height at fixed station pressure")
    rows = []
    for height in OBSERVER_HEIGHTS_M:
        observer = Observer(height_m=height)
        rows.append(
            [
                f"{height:6.0f}",
                *[
                    f"{raytrace_refraction(h, base, observer):10.4f}"
                    for h in (45.0, 10.0, 5.0, 1.0, 0.0)
                ],
            ]
        )
    out(
        format_table(
            ["observer height (m)", "45 deg", "10 deg", "5 deg", "1 deg", "0 deg"], rows
        )
    )
    out("")
    out("The refraction RISES with observer height at fixed station pressure, because")
    out("a larger radius means a smaller beta and therefore a smaller curvature")
    out("correction. The effect has the opposite sign to the one a reader might")
    out("expect from thinking about air mass, and it is small.")

    out.rule("4. vapour scale height of equation (3), at 30 C and saturation")
    rows = []
    for scale in VAPOUR_SCALE_HEIGHTS_M:
        state = AtmosphericState(
            pressure_hpa=base.pressure_hpa,
            temperature_c=30.0,
            relative_humidity=1.0,
            wavelength_um=base.wavelength_um,
            vapour_scale_height_m=scale,
        )
        rows.append(
            [
                f"{scale:6.0f}",
                *[f"{raytrace_refraction(h, state):10.4f}" for h in (45.0, 5.0, 1.0, 0.0)],
            ]
        )
    out(format_table(["vapour scale height (m)", "45 deg", "5 deg", "1 deg", "0 deg"], rows))
    vapour_range = {
        elevation: [
            raytrace_refraction(
                elevation,
                AtmosphericState(
                    pressure_hpa=base.pressure_hpa,
                    temperature_c=30.0,
                    relative_humidity=1.0,
                    wavelength_um=base.wavelength_um,
                    vapour_scale_height_m=scale,
                ),
            )
            for scale in VAPOUR_SCALE_HEIGHTS_M
        ]
        for elevation in (45.0, 5.0, 0.0)
    }
    out("")
    for elevation, values in vapour_range.items():
        out(
            f"  {elevation:5.1f} deg: range {max(values) - min(values):.4f} arcsec over a "
            f"1 km to 4 km vapour scale height"
        )
    out("")
    out("Equation (3) is a declared default, not a published profile. Its crudeness is")
    out("tolerable in the optical band because the numbers above are small; it would")
    out("not be tolerable in the radio band, which this product does not cover.")

    out.rule("5. the headline comparison: equation (22) against equation (23)")
    rows = []
    envelope_wins = 0
    for elevation in ELEVATIONS:
        point = ensemble_at(elevation, base)
        envelope, members = profile_envelope_arcsec(elevation, base)
        larger = "profile" if envelope > point.spread_arcsec else "cross-model"
        envelope_wins += int(envelope > point.spread_arcsec)
        rows.append(
            [
                f"{elevation:6.1f}",
                f"{point.spread_arcsec:12.4f}",
                f"{envelope:12.4f}",
                f"{envelope / point.spread_arcsec:8.3f}"
                if point.spread_arcsec > 0.0
                else "-",
                larger,
                f"{len(point.undefined_members)}",
            ]
        )
    out(
        format_table(
            [
                "elevation",
                "eq (22) spread",
                "eq (23) envelope",
                "ratio",
                "larger term",
                "undefined",
            ],
            rows,
        )
    )
    out("")
    out(
        f"the profile envelope is the larger term at {envelope_wins} of "
        f"{len(ELEVATIONS)} elevations tested"
    )
    out("")
    out("Read this table before the disagreement curve. Above about 10 degrees the")
    out("cross-model disagreement dominates and is the thing to budget for. Below it")
    out("the unmeasured profile dominates, and no choice among the published methods")
    out("will help, because they are all being asked about an atmosphere nobody")
    out("measured. That is why the verdict refuses on the profile envelope alone and")
    out("not only on the cross-model spread.")
    out.check(
        "the profile envelope exceeds the cross-model spread at or below 1 degree",
        all(
            profile_envelope_arcsec(elevation, base)[0]
            > ensemble_at(elevation, base).spread_arcsec
            for elevation in (1.0, 0.5, 0.0)
        ),
        "measured at 1, 0.5 and 0 degrees, which is the regime in which both the "
        "plane-parallel model and the Oriani series have left the ensemble. Between "
        "about 1 and 5 degrees the cross-model spread is the larger term only because "
        "those two forms are diverging there; among the three methods that remain "
        "defined the profile term already dominates",
    )
    out.check(
        "the cross-model spread exceeds the profile envelope at or above 20 degrees",
        all(
            ensemble_at(elevation, base).spread_arcsec
            > profile_envelope_arcsec(elevation, base)[0]
            for elevation in (90.0, 60.0, 45.0, 30.0, 20.0)
        ),
        "measured at 90, 60, 45, 30 and 20 degrees",
    )

    out.rule("6. the envelope members at a few elevations, in arcmin")
    rows = []
    for elevation in (5.0, 2.0, 1.0, 0.5, 0.0):
        _, members = profile_envelope_arcsec(elevation, base)
        rows.append(
            [
                f"{elevation:6.1f}",
                *[f"{members[label] / 60.0:9.4f}" for label, _ in PROFILE_ENVELOPE_LAPSE_RATES],
            ]
        )
    out(
        format_table(
            ["elevation", *[label for label, _ in PROFILE_ENVELOPE_LAPSE_RATES]],
            rows,
        )
    )
    out("")
    out("Equation (23) is a LOWER bound on the profile uncertainty, not an upper one:")
    out("a real surface inversion is steeper than isothermal, as section 2 shows, and")
    out("the three members are a declared bracket rather than a distribution.")
    assert np.isfinite(baseline).all()

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
