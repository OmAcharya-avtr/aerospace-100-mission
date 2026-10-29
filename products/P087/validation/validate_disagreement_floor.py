"""The disagreement curve and the measured tolerance-crossing elevation.

This is the product's headline measurement. Equation (22) is evaluated over a
dense elevation grid for a swept set of atmospheric states and declared
pointing tolerances, and the elevation at which the disagreement crosses each
tolerance is closed on by bisection and reported with its bracket.

Three things are measured here rather than asserted:

1. **The crossing elevation**, by bisection, with the bracket width and the
   spread at both bracket edges printed, so a reader can see it is a
   measurement and not a declaration.
2. **The number of crossings.** Equation (22) is not monotone in elevation,
   because members leave the ensemble at their own validity limits and the
   spread drops when the most discrepant one goes. The count is reported and
   the gap between the lowest and the highest crossing is quantified, because
   a search that assumed one crossing would return the wrong answer by that
   factor.
3. **Which pair of methods holds the extremes** at each elevation, so the
   disagreement has an attribution and not just a magnitude.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from _bootstrap import (
    Tee,
    Timer,
    bootstrap,
    default_tolerances_arcsec,
    environment_lines,
    published_state,
)

ROOT = bootstrap()

from refractpath import AtmosphericState, format_table  # noqa: E402
from refractpath.disagreement import (  # noqa: E402
    default_ensemble,
    default_floor_grid,
    ensemble_at,
    pairwise_disagreement,
    validity_floor,
)

CURVE_ELEVATIONS = (90.0, 60.0, 45.0, 30.0, 20.0, 15.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.0)
SWEPT_STATES = (
    ("published 1010 hPa 10 C dry 550 nm", published_state()),
    (
        "cold dense 1035 hPa -25 C dry 550 nm",
        AtmosphericState(pressure_hpa=1035.0, temperature_c=-25.0, wavelength_um=0.55),
    ),
    (
        "hot humid 1005 hPa 35 C 90 pct 550 nm",
        AtmosphericState(
            pressure_hpa=1005.0,
            temperature_c=35.0,
            relative_humidity=0.9,
            wavelength_um=0.55,
        ),
    ),
    (
        "high site 700 hPa 0 C dry 550 nm",
        AtmosphericState(pressure_hpa=700.0, temperature_c=0.0, wavelength_um=0.55),
    ),
    (
        "blue 1010 hPa 10 C dry 400 nm",
        AtmosphericState(pressure_hpa=1010.0, temperature_c=10.0, wavelength_um=0.40),
    ),
    (
        "near infrared 1010 hPa 10 C dry 1000 nm",
        AtmosphericState(pressure_hpa=1010.0, temperature_c=10.0, wavelength_um=1.00),
    ),
)


def main() -> int:
    out = Tee("validate_disagreement_floor_output.txt")
    state = published_state()
    names = [member.name for member in default_ensemble()]
    out.rule("refractpath -- the cross-model disagreement curve and the measured floor")
    for line in environment_lines():
        out(line)
    out("")
    out(f"ensemble  : {', '.join(names)}")
    out(f"tolerances: {default_tolerances_arcsec()} arcsec")
    out(f"grid      : {default_floor_grid().size} points, h = 90 (i/180)^2 degrees")

    out.rule("1. the ensemble and its spread against elevation, published conditions")
    rows = []
    for elevation in CURVE_ELEVATIONS:
        point = ensemble_at(elevation, state)
        row = [f"{elevation:6.1f}"]
        for name in names:
            value = point.values_arcsec.get(name)
            row.append("undefined" if value is None else f"{value:11.4f}")
        row.append(f"{point.spread_arcsec:11.4f}")
        row.append(f"{point.minimum_name} / {point.maximum_name}")
        rows.append(row)
    out(format_table(["elevation", *names, "spread", "widest pair"], rows))
    out("")
    out("The spread column is equation (22). Read the widest-pair column with it: at")
    out("moderate elevation the extremes are the Oriani series and Saemundsson's fit,")
    out("and near the horizon they are whichever members are still defined.")

    out.rule("2. the spread is NOT monotone, and this is structural")
    floor = validity_floor(state, 10.0)
    exceeded = floor.spread_arcsec >= floor.tolerance_arcsec
    transitions = np.flatnonzero(exceeded[:-1] != exceeded[1:])
    out(f"tolerance                  : {floor.tolerance_arcsec:g} arcsec")
    out(f"crossings found on the grid : {floor.crossings}")
    out(f"spread monotone on the grid  : {floor.monotone_on_grid}")
    out("")
    rows = []
    for index in transitions:
        rows.append(
            [
                f"{floor.grid_deg[index]:9.5f}",
                f"{floor.spread_arcsec[index]:12.4f}",
                f"{floor.grid_deg[index + 1]:9.5f}",
                f"{floor.spread_arcsec[index + 1]:12.4f}",
                (
                    f"{int(floor.undefined_counts[index])} -> "
                    f"{int(floor.undefined_counts[index + 1])}"
                ),
            ]
        )
    out(
        format_table(
            ["below (deg)", "spread below", "above (deg)", "spread above", "undefined count"],
            rows,
        )
    )
    first = float(floor.grid_deg[int(transitions[0])])
    out("")
    out(f"lowest crossing            : {first:.5f} deg")
    out(f"highest crossing, the floor : {floor.floor_deg:.5f} deg")
    out(f"ratio                      : {floor.floor_deg / first:.3f}")
    out("")
    out("A bisection that assumed a single crossing and walked up from the horizon")
    out("would stop at the lowest one and understate the floor by that ratio. The")
    out("non-monotonicity is caused by members LEAVING the ensemble: the undefined")
    out("count column shows plane_parallel and oriani_series dropping out at their")
    out("own validity limits, and the spread falls when the most discrepant member")
    out("goes. The floor reported by this package is always the HIGHEST crossing.")
    out.check(
        "the disagreement curve is measured to be non-monotone and the floor is the "
        "highest crossing",
        floor.crossings >= 2 and not floor.monotone_on_grid and floor.floor_deg > first,
        f"{floor.crossings} crossings, floor {floor.floor_deg:.5f} deg against lowest "
        f"crossing {first:.5f} deg",
    )

    out.rule("3. the measured floor, by tolerance and by atmospheric state")
    timer = Timer()
    rows = []
    for label, swept in SWEPT_STATES:
        row = [label]
        for tolerance in default_tolerances_arcsec():
            with timer:
                measured = validity_floor(swept, tolerance)
            row.append(
                "never"
                if measured.never_exceeded
                else (
                    ">=90"
                    if measured.exceeded_at_highest_grid_point
                    else f"{measured.floor_deg:7.3f}"
                )
            )
        rows.append(row)
    out(
        format_table(
            ["state", *[f"{t:g} arcsec" for t in default_tolerances_arcsec()]],
            rows,
        )
    )
    out("")
    out("Every cell is the apparent elevation, in degrees, below which the five")
    out("independent standard methods disagree by at least the column's tolerance.")
    out("No correction from this package is suitable for operational pointing below")
    out("the figure in the cell that matches your tolerance and your conditions.")

    out.rule("4. bracket evidence for one cell, so the bisection can be audited")
    audited = validity_floor(state, 10.0)
    out(
        format_table(
            ["quantity", "value"],
            [
                ["state", "published 1010 hPa 10 C dry 550 nm"],
                ["tolerance (arcsec)", f"{audited.tolerance_arcsec:.4f}"],
                ["floor (deg)", f"{audited.floor_deg:.6f}"],
                ["bracket low (deg)", f"{audited.bracket_deg[0]:.6f}"],
                ["bracket high (deg)", f"{audited.bracket_deg[1]:.6f}"],
                ["bracket width (deg)", f"{audited.bracket_width_deg:.3e}"],
                ["spread at bracket low (arcsec)", f"{audited.spread_below_arcsec:.6f}"],
                ["spread at bracket high (arcsec)", f"{audited.spread_above_arcsec:.6f}"],
                ["bisection iterations", str(audited.iterations)],
            ],
        )
    )
    out.check(
        "the bracket straddles the tolerance and is closed to 1e-4 degrees",
        audited.spread_below_arcsec >= audited.tolerance_arcsec
        > audited.spread_above_arcsec
        and audited.bracket_width_deg <= 1e-4,
        f"width {audited.bracket_width_deg:.3e} deg",
    )

    out.rule("5. pairwise attribution at three elevations")
    for elevation in (45.0, 10.0, 0.0):
        point = ensemble_at(elevation, state)
        pairs = pairwise_disagreement(point)
        out("")
        out(f"apparent elevation {elevation:g} deg, spread {point.spread_arcsec:.4f} arcsec")
        out(
            format_table(
                ["pair", "difference (arcsec)"],
                [
                    [f"{a} / {b}", f"{value:10.4f}"]
                    for (a, b), value in sorted(
                        pairs.items(), key=lambda item: -item[1]
                    )
                ],
            )
        )

    out.rule("6. the floor rises with a colder, denser atmosphere")
    rows = []
    for temperature in (-30.0, -10.0, 10.0, 30.0):
        swept = AtmosphericState(
            pressure_hpa=1010.0, temperature_c=temperature, wavelength_um=0.55
        )
        measured = validity_floor(swept, 10.0)
        rows.append([f"{temperature:+6.1f}", f"{measured.floor_deg:8.4f}"])
    out(format_table(["temperature (C)", "floor at 10 arcsec (deg)"], rows))
    floors = [
        validity_floor(
            AtmosphericState(pressure_hpa=1010.0, temperature_c=t, wavelength_um=0.55), 10.0
        ).floor_deg
        for t in (-30.0, -10.0, 10.0, 30.0)
    ]
    out.check(
        "the measured floor falls monotonically as the atmosphere warms and thins",
        all(later < earlier for earlier, later in pairwise(floors)),
        f"from {floors[0]:.4f} deg at -30 C to {floors[-1]:.4f} deg at +30 C",
    )

    out.rule("elapsed time -- UNSTABLE, two cores, reported for information only")
    out(
        format_table(
            ["quantity", "value"],
            [
                ["floor measurements run", str(timer.calls)],
                ["mean wall-clock per measurement (ms)", f"{timer.mean_ms:.1f}"],
                ["total wall-clock (s)", f"{timer.elapsed_s:.2f}"],
            ],
        )
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
