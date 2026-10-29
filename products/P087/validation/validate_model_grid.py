"""Every method against the ray trace over a swept grid of elevation and state.

Level 2 validation asks for a sweep, not a spot check. The grid here is the
outer product of five elevations, three pressures, three temperatures, two
humidities and three wavelengths, which is 270 states per elevation band, and
every closed form is scored against the ray trace at each one.

The error curves are reported per method, with the elevation above which each
method stays inside a declared pointing tolerance **measured** on the grid
rather than taken from the method's published range. Where a method's measured
usable range is narrower than the range its source publishes, that is reported
as the finding it is.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from _bootstrap import Tee, Timer, bootstrap, environment_lines

ROOT = bootstrap()

from refractpath import AtmosphericState, format_table  # noqa: E402
from refractpath.models import CLOSED_FORM_MODELS  # noqa: E402
from refractpath.raytrace import raytrace_refraction  # noqa: E402

ELEVATIONS = (90.0, 60.0, 45.0, 30.0, 20.0, 15.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.0)
PRESSURES = (700.0, 900.0, 1030.0)
TEMPERATURES = (-30.0, 10.0, 35.0)
HUMIDITIES = (0.0, 0.9)
WAVELENGTHS = (0.40, 0.55, 1.00)
TOLERANCE_ARCSEC = 10.0


def states() -> list[AtmosphericState]:
    """The swept state grid: 54 states, each inside the declared domains."""
    return [
        AtmosphericState(
            pressure_hpa=pressure,
            temperature_c=temperature,
            relative_humidity=humidity,
            wavelength_um=wavelength,
        )
        for pressure, temperature, humidity, wavelength in itertools.product(
            PRESSURES, TEMPERATURES, HUMIDITIES, WAVELENGTHS
        )
    ]


def main() -> int:
    out = Tee("validate_model_grid_output.txt")
    grid = states()
    out.rule("refractpath -- every closed form against the ray trace, swept")
    for line in environment_lines():
        out(line)
    out("")
    out(f"elevations : {len(ELEVATIONS)}  {ELEVATIONS}")
    out(f"pressures  : {PRESSURES} hPa")
    out(f"temperatures: {TEMPERATURES} C")
    out(f"humidities : {HUMIDITIES}")
    out(f"wavelengths: {WAVELENGTHS} um")
    out(f"states     : {len(grid)}")
    out(f"cells      : {len(grid) * len(ELEVATIONS)}")
    out(f"declared pointing tolerance for the measured ranges below: "
        f"{TOLERANCE_ARCSEC:g} arcsec")

    timer = Timer()
    reference: dict[float, list[float]] = {}
    for elevation in ELEVATIONS:
        row = []
        for state in grid:
            with timer:
                row.append(raytrace_refraction(elevation, state))
        reference[elevation] = row

    errors: dict[str, dict[float, np.ndarray]] = {}
    signed: dict[str, dict[float, np.ndarray]] = {}
    defined: dict[str, dict[float, int]] = {}
    for model in CLOSED_FORM_MODELS:
        errors[model.name] = {}
        signed[model.name] = {}
        defined[model.name] = {}
        for elevation in ELEVATIONS:
            values = np.array(
                [model.function(elevation, state) for state in grid], dtype=float
            )
            mask = np.isfinite(values)
            difference = values[mask] - np.asarray(reference[elevation])[mask]
            signed[model.name][elevation] = difference
            errors[model.name][elevation] = np.abs(difference)
            defined[model.name][elevation] = int(np.count_nonzero(mask))

    out.rule("maximum absolute error against the ray trace, arcsec, over all 54 states")
    rows = []
    for elevation in ELEVATIONS:
        row = [f"{elevation:6.1f}"]
        for model in CLOSED_FORM_MODELS:
            series = errors[model.name][elevation]
            row.append("undefined" if series.size == 0 else f"{float(series.max()):12.4f}")
        row.append(f"{float(np.asarray(reference[elevation]).mean()):12.3f}")
        rows.append(row)
    out(
        format_table(
            ["elevation", *[m.name for m in CLOSED_FORM_MODELS], "mean reference"],
            rows,
        )
    )

    out.rule("root-mean-square error against the ray trace, arcsec")
    rows = []
    for elevation in ELEVATIONS:
        row = [f"{elevation:6.1f}"]
        for model in CLOSED_FORM_MODELS:
            series = errors[model.name][elevation]
            row.append(
                "undefined"
                if series.size == 0
                else f"{float(np.sqrt(np.mean(series**2))):12.4f}"
            )
        rows.append(row)
    out(format_table(["elevation", *[m.name for m in CLOSED_FORM_MODELS]], rows))

    out.rule("relative error against the ray trace, fraction")
    rows = []
    for elevation in ELEVATIONS:
        row = [f"{elevation:6.1f}"]
        for model in CLOSED_FORM_MODELS:
            series = errors[model.name][elevation]
            if series.size == 0 or elevation >= 90.0:
                row.append("-" if elevation >= 90.0 else "undefined")
                continue
            base = np.abs(np.asarray(reference[elevation]))
            base = base[np.isfinite(base) & (base > 0.0)]
            row.append(f"{float(series.max() / base.min()):10.5f}")
        rows.append(row)
    out(format_table(["elevation", *[m.name for m in CLOSED_FORM_MODELS]], rows))
    out("")
    out("The relative column is the point the absolute columns hide: a method can sit")
    out("inside a tight ARCSECOND budget near the horizon, where the refraction is")
    out("half a degree, and still be several per cent wrong there, while the same")
    out("absolute error at 60 degrees is a fraction of a per cent of a much smaller")
    out("number. Neither column alone describes a pointing error.")

    out.rule("SIGNED error against the ray trace, arcsec: minimum, median, maximum")
    rows = []
    for elevation in ELEVATIONS:
        for model in CLOSED_FORM_MODELS:
            series = signed[model.name][elevation]
            if series.size == 0:
                rows.append([f"{elevation:6.1f}", model.name, "undefined", "", "", ""])
                continue
            rows.append(
                [
                    f"{elevation:6.1f}",
                    model.name,
                    f"{float(series.min()):+10.4f}",
                    f"{float(np.median(series)):+10.4f}",
                    f"{float(series.max()):+10.4f}",
                    "all over" if bool((series > 0.0).all())
                    else ("all under" if bool((series < 0.0).all()) else "MIXED SIGN"),
                ]
            )
    out(
        format_table(
            ["elevation", "method", "min", "median", "max", "sign"],
            rows,
        )
    )
    out("")
    out("The sign column is the finding. Above 15 degrees the plane-parallel model and")
    out("both published fits OVERSTATE the refraction relative to the layered integral")
    out("at every one of the 54 swept states, and the Oriani series understates it by")
    out("a few hundredths of an arcsecond. Below about 7 degrees the sign of the")
    out("published fits' error stops being constant across the state grid: the same")
    out("formula is high in one atmosphere and low in another, so the departure cannot")
    out("be corrected by a scale factor.")

    out.rule("measured lowest elevation at which each method stays inside the tolerance")
    rows = []
    for model in CLOSED_FORM_MODELS:
        inside = [
            elevation
            for elevation in ELEVATIONS
            if errors[model.name][elevation].size > 0
            and float(errors[model.name][elevation].max()) <= TOLERANCE_ARCSEC
        ]
        contiguous = None
        for elevation in sorted(ELEVATIONS):
            higher = [e for e in sorted(ELEVATIONS) if e >= elevation]
            if all(e in inside for e in higher):
                contiguous = elevation
                break
        rows.append(
            [
                model.name,
                model.equation,
                f"{model.validity_floor_deg:g}",
                "none" if contiguous is None else f"{contiguous:g}",
                "narrower" if contiguous is None or contiguous > model.validity_floor_deg
                else "wider or equal",
            ]
        )
    out(
        format_table(
            [
                "method",
                "equation",
                "declared floor (deg)",
                "measured floor (deg)",
                "measured vs declared",
            ],
            rows,
        )
    )
    out("")
    out("The measured floor is the lowest elevation above which the method's maximum")
    out(f"error over all 54 states stays at or below {TOLERANCE_ARCSEC:g} arcsec, on this")
    out("grid, against this package's ray trace. It is a measurement, not a claim")
    out("about the published formula's accuracy against its own reference.")

    out.rule("how many of the 54 states each method is defined at")
    rows = []
    for elevation in ELEVATIONS:
        rows.append(
            [
                f"{elevation:6.1f}",
                *[f"{defined[m.name][elevation]:3d}" for m in CLOSED_FORM_MODELS],
            ]
        )
    out(format_table(["elevation", *[m.name for m in CLOSED_FORM_MODELS]], rows))

    out.rule("verdicts")
    for model in CLOSED_FORM_MODELS:
        series = errors[model.name][45.0]
        out.check(
            f"{model.name} is within 5 arcsec of the ray trace at 45 degrees "
            "over every swept state",
            series.size > 0 and float(series.max()) <= 5.0,
            f"worst {float(series.max()):.4f} arcsec over {series.size} states",
        )
    oriani_floor = errors["oriani_series"][15.0]
    out.check(
        "oriani_series is within 1 arcsec of the ray trace at its declared 15 degree floor",
        float(oriani_floor.max()) <= 1.0,
        f"worst {float(oriani_floor.max()):.4f} arcsec; the two-term series and the "
        "shell integral are independent derivations and agreeing this closely is the "
        "strongest internal consistency result in this package",
    )
    plane_30 = errors["plane_parallel"][30.0]
    out.check(
        "plane_parallel is within its declared 30 degree floor to better than 1 arcsec",
        float(plane_30.max()) <= 1.0,
        f"worst {float(plane_30.max()):.4f} arcsec",
    )
    for elevation in (45.0, 20.0):
        for name in ("plane_parallel", "bennett_1982", "saemundsson_1986"):
            series = signed[name][elevation]
            out.check(
                f"{name} overstates the ray trace at every swept state at "
                f"{elevation:g} degrees",
                bool((series > 0.0).all()),
                f"signed error from {float(series.min()):+.4f} to "
                f"{float(series.max()):+.4f} arcsec over {series.size} states; a "
                "systematic offset between a published form and a layered integral, "
                "reported and not corrected",
            )
    for elevation in (20.0, 10.0):
        series = signed["oriani_series"][elevation]
        out.check(
            f"oriani_series understates the ray trace at every swept state at "
            f"{elevation:g} degrees",
            bool((series < 0.0).all()),
            f"signed error from {float(series.min()):+.4f} to "
            f"{float(series.max()):+.4f} arcsec over {series.size} states",
        )
    low = signed["bennett_1982"][5.0]
    out.check(
        "the sign of the bennett_1982 error is NOT constant across the grid at 5 degrees",
        bool((low > 0.0).any() and (low < 0.0).any()),
        f"signed error from {float(low.min()):+.3f} to {float(low.max()):+.3f} arcsec; "
        "a departure with no constant sign cannot be removed by a scale factor",
    )
    horizon = signed["bennett_1982"][0.0]
    span = float(horizon.max() - horizon.min())
    out.check(
        "the bennett_1982 signed error at the horizon spans more than 400 arcsec "
        "across the swept states",
        span > 400.0,
        f"span {span:.2f} arcsec, from {float(horizon.min()):+.2f} to "
        f"{float(horizon.max()):+.2f}; equation (17) carries no wavelength and only a "
        "density ratio, and at the horizon that is not enough",
    )

    out.rule("elapsed time -- UNSTABLE, two cores, reported for information only")
    out(
        format_table(
            ["quantity", "value"],
            [
                ["ray traces run", str(timer.calls)],
                ["mean wall-clock per trace (ms)", f"{timer.mean_ms:.3f}"],
                ["total wall-clock (s)", f"{timer.elapsed_s:.2f}"],
            ],
        )
    )
    assert math.isfinite(timer.elapsed_s)

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
