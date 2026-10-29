"""Refraction against apparent elevation, every method, with the low-elevation failures.

What to notice: the five methods are indistinguishable above about 20 degrees on
a logarithmic axis and separate visibly below 5 degrees. The dashed traces are
the plane-parallel model and the unguarded Oriani series continued past the
points where they stop approximating anything: one diverges upward, the other
crosses zero and goes negative.
"""

from __future__ import annotations

import numpy as np

from _common import (
    METHOD_COLOURS,
    ORIANI,
    PLANE_PARALLEL,
    bootstrap,
    published_state,
    relative,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from refractpath.models import (  # noqa: E402
    bennett_1982,
    oriani_breakdown_elevation_deg,
    oriani_series,
    oriani_series_unguarded,
    plane_parallel,
    plane_parallel_floor_deg,
    saemundsson_1986,
)
from refractpath.raytrace import raytrace_refraction  # noqa: E402


def main() -> str:
    style()
    state = published_state()
    elevations = np.concatenate(
        [np.linspace(0.0, 2.0, 160), np.linspace(2.0, 90.0, 300)[1:]]
    )
    methods = {
        "raytrace": raytrace_refraction,
        "plane_parallel": plane_parallel,
        "oriani_series": oriani_series,
        "bennett_1982": bennett_1982,
        "saemundsson_1986": saemundsson_1986,
    }
    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(8.4, 7.4), gridspec_kw={"height_ratios": [1.25, 1.0]}
    )

    for name, function in methods.items():
        values = np.array([function(float(h), state) for h in elevations])
        upper.plot(
            elevations,
            values,
            color=METHOD_COLOURS[name],
            linewidth=1.5 if name == "raytrace" else 1.1,
            label=name,
        )
    upper.set_yscale("log")
    upper.set_xlim(0.0, 90.0)
    upper.set_xlabel("apparent elevation (deg)")
    upper.set_ylabel("refraction (arcsec, log scale)")
    upper.set_title(
        "Refraction by five independent standard methods, "
        f"{state.pressure_hpa:g} hPa, {state.temperature_c:g} C, dry, "
        f"{state.wavelength_um:g} um"
    )
    upper.legend(loc="lower left", ncol=2)

    low = np.linspace(0.05, 6.0, 500)
    reference = np.array([raytrace_refraction(float(h), state) for h in low])
    lower.plot(low, reference, color=METHOD_COLOURS["raytrace"], linewidth=1.8,
               label="raytrace, equation (19)")
    plane = np.array([plane_parallel(float(h), state) for h in low])
    lower.plot(low, plane, color=PLANE_PARALLEL, linewidth=1.3,
               label="plane-parallel, equation (13)")
    raw = np.array([oriani_series_unguarded(float(h), state) for h in low])
    lower.plot(low, raw, color=ORIANI, linewidth=1.3, linestyle="--",
               label="Oriani series unguarded, equation (14)")
    guarded = np.array([oriani_series(float(h), state) for h in low])
    lower.plot(low, guarded, color=ORIANI, linewidth=2.0,
               label="Oriani series guarded by equation (14a)")
    lower.axhline(0.0, color="#333333", linewidth=0.8)
    plane_floor = plane_parallel_floor_deg(state)
    oriani_floor = oriani_breakdown_elevation_deg(state)
    lower.axvline(plane_floor, color=PLANE_PARALLEL, linewidth=1.0, linestyle=":")
    lower.axvline(oriani_floor, color=ORIANI, linewidth=1.0, linestyle=":")
    lower.annotate(
        f"equation (13) analytic floor\n{plane_floor:.3f} deg",
        xy=(plane_floor, -1500.0),
        xytext=(plane_floor + 0.45, -2600.0),
        color=PLANE_PARALLEL,
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": PLANE_PARALLEL, "linewidth": 0.8},
    )
    lower.annotate(
        f"equation (14a) breakdown\n{oriani_floor:.3f} deg",
        xy=(oriani_floor, 2200.0),
        xytext=(oriani_floor + 0.6, 2600.0),
        color=ORIANI,
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": ORIANI, "linewidth": 0.8},
    )
    lower.set_ylim(-4000.0, 4000.0)
    lower.set_xlim(0.0, 6.0)
    lower.set_xlabel("apparent elevation (deg)")
    lower.set_ylabel("refraction (arcsec, linear)")
    lower.set_title(
        "Below 6 degrees: one closed form diverges upward, the other crosses zero"
    )
    lower.legend(loc="lower right", fontsize=8)

    figure.subplots_adjust(hspace=0.33)
    path = save(figure, "refraction_curves")
    print(f"written: {path}")
    return relative(path)


if __name__ == "__main__":
    main()
