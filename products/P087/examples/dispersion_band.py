"""The four refractivity formulations across the optical band, and what they cost.

What to notice: the upper panel's four curves are indistinguishable at this
scale, which is the point: the formulation choice is worth hundredths of an
arcsecond. The lower panel is the quantity equations (15) and (16) cannot
represent at all, because they take no wavelength: the refraction difference
between 400 nm and 700 nm, which exceeds one arcsecond below 55 degrees.
"""

from __future__ import annotations

import numpy as np

from _common import FORMULATION_COLOURS, bootstrap, relative, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from refractpath import AtmosphericState  # noqa: E402
from refractpath.raytrace import raytrace_refraction  # noqa: E402
from refractpath.refractivity import REFRACTIVITY_MODELS, refractivity_spread  # noqa: E402


def main() -> str:
    style()
    wavelengths = np.linspace(0.32, 1.65, 180)
    figure, (upper, middle, lower) = plt.subplots(
        3, 1, figsize=(8.2, 8.8), gridspec_kw={"height_ratios": [1.0, 0.8, 1.0]}
    )

    for name in REFRACTIVITY_MODELS:
        values = [
            refractivity_spread(float(w), 101325.0, 15.0, 0.0)[name] for w in wavelengths
        ]
        upper.plot(wavelengths, np.array(values) * 1e4, color=FORMULATION_COLOURS[name],
                   linewidth=1.3, label=name)
    upper.set_xlabel("vacuum wavelength (um)")
    upper.set_ylabel("(n - 1) x 1e4")
    upper.set_title("Four published refractivity formulations, 15 C, 101325 Pa, dry")
    upper.legend(loc="upper right", fontsize=8)

    spreads = [
        refractivity_spread(float(w), 101325.0, 15.0, 0.0)["spread"] for w in wavelengths
    ]
    middle.plot(wavelengths, np.array(spreads) * 1e8, color="#1f4e79", linewidth=1.4,
                label="dry")
    saturated = [
        refractivity_spread(
            float(w),
            101000.0,
            30.0,
            AtmosphericState(
                pressure_hpa=1010.0, temperature_c=30.0, relative_humidity=1.0,
                wavelength_um=float(w),
            ).vapour_pressure_pa,
        )["spread"]
        for w in wavelengths
    ]
    middle.plot(wavelengths, np.array(saturated) * 1e8, color="#b0344f", linewidth=1.4,
                label="saturated at 30 C")
    middle.set_xlabel("vacuum wavelength (um)")
    middle.set_ylabel("spread in (n - 1) x 1e8")
    middle.set_title(
        "The formulations agree on dry air and disagree on the humidity term"
    )
    middle.legend(loc="upper right", fontsize=8)

    elevations = np.linspace(2.0, 89.0, 200)
    blue = np.array([
        raytrace_refraction(
            float(h),
            AtmosphericState(pressure_hpa=1013.25, temperature_c=10.0, wavelength_um=0.40),
        )
        for h in elevations
    ])
    red = np.array([
        raytrace_refraction(
            float(h),
            AtmosphericState(pressure_hpa=1013.25, temperature_c=10.0, wavelength_um=0.70),
        )
        for h in elevations
    ])
    lower.plot(elevations, blue - red, color="#6f3f8f", linewidth=1.6,
               label="equation (19), 400 nm minus 700 nm")
    lower.axhline(0.0, color="#d9850b", linewidth=1.6,
                  label="equations (15) and (16), which take no wavelength")
    lower.axhline(1.0, color="#b0344f", linewidth=0.8, linestyle=":")
    lower.set_yscale("log")
    lower.set_ylim(1e-2, 1e2)
    lower.set_xlabel("apparent elevation (deg)")
    lower.set_ylabel("atmospheric dispersion (arcsec)")
    lower.set_title("A quantity two of the five methods predict as exactly zero")
    lower.legend(loc="upper right", fontsize=8)

    figure.subplots_adjust(hspace=0.45)
    path = save(figure, "dispersion_band")
    print(f"written: {path}")
    return relative(path)


if __name__ == "__main__":
    main()
