"""The layered profile the ray trace integrates, and the refractivity it produces.

What to notice: the right panel. Refractivity falls by three orders of
magnitude over the first 40 km and the refraction integral is dominated by the
first few kilometres, which is exactly the region the library has to invent
from one surface measurement. The three traces are the declared envelope
members of equation (23).
"""

from __future__ import annotations

import numpy as np

from _common import bootstrap, published_state, relative, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from refractpath import AtmosphericState  # noqa: E402
from refractpath.atmosphere import build_profile  # noqa: E402
from refractpath.raytrace import shell_heights  # noqa: E402
from refractpath.refractivity import ciddor_1996  # noqa: E402
from refractpath.verdict import PROFILE_ENVELOPE_LAPSE_RATES  # noqa: E402


def main() -> str:
    style()
    base = published_state()
    heights = shell_heights(600, 86_000.0, 1.6)
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 4.6))
    palette = ("#b0344f", "#1f4e79", "#d9850b")

    for colour, (label, rate) in zip(palette, PROFILE_ENVELOPE_LAPSE_RATES, strict=True):
        state = AtmosphericState(
            pressure_hpa=base.pressure_hpa,
            temperature_c=base.temperature_c,
            wavelength_um=base.wavelength_um,
            surface_lapse_rate_k_per_m=rate,
        )
        profile = build_profile(state, heights, top_m=86_000.0)
        refractivity = ciddor_1996(
            state.wavelength_um,
            profile.pressure_pa,
            profile.temperature_k - 273.15,
            profile.vapour_pressure_pa,
            state.co2_ppm,
        )
        axes[0].plot(profile.temperature_k, heights / 1000.0, color=colour,
                     linewidth=1.3, label=f"{label} ({rate * 1000.0:+.1f} K/km)")
        axes[1].plot(profile.pressure_pa / 100.0, heights / 1000.0, color=colour,
                     linewidth=1.3)
        axes[2].plot(np.asarray(refractivity) * 1e6, heights / 1000.0, color=colour,
                     linewidth=1.3)

    for boundary in (11.0, 20.0, 32.0, 47.0, 51.0, 71.0):
        for panel in axes:
            panel.axhline(boundary, color="#d6d6d6", linewidth=0.7, linestyle=":")

    axes[0].set_xlabel("temperature (K)")
    axes[0].set_ylabel("height above the observer (km)")
    axes[0].set_title("equations (1) and (2)")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[1].set_xlabel("pressure (hPa)")
    axes[1].set_xscale("log")
    axes[1].set_title("hydrostatic, equation (1)")
    axes[2].set_xlabel("(n - 1) x 1e6, equation (10)")
    axes[2].set_xscale("log")
    axes[2].set_title("what equation (19) integrates")
    for panel in axes:
        panel.set_ylim(0.0, 86.0)
    figure.suptitle(
        "The layered profile anchored on one observer measurement: "
        f"{base.pressure_hpa:g} hPa, {base.temperature_c:g} C, dry, "
        f"{base.wavelength_um:g} um",
        fontsize=10,
    )
    figure.subplots_adjust(wspace=0.26, top=0.86)
    path = save(figure, "atmosphere_profile")
    print(f"written: {path}")
    return relative(path)


if __name__ == "__main__":
    main()
