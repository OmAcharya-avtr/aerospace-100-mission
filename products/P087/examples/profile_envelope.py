"""Equation (23): the part of the answer that is an input nobody supplied.

What to notice: the lower panel. Above about 10 degrees the disagreement
between five published methods is the larger term and is worth budgeting for.
Below about 1 degree the unmeasured temperature profile is the larger term by
nearly an order of magnitude, and no choice among the published methods helps,
because all of them are being asked about an atmosphere nobody measured.
"""

from __future__ import annotations

import numpy as np

from _common import ENVELOPE, bootstrap, published_state, relative, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from refractpath import AtmosphericState  # noqa: E402
from refractpath.disagreement import ensemble_at  # noqa: E402
from refractpath.raytrace import raytrace_refraction  # noqa: E402
from refractpath.verdict import PROFILE_ENVELOPE_LAPSE_RATES, profile_envelope_arcsec  # noqa: E402

LAPSE_RATES_K_PER_KM = (-9.8, -6.5, -4.0, 0.0, 10.0, 20.0)


def main() -> str:
    style()
    base = published_state()
    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(8.4, 7.2), gridspec_kw={"height_ratios": [1.0, 1.0]}
    )

    elevations = np.linspace(0.0, 10.0, 220)
    palette = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(LAPSE_RATES_K_PER_KM)))
    for colour, rate in zip(palette, LAPSE_RATES_K_PER_KM, strict=True):
        state = AtmosphericState(
            pressure_hpa=base.pressure_hpa,
            temperature_c=base.temperature_c,
            wavelength_um=base.wavelength_um,
            surface_lapse_rate_k_per_m=rate / 1000.0,
        )
        values = np.array([raytrace_refraction(float(h), state) / 60.0 for h in elevations])
        upper.plot(elevations, values, color=colour, linewidth=1.3,
                   label=f"{rate:+.1f} K/km")
    upper.set_xlabel("apparent elevation (deg)")
    upper.set_ylabel("refraction (arcmin)")
    upper.set_title(
        "The same observer measurements, six surface lapse rates: "
        f"{base.pressure_hpa:g} hPa, {base.temperature_c:g} C, dry"
    )
    upper.legend(loc="upper right", ncol=2, fontsize=8, title="surface lapse rate")

    grid = np.concatenate([np.linspace(0.0, 2.0, 90), np.linspace(2.0, 60.0, 120)[1:]])
    spread = np.array([ensemble_at(float(h), base).spread_arcsec for h in grid])
    envelope = np.array([profile_envelope_arcsec(float(h), base)[0] for h in grid])
    lower.plot(grid, spread, color="#1f4e79", linewidth=1.6,
               label="equation (22), cross-model disagreement")
    lower.plot(grid, envelope, color=ENVELOPE, linewidth=1.6,
               label="equation (23), profile envelope")
    lower.fill_between(grid, spread, envelope, where=envelope > spread,
                       color=ENVELOPE, alpha=0.3,
                       label="profile is the larger term")
    lower.set_yscale("log")
    lower.set_xscale("symlog", linthresh=1.0)
    lower.set_ylim(1e-4, 1e4)
    lower.set_xlim(0.0, 60.0)
    lower.set_xlabel("apparent elevation (deg, symmetric log below 1 deg)")
    lower.set_ylabel("arcsec")
    lower.set_title(
        "Which term is larger: the methods disagreeing, or the profile nobody measured"
    )
    lower.legend(loc="lower left", fontsize=8)
    members = ", ".join(
        f"{label} {rate * 1000.0:+.1f} K/km" for label, rate in PROFILE_ENVELOPE_LAPSE_RATES
    )
    lower.annotate(
        f"envelope members: {members}",
        xy=(0.01, 0.02), xycoords="axes fraction", fontsize=7.5, color="#555555",
    )
    figure.subplots_adjust(hspace=0.33)
    path = save(figure, "profile_envelope")
    print(f"written: {path}")
    return relative(path)


if __name__ == "__main__":
    main()
