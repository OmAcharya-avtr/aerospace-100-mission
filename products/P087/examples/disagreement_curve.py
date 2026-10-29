"""Equation (22), the cross-model disagreement, against apparent elevation.

What to notice: the curve is not monotone. The two sharp drops near 1.4 and 2.0
degrees are the plane-parallel model and the Oriani series leaving the ensemble
at their own validity limits, and the spread falls because the most discrepant
member has gone. A search for the tolerance crossing that walked up from the
horizon would stop at the first drop; the measured floor is the highest
crossing, marked for each declared tolerance.
"""

from __future__ import annotations

import numpy as np

from _common import TOLERANCE, bootstrap, published_state, relative, save, style, tolerances_arcsec

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from refractpath.disagreement import ensemble_at, validity_floor  # noqa: E402
from refractpath.models import (  # noqa: E402
    oriani_breakdown_elevation_deg,
    plane_parallel_floor_deg,
)


def main() -> str:
    style()
    state = published_state()
    elevations = np.concatenate(
        [np.linspace(0.0, 3.0, 260), np.linspace(3.0, 90.0, 240)[1:]]
    )
    spread = np.array([ensemble_at(float(h), state).spread_arcsec for h in elevations])
    undefined = np.array(
        [len(ensemble_at(float(h), state).undefined_members) for h in elevations]
    )

    figure, axes = plt.subplots(figsize=(8.6, 5.6))
    axes.plot(elevations, spread, color="#1f4e79", linewidth=1.6,
              label="equation (22), five methods")
    for tolerance in tolerances_arcsec():
        measured = validity_floor(state, tolerance)
        axes.axhline(tolerance, color=TOLERANCE, linewidth=0.7, linestyle=":")
        axes.plot(
            [measured.floor_deg], [tolerance], marker="o", markersize=5,
            color=TOLERANCE, linestyle="none",
        )
        axes.annotate(
            f"{tolerance:g} arcsec -> {measured.floor_deg:.2f} deg",
            xy=(measured.floor_deg, tolerance),
            xytext=(measured.floor_deg + 3.0, tolerance * 1.25),
            fontsize=8,
            color=TOLERANCE,
        )
    axes.fill_between(
        elevations, 1e-3, 1e5, where=undefined > 0, color="#d6d6d6", alpha=0.55,
        label="at least one method undefined",
    )
    axes.axvline(plane_parallel_floor_deg(state), color="#b0344f", linewidth=0.8,
                 linestyle="--")
    axes.axvline(oriani_breakdown_elevation_deg(state), color="#6f3f8f", linewidth=0.8,
                 linestyle="--")
    axes.set_yscale("log")
    axes.set_xscale("symlog", linthresh=1.0)
    axes.set_ylim(1e-2, 1e4)
    axes.set_xlim(0.0, 90.0)
    axes.set_xlabel("apparent elevation (deg, symmetric log below 1 deg)")
    axes.set_ylabel("cross-model disagreement (arcsec)")
    axes.set_title(
        "Equation (22) and the measured validity floor, "
        f"{state.pressure_hpa:g} hPa, {state.temperature_c:g} C, dry, "
        f"{state.wavelength_um:g} um"
    )
    axes.legend(loc="upper right", fontsize=8)
    path = save(figure, "disagreement_curve")
    print(f"written: {path}")
    return relative(path)


if __name__ == "__main__":
    main()
