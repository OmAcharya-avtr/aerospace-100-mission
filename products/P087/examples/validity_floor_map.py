"""The measured validity floor against declared pointing tolerance, for six states.

What to notice: the floor is a function of the tolerance you declare, not a
property of the atmosphere alone. Tightening a pointing budget from 30 arcsec
to 1 arcsec moves the elevation below which the published methods stop agreeing
from under 8 degrees to about 70, which is most of the sky.
"""

from __future__ import annotations

import numpy as np

from _common import bootstrap, relative, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from refractpath import AtmosphericState  # noqa: E402
from refractpath.disagreement import default_floor_grid, validity_floor  # noqa: E402

STATES = (
    ("1010 hPa, 10 C, dry, 550 nm", AtmosphericState(pressure_hpa=1010.0,
                                                     temperature_c=10.0,
                                                     wavelength_um=0.55), "#1f4e79"),
    ("1035 hPa, -25 C, dry, 550 nm", AtmosphericState(pressure_hpa=1035.0,
                                                      temperature_c=-25.0,
                                                      wavelength_um=0.55), "#6f3f8f"),
    ("1005 hPa, 35 C, 90 pct, 550 nm", AtmosphericState(pressure_hpa=1005.0,
                                                        temperature_c=35.0,
                                                        relative_humidity=0.9,
                                                        wavelength_um=0.55), "#3f8f4f"),
    ("700 hPa, 0 C, dry, 550 nm", AtmosphericState(pressure_hpa=700.0,
                                                   temperature_c=0.0,
                                                   wavelength_um=0.55), "#b0344f"),
    ("1010 hPa, 10 C, dry, 400 nm", AtmosphericState(pressure_hpa=1010.0,
                                                     temperature_c=10.0,
                                                     wavelength_um=0.40), "#1f6fb4"),
    ("1010 hPa, 10 C, dry, 1000 nm", AtmosphericState(pressure_hpa=1010.0,
                                                      temperature_c=10.0,
                                                      wavelength_um=1.00), "#d9850b"),
)


def main() -> str:
    style()
    tolerances = np.geomspace(0.5, 120.0, 11)
    # A 91-point scan grid rather than the 181-point package default, purely for
    # run time. The floors agree with the default grid to 1e-4 degrees at every
    # tolerance in validation/validate_disagreement_floor_output.txt, because
    # the grid only selects which bracket the bisection closes on and the
    # bisection then closes it to 1e-4 degrees regardless.
    grid = default_floor_grid(0.0, 90.0, 91)
    figure, axes = plt.subplots(figsize=(8.4, 5.2))
    for label, state, colour in STATES:
        floors = [
            validity_floor(state, float(t), grid_deg=grid).floor_deg for t in tolerances
        ]
        axes.plot(tolerances, floors, color=colour, linewidth=1.4, marker="o",
                  markersize=2.6, label=label)
    axes.set_xscale("log")
    axes.set_xlabel("declared pointing tolerance (arcsec)")
    axes.set_ylabel("measured validity floor (deg apparent elevation)")
    axes.set_title(
        "Below this elevation the five published methods disagree by more than\n"
        "the tolerance you declared. Measured by bisection, not asserted."
    )
    axes.legend(loc="upper right", fontsize=8)
    path = save(figure, "validity_floor_map")
    print(f"written: {path}")
    return relative(path)


if __name__ == "__main__":
    main()
