"""The verdict over a grid of apparent elevation and declared pointing tolerance.

What to notice: the boundary between amber and red moves with the tolerance you
declare, which is the whole point. The red region is where this package refuses
to supply a correction for operational pointing, and it is computed from the
measured disagreement and the measured profile envelope, not declared.
"""

from __future__ import annotations

import numpy as np

from _common import VERDICT_COLOURS, bootstrap, published_state, relative, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from refractpath.disagreement import default_ensemble, validity_floor  # noqa: E402
from refractpath.verdict import compute_verdict  # noqa: E402

LABELS = ("usable", "degraded", "refuse")


def main() -> str:
    style()
    state = published_state()
    ensemble = default_ensemble()
    elevations = np.concatenate(
        [np.linspace(0.5, 5.0, 10), np.linspace(7.0, 89.0, 18)]
    )
    tolerances = np.geomspace(0.5, 120.0, 16)

    floors = {float(t): validity_floor(state, float(t), None, ensemble) for t in tolerances}
    grid = np.zeros((tolerances.size, elevations.size), dtype=int)
    for row, tolerance in enumerate(tolerances):
        for column, elevation in enumerate(elevations):
            verdict = compute_verdict(
                float(elevation),
                state,
                float(tolerance),
                None,
                ensemble,
                floor=floors[float(tolerance)],
            )
            grid[row, column] = LABELS.index(verdict.verdict)

    figure, axes = plt.subplots(figsize=(8.6, 5.4))
    cmap = ListedColormap([VERDICT_COLOURS[label] for label in LABELS])
    axes.pcolormesh(
        elevations, tolerances, grid, cmap=cmap, vmin=0, vmax=2, shading="nearest"
    )
    axes.plot(
        [floors[float(t)].floor_deg for t in tolerances],
        tolerances,
        color="#ffffff",
        linewidth=1.6,
        label="measured validity floor",
    )
    axes.set_yscale("log")
    axes.set_xlabel("apparent elevation (deg)")
    axes.set_ylabel("declared pointing tolerance (arcsec)")
    axes.set_title(
        "The verdict, computed: "
        f"{state.pressure_hpa:g} hPa, {state.temperature_c:g} C, dry, "
        f"{state.wavelength_um:g} um"
    )
    axes.legend(
        handles=[
            Patch(facecolor=VERDICT_COLOURS[label], label=label) for label in LABELS
        ]
        + [plt.Line2D([], [], color="#ffffff", linewidth=1.6,
                      label="measured validity floor")],
        loc="upper right",
        fontsize=8,
    )
    axes.grid(False)
    path = save(figure, "verdict_map")
    print(f"written: {path}")
    return relative(path)


if __name__ == "__main__":
    main()
