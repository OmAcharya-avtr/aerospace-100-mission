"""How the three magnitude contributions trade off across a pass.

Writes ``screenshots/attribution_vs_geometry.png``. The local terminal is held
at the ascending node of an equatorial orbit and the remote terminal is swept
along an inclined orbit, so the separation and the transverse relative velocity
both change. The point of the figure is that the total and the split move
differently.
"""

from __future__ import annotations

import numpy as np

from _common import SERIES, bootstrap, plt, save, style_axes

bootstrap()

from pointahead.angle import decompose_point_ahead, reference_point_ahead  # noqa: E402
from pointahead.scenario import CircularOrbit, crosslink_from_orbits  # noqa: E402

ALTITUDE_M = 600.0e3
INCLINATION_DEG = 60.0


def main() -> int:
    local = CircularOrbit(ALTITUDE_M, 0.0, 0.0, 0.0)
    arcs = np.linspace(0.05, 1.2, 60)
    rows = []
    for arc in arcs:
        remote = CircularOrbit(ALTITUDE_M, float(np.deg2rad(INCLINATION_DEG)), 0.0, float(arc))
        link = crosslink_from_orbits(local, remote, "swept")
        decomposition = decompose_point_ahead(link)
        rows.append(
            (
                float(arc),
                link.range_m,
                reference_point_ahead(link).total_rad,
                decomposition.light_time_lead_magnitude_rad,
                decomposition.stellar_aberration_magnitude_rad,
                decomposition.velocity_aberration_magnitude_rad,
            )
        )

    print(f"local terminal          : {ALTITUDE_M / 1.0e3:.0f} km equatorial, node")
    print(f"remote terminal         : {ALTITUDE_M / 1.0e3:.0f} km, "
          f"{INCLINATION_DEG:.0f} deg inclination, swept in argument of latitude")
    print(f"samples                 : {len(rows)}")
    print()
    print(f"{'arc / rad':>10s} {'range / km':>12s} {'total / urad':>14s} "
          f"{'lead / urad':>13s} {'stellar / urad':>15s} {'velocity / urad':>16s}")
    print("-" * 84)
    for row in rows[::10]:
        print(
            f"{row[0]:>10.4f} {row[1] / 1.0e3:>12.3f} {row[2] * 1.0e6:>14.6f} "
            f"{row[3] * 1.0e6:>13.6f} {row[4] * 1.0e6:>15.6f} {row[5] * 1.0e6:>16.6f}"
        )
    totals = np.array([row[2] for row in rows])
    leads = np.array([row[3] for row in rows])
    print()
    print(f"total spans             : {totals.min() * 1.0e6:.6f} to "
          f"{totals.max() * 1.0e6:.6f} urad (factor {totals.max() / totals.min():.3f})")
    print(f"light-time lead spans   : {leads.min() * 1.0e6:.6f} to "
          f"{leads.max() * 1.0e6:.6f} urad (factor {leads.max() / leads.min():.3f})")

    figure, axes = plt.subplots(figsize=(9.0, 5.0))
    ranges = np.array([row[1] for row in rows]) / 1.0e3
    axes.plot(ranges, totals * 1.0e6, color=SERIES[0], linewidth=2.0, label="total")
    axes.plot(ranges, leads * 1.0e6, color=SERIES[1], linewidth=1.4, label="light-time lead")
    axes.plot(
        ranges,
        np.array([row[4] for row in rows]) * 1.0e6,
        color=SERIES[2],
        linewidth=1.4,
        linestyle="--",
        label="stellar aberration (received beam)",
    )
    axes.plot(
        ranges,
        np.array([row[5] for row in rows]) * 1.0e6,
        color=SERIES[3],
        linewidth=1.4,
        linestyle=":",
        label="velocity aberration (transmit beam)",
    )
    style_axes(
        axes,
        f"Point-ahead attribution across a sweep, both terminals at "
        f"{ALTITUDE_M / 1.0e3:.0f} km, {INCLINATION_DEG:.0f} deg apart in plane",
        "range / km",
        "contribution / urad",
    )
    axes.legend(fontsize=8, loc="upper left", framealpha=0.95)
    print()
    print(f"written: {save(figure, 'attribution_vs_geometry.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
