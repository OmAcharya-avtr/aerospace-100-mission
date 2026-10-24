"""The attribution is frame-dependent and the total is not, drawn.

Writes ``screenshots/frame_dependence.png``. Six inertial frames, one physical
configuration. The total sits on a flat line; the three attributed lines move
across the whole range.
"""

from __future__ import annotations

import numpy as np

from _common import SERIES, bootstrap, plt, save, style_axes

bootstrap()

from pointahead.angle import decompose_point_ahead, reference_point_ahead  # noqa: E402
from pointahead.scenario import leo_crossing_pair  # noqa: E402
from pointahead.terminal import Crosslink  # noqa: E402


def main() -> int:
    kepler = leo_crossing_pair()
    base = Crosslink(
        local=kepler.local,
        remote=kepler.remote,
        trajectory_model="linear",
        mu=kepler.mu,
        label=kepler.label,
        frame=kepler.frame,
    )
    boosts = [
        ("as supplied", np.zeros(3)),
        ("local at rest", -base.local.velocity_mps),
        ("remote at rest", -base.remote.velocity_mps),
        ("mean at rest", -0.5 * (base.local.velocity_mps + base.remote.velocity_mps)),
        ("+1 km/s x", np.array([1000.0, 0.0, 0.0])),
        ("+10 km/s z", np.array([0.0, 0.0, 10000.0])),
    ]

    labels = []
    totals = []
    leads = []
    stellars = []
    velocities = []
    for label, delta in boosts:
        link = base if not np.any(delta) else base.boosted(delta)
        decomposition = decompose_point_ahead(link)
        labels.append(label)
        totals.append(reference_point_ahead(link).total_rad)
        leads.append(decomposition.light_time_lead_magnitude_rad)
        stellars.append(decomposition.stellar_aberration_magnitude_rad)
        velocities.append(decomposition.velocity_aberration_magnitude_rad)

    print(f"geometry                : {base.label}, range {base.range_m / 1.0e3:.3f} km")
    print(f"trajectory model        : {base.trajectory_model} "
          f"(a boost of a Keplerian state is not a Keplerian state)")
    print()
    print(f"{'frame':<16s} {'total / urad':>14s} {'lead / urad':>13s} "
          f"{'stellar / urad':>15s} {'velocity / urad':>16s}")
    print("-" * 78)
    for index, label in enumerate(labels):
        print(
            f"{label:<16s} {totals[index] * 1.0e6:>14.9f} {leads[index] * 1.0e6:>13.6f} "
            f"{stellars[index] * 1.0e6:>15.6f} {velocities[index] * 1.0e6:>16.6f}"
        )
    totals_array = np.array(totals)
    leads_array = np.array(leads)
    print()
    print(f"spread of the total     : {totals_array.max() - totals_array.min():.6e} rad")
    print(f"spread of the lead      : {leads_array.max() - leads_array.min():.6e} rad")
    lead_spread = leads_array.max() - leads_array.min()
    total_spread = totals_array.max() - totals_array.min()
    print(f"ratio                   : {lead_spread / total_spread:.6e}")

    figure, axes = plt.subplots(figsize=(9.5, 5.0))
    positions = np.arange(len(labels))
    width = 0.22
    axes.bar(
        positions - 1.5 * width,
        np.array(totals) * 1.0e6,
        width=width,
        color=SERIES[0],
        label="total (frame-invariant)",
    )
    axes.bar(
        positions - 0.5 * width,
        leads_array * 1.0e6,
        width=width,
        color=SERIES[1],
        label="light-time lead",
    )
    axes.bar(
        positions + 0.5 * width,
        np.array(stellars) * 1.0e6,
        width=width,
        color=SERIES[2],
        label="stellar aberration",
    )
    axes.bar(
        positions + 1.5 * width,
        np.array(velocities) * 1.0e6,
        width=width,
        color=SERIES[3],
        label="velocity aberration",
    )
    axes.axhline(
        float(totals_array[0]) * 1.0e6,
        color="#555555",
        linestyle="--",
        linewidth=1.0,
    )
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, fontsize=8)
    style_axes(
        axes,
        "One configuration, six inertial frames: the total is fixed, the split is not",
        "inertial frame (constant velocity added to both terminals)",
        "contribution / urad",
    )
    axes.legend(fontsize=8, loc="upper left", framealpha=0.95)
    print()
    print(f"written: {save(figure, 'frame_dependence.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
