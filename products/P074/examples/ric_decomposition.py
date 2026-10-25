"""The divergence split into radial, in-track and cross-track components.

What to notice: the in-track component is the whole of the divergence to within
a per cent for almost every object. The radial and cross-track components are
periodic rather than secular, so they do not grow, and the growth curve the
product reports is an along-track growth curve. This is why a mean-motion error
is the right thing to think about and why a radial position tolerance would be
the wrong tolerance to declare.
"""

from __future__ import annotations

import numpy as np

from _common import SERIES, bootstrap, load_dataset, save, style

bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    table = load_dataset()
    objects = table.column("object_index")
    ages = table.column("epoch_age_days")
    total = table.column("drift_km")
    radial = table.column("drift_radial_km")
    in_track = table.column("drift_in_track_km")
    cross = table.column("drift_cross_track_km")
    perigee = table.column("perigee_altitude_km")

    unique = np.unique(objects)
    chosen = unique[np.argsort([perigee[objects == value][0] for value in unique])]
    sample = chosen[:: max(1, chosen.size // 3)][:3]

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))

    for position, value in enumerate(sample):
        mask = objects == value
        axis = axes[position]
        for label, series, colour in (
            ("in-track", in_track[mask], SERIES[0]),
            ("radial", radial[mask], SERIES[1]),
            ("cross-track", cross[mask], SERIES[2]),
        ):
            axis.plot(ages[mask], series, color=colour, linewidth=1.3, label=label)
        axis.set_yscale("log")
        axis.set_xlabel("epoch age, days")
        if position == 0:
            axis.set_ylabel("component of the divergence, km (window RMS)")
            axis.legend(fontsize=8)
        axis.set_title(f"perigee {perigee[mask][0]:.0f} km")

    share = in_track / total
    figure.suptitle(
        "orbitdrift: the divergence is along-track. Median in-track share "
        f"{float(np.median(share)):.4f} over {share.size} rows",
        fontsize=10,
    )
    figure.tight_layout()
    print(f"wrote {save(figure, 'ric_decomposition')}")
    print(f"median in-track share        : {float(np.median(share)):.6f}")
    print(f"5th percentile in-track share: {float(np.percentile(share, 5)):.6f}")
    print(f"median radial share          : {float(np.median(radial / total)):.6f}")
    print(f"median cross-track share     : {float(np.median(cross / total)):.6f}")
    print(
        "rows where in-track is the largest component: "
        f"{float(np.mean((in_track > radial) & (in_track > cross))):.6f}"
    )


if __name__ == "__main__":
    main()
