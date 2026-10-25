"""Divergence against epoch age, for objects spanning the altitude range.

What to notice: the curves are straight lines on logarithmic axes, so the
growth is a power law; the slope is the exponent, and it is steeper for the
low-altitude objects where a drag-rate error dominates than for the high ones
where a constant mean-motion error does. The spread at seven days is four
orders of magnitude, which is why a single kilometres-per-day rule of thumb
cannot work.
"""

from __future__ import annotations

import numpy as np

from _common import SERIES, bootstrap, load_dataset, save, style

bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402

from orbitdrift.growth import fit_power_law  # noqa: E402


def main() -> None:
    table = load_dataset()
    objects = table.column("object_index")
    ages = table.column("epoch_age_days")
    drift = table.column("drift_km")
    perigee = table.column("perigee_altitude_km")

    unique = np.unique(objects)
    altitudes = np.array([perigee[objects == value][0] for value in unique])
    bands = ((300.0, 450.0), (450.0, 600.0), (600.0, 800.0), (800.0, 1100.0), (1100.0, 1700.0))

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    for index, (low, high) in enumerate(bands):
        inside = unique[(altitudes >= low) & (altitudes < high)]
        if inside.size == 0:
            continue
        chosen = inside[: min(4, inside.size)]
        for position, value in enumerate(chosen):
            mask = objects == value
            axes[0].plot(
                ages[mask],
                drift[mask],
                color=SERIES[index % len(SERIES)],
                linewidth=1.1,
                alpha=0.85,
                label=f"{low:.0f}-{high:.0f} km" if position == 0 else None,
            )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch age, days")
    axes[0].set_ylabel("model divergence, km (window RMS)")
    axes[0].set_title("SGP4 against the independent force model")
    axes[0].legend(title="perigee altitude", fontsize=8, title_fontsize=8)

    last = np.array(
        [float(drift[objects == value][int(np.argmax(ages[objects == value]))]) for value in unique]
    )
    exponents = np.array(
        [
            fit_power_law(ages[objects == value], drift[objects == value]).exponent
            for value in unique
        ]
    )
    scatter = axes[1].scatter(
        altitudes, last, c=exponents, cmap="viridis", s=18, edgecolor="none"
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("perigee altitude, km")
    axes[1].set_ylabel("divergence at 6.79 days, km")
    axes[1].set_title("divergence at the end of the span, by altitude")
    bar = figure.colorbar(scatter, ax=axes[1])
    bar.set_label("fitted growth exponent p")

    figure.suptitle(
        "orbitdrift: model divergence against epoch age, "
        f"{unique.size} synthetic objects, 7-day span",
        fontsize=10,
    )
    figure.tight_layout()
    print(f"wrote {save(figure, 'divergence_curves')}")
    print(f"objects plotted: {unique.size}")
    print(f"divergence at the end of the span: {last.min():.4f} to {last.max():.1f} km")
    print(f"fitted exponent: {exponents.min():.4f} to {exponents.max():.4f}")


if __name__ == "__main__":
    main()
