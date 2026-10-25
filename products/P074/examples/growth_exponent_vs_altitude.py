"""The fitted growth exponent against perigee altitude: the central physical result.

What to notice: above about 1100 km the exponent sits on the horizontal line at
1, which is the linear-in-time growth a constant mean-motion error produces.
Below 450 km it rises toward 2, the quadratic growth a constant drag-rate error
produces. The transition is the physics, and it is the reason a single analytic
form fitted globally cannot do as well as one whose exponent depends on
altitude.
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
    drag = table.column("log10_drag_parameter")

    unique = np.unique(objects)
    altitudes = np.empty(unique.size)
    drags = np.empty(unique.size)
    exponents = np.empty(unique.size)
    errors = np.empty(unique.size)
    for index, value in enumerate(unique):
        mask = objects == value
        fit = fit_power_law(ages[mask], drift[mask])
        altitudes[index] = perigee[mask][0]
        drags[index] = drag[mask][0]
        exponents[index] = fit.exponent
        errors[index] = fit.exponent_stderr

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    axes[0].errorbar(
        altitudes,
        exponents,
        yerr=errors,
        fmt="o",
        markersize=3.4,
        linewidth=0.7,
        color=SERIES[0],
        ecolor="#9ab2cc",
        capsize=1.6,
    )
    axes[0].axhline(1.0, color=SERIES[2], linestyle="--", linewidth=1.1)
    axes[0].axhline(2.0, color=SERIES[1], linestyle="--", linewidth=1.1)
    axes[0].text(
        1650.0, 1.02, "linear in time:\nconstant mean-motion error", fontsize=7.5,
        color=SERIES[2], ha="right", va="bottom",
    )
    axes[0].text(
        1650.0, 1.96, "quadratic in time:\nconstant drag-rate error", fontsize=7.5,
        color=SERIES[1], ha="right", va="top",
    )
    axes[0].set_xlabel("perigee altitude, km")
    axes[0].set_ylabel("fitted growth exponent p")
    axes[0].set_title("p against altitude, with the two predicted regimes")

    bands = ((300.0, 450.0), (450.0, 600.0), (600.0, 800.0), (800.0, 1100.0), (1100.0, 1700.0))
    centres = []
    means = []
    spreads = []
    counts = []
    for low, high in bands:
        selected = (altitudes >= low) & (altitudes < high)
        if not np.any(selected):
            continue
        centres.append(0.5 * (low + high))
        means.append(float(exponents[selected].mean()))
        spreads.append(float(exponents[selected].std()))
        counts.append(int(np.count_nonzero(selected)))

    axes[1].scatter(drags, exponents, s=16, color=SERIES[3], edgecolor="none", alpha=0.8)
    coefficients = np.polyfit(drags, exponents, 1)
    grid = np.linspace(drags.min(), drags.max(), 50)
    axes[1].plot(
        grid,
        np.polyval(coefficients, grid),
        color=SERIES[1],
        linewidth=1.3,
        label=f"least squares, slope {coefficients[0]:+.4f}",
    )
    correlation = float(np.corrcoef(drags, exponents)[0, 1])
    axes[1].set_xlabel(r"$\log_{10}(C_D A/m \cdot \rho_{\rm perigee})$")
    axes[1].set_ylabel("fitted growth exponent p")
    axes[1].set_title(f"p against the drag parameter, r = {correlation:+.4f}")
    axes[1].legend(fontsize=8)

    figure.suptitle(
        "orbitdrift: the growth exponent is the physics, measured over "
        f"{unique.size} objects",
        fontsize=10,
    )
    figure.tight_layout()
    print(f"wrote {save(figure, 'growth_exponent_vs_altitude')}")
    for (low, high), mean, spread, count in zip(bands, means, spreads, counts, strict=True):
        print(f"  {low:.0f}-{high:.0f} km: n={count} mean p={mean:.4f} sd={spread:.4f}")
    print(f"  correlation of p with the log drag parameter: {correlation:+.4f}")


if __name__ == "__main__":
    main()
