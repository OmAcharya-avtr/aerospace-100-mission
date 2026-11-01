"""Bennett's formula against an independently written ray trace.

What to notice: the normalised shapes agree to 2 per cent over 10 to 85 degrees
and the absolute refractions differ by a few arcseconds, most of which is the
difference between the optical and the dry-air radio refractivity constants.
The shape is what the design matrix uses; the scale is what the fitted
coefficient absorbs.
"""

from __future__ import annotations

import numpy as np

from _common import BOUND, INDEPENDENT, LEAST, NEUTRAL, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from pointmodel import refraction  # noqa: E402

GRID = np.linspace(10.0, 85.0, 31)
MEEUS_ARCSEC = 0.07 * 60.0


def main() -> int:
    style()
    radio = refraction.compare_refraction(
        GRID, refractivity_constant=refraction.K_RADIO_DRY
    )
    optical = refraction.compare_refraction(
        GRID, refractivity_constant=refraction.K_OPTICAL_550NM
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))

    axes[0].semilogy(GRID, radio.bennett_arcsec, color=LEAST, lw=1.8, label="Bennett 1982")
    axes[0].semilogy(
        GRID, radio.raytrace_arcsec, color=INDEPENDENT, lw=1.4, ls="--",
        label="ray trace, radio dry",
    )
    axes[0].semilogy(
        GRID, optical.raytrace_arcsec, color=BOUND, lw=1.4, ls=":",
        label="ray trace, optical",
    )
    axes[0].set_xlabel("true elevation, degrees")
    axes[0].set_ylabel("refraction, arcsec")
    axes[0].set_title("absolute refraction")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(
        GRID,
        radio.bennett_arcsec - radio.raytrace_arcsec,
        color=INDEPENDENT,
        lw=1.6,
        label="radio dry constant",
    )
    axes[1].plot(
        GRID,
        optical.bennett_arcsec - optical.raytrace_arcsec,
        color=BOUND,
        lw=1.6,
        label="optical constant",
    )
    axes[1].axhline(MEEUS_ARCSEC, color=NEUTRAL, ls="--", lw=1.0)
    axes[1].axhline(-MEEUS_ARCSEC, color=NEUTRAL, ls="--", lw=1.0)
    axes[1].text(
        40.0, MEEUS_ARCSEC + 0.2, "Meeus: equation (16.4) good to 0.07' = 4.2\"",
        fontsize=8, color=NEUTRAL,
    )
    axes[1].set_xlabel("true elevation, degrees")
    axes[1].set_ylabel("Bennett minus ray trace, arcsec")
    axes[1].set_title("absolute disagreement, against a published accuracy")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(
        GRID,
        100.0 * (radio.shape_bennett - radio.shape_raytrace) / radio.shape_raytrace,
        color=INDEPENDENT,
        lw=1.6,
        label="radio dry constant",
    )
    axes[2].plot(
        GRID,
        100.0 * (optical.shape_bennett - optical.shape_raytrace) / optical.shape_raytrace,
        color=BOUND,
        lw=1.6,
        ls="--",
        label="optical constant",
    )
    axes[2].axhline(0.0, color="#1a1a1a", lw=0.8)
    axes[2].set_xlabel("true elevation, degrees")
    axes[2].set_ylabel("shape disagreement, per cent")
    axes[2].set_title("the normalised shape, which the model uses")
    axes[2].legend(loc="lower left", fontsize=8)

    figure.suptitle(
        "atmospheric refraction: an empirical fit against a ray trace written here",
        fontsize=11,
    )
    figure.subplots_adjust(top=0.84, wspace=0.30)
    path = save(figure, "refraction_crosscheck")
    print(f"written: {path}")
    print("")
    print(f"max |absolute difference|, radio dry   : {radio.max_absolute_arcsec:.4f} arcsec")
    print(f"max |absolute difference|, optical     : {optical.max_absolute_arcsec:.4f} arcsec")
    print(f"Meeus's quoted accuracy for eq (16.4)  : {MEEUS_ARCSEC:.4f} arcsec")
    print(f"mean Bennett / ray trace, radio dry    : {radio.mean_ratio:.6f}")
    print(f"mean Bennett / ray trace, optical      : {optical.mean_ratio:.6f}")
    print(f"max relative shape difference, radio   : {radio.max_shape_relative:.6e}")
    print(f"max relative shape difference, optical : {optical.max_shape_relative:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
