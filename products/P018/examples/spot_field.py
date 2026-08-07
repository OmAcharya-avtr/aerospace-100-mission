"""Simulated Shack-Hartmann spot pattern with the measured slope vectors overlaid.

Run from products/P018/:

    PYTHONPATH=src python examples/spot_field.py     # ~3 s

Writes ../screenshots/spot_field.png.

The wavefront is a global tilt plus a defocus term, so the true slope field is
known exactly: a uniform vector plus a radial fan. The figure shows the raw
detector frame, the slopes measured by the thresholded centre of gravity as a
quiver plot over the lenslet grid, and the per-subaperture residual against
truth.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from shacksim import (  # noqa: E402
    LensletArray,
    cog_slopes,
    correlation_slopes,
    defocus_slopes,
    simulate_frame,
    tilt_slopes,
)

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "spot_field.png"

PHOTONS = 2000.0
BACKGROUND = 1.0
READ_NOISE = 3.0
THRESHOLD = BACKGROUND + 3 * READ_NOISE
TILT_X, TILT_Y = 8.0e-4, -4.0e-4
CURVATURE = 0.35  # [1/m] -> 2c R = 0.35 * 4e-3 = 1.4e-3 rad at the pupil edge


def main() -> None:
    array = LensletArray()
    truth = tilt_slopes(array, TILT_X, TILT_Y) + defocus_slopes(array, CURVATURE)
    frame = simulate_frame(
        array, truth, photons=PHOTONS, background=BACKGROUND, read_noise=READ_NOISE, seed=2026
    )
    measured = cog_slopes(frame, array, threshold=THRESHOLD)
    measured_corr = correlation_slopes(frame, array)

    centres_px = (array.valid_centres() / array.pitch + (array.n_lenslets - 1) / 2.0)
    centres_px = centres_px * array.pixels_per_sub + (array.pixels_per_sub - 1) / 2.0
    disp = array.slope_to_displacement(measured)
    disp_true = array.slope_to_displacement(truth)
    resid = array.slope_to_displacement(measured - truth)
    resid_corr = array.slope_to_displacement(measured_corr - truth)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.9))

    ax = axes[0]
    ax.imshow(frame, origin="upper", cmap="inferno", vmin=0.0,
              vmax=np.percentile(frame, 99.9))
    for k in range(1, array.n_lenslets):
        ax.axhline(k * array.pixels_per_sub - 0.5, color="w", lw=0.35, alpha=0.35)
        ax.axvline(k * array.pixels_per_sub - 0.5, color="w", lw=0.35, alpha=0.35)
    scale = 4.0
    ax.quiver(
        centres_px[:, 0], centres_px[:, 1], disp[:, 0] * scale, disp[:, 1] * scale,
        color="cyan", angles="xy", scale_units="xy", scale=1.0, width=0.004,
        headwidth=3.5, label=f"measured slope (x{scale:.0f})",
    )
    ax.set_title(
        f"Shack-Hartmann frame, {array.n_valid} illuminated subapertures\n"
        f"{PHOTONS:.0f} e-/subap, B = {BACKGROUND:.0f} e-/px, R = {READ_NOISE:.0f} e- RMS"
    )
    ax.set_xlabel("detector column [px]")
    ax.set_ylabel("detector row [px]")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    ax = axes[1]
    ax.quiver(
        centres_px[:, 0], centres_px[:, 1], disp_true[:, 0] * scale, disp_true[:, 1] * scale,
        color="0.55", angles="xy", scale_units="xy", scale=1.0, width=0.005,
        label=f"true slope (x{scale:.0f})",
    )
    err_scale = 200.0
    ax.quiver(
        centres_px[:, 0], centres_px[:, 1],
        resid[:, 0] * err_scale, resid[:, 1] * err_scale,
        color="tab:red", angles="xy", scale_units="xy", scale=1.0, width=0.003,
        label=f"CoG residual (x{err_scale:.0f})",
    )
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("detector column [px]")
    ax.set_ylabel("detector row [px]")
    rms_cog = float(np.sqrt(np.mean(resid**2)))
    rms_corr = float(np.sqrt(np.mean(resid_corr**2)))
    ax.set_title(
        f"True slope field and CoG residual (x{err_scale:.0f})\n"
        f"residual RMS: CoG {rms_cog:.4f} px, correlation {rms_corr:.4f} px"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Global tilt (gx = 8.0e-4, gy = -4.0e-4 rad) plus defocus (c = 0.35 /m)",
        fontsize=11,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
