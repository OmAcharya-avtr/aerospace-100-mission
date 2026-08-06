"""Example: Kolmogorov phase screen and the intensity speckle it produces.

Left: one FFT-synthesised Kolmogorov phase screen (rad). Right: intensity
of an initially uniform plane wave after split-step propagation through
8 such screens over a 2 km path (weak-fluctuation speckle).

Saves ../screenshots/phase_screen_speckle.png. Runtime: ~5 s.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from scintinet import angular_spectrum_propagate, kolmogorov_phase_screen  # noqa: E402

CN2 = 1e-15  # m^(-2/3)
LAM = 1.55e-6  # m
LENGTH = 2000.0  # m
N = 256
WIDTH = 0.5  # m
N_SCREENS = 8
SEED = 12


def main() -> None:
    dx = WIDTH / N
    dz = LENGTH / N_SCREENS
    rng = np.random.default_rng(SEED)

    first_screen = None
    u = np.ones((N, N), dtype=complex)
    for i in range(N_SCREENS):
        u = angular_spectrum_propagate(u, LAM, dx, dz / 2.0)
        screen = kolmogorov_phase_screen(rng, N, dx, CN2 * dz, LAM)
        if i == 0:
            first_screen = screen
        u = u * np.exp(1j * screen)
        u = angular_spectrum_propagate(u, LAM, dx, dz / 2.0)
    intensity = np.abs(u) ** 2

    extent = (0.0, WIDTH * 100.0, 0.0, WIDTH * 100.0)  # cm
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    im0 = axes[0].imshow(first_screen, cmap="RdBu_r", extent=extent, origin="lower")
    axes[0].set_title(f"Kolmogorov phase screen (Cn$^2\\,\\Delta z$={CN2 * dz:.1e} m$^{{1/3}}$)")
    axes[0].set_xlabel("x [cm]")
    axes[0].set_ylabel("y [cm]")
    fig.colorbar(im0, ax=axes[0], label="phase [rad]")
    im1 = axes[1].imshow(intensity, cmap="inferno", extent=extent, origin="lower")
    axes[1].set_title(f"Intensity after {LENGTH:.0f} m, {N_SCREENS} screens "
                      f"($\\sigma_I^2$={intensity.var() / intensity.mean() ** 2:.3f})")
    axes[1].set_xlabel("x [cm]")
    axes[1].set_ylabel("y [cm]")
    fig.colorbar(im1, ax=axes[1], label="I / I$_0$")
    fig.tight_layout()
    out = HERE.parent / "screenshots" / "phase_screen_speckle.png"
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
