"""Kolmogorov phase screens: appearance, structure function and modal content.

Produces ``screenshots/phase_screen_gallery.png``.

Run from products/P011:
    PYTHONPATH=src python examples/phase_screen_gallery.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from waveforge.atmosphere import (  # noqa: E402
    band_limited_structure_function,
    phase_screen,
    structure_function,
)
from waveforge.pupil import PupilGrid, piston_removed  # noqa: E402
from waveforge.statistics import phase_structure_function, zernike_variance  # noqa: E402
from waveforge.zernike import zernike_basis  # noqa: E402

OUTPUT = Path(__file__).resolve().parent.parent / "screenshots" / "phase_screen_gallery.png"
R0 = 0.10
PIXEL_SCALE = 0.02
N_SCREEN = 128
N_ENSEMBLE = 40


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    screen = phase_screen(N_SCREEN, PIXEL_SCALE, R0, n_subharmonics=4, rng=2026)
    extent = [0.0, N_SCREEN * PIXEL_SCALE, 0.0, N_SCREEN * PIXEL_SCALE]
    image = axes[0].imshow(screen, origin="lower", extent=extent, cmap="RdBu_r")
    axes[0].set_title(f"Kolmogorov screen, $r_0$ = {R0} m\n(4 subharmonic levels, seed 2026)")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    fig.colorbar(image, ax=axes[0], label="phase [rad]")

    lags = np.arange(1, N_SCREEN // 8 + 1)
    separations = lags * PIXEL_SCALE
    continuous = phase_structure_function(separations, R0)
    band = band_limited_structure_function(N_SCREEN, PIXEL_SCALE, R0, lags)
    axes[1].loglog(separations, continuous, "k-", lw=2, label=r"$6.884\,(r/r_0)^{5/3}$")
    axes[1].loglog(separations, band, "k--", lw=1.4, label="exact discrete spectrum")
    for levels, colour in ((0, "tab:red"), (2, "tab:orange"), (4, "tab:green")):
        acc = np.zeros(len(lags))
        for k in range(N_ENSEMBLE):
            acc += structure_function(
                phase_screen(N_SCREEN, PIXEL_SCALE, R0, n_subharmonics=levels, rng=900 + k),
                max_lag=len(lags),
            )[1]
        axes[1].loglog(
            separations, acc / N_ENSEMBLE, "o-", ms=3.5, color=colour,
            label=f"measured, {levels} subharmonics",
        )
    axes[1].set_xlabel("separation r [m]")
    axes[1].set_ylabel(r"$D_\varphi(r)$ [rad$^2$]")
    axes[1].set_title("Structure function\n(measured vs theory vs band limit)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7.5, loc="upper left")

    pupil = PupilGrid(64, 0.5)
    rho, theta = pupil.polar()
    basis = zernike_basis(21, rho, theta, mask=pupil.mask, include_piston=False)
    pinv = np.linalg.pinv(basis.T)
    accumulator = np.zeros(20)
    n_modal = 120
    for k in range(n_modal):
        raw = phase_screen(512, pupil.sample_spacing_m, R0, n_subharmonics=6, rng=5000 + 3 * k)
        window = piston_removed(raw[:64, :64], pupil.mask)
        accumulator += (pinv @ window[pupil.mask]) ** 2
    accumulator /= n_modal
    j_values = np.arange(2, 22)
    analytic = np.array([zernike_variance(int(j), 5.0) for j in j_values])
    axes[2].semilogy(j_values, analytic, "k-o", ms=4, label="Noll analytic, D/$r_0$ = 5")
    axes[2].semilogy(j_values, accumulator, "s", ms=5, color="tab:blue", label="measured screens")
    axes[2].set_xlabel("Noll index j")
    axes[2].set_ylabel(r"$\langle a_j^2\rangle$ [rad$^2$]")
    axes[2].set_title(f"Modal variance spectrum\n({n_modal} screens, 64-pixel pupil)")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend(fontsize=8)
    axes[2].set_xticks([2, 5, 8, 11, 14, 17, 20])

    fig.suptitle(
        "WaveForge — Kolmogorov phase screens (research-grade, not flight-qualified)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
