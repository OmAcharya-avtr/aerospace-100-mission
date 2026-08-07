"""Slope error versus photon flux: classical baselines vs the learned estimator.

Run from products/P018/:

    PYTHONPATH=src python examples/slope_error_vs_flux.py     # ~55 s

Writes ../screenshots/slope_error_vs_flux.png.

This is a *reduced* version of validation section 5 (smaller training set and
ensemble) so that the example runs quickly. The full, characterized numbers are
in validation/VALIDATION.md; expect this script to reproduce the qualitative
picture — including the crossover — but not the exact figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from shacksim import (  # noqa: E402
    LensletArray,
    MLSlopeEstimator,
    cog_displacement,
    cog_noise_sigma,
    correlation_displacement,
    generate_subaperture_dataset,
    reference_template,
)

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "slope_error_vs_flux.png"

BACKGROUND = 1.0
READ_NOISE = 3.0
PHOTONS = (30.0, 50.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
THRESHOLDS = {30.0: 4.0, 50.0: 7.0, 100.0: 10.0, 300.0: 13.0,
              1000.0: 13.0, 3000.0: 10.0, 10000.0: 10.0}
N_TEST = 800


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def main() -> None:
    array = LensletArray()
    train_x, train_y = generate_subaperture_dataset(
        array, 6000, photons=(30.0, 30000.0), background=BACKGROUND, read_noise=READ_NOISE,
        elongation=(1.0, 3.0), seed=100,
    )
    model = MLSlopeEstimator(
        array, n_estimators=3, hidden_layer_sizes=(96, 48), random_state=0
    ).fit(train_x, train_y)
    template = reference_template(array)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    for ax, elong in zip(axes, (1.0, 3.0)):
        cog, corr, ml, spread = [], [], [], []
        for n_ph in PHOTONS:
            stamps, slopes = generate_subaperture_dataset(
                array, N_TEST, photons=n_ph, background=BACKGROUND, read_noise=READ_NOISE,
                elongation=elong, seed=int(9000 + n_ph + 7 * elong),
            )
            d_true = array.slope_to_displacement(slopes)
            cog.append(rms(cog_displacement(stamps, threshold=THRESHOLDS[n_ph]) - d_true))
            corr.append(rms(correlation_displacement(stamps, template) - d_true))
            pred, std = model.predict(stamps, return_std=True)
            ml.append(rms(array.slope_to_displacement(pred) - d_true))
            spread.append(float(np.mean(array.slope_to_displacement(std))))

        theory = [
            cog_noise_sigma(array, n, BACKGROUND, READ_NOISE, elongation=elong,
                            displacement_px=0.6 * array.pixels_per_sub / 2 / np.sqrt(3))
            / array.pixel_angle
            for n in PHOTONS
        ]
        ax.loglog(PHOTONS, cog, "o-", label="thresholded CoG")
        ax.loglog(PHOTONS, corr, "^-", label="correlation")
        ax.loglog(PHOTONS, ml, "s-", color="tab:green", label="ML ensemble")
        ax.fill_between(PHOTONS, np.array(ml) - np.array(spread),
                        np.array(ml) + np.array(spread), color="tab:green", alpha=0.18,
                        label="ML ensemble spread")
        ax.loglog(PHOTONS, theory, "k--", lw=1.0, label="linear-CoG noise theory")
        better = np.array(ml) < np.array(cog)
        if better.any() and not better.all():
            idx = int(np.max(np.where(better)))
            cross = np.sqrt(PHOTONS[idx] * PHOTONS[idx + 1])
            ax.axvline(cross, color="0.4", ls=":", lw=1.2)
            ax.annotate(f"crossover\n~{cross:.0f} e-", xy=(cross, max(cog) * 0.6),
                        fontsize=8, ha="center", color="0.25")
        ax.set_xlabel("photons per subaperture [e-]")
        ax.set_title(f"elongation {elong:.0f}x")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("RMS slope error [px of spot displacement]")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Slope-error vs flux (reduced run: 6000 training stamps, 3 members). "
        f"B = {BACKGROUND:.0f} e-/px, R = {READ_NOISE:.0f} e- RMS",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
