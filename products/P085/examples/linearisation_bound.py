"""The design matrix against the bound derived from a different mathematics.

What to notice: the measured error tracks the bound over five decades with a
ratio near 0.3, so the bound is neither violated nor vacuous; and a single
sign flip on one design-matrix column puts the reconstruction orders of
magnitude outside it.
"""

from __future__ import annotations

import numpy as np

from _common import BOUND, LEAST, NEUTRAL, REFUSED, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from pointmodel import geometry, terms  # noqa: E402

AMPLITUDES = np.array([1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0])
SAMPLES = 120


def main() -> int:
    style()
    rng = np.random.default_rng(20260101)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, SAMPLES)
    elevation = np.radians(rng.uniform(3.0, 88.0, SAMPLES))
    direction = rng.uniform(-1.0, 1.0, terms.N_TERMS)

    errors, bounds, corrupted = [], [], []
    cross_block, elev_block = terms.design_matrix(azimuth, elevation)
    flipped = cross_block.copy()
    flipped[:, terms.TERM_INDEX["NPAE"]] *= -1.0
    for amplitude in AMPLITUDES:
        coefficients = amplitude * direction
        error, bound = geometry.linearisation_error(azimuth, elevation, coefficients)
        errors.append(float(np.max(error)))
        bounds.append(float(np.max(bound)))
        worst = 0.0
        for position in range(SAMPLES):
            exact = geometry.beam_direction(
                float(azimuth[position]), float(elevation[position]), coefficients
            )
            nominal = geometry.sky_vector(
                float(azimuth[position]), float(elevation[position])
            )
            tangent_cross, tangent_elev = geometry.tangent_basis(
                float(azimuth[position]), float(elevation[position])
            )
            linear = (
                nominal
                + float(flipped[position] @ coefficients)
                * geometry.RAD_PER_ARCSEC
                * tangent_cross
                + float(elev_block[position] @ coefficients)
                * geometry.RAD_PER_ARCSEC
                * tangent_elev
            )
            worst = max(worst, float(np.linalg.norm(exact - linear)))
        corrupted.append(worst)

    figure, axes = plt.subplots(
        2, 1, figsize=(8.0, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    axes[0].loglog(AMPLITUDES, bounds, color=BOUND, lw=2.0, label="bound B(s), equation (15)")
    axes[0].loglog(
        AMPLITUDES, errors, color=LEAST, lw=1.6, marker="o", ms=4,
        label="measured error, correct design matrix",
    )
    axes[0].loglog(
        AMPLITUDES, corrupted, color=REFUSED, lw=1.6, ls="--", marker="s", ms=4,
        label="measured error, NPAE column sign flipped",
    )
    axes[0].set_ylabel("Euclidean error of the beam vector")
    axes[0].set_title(
        "the design matrix against a bound derived from the matrix-exponential\n"
        "Taylor remainder, which no part of the design matrix uses"
    )
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].semilogx(
        AMPLITUDES,
        np.array(errors) / np.array(bounds),
        color=LEAST,
        lw=1.6,
        marker="o",
        ms=4,
        label="correct",
    )
    axes[1].semilogx(
        AMPLITUDES,
        np.array(corrupted) / np.array(bounds),
        color=REFUSED,
        lw=1.6,
        ls="--",
        marker="s",
        ms=4,
        label="sign flipped",
    )
    axes[1].axhline(1.0, color=NEUTRAL, ls="--", lw=1.0)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("error / bound")
    axes[1].set_xlabel("misalignment amplitude, arcsec")
    axes[1].legend(loc="upper right", fontsize=8)
    figure.subplots_adjust(hspace=0.12)
    path = save(figure, "linearisation_bound")
    print(f"written: {path}")
    print("")
    print("amplitude arcsec        bound       error  error/bound   flipped  flipped/bound")
    for index, amplitude in enumerate(AMPLITUDES):
        print(
            f"{amplitude:16.0f}  {bounds[index]:11.4e}  {errors[index]:10.4e}  "
            f"{errors[index] / bounds[index]:11.4f}  {corrupted[index]:8.2e}  "
            f"{corrupted[index] / bounds[index]:13.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
