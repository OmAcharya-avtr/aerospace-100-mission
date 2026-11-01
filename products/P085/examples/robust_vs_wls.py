"""Least squares, Huber and RANSAC against the contamination fraction.

What to notice: least squares degrades from the first failed acquisition.
RANSAC is the best of the three until about fifteen per cent contamination and
then collapses, because the probability that a 27-observation subset is clean
falls below what 200 trials can find. Huber degrades gradually instead.
"""

from __future__ import annotations

import numpy as np

from _common import HUBER, LEAST, NEUTRAL, RANSAC, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from pointmodel import robust, synthetic, terms, wls  # noqa: E402

FRACTIONS = np.array([0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
OBSERVATIONS = 300
SIGMA = 2.0
OUTLIER = 300.0
REPEATS = 6
SUBSET = 3 * terms.N_TERMS
TRIALS = 200


def main() -> int:
    style()
    curves = {"least squares": [], "huber": [], "ransac": []}
    for fraction in FRACTIONS:
        batch = {key: [] for key in curves}
        for repeat in range(REPEATS):
            residuals = synthetic.generate(
                "full_sky",
                OBSERVATIONS,
                sigma_arcsec=SIGMA,
                outlier_fraction=float(fraction),
                outlier_arcsec=OUTLIER,
                seed=20260101 + 1000 * repeat,
            )
            truth = residuals.coefficients
            batch["least squares"].append(
                float(
                    np.max(
                        np.abs(
                            wls.fit(
                                residuals.design, residuals.observed, SIGMA
                            ).coefficients
                            - truth
                        )
                    )
                )
            )
            batch["huber"].append(
                float(
                    np.max(
                        np.abs(
                            robust.huber_fit(
                                residuals.design, residuals.observed, SIGMA
                            ).coefficients
                            - truth
                        )
                    )
                )
            )
            try:
                estimate = robust.ransac_fit(
                    residuals.design, residuals.observed, SIGMA, seed=4242
                ).coefficients
                batch["ransac"].append(float(np.max(np.abs(estimate - truth))))
            except ValueError:
                batch["ransac"].append(np.nan)
        for key in curves:
            curves[key].append(float(np.nanmean(batch[key])))

    figure, axes = plt.subplots(
        2, 1, figsize=(8.0, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    for key, colour in (
        ("least squares", LEAST),
        ("huber", HUBER),
        ("ransac", RANSAC),
    ):
        axes[0].semilogy(
            100.0 * FRACTIONS, curves[key], marker="o", ms=4, lw=1.6, color=colour, label=key
        )
    axes[0].axhline(SIGMA, color=NEUTRAL, ls="--", lw=1.0)
    axes[0].text(30.0, SIGMA * 1.1, "one declared sigma", fontsize=8, color=NEUTRAL)
    axes[0].set_ylabel("worst |coefficient error| against the truth, arcsec")
    axes[0].set_title(
        "three estimators against acquisition-failure contamination\n"
        f"{OBSERVATIONS} observations, {REPEATS} seeds per point, failures at "
        f"{OUTLIER:g} arcsec"
    )
    axes[0].legend(loc="lower right")

    purity = TRIALS * (1.0 - FRACTIONS) ** SUBSET
    axes[1].semilogy(100.0 * FRACTIONS, purity, color=RANSAC, lw=1.6, marker="s", ms=4)
    axes[1].axhline(1.0, color=NEUTRAL, ls="--", lw=1.0)
    axes[1].text(2.0, 1.3, "one expected clean subset", fontsize=8, color=NEUTRAL)
    axes[1].set_ylabel(f"expected clean subsets\nin {TRIALS} trials")
    axes[1].set_xlabel("contamination, per cent of acquisitions")
    figure.subplots_adjust(hspace=0.12)
    path = save(figure, "robust_vs_wls")
    print(f"written: {path}")
    print("")
    print("contamination %   least squares      huber     ransac   expected clean subsets")
    for index, fraction in enumerate(FRACTIONS):
        print(
            f"{100 * fraction:14.0f}   {curves['least squares'][index]:13.4f}   "
            f"{curves['huber'][index]:8.4f}   {curves['ransac'][index]:8.4f}   "
            f"{purity[index]:20.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
