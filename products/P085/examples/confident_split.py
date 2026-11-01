"""The confident split: a repeatable, wrong, tightly quoted coefficient.

What to notice: on the single-azimuth geometry the least-squares estimate of
the azimuth-axis tilt is repeatable across noise realisations to about two
arcseconds and sits sixteen arcseconds from the truth. A reader given the
estimate and its error bar has no way to see that.
"""

from __future__ import annotations

import numpy as np

from _common import LEAST, REFUSED, TRUTH, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from pointmodel import identifiability, synthetic, terms, wls  # noqa: E402

OBSERVATIONS = 300
SIGMA = 2.0
TRIALS = 200
SHOWN = ("AN", "AW", "NPAE", "IE")


def run(name: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    estimates = np.empty((TRIALS, terms.N_TERMS))
    base = synthetic.generate(name, OBSERVATIONS, sigma_arcsec=SIGMA, seed=20260101)
    for trial in range(TRIALS):
        realisation = synthetic.generate(
            name,
            OBSERVATIONS,
            sigma_arcsec=SIGMA,
            seed=20260101,
            noise_seed=300_000 + trial,
        )
        estimates[trial] = wls.fit(
            realisation.design, realisation.observed, SIGMA
        ).coefficients
    result = wls.fit(base.design, base.observed, SIGMA)
    report = identifiability.analyse_fit(base.design, result)
    return estimates, base.coefficients, report.unidentifiable_terms


def main() -> int:
    style()
    figure, axes = plt.subplots(2, len(SHOWN), figsize=(12.0, 5.6), sharex="col")
    for column, name in enumerate(SHOWN):
        index = terms.TERM_INDEX[name]
        for row, geometry in enumerate(("full_sky", "single_az")):
            estimates, truth, refused = run(geometry)
            axis = axes[row, column]
            colour = REFUSED if name in refused else LEAST
            axis.hist(estimates[:, index], bins=24, color=colour, alpha=0.75)
            axis.axvline(truth[index], color=TRUTH, lw=1.6)
            bias = float(np.mean(estimates[:, index]) - truth[index])
            spread = float(np.std(estimates[:, index], ddof=1))
            axis.set_title(
                f"{geometry} {name}\nbias {bias:+.2f}\", sd {spread:.2f}\""
                + ("  REFUSED" if name in refused else ""),
                fontsize=9,
            )
            if row == 1:
                axis.set_xlabel("estimate, arcsec")
            if column == 0:
                axis.set_ylabel("realisations")
    figure.suptitle(
        "the same estimator on two geometries: the black line is the truth\n"
        f"{TRIALS} noise realisations, geometry fixed, sigma {SIGMA:g} arcsec",
        fontsize=11,
    )
    figure.subplots_adjust(top=0.82, hspace=0.42, wspace=0.26)
    path = save(figure, "confident_split")
    print(f"written: {path}")
    print("")
    for geometry in ("full_sky", "single_az"):
        estimates, truth, refused = run(geometry)
        print(f"{geometry}: refused {', '.join(refused) or 'nothing'}")
        for name in SHOWN:
            index = terms.TERM_INDEX[name]
            bias = float(np.mean(estimates[:, index]) - truth[index])
            spread = float(np.std(estimates[:, index], ddof=1))
            print(
                f"  {name:<4s} truth {truth[index]:8.3f}  bias {bias:+9.3f}  "
                f"sd {spread:8.3f}  |bias|/sd {abs(bias) / spread:7.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
