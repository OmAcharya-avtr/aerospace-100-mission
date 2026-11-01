"""What the learned model knows that no marginal bound can: where the twin is worst.

What to notice: the model's conditional band rises through the transonic
corner, tracking the true conditional mean gap of equation (22), while the
distribution-free limits are flat lines that cannot vary with the operating
condition at all. The model's band is also BELOW the true conditional 0.95
quantile over most of the envelope, which is the under-coverage measured in
validation/validate_learned_model_output.txt shown as a picture.
"""

from __future__ import annotations

import numpy as np

from _common import (
    DISTRIBUTION_FREE,
    LEARNED,
    LEARNED_CONFORMAL,
    NEUTRAL,
    TRUTH,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from transfergap import synthetic  # noqa: E402
from transfergap.bounds import normal_tolerance_upper  # noqa: E402
from transfergap.learned import ConformalisedDiscrepancyModel, DiscrepancyModel  # noqa: E402
from transfergap.resample import split_conformal_upper  # noqa: E402

N = 60
PROPORTION = 0.95
CONFIDENCE = 0.95
R = synthetic.BOUND_RANGE_KNM


def main() -> int:
    style()
    sample = synthetic.generate(N, "random", seed=202)
    model = DiscrepancyModel(
        upper_level=PROPORTION, lower_level=1.0 - PROPORTION, seed=990000
    ).fit(sample.features, sample.gaps)
    calibrated = ConformalisedDiscrepancyModel(confidence=PROPORTION, seed=990000)
    calibrated.fit(sample.features, sample.gaps)

    # A Mach sweep through the middle of the envelope, every other condition
    # held at the midpoint of its declared range.
    mach = np.linspace(synthetic.ENVELOPE[0][0], synthetic.ENVELOPE[0][1], 200)
    held = np.array([0.5 * (low + high) for low, high in synthetic.ENVELOPE[1:]])
    grid = np.column_stack([mach, np.tile(held, (mach.size, 1))])

    prediction = model.predict_with_uncertainty(grid)
    band = calibrated.predict_upper(grid)
    truth_mean = synthetic.conditional_mean_gap(grid)

    # The TRUE conditional quantile at each condition, by direct Monte Carlo on
    # the generator's own equations (19)-(21). Available only because the data
    # is synthetic.
    rng = np.random.default_rng(5150)
    bias = synthetic._bias(grid)
    scale = synthetic._scale(grid)
    draws = np.abs(bias[:, None] + scale[:, None] * rng.standard_normal((mach.size, 4000)))
    true_quantile = np.quantile(np.minimum(draws, R), PROPORTION, axis=1)

    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    axis.fill_between(mach, prediction.lower, prediction.upper, color=LEARNED, alpha=0.16,
                      label=f"learned conditional interval, {1 - PROPORTION:g} to "
                            f"{PROPORTION:g}")
    axis.plot(mach, prediction.point, color=LEARNED, lw=1.8,
              label="learned conditional median")
    axis.plot(mach, prediction.upper, color=LEARNED, lw=1.3, ls="--",
              label=f"learned conditional {PROPORTION:g} quantile")
    axis.plot(mach, band, color=LEARNED_CONFORMAL, lw=1.6, ls="-.",
              label=f"conformalised, offset {calibrated.offset:+.2f} "
                    f"{synthetic.UNIT}")
    axis.plot(mach, truth_mean, color=TRUTH, lw=1.8,
              label="TRUE conditional mean gap, equation (22)")
    axis.plot(mach, true_quantile, color=TRUTH, lw=1.4, ls=":",
              label=f"TRUE conditional {PROPORTION:g} quantile")

    conformal = split_conformal_upper(sample.gaps, R, PROPORTION).upper
    normal = normal_tolerance_upper(sample.gaps, R, PROPORTION, CONFIDENCE).upper
    axis.axhline(conformal, color=DISTRIBUTION_FREE, lw=1.5,
                 label=f"split conformal, marginal: {conformal:.2f} {synthetic.UNIT}")
    axis.axhline(normal, color=NEUTRAL, lw=1.2, ls="--",
                 label=f"normal tolerance, marginal: {normal:.2f} {synthetic.UNIT}")

    axis.set_xlabel("Mach number (every other operating condition at its envelope midpoint)")
    axis.set_ylabel(f"sim-to-field discrepancy [{synthetic.UNIT}]")
    axis.set_title(
        f"the learned model varies with the operating condition; a marginal bound "
        f"cannot\nfitted on n = {N} paired measurements, seed 202",
        fontsize=10,
    )
    axis.legend(loc="upper left", fontsize=7.5, ncol=2)
    axis.set_ylim(0.0, max(float(np.max(true_quantile)), normal, conformal) * 1.45)
    figure.subplots_adjust(top=0.87)
    path = save(figure, "learned_discrepancy")
    print(f"written: {path}")
    print(f"conformal offset            {calibrated.offset:+.4f} {synthetic.UNIT}")
    print(f"learned band, mean over grid {float(np.mean(prediction.upper)):.4f}")
    print(f"true 0.95 quantile, mean     {float(np.mean(true_quantile)):.4f}")
    print(f"fraction of the Mach sweep where the learned band is BELOW the truth: "
          f"{float(np.mean(prediction.upper < true_quantile)):.4f}")
    print(f"quantile crossing rate on the grid: {prediction.crossing_rate:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
