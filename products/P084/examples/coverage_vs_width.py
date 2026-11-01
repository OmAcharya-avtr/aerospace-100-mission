"""Achieved coverage against width: the plot this product exists to make.

What to notice: the learned conditional quantile sits furthest left -- the
narrowest limit in the figure -- and furthest from the declared coverage line.
Every limit that reaches the line is wider. A point to the left of the line is
not a better answer; it is a bound that does not hold.

The repetition count here is lower than in ``validation/`` so the example runs
in seconds; the numbers quoted in README.md come from the validation script,
not from this figure.
"""

from __future__ import annotations

import numpy as np

from _common import METHOD_COLOURS, METHOD_MARKERS, REFUSED, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from transfergap import synthetic  # noqa: E402
from transfergap.bounds import normal_tolerance_upper, wilks_tolerance_upper  # noqa: E402
from transfergap.learned import ConformalisedDiscrepancyModel, DiscrepancyModel  # noqa: E402
from transfergap.resample import split_conformal_upper  # noqa: E402
from transfergap.scoring import score_prediction_bounds  # noqa: E402

CONFIDENCE = 0.95
PROPORTION = 0.95
SIZES = (15, 25, 40, 60)
REPETITIONS = 40
POOL = 2000
R = synthetic.BOUND_RANGE_KNM


def main() -> int:
    style()
    pool = synthetic.generate(POOL, "random", seed=880000)

    def draw(n: int, seed: int) -> synthetic.GapSample:
        return synthetic.generate(n, "random", seed=seed)

    methods = {
        "wilks_tolerance": lambda s, _c: wilks_tolerance_upper(
            s.gaps, R, PROPORTION, CONFIDENCE
        ).upper,
        "split_conformal": lambda s, _c: split_conformal_upper(s.gaps, R, PROPORTION).upper,
        "normal_tolerance": lambda s, _c: normal_tolerance_upper(
            s.gaps, R, PROPORTION, CONFIDENCE
        ).upper,
        "learned_quantile": lambda s, c: DiscrepancyModel(
            upper_level=PROPORTION, lower_level=1.0 - PROPORTION, seed=990000,
            fit_full=False,
        ).fit(s.features, s.gaps).predict_upper(c),
        "learned_quantile_conformal": lambda s, c: ConformalisedDiscrepancyModel(
            confidence=PROPORTION, seed=990000, fit_full=False
        ).fit(s.features, s.gaps).predict_upper(c),
    }
    flags = {
        "wilks_tolerance": True, "split_conformal": True, "normal_tolerance": False,
        "learned_quantile": False, "learned_quantile_conformal": True,
    }

    scores = []
    for n in SIZES:
        scores += score_prediction_bounds(
            methods, draw, pool.features, pool.gaps, n, REPETITIONS, 770000 + 29 * n,
            PROPORTION, flags,
        )

    figure, axis = plt.subplots(figsize=(7.6, 5.2))
    axis.axvline(PROPORTION, color=REFUSED, lw=1.4, ls="--")
    axis.axvspan(0.80, PROPORTION, color=REFUSED, alpha=0.07)
    sizes_by_marker = {15: 40.0, 25: 70.0, 40: 110.0, 60: 160.0}
    seen = set()
    for score in scores:
        if np.isnan(score.coverage):
            continue
        label = score.method if score.method not in seen else None
        seen.add(score.method)
        axis.scatter(
            score.coverage, score.mean_upper,
            s=sizes_by_marker[score.n], marker=METHOD_MARKERS[score.method],
            color=METHOD_COLOURS[score.method], edgecolor="white", linewidth=0.6,
            label=label, zorder=3,
        )
        axis.annotate(
            f"{score.n}", (score.coverage, score.mean_upper),
            textcoords="offset points", xytext=(7, -3), fontsize=7, color="#555555",
        )
    axis.set_xlabel(f"achieved coverage of a future field measurement (declared {PROPORTION:g})")
    axis.set_ylabel(f"mean width of the limit [{synthetic.UNIT}]")
    axis.set_title(
        "coverage is the scoring rule; width is reported beside it, never instead\n"
        f"{REPETITIONS} campaigns per point, {POOL}-point held-out pool, label = n",
        fontsize=10,
    )
    axis.text(0.862, axis.get_ylim()[1] * 0.97, "limits here do not hold", fontsize=8,
              color=REFUSED, va="top")
    axis.legend(loc="lower left", fontsize=8)
    figure.subplots_adjust(top=0.88)
    path = save(figure, "coverage_vs_width")
    print(f"written: {path}")
    for score in scores:
        if not np.isnan(score.coverage):
            print(f"  {score.method:<28s} n={score.n:<3d} coverage {score.coverage:.4f} "
                  f"width {score.mean_upper:.3f} {synthetic.UNIT}")
        else:
            print(f"  {score.method:<28s} n={score.n:<3d} no limit exists at this n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
