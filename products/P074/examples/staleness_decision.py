"""The decision the product exists for: how old can this element set be.

What to notice: the two horizons differ by a factor of several. The median
horizon is the optimistic answer a point prediction gives, and half the time
the drift is already worse than it says. The conservative horizon is the upper
end of a prediction interval whose coverage has been measured on held-out
objects, and the measured figure is printed on the figure because a
conservative horizon drawn from an interval that under-covers is not
conservative.

The three objects are real members of the committed population, held out of
the fit. They are **not** hand-assembled feature vectors, and that is
deliberate: the feature set is strongly collinear and the model's answer off
that manifold is an extrapolation of a collinear fit rather than a prediction.
The leverage printed below is the diagnostic, and it is the reason the
documentation says to build features with ``features_from_satrec`` from an
actual element set.
"""

from __future__ import annotations

import numpy as np

from _common import SERIES, bootstrap, load_dataset, save, style

bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402

from orbitdrift import (  # noqa: E402
    AnalyticGrowthPredictor,
    ConformalPredictor,
    ObjectFeatures,
    design_matrix,
    split_by_object,
    staleness_verdict,
)
from orbitdrift.calibration import interval_metrics  # noqa: E402
from orbitdrift.features import FEATURE_NAMES  # noqa: E402

TOLERANCES_KM = (1.0, 5.0, 25.0, 100.0)


def main() -> None:
    table = load_dataset()
    train, test = split_by_object(table, test_fraction=0.3)
    design = table.design()
    drift = table.column("drift_km")
    objects = table.column("object_index")
    perigee = table.column("perigee_altitude_km")

    base = AnalyticGrowthPredictor(form="power_altitude")
    predictor = ConformalPredictor(base, coverage=0.90)
    predictor.fit(design[train], drift[train], groups=objects[train])
    lower, upper = predictor.predict_interval_km(design[test])
    metrics = interval_metrics(lower, upper, drift[test], nominal=0.90)
    fitted = predictor.base

    # Three real held-out objects, chosen by altitude. Their features come from
    # the dataset, which built them from actual element sets.
    held_out = np.unique(objects[test])
    altitudes = np.array([perigee[objects == value][0] for value in held_out])
    cases = []
    for target in (420.0, 700.0, 1300.0):
        value = held_out[int(np.argmin(np.abs(altitudes - target)))]
        row = design[objects == value][0, :-1]
        cases.append((value, ObjectFeatures(**dict(zip(FEATURE_NAMES, row, strict=True)))))

    ages = np.linspace(0.05, 14.0, 280)
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), sharey=True)

    rows = []
    for position, (object_index, features) in enumerate(cases):
        axis = axes[position]
        static = np.tile(features.as_array(), (ages.size, 1))
        matrix, _ = design_matrix(static, ages)
        median = predictor.predict_median_km(matrix)
        low, high = predictor.predict_interval_km(matrix)
        leverage = float(fitted.leverage(matrix)[0])
        axis.fill_between(
            ages, low, high, color=SERIES[0], alpha=0.18, label="90 per cent interval"
        )
        axis.plot(ages, median, color=SERIES[0], linewidth=1.4, label="median prediction")

        mask = objects == object_index
        axis.plot(
            table.column("epoch_age_days")[mask],
            drift[mask],
            "k.",
            markersize=3.0,
            label="measured divergence",
        )
        for index, tolerance in enumerate(TOLERANCES_KM):
            verdict = staleness_verdict(
                predictor,
                features,
                tolerance,
                coverage=0.90,
                measured_coverage=metrics.coverage,
            )
            rows.append(
                (
                    features.perigee_altitude_km,
                    tolerance,
                    verdict.median_horizon_days,
                    verdict.conservative_horizon_days,
                    leverage,
                )
            )
            colour = SERIES[(index + 1) % len(SERIES)]
            axis.axhline(tolerance, color=colour, linestyle=":", linewidth=0.9)
            if np.isfinite(verdict.conservative_horizon_days):
                axis.plot(
                    [verdict.conservative_horizon_days], [tolerance], "v",
                    color=colour, markersize=6,
                )
            if np.isfinite(verdict.median_horizon_days):
                axis.plot(
                    [verdict.median_horizon_days], [tolerance], "o",
                    color=colour, markersize=4, markerfacecolor="white",
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("epoch age, days")
        axis.set_title(
            f"object {int(object_index)}, perigee {features.perigee_altitude_km:.0f} km\n"
            f"leverage {leverage:.2f} (training max {fitted.training_leverage_max_:.2f})",
            fontsize=8.5,
        )
        if position == 0:
            axis.set_ylabel("model divergence, km")
            axis.legend(fontsize=7.5, loc="lower right")

    figure.suptitle(
        "orbitdrift: staleness horizons on three held-out objects. Triangles are the "
        f"conservative horizon, open circles the median. Measured coverage "
        f"{metrics.coverage:.4f} against 0.9000 nominal.",
        fontsize=9.5,
    )
    figure.tight_layout()
    print(f"wrote {save(figure, 'staleness_decision')}")
    print(f"predictor                 : {predictor.name}")
    print(
        f"measured coverage         : {metrics.coverage:.4f} "
        f"[{metrics.coverage_ci[0]:.4f}, {metrics.coverage_ci[1]:.4f}] on "
        f"{metrics.n} held-out rows"
    )
    print(f"training leverage maximum : {fitted.training_leverage_max_:.4f}")
    print("  perigee km   tolerance km   median horizon d   conservative horizon d   leverage")
    for altitude, tolerance, median_days, conservative_days, leverage in rows:
        print(
            f"  {altitude:10.1f}   {tolerance:12.1f}   {median_days:16.4f}   "
            f"{conservative_days:22.4f}   {leverage:8.2f}"
        )


if __name__ == "__main__":
    main()
