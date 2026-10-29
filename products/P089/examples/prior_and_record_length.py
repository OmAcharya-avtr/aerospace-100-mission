"""What the regularised baseline's win is made of, and what it costs.

Left: the RMSE of all four estimators against record length, on a fixed
well-excited configuration. Middle: the MAP advantage over the unbiased solution
as a ratio, which collapses toward 1 as the record gains information. Right: the
same estimator with a mis-specified prior, which loses to the unbiased solution
at every length. The analytic win of the middle panel is a statement about
knowing the population, not a free lunch.
"""

from __future__ import annotations

import sys
from pathlib import Path

from _common import BATCH, BOUND, LEARNED, MAP, TRIVIAL, bootstrap, save, style

bootstrap()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
style()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aligncal import batch, benchmark, dataset, features, learned, observability  # noqa: E402

from _bootstrap import PRIOR_SIGMA_RAD, fit_estimators  # noqa: E402

LENGTHS = (150, 300, 600, 1200, 2400, 4800, 9600)
RECORDS = 40


def main() -> str:
    models = fit_estimators(count=1500, samples=150, seed=20261029)
    plain_curve = []
    map_curve = []
    tight_curve = []
    loose_curve = []
    learned_curve = []
    trivial_curve = []
    bound_curve = []

    for length in LENGTHS:
        population = [
            dataset.generate(
                profile="three_axis",
                samples=length,
                amplitude=0.05,
                noise_sigma=2.0e-4,
                theta_max_rad=3.0e-3,
                seed=70120261 + 3000 + index,
            )
            for index in range(RECORDS)
        ]
        matrix, targets, _ = features.build_population(population)
        sensors = []
        errors = {"plain": [], "map": [], "tight": [], "loose": []}
        bounds = []
        for generated in population:
            verdict = observability.assess(generated.record)
            sensor = verdict["secondary_1"]
            sensors.append(sensor)
            truth = generated.truth["secondary_1"]
            values = np.array([d.eigenvalue for d in sensor.degrees_of_freedom])
            bounds.append(float(np.sqrt(np.sum(1.0 / values[sensor.observable_mask]))))
            errors["plain"].append(
                batch.solve(generated.record, verdict, "secondary_1").theta_rad - truth
            )
            for key, prior in (
                ("map", PRIOR_SIGMA_RAD),
                ("tight", PRIOR_SIGMA_RAD / 10.0),
                ("loose", PRIOR_SIGMA_RAD * 10.0),
            ):
                errors[key].append(
                    batch.solve_map(
                        generated.record, verdict, "secondary_1", prior_sigma_rad=prior
                    ).theta_rad
                    - truth
                )
        for key, curve in (
            ("plain", plain_curve),
            ("map", map_curve),
            ("tight", tight_curve),
            ("loose", loose_curve),
        ):
            curve.append(
                benchmark.score(key, np.array(errors[key]), sensors).rmse_observable_mrad
            )
        bound_curve.append(float(np.median(bounds)) * 1e3)
        best = min(
            benchmark.score(
                name, models[name].predict(matrix).theta_rad - targets, sensors
            ).rmse_observable_mrad
            for name in learned.CONFIGURATIONS
        )
        learned_curve.append(best)
        trivial_curve.append(
            benchmark.score(
                "trivial", models["trivial"].predict(matrix).theta_rad - targets, sensors
            ).rmse_observable_mrad
        )

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    figure.subplots_adjust(wspace=0.32)

    left = axes[0]
    left.loglog(LENGTHS, bound_curve, color=BOUND, linestyle="--", marker="o", markersize=4,
                label="Cramer-Rao one sigma, eq (12)")
    left.loglog(LENGTHS, plain_curve, color=BATCH, marker="s", markersize=4,
                label="batch, eq (14)-(16)")
    left.loglog(LENGTHS, map_curve, color=MAP, marker="^", markersize=4,
                label="batch MAP, eq (18), true prior")
    left.loglog(LENGTHS, learned_curve, color=LEARNED, marker="D", markersize=4,
                label="best learned configuration")
    left.loglog(LENGTHS, trivial_curve, color=TRIVIAL, marker="v", markersize=4,
                label="trivial baseline")
    left.set_xticks(list(LENGTHS))
    left.set_xticklabels([str(value) for value in LENGTHS], fontsize=7)
    left.minorticks_off()
    left.set_xlabel("record length, samples")
    left.set_ylabel("RMSE on the observable subspace, mrad")
    left.set_title(
        f"fixed three_axis excitation, {RECORDS} records per point\n"
        "the learned estimator was trained at 150 samples"
    )
    left.legend(fontsize=7, loc="lower left")

    middle = axes[1]
    ratio = np.array(map_curve) / np.array(plain_curve)
    middle.semilogx(LENGTHS, ratio, color=MAP, marker="^", markersize=5)
    middle.axhline(1.0, color=BATCH, linestyle="--", linewidth=1.2,
                   label="no advantage over the unbiased solution")
    for length, value in zip(LENGTHS, ratio, strict=True):
        middle.annotate(f"{value:.3f}", (length, value), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7)
    middle.set_xticks(list(LENGTHS))
    middle.set_xticklabels([str(value) for value in LENGTHS], fontsize=7)
    middle.minorticks_off()
    middle.set_ylim(0.5, 1.12)
    middle.set_xlabel("record length, samples")
    middle.set_ylabel("MAP RMSE / batch RMSE")
    middle.set_title(
        "the MAP advantage collapses as the record\ngains information. What is left at the\n"
        "long end is this measurement's own noise."
    )
    middle.legend(fontsize=8, loc="lower right")

    right = axes[2]
    right.loglog(LENGTHS, plain_curve, color=BATCH, marker="s", markersize=4,
                 label="batch, eq (14)-(16)")
    right.loglog(LENGTHS, map_curve, color=MAP, marker="^", markersize=4,
                 label="MAP, true prior 1.0 mrad")
    right.loglog(LENGTHS, tight_curve, color="#b0344f", marker="x", markersize=5,
                 label="MAP, prior 10x too tight")
    right.loglog(LENGTHS, loose_curve, color="#6f3f8f", marker="+", markersize=6,
                 label="MAP, prior 10x too loose")
    right.set_xticks(list(LENGTHS))
    right.set_xticklabels([str(value) for value in LENGTHS], fontsize=7)
    right.minorticks_off()
    right.set_xlabel("record length, samples")
    right.set_ylabel("RMSE on the observable subspace, mrad")
    right.set_title(
        "mis-specify the prior and the analytic win\nbecomes an analytic loss"
    )
    right.legend(fontsize=7, loc="lower left")

    figure.suptitle(
        "What the regularised baseline's win is made of. It is bought with a prior, "
        "and the price is visible on the right.",
        fontsize=11,
    )
    figure.subplots_adjust(top=0.82)
    return save(figure, "prior_and_record_length")


if __name__ == "__main__":
    print(main())
