"""The learned estimator against both analytic baselines and a trivial one.

Left: per-component RMSE on the observable subspace, over a held-out
short-record population. Middle: the same split by excitation profile, so the
aggregate is not hiding a reversal. Right: the unobservable-subspace column,
which every estimator carries in full, including the one that refuses.
"""

from __future__ import annotations

import sys
from pathlib import Path

from _common import BATCH, LEARNED, MAP, TRIVIAL, bootstrap, save, style

bootstrap()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
style()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aligncal import batch, benchmark, dataset, features, learned, observability  # noqa: E402

from _bootstrap import PRIOR_SIGMA_RAD, fit_estimators  # noqa: E402

TRAIN = 1500
TEST = 400
SAMPLES = 150

COLOURS = {
    "batch (14)-(16)": BATCH,
    "batch MAP (18)": MAP,
    "learned forest": LEARNED,
    "learned forest_deep": "#7ab87a",
    "learned bagged_ridge": "#2f6f3f",
    "learned trivial": TRIVIAL,
}


def _tables(population, models):
    matrix, targets, profiles = features.build_population(population)
    sensors = []
    plain = []
    regularised = []
    for generated in population:
        verdict = observability.assess(generated.record)
        sensors.append(verdict["secondary_1"])
        truth = generated.truth["secondary_1"]
        plain.append(batch.solve(generated.record, verdict, "secondary_1").theta_rad - truth)
        regularised.append(
            batch.solve_map(
                generated.record, verdict, "secondary_1", prior_sigma_rad=PRIOR_SIGMA_RAD
            ).theta_rad
            - truth
        )
    tables = {
        "batch (14)-(16)": benchmark.score("b", np.array(plain), sensors),
        "batch MAP (18)": benchmark.score("m", np.array(regularised), sensors),
    }
    for name in list(learned.CONFIGURATIONS) + ["trivial"]:
        prediction = models[name].predict(matrix)
        tables[f"learned {name}"] = benchmark.score(
            name, prediction.theta_rad - targets, sensors
        )
    return tables, profiles


def main() -> str:
    models = fit_estimators(count=TRAIN, samples=SAMPLES, seed=20261029)
    test = dataset.training_population(count=TEST, samples=SAMPLES, seed=70120261)
    tables, profiles = _tables(test, models)
    order = list(COLOURS)

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))
    figure.subplots_adjust(wspace=0.34)

    left = axes[0]
    values = [tables[label].rmse_observable_mrad for label in order]
    bars = left.bar(range(len(order)), values,
                    color=[COLOURS[label] for label in order], edgecolor="white")
    best = int(np.argmin(values))
    bars[best].set_edgecolor("#000000")
    bars[best].set_linewidth(1.6)
    for position, value in enumerate(values):
        left.text(position, value, f"{value:.4f}", ha="center", va="bottom", fontsize=7)
    left.set_xticks(range(len(order)))
    left.set_xticklabels([label.replace(" ", "\n", 1) for label in order], fontsize=7,
                         rotation=20, ha="right")
    left.set_ylabel("RMSE on the observable subspace, mrad")
    left.set_title(
        f"held-out short records ({TEST} of {SAMPLES} samples)\n"
        f"lowest: {order[best]}"
    )

    middle = axes[1]
    unique = sorted(set(profiles))
    width = 0.8 / len(order)
    for index, label in enumerate(order):
        heights = []
        for profile in unique:
            subset = [test[k] for k, value in enumerate(profiles) if value == profile]
            subset_tables, _ = _tables(subset, models)
            heights.append(subset_tables[label].rmse_observable_mrad)
        middle.bar(
            np.arange(len(unique)) + index * width - 0.4 + width / 2,
            heights,
            width=width,
            color=COLOURS[label],
            label=label,
            edgecolor="white",
            linewidth=0.4,
        )
    middle.set_xticks(range(len(unique)))
    middle.set_xticklabels([p.replace("_", "\n") for p in unique], fontsize=7)
    middle.set_ylabel("RMSE on the observable subspace, mrad")
    middle.set_title("by excitation profile: no profile reverses\nthe aggregate conclusion")
    middle.legend(fontsize=6.5, ncol=2)

    right = axes[2]
    observable = [tables[label].rmse_observable_mrad for label in order]
    unobservable = [tables[label].rmse_unobservable_mrad for label in order]
    right.bar(range(len(order)), observable, color=[COLOURS[label] for label in order],
              edgecolor="white", label="observable subspace")
    right.bar(range(len(order)), unobservable, bottom=observable,
              color=[COLOURS[label] for label in order], alpha=0.35, hatch="//",
              edgecolor="white", label="unobservable subspace")
    right.set_xticks(range(len(order)))
    right.set_xticklabels([label.replace(" ", "\n", 1) for label in order], fontsize=7,
                          rotation=20, ha="right")
    right.set_ylabel("RMSE, mrad")
    right.set_title(
        "the hatched part is carried in full by EVERY\nestimator, including the one that refuses"
    )
    right.legend(fontsize=8)

    figure.suptitle(
        "The baseline wins. The regularised analytic solution of (18) beats every "
        "learned configuration, with no training, no features and no model.",
        fontsize=11,
    )
    figure.subplots_adjust(top=0.84)
    return save(figure, "learned_versus_baselines")


if __name__ == "__main__":
    print(main())
