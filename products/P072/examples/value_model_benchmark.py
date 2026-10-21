"""The learned value model against the declared passthrough, scored end to end.

What to notice: the left panel is the regret -- the value the selection gives up
against an oracle that knows the realised values -- for each value source in
five scenarios, through the same exact solver on the same instances. The
declared-value passthrough wins when the declarations are complete and fresh,
which is the result the specification predicted. It loses once the declarations
are stale or absent, which is the case the learned model exists for. The right
panel is why ranking matters more than level: the rank correlation tracks the
regret and the mean absolute error does not.
"""

from __future__ import annotations

import numpy as np

from _common import ANALYTIC, EXACT, GREEDY, LEARNED, LEFT_BEHIND, ORACLE, bootstrap, save, style

bootstrap()

from downlinkpick.benchmark import benchmark_value_sources  # noqa: E402
from downlinkpick.dataset import (  # noqa: E402
    generate_instance,
    generate_training_table,
    value_feature_row,
)
from downlinkpick.exact import exact_select  # noqa: E402
from downlinkpick.valuemodel import (  # noqa: E402
    ValueModel,
    declared_values,
    fit_size_age_heuristic,
    hybrid_values,
)

SEEDS = tuple(range(5001, 5021))

SCENARIOS = (
    ("complete\nand fresh", dict(missing_declared_fraction=0.0, stale_declared_fraction=0.0)),
    ("complete\nbut stale", dict(missing_declared_fraction=0.0, stale_declared_fraction=1.0)),
    ("30% missing", dict(missing_declared_fraction=0.3, stale_declared_fraction=0.0)),
    ("all missing", dict(missing_declared_fraction=1.0, stale_declared_fraction=0.0)),
    (
        "40% missing,\nhalf stale",
        dict(missing_declared_fraction=0.4, stale_declared_fraction=0.5),
    ),
)

ARM_COLOURS = {
    "declared-median": ANALYTIC,
    "size-age": GREEDY,
    "learned": LEARNED,
    "hybrid": EXACT,
}


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    table = generate_training_table(6000, 1001)
    model = ValueModel().fit(table.value_features, table.value_target)
    heuristic = fit_size_age_heuristic(table.value_features, table.value_target)

    def learned(instance):
        features = np.array(
            [value_feature_row(p, instance.inventory) for p in instance.inventory],
            dtype=float,
        )
        return dict(
            zip(instance.inventory.names, map(float, model.predict(features)), strict=True)
        )

    arms = {
        "declared-median": lambda instance: declared_values(instance.inventory),
        "size-age": lambda instance: heuristic.predict_inventory(instance.inventory),
        "learned": learned,
        "hybrid": lambda instance: hybrid_values(instance.inventory, learned(instance)),
    }

    regrets: dict[str, list[float]] = {name: [] for name in arms}
    spearmans: list[float] = []
    maes: list[float] = []
    points: list[float] = []
    for _label, kwargs in SCENARIOS:
        instances = [
            generate_instance(
                seed, n_products=24, n_windows=3, capacity_fraction=0.45, **kwargs
            )
            for seed in SEEDS
        ]
        results = {
            result.name: result
            for result in benchmark_value_sources(instances, arms, selector=exact_select)
        }
        for name in arms:
            regrets[name].append(results[name].regret_fraction)
            if np.isfinite(results[name].spearman):
                spearmans.append(results[name].spearman)
                maes.append(results[name].mae)
                points.append(results[name].regret_fraction)

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.3))
    figure.subplots_adjust(wspace=0.28, bottom=0.2, top=0.88)

    width = 0.2
    base = np.arange(len(SCENARIOS))
    for index, (name, colour) in enumerate(ARM_COLOURS.items()):
        left.bar(
            base + (index - 1.5) * width,
            regrets[name],
            width,
            color=colour,
            label=name,
        )
    left.set_xticks(base)
    left.set_xticklabels([label for label, _k in SCENARIOS], fontsize=8)
    left.set_ylabel("regret fraction against the oracle selection")
    left.set_title(f"end-to-end regret, {len(SEEDS)} instances per scenario")
    left.set_yscale("log")
    left.legend(fontsize=8, ncols=2)

    right.scatter(spearmans, points, s=28, color=EXACT, label="rank correlation")
    secondary = right.twiny()
    secondary.scatter(maes, points, s=28, color=LEFT_BEHIND, marker="s", label="MAE")
    right.set_xlabel("Spearman rank correlation with the realised value", color=EXACT)
    secondary.set_xlabel("mean absolute error [value units]", color=LEFT_BEHIND)
    right.set_ylabel("regret fraction")
    right.set_yscale("log")
    right.set_title("ranking predicts regret; level does not")
    right.axhline(
        min(points), color=ORACLE, linewidth=0.8, linestyle="--", label="best measured"
    )

    figure.text(
        0.5,
        0.01,
        "the declared-value passthrough is the best arm in the complete-and-fresh "
        "scenario and in no other; that is the measured result and it is the one the "
        "specification predicted",
        ha="center",
        fontsize=7,
    )
    return save(figure, "value_model_benchmark")


if __name__ == "__main__":
    print(main())
