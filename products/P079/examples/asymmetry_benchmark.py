"""The learned asymmetry corrector against the baselines, on held-out data.

Writes ``screenshots/asymmetry_benchmark.png``. What to notice: the symmetric
assumption is the tallest bar everywhere, the two closed-form estimators take
most of the available improvement, and the gradient-boosted model never beats
the linear ridge on the same sixteen features. At zero coupling -- where the
static asymmetry is unlearnable by construction -- the median filter is within
a few per cent of the learned model, which is the result the specification
predicted.
"""

from __future__ import annotations

import numpy as np
from _common import SERIES, bootstrap, save, style

ROOT = bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402

from timetransfer.asym import (  # noqa: E402
    FEATURE_NAMES,
    GeneratorConfig,
    LearnedAsymmetryCorrector,
    MeanDifferenceEstimator,
    MedianFilterEstimator,
    RidgeAsymmetryCorrector,
    SymmetricAssumption,
    build_dataset,
    score_predictions,
)

COUPLINGS = (0.0, 1.0, 2.0)
N_SCENARIOS = 900
SEED = 20261024
NAMES = ("symmetric_assumption", "median_filter", "mean_difference",
         "learned_ridge", "learned_gbr")


def main() -> None:
    rmse: dict[str, list[float]] = {name: [] for name in NAMES}
    importances = None
    print(f"{'coupling':>9} " + " ".join(f"{name:>22}" for name in NAMES))
    for coupling in COUPLINGS:
        config = GeneratorConfig(coupling=coupling)
        dataset = build_dataset(N_SCENARIOS, seed=SEED, config=config)
        train, test = dataset.split(train_fraction=0.7)
        reference = None
        row_text = f"{coupling:>9.1f} "
        for estimator in (
            SymmetricAssumption(),
            MedianFilterEstimator(),
            MeanDifferenceEstimator(),
            RidgeAsymmetryCorrector(),
            LearnedAsymmetryCorrector(),
        ):
            estimator.fit(dataset.features[train], dataset.target_s[train])
            scored = score_predictions(
                estimator.name,
                estimator.predict(dataset.features[test]),
                dataset.target_s[test],
                reference_rmse_s=reference,
            )
            reference = reference or scored.rmse_s
            rmse[estimator.name].append(scored.rmse_s)
            row_text += f"{scored.rmse_s:>22.6e} "
            if coupling == 1.0 and isinstance(estimator, LearnedAsymmetryCorrector):
                importances = estimator.feature_importances
        print(row_text)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    left, right = axes
    positions = np.arange(len(COUPLINGS))
    width = 0.16
    for index, name in enumerate(NAMES):
        offsets = positions + (index - 2) * width
        left.bar(
            offsets,
            np.asarray(rmse[name]) * 1e6,
            width,
            color=SERIES[index % len(SERIES)],
            alpha=1.0 if "learned" in name else 0.75,
            hatch="//" if "learned" in name else None,
            label=name,
        )
    left.set_xticks(positions)
    left.set_xticklabels([f"coupling {c:.1f}" for c in COUPLINGS])
    left.set_ylabel("held-out RMSE on the asymmetry bias [us]")
    left.set_title(
        f"{N_SCENARIOS} scenarios per coupling, 70/30 split\n"
        f"lower is better; hatched bars are the learned models"
    )
    left.legend(loc="upper left", fontsize=7.2)

    assert importances is not None
    order = np.argsort(importances)[::-1][:8]
    right.barh(
        np.arange(len(order)),
        importances[order],
        color=SERIES[0],
    )
    right.set_yticks(np.arange(len(order)))
    right.set_yticklabels([FEATURE_NAMES[i] for i in order], fontsize=7.4)
    right.invert_yaxis()
    right.set_xlabel("impurity-based importance")
    right.set_title("what the boosted model leans on, coupling 1.0")

    fig.tight_layout()
    print(save(fig, "asymmetry_benchmark"))
    for coupling_index, coupling in enumerate(COUPLINGS):
        median = rmse["median_filter"][coupling_index]
        gbr = rmse["learned_gbr"][coupling_index]
        ridge = rmse["learned_ridge"][coupling_index]
        print(
            f"coupling {coupling:.1f}: median/gbr {median / gbr:.4f}, "
            f"gbr/ridge {gbr / ridge:.4f}"
        )


if __name__ == "__main__":
    main()
