"""The learned conflict predictor against the contention-ratio baselines.

What to notice: the per-station-window ratio the product specification names is
the weakest of the three contention baselines, the per-request ratio is far
stronger, and the one-feature logistic baseline is BETTER CALIBRATED than the
learned model even though it ranks worse. A model that ranks well and forecasts
badly is a different tool from one that forecasts well.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import BASELINE, LEARNED, STARVED, bootstrap, save, style

bootstrap()

from contactsched.dataset import build_dataset, split_by_instance  # noqa: E402
from contactsched.features import BASELINE_FEATURES, FEATURE_NAMES  # noqa: E402
from contactsched.predictors import (  # noqa: E402
    LearnedConflictPredictor,
    RatioLogisticBaseline,
    average_precision,
    brier_score,
    expected_calibration_error,
    reliability_table,
    roc_auc,
)

TRAIN_SEEDS = range(0, 120)
TEST_SEEDS = range(120, 160)
HORIZON_S = 43200.0


def matrix(frame) -> np.ndarray:
    """Design matrix with the declared finite sentinel for infinities."""
    values = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    return np.where(np.isfinite(values), values, 1.0e6)


def roc_points(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False and true positive rates, from sklearn's own curve."""
    from sklearn.metrics import roc_curve

    false_positive, true_positive, _thresholds = roc_curve(y_true, scores)
    return false_positive, true_positive


def main() -> None:
    style()
    frame = build_dataset(list(TRAIN_SEEDS) + list(TEST_SEEDS), horizon_s=HORIZON_S)
    train, test = split_by_instance(frame, TRAIN_SEEDS, TEST_SEEDS)
    x_train, y_train = matrix(train), train["unsatisfiable"].to_numpy(dtype=int)
    x_test, y_test = matrix(test), test["unsatisfiable"].to_numpy(dtype=int)

    models = []
    for index, feature in enumerate(BASELINE_FEATURES):
        shade = ("#4a4a4a", "#8c8c8c", "#bdbdbd")[index % 3]
        models.append(
            (
                f"logistic[{feature}]",
                RatioLogisticBaseline(feature).fit(x_train, y_train),
                shade,
                "--",
            )
        )
    learned = LearnedConflictPredictor().fit(x_train, y_train)
    models.append(("learned-hgb", learned, LEARNED, "-"))

    figure, (roc, calibration) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    roc.plot([0, 1], [0, 1], color=STARVED, lw=0.9, ls=":", label="coin flip")
    for name, model, colour, dashes in models:
        scores = model.score(x_test)
        false_positive, true_positive = roc_points(y_test, scores)
        roc.plot(
            false_positive,
            true_positive,
            color=colour,
            ls=dashes,
            lw=1.7,
            label=f"{name}: AUC {roc_auc(y_test, scores):.4f}, "
            f"AP {average_precision(y_test, scores):.4f}",
        )
    roc.set_xlabel("false positive rate")
    roc.set_ylabel("true positive rate")
    roc.set_title(f"Predicting starvation on {len(y_test)} held-out requests")
    roc.legend(fontsize=7, loc="lower right")

    calibration.plot([0, 1], [0, 1], color="#9a9a9a", lw=0.9, ls="--", label="perfect")
    for name, model, colour, dashes in models:
        probabilities = model.predict_proba(x_test)
        table = reliability_table(y_test, probabilities, bins=10)
        calibration.plot(
            [row[1] for row in table],
            [row[2] for row in table],
            marker="o",
            markersize=4,
            color=colour,
            ls=dashes,
            lw=1.4,
            label=f"{name}: Brier {brier_score(y_test, probabilities):.4f}, "
            f"ECE {expected_calibration_error(y_test, probabilities):.4f}",
        )
    calibration.set_xlabel("mean predicted probability in bin")
    calibration.set_ylabel("observed starvation rate in bin")
    calibration.set_title("Calibration of the confidence output, ten equal-width bins")
    calibration.set_xlim(-0.03, 1.03)
    calibration.set_ylim(-0.03, 1.03)
    calibration.legend(fontsize=7, loc="upper left")

    figure.tight_layout()
    path = save(figure, "predictor_benchmark")

    print(f"train rows {len(y_train)} (positive rate {y_train.mean():.4f}), "
          f"test rows {len(y_test)} (positive rate {y_test.mean():.4f})")
    print(f"{'predictor':<30} {'AUC':>8} {'AP':>8} {'Brier':>8} {'ECE':>8}")
    for name, model, _colour, _dashes in models:
        scores = model.score(x_test)
        probabilities = model.predict_proba(x_test)
        print(
            f"{name:<30} {roc_auc(y_test, scores):>8.4f} "
            f"{average_precision(y_test, scores):>8.4f} "
            f"{brier_score(y_test, probabilities):>8.4f} "
            f"{expected_calibration_error(y_test, probabilities):>8.4f}"
        )
    constant = float(y_train.mean())
    print(
        f"{'constant at the base rate':<30} {0.5:>8.4f} {y_test.mean():>8.4f} "
        f"{brier_score(y_test, np.full(len(y_test), constant)):>8.4f} {'-':>8}"
    )
    print(f"colour note: the baselines are grey, the learned model {LEARNED}, "
          f"the coin flip {BASELINE}")
    print(f"figure written: {path}")


if __name__ == "__main__":
    main()
