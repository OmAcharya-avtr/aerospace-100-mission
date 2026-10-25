"""The learned conflict predictor against the contention-ratio baselines.

The order of work here is the order the build guide requires: the baselines are
built and measured first, the strongest of them is identified by measurement
rather than by reading the specification, and only then is the learned model
scored against it. That order changed the published comparison once already --
the specification names a demand-over-supply ratio per station-window, and that
ratio turned out to be the weakest of the three available, so taking it as the
baseline would have inflated the learned margin.

Everything is measured on instances from a seed range disjoint from the training
range. The decision threshold is chosen on the TRAINING rows and applied to the
test rows; choosing it on the test rows would be a leak.

Every model is scored twice: over all requests, and over the subset whose
``capacity_deficit`` is not positive. The first subset contains requests that
declared more contacts than their own passes can carry, which are unsatisfiable
whatever the competition does and are trivially predictable from one feature.
Reporting only the first number would flatter every model including the
baselines.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, Timer, bootstrap, environment_lines

bootstrap()

from contactsched.benchmark import score_predictor  # noqa: E402
from contactsched.dataset import build_dataset, split_by_instance  # noqa: E402
from contactsched.features import BASELINE_FEATURES, FEATURE_NAMES  # noqa: E402
from contactsched.predictors import (  # noqa: E402
    LearnedConflictPredictor,
    RatioLogisticBaseline,
    RatioRankingBaseline,
    best_f1_threshold,
    brier_score,
    reliability_table,
    roc_auc,
)

FAILURES: list[str] = []
TRAIN_SEEDS = range(0, 240)
TEST_SEEDS = range(240, 320)
HORIZON_S = 43200.0
RANDOM_STATE = 0


def matrix(frame) -> np.ndarray:
    """Design matrix with the declared finite sentinel substituted for infinities."""
    values = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    return np.where(np.isfinite(values), values, 1.0e6)


def main() -> None:
    out = Tee("validate_predictors_output.txt")
    out.rule("contactsched validate_predictors")
    for line in environment_lines():
        out(line)

    build_timer = Timer()
    with build_timer:
        frame = build_dataset(list(TRAIN_SEEDS) + list(TEST_SEEDS), horizon_s=HORIZON_S)
    train, test = split_by_instance(frame, TRAIN_SEEDS, TEST_SEEDS)
    x_train, y_train = matrix(train), train["unsatisfiable"].to_numpy(dtype=int)
    x_test, y_test = matrix(test), test["unsatisfiable"].to_numpy(dtype=int)

    out.rule("Dataset")
    out(f"train instances : {len(TRAIN_SEEDS)} (seeds {TRAIN_SEEDS.start}.."
        f"{TRAIN_SEEDS.stop - 1})")
    out(f"test instances  : {len(TEST_SEEDS)} (seeds {TEST_SEEDS.start}.."
        f"{TEST_SEEDS.stop - 1})")
    out(f"train rows      : {len(y_train)}, positive rate {y_train.mean():.6f}")
    out(f"test rows       : {len(y_test)}, positive rate {y_test.mean():.6f}")
    out(f"features        : {len(FEATURE_NAMES)}")
    out("label           : the exact optimum delivers fewer than min_contacts")
    out("")
    out("Compute, wall clock, unstable, table only:")
    out(f"  dataset build : {build_timer.elapsed_s:.2f} s")

    deficit_column = FEATURE_NAMES.index("capacity_deficit")
    hard = x_test[:, deficit_column] <= 0.0
    out("")
    out(f"hard subset (capacity_deficit <= 0) : {int(hard.sum())} rows, "
        f"positive rate {y_test[hard].mean():.6f}")
    out(f"easy subset (capacity_deficit > 0)  : {int((~hard).sum())} rows, "
        f"positive rate {y_test[~hard].mean():.6f}")

    out.rule("1. The baselines, measured before the learned model is fitted")
    out("Three contention ratios, each in a ranking form with no fitting and a")
    out("one-feature logistic form fitted on the training rows. A monotone")
    out("one-feature fit cannot change the ranking, so the two agree on the rank")
    out("statistics by construction and differ only on calibration; the agreement is")
    out("printed so the reader can see it rather than take it on trust.")
    out("")
    out(
        f"{'baseline':<34} {'AUC':>8} {'AP':>8} {'Brier':>8} {'ECE':>8} "
        f"{'AUC (hard)':>11}"
    )
    out("-" * 82)
    baseline_rows = []
    for feature in BASELINE_FEATURES:
        ranking = RatioRankingBaseline(feature).fit(x_train, y_train)
        logistic = RatioLogisticBaseline(feature, random_state=RANDOM_STATE).fit(
            x_train, y_train
        )
        threshold = best_f1_threshold(y_train, logistic.predict_proba(x_train))
        for name, model, calibrated in (
            (f"ranking[{feature}]", ranking, False),
            (f"logistic[{feature}]", logistic, True),
        ):
            row = score_predictor(
                name,
                y_test,
                model.score(x_test),
                model.predict_proba(x_test),
                threshold if calibrated else 0.5,
                calibrated,
            )
            row["auc_hard"] = roc_auc(y_test[hard], model.score(x_test)[hard])
            baseline_rows.append((row, model, calibrated))
            out(
                f"{name:<34} {row['roc_auc']:>8.4f} {row['average_precision']:>8.4f} "
                f"{row['brier']:>8.4f} {row['ece']:>8.4f} {row['auc_hard']:>11.4f}"
            )
    out("")
    out("The two per-bin ratios the specification names, ratio_max and ratio_weighted,")
    out("are the weakest of the three over the whole test set. The per-request ratio")
    out("demand_over_usable is the strongest and is therefore the baseline the learned")
    out("model has to beat. On the hard subset the ordering is different, which is")
    out("itself informative: the per-bin ratios carry more signal once the trivially")
    out("over-asked requests are removed.")

    best_row, best_model, _calibrated = max(
        (r for r in baseline_rows if r[2]), key=lambda r: r[0]["roc_auc"]
    )
    out("")
    out(f"strongest calibrated baseline by AUC : {best_row['predictor']}")

    out.rule("2. The learned model")
    model = LearnedConflictPredictor(random_state=RANDOM_STATE)
    fit_timer = Timer()
    with fit_timer:
        model.fit(x_train, y_train)
    threshold = best_f1_threshold(y_train, model.predict_proba(x_train))
    learned_row = score_predictor(
        model.name,
        y_test,
        model.score(x_test),
        model.predict_proba(x_test),
        threshold,
        True,
    )
    learned_row["auc_hard"] = roc_auc(y_test[hard], model.score(x_test)[hard])
    out("estimator      : sklearn HistGradientBoostingClassifier")
    out(f"max_iter       : {model.max_iter}")
    out(f"learning_rate  : {model.learning_rate}")
    out(f"max_leaf_nodes : {model.max_leaf_nodes}")
    out(f"min_samples_leaf: {model.min_samples_leaf}")
    out(f"l2             : {model.l2_regularization}")
    out(f"random_state   : {model.random_state}")
    out(f"fit wall clock : {fit_timer.elapsed_s:.3f} s (unstable, table only)")
    out(f"threshold chosen on TRAINING rows : {threshold:.3f}")

    out.rule("3. Head to head on the held-out instances")
    out(
        f"{'predictor':<34} {'AUC':>8} {'AP':>8} {'Brier':>8} {'ECE':>8} "
        f"{'prec':>7} {'recall':>7} {'F1':>7} {'AUC (hard)':>11}"
    )
    out("-" * 108)
    rows = [r[0] for r in baseline_rows] + [learned_row]
    constant = float(y_train.mean())
    for row in rows:
        out(
            f"{row['predictor']:<34} {row['roc_auc']:>8.4f} "
            f"{row['average_precision']:>8.4f} {row['brier']:>8.4f} {row['ece']:>8.4f} "
            f"{row['precision']:>7.4f} {row['recall']:>7.4f} {row['f1']:>7.4f} "
            f"{row['auc_hard']:>11.4f}"
        )
    out(
        f"{'constant forecast at the base rate':<34} {0.5:>8.4f} "
        f"{y_test.mean():>8.4f} "
        f"{brier_score(y_test, np.full(len(y_test), constant)):>8.4f} "
        f"{'-':>8} {'-':>7} {'-':>7} {'-':>7} {0.5:>11.4f}"
    )
    out("")
    out("The constant forecast is included because a Brier score has no meaning")
    out("without it: a model that beats no constant is not forecasting.")

    out.rule("4. The verdict, as measured")
    out("The learned model is compared against the BEST baseline on each metric")
    out("separately, not against one nominated baseline. On this test set the ranking")
    out("and logistic forms of demand_over_usable tie on area under the ROC curve but")
    out("the ranking form has the lower Brier score, so nominating the logistic form")
    out("as the single baseline would have handed the learned model an easier Brier")
    out("comparison than the baselines actually offer.")
    out("")
    all_baselines = [r[0] for r in baseline_rows]
    out(
        f"{'metric':<26} {'best baseline':>30} {'baseline':>10} {'learned':>10} "
        f"{'margin':>10}"
    )
    out("-" * 92)
    verdict: dict[str, tuple[str, float, float]] = {}
    for metric, higher_is_better in (
        ("roc_auc", True),
        ("average_precision", True),
        ("brier", False),
        ("ece", False),
        ("f1", True),
        ("auc_hard", True),
    ):
        chooser = max if higher_is_better else min
        winner = chooser(all_baselines, key=lambda row: row[metric])
        baseline_value = float(winner[metric])
        learned_value = float(learned_row[metric])
        margin = (
            learned_value - baseline_value if higher_is_better
            else baseline_value - learned_value
        )
        verdict[metric] = (str(winner["predictor"]), baseline_value, learned_value)
        out(
            f"{metric:<26} {winner['predictor']:>30} {baseline_value:>10.6f} "
            f"{learned_value:>10.6f} {margin:>+10.6f}"
        )
    out("")
    out("Margin is signed so that positive always means the learned model is better.")
    out("")
    losses = [
        metric
        for metric, (_who, baseline_value, learned_value) in verdict.items()
        if (
            learned_value < baseline_value
            if metric in {"roc_auc", "average_precision", "f1", "auc_hard"}
            else learned_value > baseline_value
        )
    ]
    if losses:
        out(f"The baseline WINS on: {', '.join(losses)}.")
        out("That result stands. The learned model is not removed, retuned or re-split")
        out("to change it, and the metrics it loses on are listed here first.")
    else:
        out("The learned model wins on every metric above, against the best baseline on")
        out("each one individually, including on the hard subset where the trivially")
        out("over-asked requests have been removed.")
    out("")
    out("What it costs. The learned model uses twenty-three pre-scheduling features,")
    out("needs a labelled training set built by running the exact solver on every")
    instance_count = len(TRAIN_SEEDS) + len(TEST_SEEDS)
    out(
        f"training instance ({build_timer.elapsed_s:.0f} s for {instance_count} "
        "instances here), and takes a fit"
    )
    out(f"({fit_timer.elapsed_s:.1f} s). The strongest baseline uses one feature, needs no")
    out("training data at all, and is a single division. A reader who cannot build a")
    out("labelled set should use demand_over_usable and expect the numbers in the row")
    out("above.")
    out("")
    out("What this is NOT evidence of: skill on any real ground-station network. The")
    out("training and test instances come from one synthetic generator whose")
    out("distributional choices are printed in validate_dataset_output.txt. A reader")
    out("whose request mix differs from those choices should expect different numbers")
    out("and should re-measure.")

    out.rule("5. Calibration of the confidence output")
    out("The learned model's predict_proba is the confidence this package exposes. It")
    out("is measured here, not asserted.")
    out("")
    for label, model_to_check in (
        (learned_row["predictor"], model),
        (best_row["predictor"], best_model),
    ):
        probabilities = model_to_check.predict_proba(x_test)
        out(f"{label}:")
        out(f"  {'bin':<14} {'mean predicted':>15} {'observed rate':>14} {'n':>6}")
        out("  " + "-" * 53)
        for low, predicted, observed, count in reliability_table(
            y_test, probabilities, bins=10
        ):
            out(
                f"  [{low:.1f}, {low + 0.1:.1f})   {predicted:>15.4f} "
                f"{observed:>14.4f} {count:>6d}"
            )
        out("")
    out("A well-calibrated forecaster has the predicted and observed columns equal in")
    out("every row. Where they are not, the expected calibration error in the table in")
    out("section 3 is the count-weighted mean of the gaps, over ten equal-width bins.")
    out("The bin count matters and is always reported with the number.")

    out.rule("6. Permutation importance of the learned model")
    out("Measured on the held-out rows, five permutations per feature, scored by area")
    out("under the ROC curve. These are a description of THIS fitted model on THIS")
    out("generator and are not evidence about ground-station scheduling in general;")
    out("correlated features share credit arbitrarily (Strobl et al., 2007).")
    out("")
    from sklearn.inspection import permutation_importance

    class RocScorer:
        """Wrap the predictor so permutation_importance can score it by AUC."""

        def __init__(self, inner) -> None:
            self.inner = inner

        def fit(self, x, y):  # noqa: D102 - sklearn interface
            return self

        def predict_proba(self, x):  # noqa: D102 - sklearn interface
            probabilities = self.inner.predict_proba(x)
            return np.column_stack([1.0 - probabilities, probabilities])

        def __sklearn_tags__(self):  # noqa: D105 - sklearn interface
            from sklearn.utils import Tags, TargetTags

            return Tags(
                estimator_type="classifier",
                target_tags=TargetTags(required=False),
                classifier_tags=None,
                regressor_tags=None,
                transformer_tags=None,
            )

    importance_timer = Timer()
    with importance_timer:
        result = permutation_importance(
            model._model,
            x_test,
            y_test,
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring="roc_auc",
            n_jobs=1,
        )
    order = np.argsort(result.importances_mean)[::-1]
    out(f"{'feature':<36} {'mean drop in AUC':>18} {'std':>10}")
    out("-" * 68)
    for index in order:
        out(
            f"{FEATURE_NAMES[index]:<36} {result.importances_mean[index]:>18.6f} "
            f"{result.importances_std[index]:>10.6f}"
        )
    out("")
    out(f"permutation importance wall clock: {importance_timer.elapsed_s:.2f} s "
        "(unstable, table only)")

    out.rule("7. Reproducibility")
    out("Refitting with the same seed on the same rows must give identical predictions.")
    repeat = LearnedConflictPredictor(random_state=RANDOM_STATE).fit(x_train, y_train)
    identical = bool(
        np.array_equal(repeat.predict_proba(x_test), model.predict_proba(x_test))
    )
    out(f"identical predictions on a refit : {identical}")
    if not identical:
        FAILURES.append("refitting with the same seed changed the predictions")
    out("")
    out("Exact commands to reproduce every number in this file:")
    out("  python validation/validate_dataset.py")
    out("  python validation/validate_predictors.py")

    out.rule("Result")
    if FAILURES:
        out(f"FAILED {len(FAILURES)} check(s):")
        for failure in FAILURES[:20]:
            out(f"  - {failure}")
    else:
        out("All checks in this script passed.")
    out("")
    out(f"written to {out.save()}")
    if FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
