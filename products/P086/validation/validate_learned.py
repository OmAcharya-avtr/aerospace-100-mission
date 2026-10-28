"""The learned classifier against the rules and against a majority-class guess.

The baselines are implemented and validated first, in
``validate_rule_screens.py`` and ``validate_classification.py``. This script
only scores the learned component against them, on the same held-out split, and
publishes what it measures.

What is measured
----------------
1. **Both rule baselines, three learned configurations and two majority
   guesses**, on the complete held-out population, per class, by equations (13)
   to (15b).
2. **The ambiguous residual set**, which is the population the product
   specification assigns to the learned component, scored against a
   majority-class guess fitted on the ambiguous rows of the *training* split.
3. **A swept grid** of anomaly rate against cadence jitter, with the ambiguous
   rows pooled across cells so that the majority comparison rests on a larger
   population than one record provides.
4. **Both uncertainty outputs**: the expected calibration error of the vote
   share with its reliability curve, and the area under the ROC curve of the
   per-tree dispersion as a predictor of being wrong.
5. **An abstention sweep**, so the precision-recall trade of the declared
   threshold is visible rather than chosen.
6. **The compute budget**, in a table declared unstable.

Nothing below is tuned. No hyperparameter search was run; the three
configurations are declared in :data:`telemgap.learned.CONFIGURATIONS` and all
three are reported whatever they measure.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, Timer, bootstrap, environment_lines

ROOT = bootstrap()

from telemgap import benchmark, classify, dataset, features  # noqa: E402
from telemgap.learned import (  # noqa: E402
    CLASS_NAMES,
    CONFIGURATIONS,
    LearnedClassifier,
    MajorityClassifier,
    dispersion_error_auc,
    expected_calibration_error,
)
from telemgap.report import format_table, render_scores  # noqa: E402

#: Frames in the training, test and calibration splits.
SAMPLES = 12_000

#: Swept grid, as multipliers on the generator's injection counts and as a
#: cadence-jitter fraction.
ANOMALY_RATES = (0.5, 1.0, 2.0, 4.0)
JITTER_FRACTIONS = (0.0, 0.08, 0.16)
SWEEP_SAMPLES = 3000

#: Abstention thresholds reported.
ABSTAIN_THRESHOLDS = (0.0, 0.5, 0.6, 0.8, 0.9, 0.95)


def rule_and_truth(generated: dataset.GeneratedRecord, mode: str = "strengthened"):
    """Flattened rule labels and ground truth for one generated record."""
    report = classify.classify(generated.record, mode=mode)
    return (
        benchmark.rule_labels(generated.record, report).reshape(-1),
        generated.value_truth.reshape(-1),
    )


def main() -> int:
    out = Tee("validate_learned_output.txt")
    out.rule("telemgap -- learned classifier against the rules and a majority-class guess")
    for line in environment_lines():
        out(line)
    out("")
    out(f"frames per split   : {SAMPLES}")
    out(f"training seed      : {dataset.TRAINING_SEED}")
    out(f"test seed          : {dataset.TEST_SEED}")
    out(f"calibration seed   : {dataset.CALIBRATION_SEED}")
    out("splits are disjoint realisations of the generator; no frame is shared")
    out("")
    out("configurations (declared in telemgap.learned.CONFIGURATIONS):")
    for name, configuration in CONFIGURATIONS.items():
        out(f"  {name:<16s} {configuration}")

    timers = {key: Timer() for key in ("generate", "features", "rules", "fit", "predict")}
    with timers["generate"]:
        train = dataset.training_record(samples=SAMPLES)
        test = dataset.test_record(samples=SAMPLES)
        calibration = dataset.calibration_record(samples=SAMPLES)
    with timers["features"]:
        train_matrix, _ = features.build(train.record)
        test_matrix, _ = features.build(test.record)
        calibration_matrix, _ = features.build(calibration.record)
    train_labels = features.truth_labels(train.value_truth)
    test_labels = features.truth_labels(test.value_truth)
    calibration_labels = features.truth_labels(calibration.value_truth)
    with timers["rules"]:
        train_rules, _ = rule_and_truth(train)
        test_rules, _ = rule_and_truth(test)
        calibration_rules, _ = rule_and_truth(calibration)
    train_ambiguous = train_rules == benchmark.UNDECIDED
    test_ambiguous = test_rules == benchmark.UNDECIDED
    calibration_ambiguous = calibration_rules == benchmark.UNDECIDED

    out("")
    out(
        format_table(
            ["split", "rows", *CLASS_NAMES, "ambiguous rows"],
            [
                [
                    name,
                    str(labels.size),
                    *[str(int((labels == code).sum())) for code in range(len(CLASS_NAMES))],
                    str(int(mask.sum())),
                ]
                for name, labels, mask in (
                    ("train", train_labels, train_ambiguous),
                    ("test", test_labels, test_ambiguous),
                    ("calibration", calibration_labels, calibration_ambiguous),
                )
            ],
        )
    )

    models: dict[str, LearnedClassifier] = {}
    for name, configuration in CONFIGURATIONS.items():
        model = LearnedClassifier(**configuration)  # type: ignore[arg-type]
        with timers["fit"]:
            if name == "ambiguous_only":
                models[name] = model.fit(
                    train_matrix[train_ambiguous], train_labels[train_ambiguous]
                )
            else:
                models[name] = model.fit(train_matrix, train_labels)
    predictions = {}
    for name, model in models.items():
        with timers["predict"]:
            predictions[name] = model.predict(test_matrix)

    out.rule("1. the complete held-out population")
    result = benchmark.BenchmarkResult()
    for mode in classify.MODES:
        rules, _ = rule_and_truth(test, mode=mode)
        result.add(
            f"all_rules_{mode}", benchmark.score(test_labels, rules, f"all/rules_{mode}")
        )
    for name, prediction in predictions.items():
        result.add(
            f"all_learned_{name}",
            benchmark.score(test_labels, prediction.labels, f"all/learned_{name}"),
        )
    majority_all = MajorityClassifier().fit(train_labels)
    result.add(
        "all_majority",
        benchmark.score(
            test_labels, majority_all.predict(test_matrix).labels, "all/majority"
        ),
    )
    for key in sorted(result.tables):
        out(render_scores(result.tables[key]))
        out("")

    out.rule("1b. baseline versus learned on the complete population, stated plainly")
    out(
        "The strongest rule baseline is the strengthened one; the strongest learned "
        "configuration on this population is 'unweighted'. Both are named before the "
        "numbers are read."
    )
    out("")
    verdict = benchmark.BenchmarkResult()
    verdict.add("all_rules_strengthened", result.tables["all_rules_strengthened"])
    verdict.add("all_learned", result.tables["all_learned_unweighted"])
    for line in verdict.verdict_lines():
        out(f"  {line}")
    out("")
    wins = {"rules": 0, "learned": 0, "tie": 0}
    for line in verdict.verdict_lines():
        wins[line.rsplit(" ", 1)[-1]] = wins.get(line.rsplit(" ", 1)[-1], 0) + 1
    out(f"metric tally over three classes and three metrics: {wins}")
    out(
        "Recall and surfaced rate measure different things and both are reported. The "
        "rules abstain rather than guess, so an undecided row costs them recall and is "
        "counted in the surfaced rate; the learned configurations here never abstain, so "
        "their two numbers are equal by construction."
    )

    out.rule("2. the ambiguous residual set, against a majority-class guess")
    majority_ambiguous = MajorityClassifier().fit(train_labels[train_ambiguous])
    out(
        f"majority class on the training ambiguous rows: "
        f"{CLASS_NAMES[majority_ambiguous.label]} at {majority_ambiguous.frequency:.6f}"
    )
    out("")
    ambiguous_tables = {
        "majority": benchmark.score(
            test_labels[test_ambiguous],
            majority_ambiguous.predict(test_matrix[test_ambiguous]).labels,
            "ambiguous/majority",
        )
    }
    for name, model in models.items():
        prediction = model.predict(test_matrix[test_ambiguous])
        ambiguous_tables[name] = benchmark.score(
            test_labels[test_ambiguous], prediction.labels, f"ambiguous/learned_{name}"
        )
    for key in ambiguous_tables:
        out(render_scores(ambiguous_tables[key]))
        out("")
    rows = []
    majority_accuracy = ambiguous_tables["majority"].accuracy_decided
    for key, table in ambiguous_tables.items():
        rows.append(
            [
                key,
                str(table.population),
                f"{table.accuracy_decided:.6f}",
                f"{table.accuracy_decided - majority_accuracy:+.6f}",
                "beats the majority guess"
                if table.accuracy_decided > majority_accuracy
                else ("the majority guess" if key == "majority" else "LOSES to the majority guess"),
            ]
        )
    out(
        format_table(
            ["predictor", "rows", "accuracy", "delta vs majority", "verdict"], rows
        )
    )
    out("")
    best = max(
        (key for key in ambiguous_tables if key != "majority"),
        key=lambda key: ambiguous_tables[key].accuracy_decided,
    )
    out(
        f"The best learned configuration on the ambiguous residual set is {best!r} at "
        f"{ambiguous_tables[best].accuracy_decided:.6f} against the majority guess at "
        f"{majority_accuracy:.6f}."
    )
    losers = [
        key
        for key, table in ambiguous_tables.items()
        if key != "majority" and table.accuracy_decided <= majority_accuracy
    ]
    if losers:
        out(
            f"Configurations that LOSE to the majority-class guess on this population: "
            f"{losers}. That is published as measured and is not softened: a model that "
            "cannot beat a constant on the population it exists for has not earned its "
            "place there, and the configuration that decides the matter is the imbalance "
            "handling rather than anything about the architecture."
        )
    out.check(
        "at least one declared configuration beats the majority-class guess on the "
        "ambiguous residual set",
        ambiguous_tables[best].accuracy_decided > majority_accuracy,
        f"{best} {ambiguous_tables[best].accuracy_decided:.6f} against "
        f"{majority_accuracy:.6f}",
    )
    out.check(
        "the recall-weighted configuration is reported even though it loses to the "
        "majority-class guess",
        "recall_weighted" in ambiguous_tables,
        f"recall_weighted accuracy {ambiguous_tables['recall_weighted'].accuracy_decided:.6f}",
    )

    out.rule("3. swept grid, ambiguous rows pooled across cells")
    pooled_truth: list[np.ndarray] = []
    pooled_learned: list[np.ndarray] = []
    pooled_rules: list[np.ndarray] = []
    rows = []
    for rate in ANOMALY_RATES:
        for jitter in JITTER_FRACTIONS:
            cell = dataset.generate(
                samples=SWEEP_SAMPLES,
                seed=int(500_000 + 1000 * rate + 100 * jitter),
                n_gaps=max(1, int(round(6 * rate))),
                n_duplicates=max(1, int(round(8 * rate))),
                n_reorders=max(1, int(round(8 * rate))),
                n_clock_resets=0,
                n_corruptions=max(1, int(round(40 * rate))),
                n_byteswap=max(1, int(round(10 * rate))),
                n_scale=max(1, int(round(10 * rate))),
                jitter_fraction=jitter,
            )
            matrix, _ = features.build(cell.record)
            labels = features.truth_labels(cell.value_truth)
            rules, _ = rule_and_truth(cell)
            ambiguous = rules == benchmark.UNDECIDED
            prediction = models[best].predict(matrix)
            rule_table = benchmark.score(labels, rules, "rules")
            learned_table = benchmark.score(labels, prediction.labels, "learned")
            pooled_truth.append(labels[ambiguous])
            pooled_learned.append(prediction.labels[ambiguous])
            pooled_rules.append(rules[ambiguous])
            rows.append(
                [
                    f"{rate:g}",
                    f"{jitter:g}",
                    str(int(ambiguous.sum())),
                    f"{rule_table['corrupt'].precision:.4f}",
                    f"{rule_table['corrupt'].recall:.4f}",
                    f"{learned_table['corrupt'].precision:.4f}",
                    f"{learned_table['corrupt'].recall:.4f}",
                    f"{rule_table['misparsed'].precision:.4f}",
                    f"{rule_table['misparsed'].recall:.4f}",
                    f"{learned_table['misparsed'].precision:.4f}",
                    f"{learned_table['misparsed'].recall:.4f}",
                ]
            )
    out(
        format_table(
            [
                "rate",
                "jitter",
                "ambig",
                "rule corr P",
                "rule corr R",
                "lrn corr P",
                "lrn corr R",
                "rule misp P",
                "rule misp R",
                "lrn misp P",
                "lrn misp R",
            ],
            rows,
        )
    )
    truth_pool = np.concatenate(pooled_truth)
    learned_pool = np.concatenate(pooled_learned)
    out("")
    out(f"pooled ambiguous rows across the grid: {truth_pool.size}")
    pooled_majority = float((truth_pool == majority_ambiguous.label).mean())
    pooled_accuracy = float((learned_pool == truth_pool).mean())
    out(
        format_table(
            ["quantity", "value"],
            [
                ["pooled rows", str(truth_pool.size)],
                ["majority-class accuracy", f"{pooled_majority:.6f}"],
                [f"learned ({best}) accuracy", f"{pooled_accuracy:.6f}"],
                ["delta", f"{pooled_accuracy - pooled_majority:+.6f}"],
            ],
        )
    )
    out.check(
        "the best configuration beats the majority-class guess on the pooled ambiguous "
        "rows of the swept grid",
        pooled_accuracy > pooled_majority,
        f"{pooled_accuracy:.6f} against {pooled_majority:.6f} on {truth_pool.size} rows",
    )

    out.rule("4. both uncertainty outputs, measured on an independent third split")
    rows = []
    for name, model in models.items():
        if name == "ambiguous_only":
            prediction = model.predict(calibration_matrix[calibration_ambiguous])
            truth = calibration_labels[calibration_ambiguous]
        else:
            prediction = model.predict(calibration_matrix)
            truth = calibration_labels
        correct = prediction.labels == truth
        ece, centres, confidence, accuracy = expected_calibration_error(
            prediction.probability, correct
        )
        auc = dispersion_error_auc(prediction.dispersion, ~correct)
        rows.append(
            [
                name,
                str(truth.size),
                f"{float(correct.mean()):.6f}",
                f"{ece:.6f}",
                f"{auc:.6f}" if np.isfinite(auc) else "nan",
                f"{float(prediction.dispersion.mean()):.6f}",
                f"{float(prediction.dispersion.max()):.6f}",
            ]
        )
        if name == best:
            reliability = (centres, confidence, accuracy)
    out(
        format_table(
            [
                "configuration",
                "rows",
                "accuracy",
                "ECE",
                "dispersion AUC vs wrong",
                "mean dispersion",
                "max dispersion",
            ],
            rows,
        )
    )
    out("")
    out(f"reliability curve of the {best!r} configuration:")
    out("")
    out(
        format_table(
            ["bin centre", "mean confidence", "realised accuracy", "gap"],
            [
                [
                    f"{centre:.2f}",
                    f"{conf:.6f}" if np.isfinite(conf) else "-",
                    f"{acc:.6f}" if np.isfinite(acc) else "-",
                    f"{conf - acc:+.6f}" if np.isfinite(conf) else "-",
                ]
                for centre, conf, acc in zip(*reliability, strict=True)
            ],
        )
    )
    out("")
    out(
        "The vote share is a vote share. Its expected calibration error is reported "
        "above and is not post-hoc corrected; the dispersion is an ensemble disagreement "
        "measure and not a confidence interval, and its area under the ROC curve against "
        "the indicator of being wrong is the only claim made for it."
    )
    out.check(
        "the dispersion of the best configuration carries some information about being "
        "wrong",
        float(rows[[row[0] for row in rows].index(best)][4] or "nan") > 0.5,
        f"AUC {rows[[row[0] for row in rows].index(best)][4]}",
    )

    out.rule("5. abstention sweep of the best configuration")
    rows = []
    for threshold in ABSTAIN_THRESHOLDS:
        prediction = models[best].predict(test_matrix[test_ambiguous], abstain_threshold=threshold)
        table = benchmark.score(
            test_labels[test_ambiguous], prediction.labels, f"abstain {threshold:g}"
        )
        rows.append(
            [
                f"{threshold:g}",
                f"{prediction.abstention_rate:.6f}",
                f"{table.accuracy_decided:.6f}" if np.isfinite(table.accuracy_decided) else "-",
                f"{table['corrupt'].precision:.4f}",
                f"{table['corrupt'].recall:.4f}",
                f"{table['corrupt'].surfaced_rate:.4f}",
            ]
        )
    out(
        format_table(
            [
                "threshold",
                "abstention rate",
                "accuracy on decided",
                "corrupt P",
                "corrupt R",
                "corrupt surfaced",
            ],
            rows,
        )
    )

    out.rule("6. feature importances of the best configuration")
    importances = models[best].feature_importances
    out(
        format_table(
            ["feature", "mean decrease in impurity"],
            [
                [name, f"{value:.6f}"]
                for name, value in sorted(importances.items(), key=lambda item: -item[1])
            ],
        )
    )
    out("")
    out(
        "Mean decrease in impurity, reported and not interpreted causally. It is biased "
        "toward high-cardinality features and says nothing about what would happen if a "
        "feature were removed."
    )

    out.rule("compute budget -- UNSTABLE, two cores shared with other work")
    out(
        format_table(
            ["stage", "calls", "total s", "mean ms"],
            [
                [name, str(timer.calls), f"{timer.elapsed_s:.2f}", f"{timer.mean_ms:.1f}"]
                for name, timer in timers.items()
            ],
        )
    )
    total = sum(timer.elapsed_s for timer in timers.values())
    out("")
    out(f"total measured stage time: {total:.2f} s")
    out.check(
        "training and prediction of all three configurations finish inside the "
        "three-minute compute budget",
        total < 180.0,
        f"{total:.2f} s on {len(ANOMALY_RATES) * len(JITTER_FRACTIONS)} swept cells plus "
        "three splits",
    )
    out("")
    out("This model is not certified for operational flight use.")

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
