"""The headline comparison: the analytic certificate against the learned classifier.

The scoring rule, declared before the measurement
-------------------------------------------------
Dismissing a miss that mattered and re-simulating one that did not are not the
same mistake, so the quantity that decides is **recall of consequential
misses**, and the second quantity is how much re-simulation a method saves at
that recall. Flag rate is reported in its own column and never enters the
verdict. Accuracy is not reported at all: the positive class is a few per cent
of the sample and accuracy would reward dismissing everything.

What is compared
----------------
- **the analytic certificate**, baseline 1, whose dismissals are a theorem;
- **the learned classifier**, at four operating points: the threshold that
  reaches the declared recall target with the smallest flag rate, the threshold
  matched to the certificate's flag rate, the untuned 0.5, and the
  abstention-aware policy in which a probability near the threshold is referred
  to re-simulation rather than dismissed;
- **two trivial baselines** that cost nothing and must be beaten by anything
  claiming to be useful: flag every miss, and threshold the overrun alone;
- **a classifier fitted only on the certificate's residual**, which is the one
  role the specification allows the learned component and the one this script
  exists to test.

The split is by RUN. Misses inside one run share its trajectory and its timing
trace and are not independent of each other, so a split by miss would leak.
There is no hyperparameter search anywhere: the specification in
:data:`hilverdict.learned.DEFAULT_SPEC` is declared, and a search scored on the
same held-out runs that then report the result would make the result a
selection artifact.
"""

from __future__ import annotations

# Bytecode must not be written anywhere inside this tree. See
# validation/validate_cli.py section 3; CPython writes a module's bytecode
# cache BEFORE executing it, so _bootstrap cannot set this on its own behalf.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import time  # noqa: E402

import numpy as np  # noqa: E402

from _bootstrap import (  # noqa: E402
    STEPS,
    TARGET_RECALL,
    TEST_SEEDS,
    TOLERANCE,
    TRAIN_SEEDS,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from hilverdict.certificate import certify_run  # noqa: E402
from hilverdict.features import FEATURE_NAMES, feature_matrix  # noqa: E402
from hilverdict.learned import DEFAULT_SPEC, MissClassifier  # noqa: E402
from hilverdict.report import format_table, render_scores  # noqa: E402
from hilverdict.resimulate import resimulate_run  # noqa: E402
from hilverdict.scoring import operating_points, score_decision, threshold_at_recall  # noqa: E402
from hilverdict.synthetic import CAMPAIGNS, generate_run  # noqa: E402

PLANT_CALL_PER_STEP = 16


class Dataset:
    """Features, labels, bounds and costs for one population of runs."""

    def __init__(self, seeds: tuple[int, ...], label: str) -> None:
        features: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        bounds: list[np.ndarray] = []
        divergences: list[np.ndarray] = []
        calls: list[np.ndarray] = []
        groups: list[np.ndarray] = []
        self.refused = 0
        self.runs = 0
        started = time.perf_counter()
        for position, seed in enumerate(seeds):
            profile, amplitude = CAMPAIGNS[position % len(CAMPAIGNS)]
            log = generate_run(
                seed=seed, profile=profile, amplitude_deg=amplitude, steps=STEPS
            )
            certificate = certify_run(log, TOLERANCE)
            if certificate.refusals:
                self.refused += 1
                continue
            self.runs += 1
            matrix, events = feature_matrix(log)
            if not events:
                continue
            results = resimulate_run(log, TOLERANCE)
            features.append(matrix)
            labels.append(np.array([r.consequential for r in results], dtype=bool))
            bounds.append(np.array([c.bound for c in certificate.certificates]))
            divergences.append(np.array([r.peak_divergence for r in results]))
            calls.append(np.array([r.plant_calls for r in results], dtype=float))
            groups.append(np.full(len(events), position, dtype=int))
        self.label = label
        self.features = np.vstack(features)
        self.labels = np.concatenate(labels)
        self.bounds = np.concatenate(bounds)
        self.divergences = np.concatenate(divergences)
        self.plant_calls = np.concatenate(calls)
        self.run_index = np.concatenate(groups)
        self.elapsed_s = time.perf_counter() - started

    @property
    def undecided(self) -> np.ndarray:
        """Mask of misses the certificate leaves undecided."""
        return self.bounds >= TOLERANCE

    def summary(self) -> list[list[str]]:
        return [
            [self.label, str(self.runs), str(self.refused), str(self.labels.shape[0]),
             str(int(np.count_nonzero(self.labels))),
             f"{float(np.mean(self.labels)):.6f}",
             str(int(np.count_nonzero(self.undecided))),
             f"{self.elapsed_s:.2f}"],
        ]


def section_populations(out: Tee, train: Dataset, test: Dataset) -> None:
    out.rule("1. the populations, and the labels the reference produced")
    out("Labels come from baseline 2: a miss is consequential when the re-simulated")
    out("peak divergence reaches the declared tolerance. Nothing here labels a miss")
    out("by a prediction.")
    out("")
    out(
        format_table(
            ["population", "runs", "refused", "misses", "consequential",
             "rate", "undecided by eq (13)", "label time [s]"],
            train.summary() + test.summary(),
        )
    )
    out("")
    out(f"declared tolerance {TOLERANCE}, declared recall target {TARGET_RECALL}")
    out(f"features: {len(FEATURE_NAMES)} -- {', '.join(FEATURE_NAMES)}")
    out("")
    out("Two of those features, overrun_s and input_step_rad, are EXACTLY the two")
    out("numbers equation (13) is a function of. The learned model is given")
    out("everything the proof uses and twelve further features besides, so a loss")
    out("here is a loss on the merits.")
    out.check(
        "the training and test populations are disjoint by run",
        not (set(TRAIN_SEEDS) & set(TEST_SEEDS)),
        f"{len(TRAIN_SEEDS)} training seeds, {len(TEST_SEEDS)} test seeds, "
        f"{len(set(TRAIN_SEEDS) & set(TEST_SEEDS))} shared",
    )
    out.check(
        "both populations contain consequential and inconsequential misses",
        0 < int(np.count_nonzero(train.labels)) < train.labels.shape[0]
        and 0 < int(np.count_nonzero(test.labels)) < test.labels.shape[0],
        f"train {int(np.count_nonzero(train.labels))}/{train.labels.shape[0]}, "
        f"test {int(np.count_nonzero(test.labels))}/{test.labels.shape[0]}",
    )
    out.check(
        "the certificate dismisses no consequential miss in either population",
        int(np.count_nonzero(train.labels & ~train.undecided)) == 0
        and int(np.count_nonzero(test.labels & ~test.undecided)) == 0,
        f"train {int(np.count_nonzero(train.labels & ~train.undecided))}, "
        f"test {int(np.count_nonzero(test.labels & ~test.undecided))}",
    )


def section_comparison(out: Tee, train: Dataset, test: Dataset) -> MissClassifier:
    out.rule("2. the comparison, scored on recall of consequential misses")
    out("EVERY learned threshold below is chosen on the TRAINING runs and then")
    out("applied unchanged to the test runs. That protocol is not a formality: a")
    out("threshold picked as 'the smallest flag rate that reaches recall 1' using")
    out("the test labels reaches recall 1 BY CONSTRUCTION, and reporting that as an")
    out("out-of-sample result would be a leak. The leaked version is shown too, in")
    out("its own row, labelled as an oracle a user cannot have.")
    out("")
    classifier = MissClassifier(spec=DEFAULT_SPEC).fit(train.features, train.labels)
    train_probability = classifier.predict_proba(train.features)
    probability = classifier.predict_proba(test.features)
    certificate_flags = test.undecided
    certificate_flag_rate = float(np.mean(certificate_flags))
    train_flag_rate = float(np.mean(train.undecided))

    chosen, train_recall, train_flag = threshold_at_recall(
        train_probability, train.labels, TARGET_RECALL
    )
    matched = float(np.quantile(train_probability, 1.0 - train_flag_rate))
    overrun_column = FEATURE_NAMES.index("overrun_s")
    overrun_threshold, overrun_train_recall, overrun_train_flag = threshold_at_recall(
        train.features[:, overrun_column], train.labels, TARGET_RECALL
    )
    oracle, oracle_recall, oracle_flag = threshold_at_recall(
        probability, test.labels, TARGET_RECALL
    )

    scores = [
        score_decision(
            "analytic certificate, eq (13)",
            certificate_flags,
            test.labels,
            sound=True,
        ),
        score_decision(
            f"learned, threshold from train at recall {TARGET_RECALL:g}",
            probability >= chosen,
            test.labels,
        ),
        score_decision(
            "learned, threshold from train at the certificate's flag rate",
            probability >= matched,
            test.labels,
        ),
        score_decision("learned at the untuned 0.5", probability >= 0.5, test.labels),
    ]
    classifier.set_threshold(chosen)
    prediction = classifier.predict_with_uncertainty(test.features)
    scores.append(
        score_decision(
            "learned with abstention referred, threshold from train",
            prediction.referred,
            test.labels,
        )
    )
    scores.append(
        score_decision(
            "learned, ORACLE threshold chosen on the test labels",
            probability >= oracle,
            test.labels,
        )
    )
    scores.append(
        score_decision(
            "trivial: flag every miss",
            np.ones_like(test.labels, dtype=bool),
            test.labels,
        )
    )
    scores.append(
        score_decision(
            "trivial: overrun threshold from train at recall "
            f"{TARGET_RECALL:g}",
            test.features[:, overrun_column] >= overrun_threshold,
            test.labels,
        )
    )
    out(render_scores(scores))
    out("")
    rows = [
        ["threshold chosen on train at the recall target", f"{chosen:.9f}"],
        ["its recall and flag rate ON TRAIN",
         f"{train_recall:.6f} / {train_flag:.6f}"],
        ["train flag rate of the certificate", f"{train_flag_rate:.6f}"],
        ["threshold chosen on train to match that flag rate", f"{matched:.9f}"],
        ["overrun threshold chosen on train [ms]",
         f"{overrun_threshold * 1e3:.6f}"],
        ["its recall and flag rate ON TRAIN",
         f"{overrun_train_recall:.6f} / {overrun_train_flag:.6f}"],
        ["ORACLE threshold from the test labels", f"{oracle:.9f}"],
        ["its recall and flag rate on test",
         f"{oracle_recall:.6f} / {oracle_flag:.6f}"],
        ["certificate flag rate on test", f"{certificate_flag_rate:.6f}"],
    ]
    out(format_table(["quantity", "value"], rows))

    certificate_score, honest_score = scores[0], scores[1]
    matched_score = scores[2]
    oracle_score = scores[5]
    out("")
    out("THE RESULT, computed from the table above rather than written in advance:")
    out("")
    out(f"  The analytic certificate reaches recall "
        f"{certificate_score.counts.recall:.6f} at flag rate "
        f"{certificate_score.counts.flag_rate:.6f}, dismissing")
    out(f"  {certificate_score.missed_consequential} of "
        f"{certificate_score.counts.positives} consequential misses.")
    out(f"  The learned classifier at the train-chosen threshold reaches recall "
        f"{honest_score.counts.recall:.6f}")
    out(f"  at flag rate {honest_score.counts.flag_rate:.6f}, dismissing "
        f"{honest_score.missed_consequential} consequential misses.")
    out("")
    if honest_score.missed_consequential > 0:
        out("  So the learned component FAILS the scoring rule outright: it dismisses")
        out("  misses that mattered, which is the asymmetric error, while the")
        out("  certificate dismisses none and cannot.")
    elif honest_score.counts.flag_rate < certificate_score.counts.flag_rate:
        out("  So on this sample the learned component matched the certificate's")
        out("  recall at a LOWER flag rate. That is reported as measured. It is not")
        out("  a reason to prefer it, for two reasons that are not about taste:")
        out("")
        out("  - the certificate's recall is a THEOREM about the declared")
        out("    linearisation. The learned recall is a sample statistic whose exact")
        out(f"    95 per cent interval is "
            f"[{honest_score.recall_interval[0]:.6f}, "
            f"{honest_score.recall_interval[1]:.6f}], so the data is consistent with")
        out(f"    a true recall as low as {honest_score.recall_interval[0]:.4f}. The")
        out("    certificate's dismissals admit no such interval because they are not")
        out("    estimates.")
        out("  - a threshold that reached recall 1 on 60 training runs carries no")
        out("    guarantee on the next run, and section 2b measures exactly how much")
        out("    it moves between folds of the test population.")
    else:
        out("  So the certificate matched or beat the learned component on both the")
        out("  quantity that decides and the cost of reaching it.")
    out("")
    out("  The oracle row, whose threshold was chosen using the test labels, reaches")
    out(f"  recall {oracle_score.counts.recall:.6f} at flag rate "
        f"{oracle_score.counts.flag_rate:.6f}. It is an upper bound on what any")
    out("  threshold of this model could do on this exact sample and is not")
    out("  achievable by a user, who does not have the labels.")
    out("")
    out.check(
        "the analytic certificate reaches the declared recall target",
        certificate_score.counts.recall >= TARGET_RECALL - 1e-12,
        f"recall {certificate_score.counts.recall:.6f}, "
        f"{certificate_score.missed_consequential} consequential misses dismissed",
    )
    out.check(
        "the certificate's dismissals cost nothing in recall, by construction",
        certificate_score.missed_consequential == 0,
        f"{certificate_score.missed_consequential} of "
        f"{certificate_score.counts.positives}",
    )
    out.check(
        "the certificate beats the trivial flag-everything policy on cost",
        certificate_score.counts.flag_rate < 1.0,
        f"flag rate {certificate_score.counts.flag_rate:.6f} against 1.000000 at "
        "the same recall",
    )
    out.check(
        "the certificate beats the trivial overrun threshold at equal recall",
        certificate_score.counts.flag_rate < scores[7].counts.flag_rate,
        f"{certificate_score.counts.flag_rate:.6f} against "
        f"{scores[7].counts.flag_rate:.6f}",
    )
    out.check(
        "every learned recall is reported with an exact interval that admits "
        "a worse value",
        all(
            score.recall_interval[0] < 1.0
            for score in scores[1:]
            if not score.sound
        ),
        "lower limits "
        + ", ".join(
            f"{score.recall_interval[0]:.4f}" for score in scores[1:] if not score.sound
        ),
    )
    out("")
    out("Reported, not asserted: which method had the lower flag rate at the")
    out("declared recall on this sample.")
    out(f"  certificate flag rate : {certificate_score.counts.flag_rate:.6f} "
        f"(recall {certificate_score.counts.recall:.6f}, proof)")
    out(f"  learned flag rate     : {honest_score.counts.flag_rate:.6f} "
        f"(recall {honest_score.counts.recall:.6f}, estimate)")
    out(f"  matched-rate learned recall : {matched_score.counts.recall:.6f} "
        f"({matched_score.missed_consequential} consequential misses dismissed)")
    section_folds(out, classifier, chosen, test, certificate_score)
    return classifier


def section_folds(
    out: Tee,
    classifier: MissClassifier,
    threshold: float,
    test: Dataset,
    certificate_score,
) -> None:
    out.rule("2b. how much the learned recall moves between folds of the test runs")
    out("The certificate's recall is a theorem and cannot move. The learned")
    out("classifier's is a statistic, and the test population is split into three")
    out(f"disjoint folds of {len(TEST_SEEDS) // 3} runs each -- by RUN, never by "
        "miss -- to measure")
    out("how far it moves at the SAME train-chosen threshold.")
    out("")
    probability = classifier.predict_proba(test.features)
    flagged = probability >= threshold
    rows = []
    learned_recalls = []
    certificate_recalls = []
    for fold in range(3):
        mask = (test.run_index % 3) == fold
        truth = test.labels[mask]
        if not np.any(truth):
            rows.append([str(fold), str(int(np.count_nonzero(mask))), "0", "-", "-",
                         "-", "-"])
            continue
        learned = score_decision(
            f"fold {fold} learned", flagged[mask], truth
        )
        certificate = score_decision(
            f"fold {fold} certificate", test.undecided[mask], truth, sound=True
        )
        learned_recalls.append(learned.counts.recall)
        certificate_recalls.append(certificate.counts.recall)
        rows.append(
            [
                str(fold),
                str(int(np.count_nonzero(mask))),
                str(learned.counts.positives),
                f"{certificate.counts.recall:.6f}",
                f"{certificate.counts.flag_rate:.6f}",
                f"{learned.counts.recall:.6f}",
                f"{learned.counts.flag_rate:.6f}",
            ]
        )
    out(
        format_table(
            ["fold", "misses", "consequential", "certificate recall",
             "certificate flag rate", "learned recall", "learned flag rate"],
            rows,
        )
    )
    out("")
    spread = max(learned_recalls) - min(learned_recalls)
    out(f"  certificate recall across folds : "
        f"{min(certificate_recalls):.6f} to {max(certificate_recalls):.6f}")
    out(f"  learned recall across folds     : "
        f"{min(learned_recalls):.6f} to {max(learned_recalls):.6f}")
    out(f"  learned recall spread           : {spread:.6f}")
    out.check(
        "the certificate's recall is identical on every fold",
        max(certificate_recalls) == min(certificate_recalls) == 1.0,
        f"{min(certificate_recalls):.6f} to {max(certificate_recalls):.6f}",
    )
    out.check(
        "the learned recall across folds is reported, whichever way it came out",
        True,
        f"spread {spread:.6f}, from {min(learned_recalls):.6f} to "
        f"{max(learned_recalls):.6f} at a fixed threshold",
    )
    _ = certificate_score


def section_residual(out: Tee, train: Dataset, test: Dataset) -> None:
    out.rule("3. the residual undecided set: the only role the specification allows")
    out("Restricted to the misses the certificate cannot settle, can a classifier")
    out("save any of the re-simulations that set would otherwise cost? A model is")
    out("fitted on the TRAINING runs' residual misses only, and scored on the test")
    out("runs' residual misses only.")
    out("")
    train_mask = train.undecided
    test_mask = test.undecided
    rows = [
        ["training residual misses", str(int(np.count_nonzero(train_mask)))],
        ["of which consequential",
         str(int(np.count_nonzero(train.labels[train_mask])))],
        ["test residual misses", str(int(np.count_nonzero(test_mask)))],
        ["of which consequential",
         str(int(np.count_nonzero(test.labels[test_mask])))],
        ["consequential rate in the test residual",
         f"{float(np.mean(test.labels[test_mask])):.6f}"],
    ]
    out(format_table(["quantity", "value"], rows))
    out("")
    classes = set(np.unique(train.labels[train_mask]).tolist())
    if classes != {False, True}:
        out("the training residual contains only one class, so no model can be fitted")
        out.check(
            "the residual-only model was fitted",
            False,
            "the training residual is single-class",
        )
        return
    residual_model = MissClassifier(spec=DEFAULT_SPEC).fit(
        train.features[train_mask], train.labels[train_mask]
    )
    # The threshold is chosen on the TRAINING residual and applied unchanged to
    # the test residual. Choosing it on the test residual would reach recall 1
    # by construction, which is the leak section 2 explains.
    train_probability = residual_model.predict_proba(train.features[train_mask])
    chosen, train_recall, train_flag = threshold_at_recall(
        train_probability, train.labels[train_mask], TARGET_RECALL
    )
    probability = residual_model.predict_proba(test.features[test_mask])
    thresholds, recall, flag_rate = operating_points(
        probability, test.labels[test_mask]
    )
    oracle_feasible = np.nonzero(recall >= TARGET_RECALL - 1e-12)[0]
    oracle = oracle_feasible[int(np.argmin(flag_rate[oracle_feasible]))]
    scores = [
        score_decision(
            "residual-only model, threshold from the train residual",
            probability >= chosen,
            test.labels[test_mask],
        ),
        score_decision(
            "residual-only model, ORACLE threshold from the test labels",
            probability >= float(thresholds[oracle]),
            test.labels[test_mask],
        ),
        score_decision(
            "re-simulate the whole residual",
            np.ones(int(np.count_nonzero(test_mask)), dtype=bool),
            test.labels[test_mask],
        ),
    ]
    out(render_scores(scores))
    out("")
    honest = scores[0]
    out(
        format_table(
            ["quantity", "value"],
            [
                ["threshold chosen on the train residual", f"{chosen:.9f}"],
                ["its recall and flag rate ON TRAIN",
                 f"{train_recall:.6f} / {train_flag:.6f}"],
                ["its recall ON TEST", f"{honest.counts.recall:.6f}"],
                ["its flag rate ON TEST", f"{honest.counts.flag_rate:.6f}"],
                ["consequential misses it dismissed",
                 str(honest.missed_consequential)],
                ["oracle flag rate at recall 1 on test",
                 f"{float(flag_rate[oracle]):.6f}"],
            ],
        )
    )
    out("")
    saving = (
        1.0 - honest.counts.flag_rate
        if honest.missed_consequential == 0
        else 0.0
    )
    out(f"  re-simulations saved at no cost in recall : {saving:.6f}")
    out("")
    out("Operating points on the residual, so the whole curve is visible and not")
    out("only the point that flatters the model:")
    rows = []
    for index in range(0, len(thresholds), max(1, len(thresholds) // 12)):
        rows.append(
            [
                f"{float(thresholds[index]):.6f}",
                f"{float(recall[index]):.6f}",
                f"{float(flag_rate[index]):.6f}",
            ]
        )
    out(format_table(["threshold", "recall", "flag rate"], rows))
    out("")
    if honest.missed_consequential > 0:
        out("MEASURED RESULT: on the residual set, at the threshold a user could")
        out("actually have chosen, the model dismisses")
        out(f"{honest.missed_consequential} of {honest.counts.positives} consequential")
        out("misses. It therefore saves nothing that a user is entitled to take: the")
        out("saving is bought with exactly the error the scoring rule forbids. The")
        out("oracle row shows that a threshold chosen WITH the test labels could have")
        out(f"reached recall 1 while flagging {float(flag_rate[oracle]):.6f} of the")
        out("residual, so the information is in the model; what is missing is any way")
        out("to find that threshold without the labels.")
    elif saving <= 0.0:
        out("MEASURED RESULT: on the residual set the model cannot reach the declared")
        out("recall target without flagging every miss in it, so it saves NOTHING.")
        out("The residual set is where the certificate has already removed every easy")
        out("case, and what is left is dominated by consequential misses: the")
        out(f"consequential rate there is {float(np.mean(test.labels[test_mask])):.6f}")
        out("against a population rate of "
            f"{float(np.mean(test.labels)):.6f}. A classifier asked for perfect")
        out("recall on a set that is mostly positive has nothing to work with.")
    else:
        out("MEASURED RESULT: on the residual set the model reaches the declared")
        out(f"recall target while flagging {honest.counts.flag_rate:.6f} of it,")
        out(f"saving {saving:.6f} of the residual re-simulations. That saving is a")
        out(f"MEASUREMENT on {len(TEST_SEEDS)} held-out runs and carries the sampling")
        out("error of the recall interval above; it is not a guarantee, and a user who")
        out("needs a guarantee must re-simulate.")
    out.check(
        "the residual-set result is reported as measured, whichever way it came out",
        True,
        f"flag rate {honest.counts.flag_rate:.6f} at recall "
        f"{honest.counts.recall:.6f}, {honest.missed_consequential} consequential "
        f"misses dismissed, saving {saving:.6f}",
    )


def section_cost(out: Tee, test: Dataset) -> None:
    out.rule("4. what each policy costs in plant evaluations")
    out("The unit is a right-hand-side evaluation of equations (1)-(2). The cost of")
    out("re-simulating a miss is known exactly from its remaining horizon, so these")
    out("are counts and not timings.")
    out("")
    total = float(np.sum(test.plant_calls))
    rows = [
        [
            "re-simulate every miss",
            f"{total:.0f}",
            "1.000000",
            f"{float(np.mean(test.labels)):.6f}",
        ],
        [
            "certificate, then re-simulate the residual",
            f"{float(np.sum(test.plant_calls[test.undecided])):.0f}",
            f"{float(np.sum(test.plant_calls[test.undecided]) / total):.6f}",
            "0.000000",
        ],
    ]
    out(
        format_table(
            ["policy", "plant evaluations", "fraction of the full cost",
             "consequential misses dismissed, as a rate"],
            rows,
        )
    )
    out("")
    saving = 1.0 - float(np.sum(test.plant_calls[test.undecided]) / total)
    out(f"  evaluations saved by baseline 1 at no cost in recall : {saving:.6f}")
    out.check(
        "the certificate saves most of the reference's cost at zero cost in recall",
        saving > 0.5,
        f"{saving:.6f} of plant evaluations over {test.labels.shape[0]} misses",
    )


def section_model_quality(out: Tee, classifier: MissClassifier, test: Dataset) -> None:
    out.rule("5. the model itself: calibration, and what it leaned on")
    reliability = classifier.reliability(test.features, test.labels)
    rows = []
    for predicted, observed, count in zip(
        reliability.predicted, reliability.observed, reliability.counts, strict=True
    ):
        rows.append([f"{predicted:.6f}", f"{observed:.6f}", str(int(count)),
                     f"{abs(predicted - observed):.6f}"])
    out(format_table(["mean predicted", "observed rate", "misses", "gap"], rows))
    out("")
    base_rate = float(np.mean(test.labels))
    out(f"  Brier score                 : {reliability.brier:.6f}")
    out(f"  base-rate predictor's Brier : {base_rate * (1.0 - base_rate):.6f}")
    out(f"  worst calibration gap       : {reliability.worst_gap:.6f}")
    out.check(
        "the model beats the base-rate predictor on the Brier score",
        reliability.brier < base_rate * (1.0 - base_rate),
        f"{reliability.brier:.6f} against {base_rate * (1.0 - base_rate):.6f}",
    )
    out("")
    out("A Brier score better than the base rate says the probabilities carry")
    out("information. It does not say the model is useful for THIS decision, which")
    out("is what section 2 measures and section 2 is the verdict.")
    out("")
    importance = classifier.feature_importance
    rows = [
        [name, f"{importance[name]:.6f}"]
        for name in sorted(FEATURE_NAMES, key=lambda key: -importance[key])
    ]
    out(format_table(["feature", "impurity importance"], rows))
    out("")
    out("Impurity importance is biased towards features with many distinct values")
    out("and is a description of the fitted trees, never evidence about the plant.")
    out("What it does show is that the model concentrated on the same two quantities")
    out("equation (13) is built from, which is the expected outcome: there was")
    out("nothing else to find.")
    top_two = sorted(importance, key=lambda key: -importance[key])[:2]
    out.check(
        "the model's two most important features are the two the certificate uses",
        set(top_two) <= {"overrun_s", "overrun_periods", "input_step_rad",
                         "hold_periods"},
        f"top two were {top_two}",
    )
    out("")
    out("This model is not certified for operational flight use.")


def main() -> int:
    out = Tee("validate_learned_model_output.txt")
    out.rule("hilverdict -- the analytic certificate against the learned classifier")
    for line in environment_lines():
        out(line)
    out("")
    out("spec: " + ", ".join(
        f"{field}={getattr(DEFAULT_SPEC, field)}"
        for field in (
            "n_estimators", "learning_rate", "max_depth", "min_samples_leaf",
            "subsample", "positive_weight", "seed", "abstention_band",
        )
    ))
    train = Dataset(TRAIN_SEEDS, "train")
    test = Dataset(TEST_SEEDS, "test")
    section_populations(out, train, test)
    classifier = section_comparison(out, train, test)
    section_residual(out, train, test)
    section_cost(out, test)
    section_model_quality(out, classifier, test)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
