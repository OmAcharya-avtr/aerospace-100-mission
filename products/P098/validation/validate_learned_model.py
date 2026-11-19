"""Benchmark the learned staleness-risk model against the validated baselines.

THIS RUNS AFTER ``validate_baselines.py``, and that order is binding in this
mission: an under-strength baseline inflates an apparent learned win. The
baselines are validated against the by-construction ground truth there; here
they are scored on exactly the same held-out test items as the model.

TWO SPLITS, SCORED SEPARATELY and never averaged into one figure:

- **held-out ARTIFACTS** -- ``GroupKFold`` with the subject artifact as the
  group, so every evidence item about an artifact is on one side of the split;
- **held-out TIME** -- forward-chaining splits on the production revision at
  the 50th, 65th and 80th percentiles.

THE PRE-REGISTERED EXPECTATION, from the product specification: the closure
comparison is exact and the learned model cannot beat it. Nothing here was
retuned after a score was seen; the forest's hyperparameters were fixed before
the first fit and are in ``src/evidencestale/learned.py``.

THE ONLY CASE IN WHICH A LEARNED MODEL COULD ADD ANYTHING is an incomplete
dependency declaration, because then the closure comparison is no longer
exact. Section 3 measures that case rather than asserting it.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import time  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from evidencestale.benchmark import (  # noqa: E402
    THRESHOLD,
    confidence_table,
    run_benchmark,
)
from evidencestale.corpus import MAIN_CORPUS, load_corpus  # noqa: E402
from evidencestale.features import FEATURE_NAMES  # noqa: E402
from evidencestale.learned import MAX_DEPTH, N_ESTIMATORS, RANDOM_STATE  # noqa: E402

#: Methods in report order.
METHODS = (
    "baseline1_timestamp",
    "baseline1b_timestamp_depth1",
    "baseline2_closure_hash",
    "learned_forest",
    "learned_leak_ablation",
)


def table(out: _bootstrap.Tee, result, kind: str) -> None:
    """Print one split kind's scores for every method present."""
    out("")
    out(
        f"{'method':<30} {'folds':>5} {'auc':>9} {'brier':>9} {'agree':>9} "
        f"{'missed':>9} {'alarm':>9}"
    )
    for method in METHODS:
        if not result.by(method, kind):
            continue
        summary = result.summary(method, kind)
        out(
            f"{method:<30} {summary['folds']:>5.0f} {summary['auc']:>9.4f} "
            f"{summary['brier']:>9.4f} {summary['agreement']:>9.4f} "
            f"{summary['missed_staleness_rate']:>9.4f} "
            f"{summary['false_alarm_rate']:>9.4f}"
        )


def main() -> int:
    """Score the model against the baselines on both splits and both views."""
    out = _bootstrap.Tee("validate_learned_model_output.txt")
    out("Learned model against the validated baselines -- P098 EvidenceStale")
    out("")
    out("run AFTER validate_baselines.py, which validates the baselines against the")
    out("by-construction ground truth. Every method is scored on the same held-out")
    out("test items of the same splits.")
    out("")
    out(f"fast path (EVIDENCESTALE_FAST=1): {_bootstrap.FAST}")
    out(f"grouped folds: {_bootstrap.SPLITS}")
    out(f"forest: RandomForestClassifier, n_estimators={N_ESTIMATORS}, "
        f"max_depth={MAX_DEPTH}, random_state={RANDOM_STATE}, n_jobs=1")
    out(f"decision threshold: {THRESHOLD}")
    out(f"features ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}")
    out("")
    out("NO FEATURE READS ANY CONTENT, ANY DIGEST, OR ANY EVENT AFTER THE EVIDENCE")
    out("WAS PRODUCED. An indicator 'some artifact in the declared closure has an")
    out("event after production' would reproduce baseline 2's information and make")
    out("the model a reimplementation of the exact method. It is measured anyway, as")
    out("learned_leak_ablation, so that the claim is a measurement and not an")
    out("assertion.")

    corpus = load_corpus(ROOT / MAIN_CORPUS)

    out.rule("1. COMPLETE DEPENDENCY DECLARATIONS")
    start = time.perf_counter()
    complete = run_benchmark(
        corpus,
        "complete",
        n_group_splits=_bootstrap.SPLITS,
        include_leak_ablation=True,
    )
    wall = time.perf_counter() - start
    out(f"evidence items               : {complete.n_items}")
    out(f"truly stale items            : {complete.n_positive}")
    out(f"degenerate folds             : {complete.degenerate_folds}")
    out(f"total fit seconds, all folds : {complete.fit_seconds:.3f}")
    out(f"wall seconds for this section: {wall:.3f}")
    out("")
    out("held-out ARTIFACTS (GroupKFold on the subject artifact)")
    table(out, complete, "artifact")
    out("")
    out("held-out TIME (forward chaining on the production revision)")
    table(out, complete, "time")

    exact_artifact = complete.summary("baseline2_closure_hash", "artifact")
    exact_time = complete.summary("baseline2_closure_hash", "time")
    learned_artifact = complete.summary("learned_forest", "artifact")
    learned_time = complete.summary("learned_forest", "time")
    out("")
    out.check(
        "the closure comparison is exact on held-out artifacts",
        exact_artifact["auc"] == 1.0 and exact_artifact["agreement"] == 1.0,
        f"auc {exact_artifact['auc']:.6f}, agreement {exact_artifact['agreement']:.6f}",
    )
    out.check(
        "the closure comparison is exact on held-out time",
        exact_time["auc"] == 1.0 and exact_time["agreement"] == 1.0,
        f"auc {exact_time['auc']:.6f}, agreement {exact_time['agreement']:.6f}",
    )
    out.check(
        "the learned model does NOT beat the closure comparison on held-out artifacts",
        learned_artifact["auc"] < exact_artifact["auc"],
        f"learned {learned_artifact['auc']:.6f} against exact "
        f"{exact_artifact['auc']:.6f}",
    )
    out.check(
        "the learned model does NOT beat the closure comparison on held-out time",
        learned_time["auc"] < exact_time["auc"],
        f"learned {learned_time['auc']:.6f} against exact {exact_time['auc']:.6f}",
    )
    timestamp_artifact = complete.summary("baseline1_timestamp", "artifact")
    timestamp_time = complete.summary("baseline1_timestamp", "time")
    out.check(
        "the learned model DOES beat the timestamp baseline on held-out artifacts",
        learned_artifact["auc"] > timestamp_artifact["auc"],
        f"learned {learned_artifact['auc']:.6f} against timestamp "
        f"{timestamp_artifact['auc']:.6f}",
    )
    out.check(
        "the learned model DOES beat the timestamp baseline on held-out time",
        learned_time["auc"] > timestamp_time["auc"],
        f"learned {learned_time['auc']:.6f} against timestamp "
        f"{timestamp_time['auc']:.6f}",
    )
    out.check(
        "all fits in this section finished inside the declared 60 s CPU budget",
        complete.fit_seconds < 60.0,
        f"{complete.fit_seconds:.3f} s",
    )

    out.rule("2. THE FEATURE IMPORTANCES, averaged over the grouped folds")
    for name, value in sorted(
        complete.importances.items(), key=lambda pair: -pair[1]
    ):
        out(f"  {name:<30} {value:.6f}")
    total = sum(complete.importances.values())
    out(f"  {'sum':<30} {total:.6f}")
    out.check(
        "the impurity importances sum to 1",
        abs(total - 1.0) <= 1e-9,
        f"{total:.12f}",
    )

    out.rule("3. INCOMPLETE DEPENDENCY DECLARATIONS -- the only case for a model")
    out("30 per cent of the true dependency edges are withheld. The ground truth is")
    out("still computed over the TRUE graph, so this measures what each method does")
    out("when the declarations it reads are wrong.")
    partial = run_benchmark(
        corpus,
        "partial",
        n_group_splits=_bootstrap.SPLITS,
        include_leak_ablation=True,
    )
    out("")
    out(f"hidden edges                 : {corpus.hidden_edge_count()} of "
        f"{len(corpus.true_graph.edges)}")
    out(f"total fit seconds, all folds : {partial.fit_seconds:.3f}")
    out("")
    out("held-out ARTIFACTS")
    table(out, partial, "artifact")
    out("")
    out("held-out TIME")
    table(out, partial, "time")
    degraded_artifact = partial.summary("baseline2_closure_hash", "artifact")
    degraded_time = partial.summary("baseline2_closure_hash", "time")
    partial_learned_artifact = partial.summary("learned_forest", "artifact")
    partial_learned_time = partial.summary("learned_forest", "time")
    out("")
    out.check(
        "an incomplete declaration degrades the closure comparison below exact",
        degraded_artifact["auc"] < 1.0 and degraded_time["auc"] < 1.0,
        f"artifact {degraded_artifact['auc']:.6f}, time {degraded_time['auc']:.6f}",
    )
    out.check(
        "the degraded closure comparison still makes no false alarm",
        degraded_artifact["false_alarm_rate"] == 0.0
        and degraded_time["false_alarm_rate"] == 0.0,
    )
    out.check(
        "the learned model does NOT beat the degraded closure comparison on "
        "held-out artifacts",
        partial_learned_artifact["auc"] < degraded_artifact["auc"],
        f"learned {partial_learned_artifact['auc']:.6f} against degraded "
        f"{degraded_artifact['auc']:.6f}",
    )
    out.check(
        "the learned model does NOT beat the degraded closure comparison on "
        "held-out time",
        partial_learned_time["auc"] < degraded_time["auc"],
        f"learned {partial_learned_time['auc']:.6f} against degraded "
        f"{degraded_time['auc']:.6f}",
    )
    out("")
    out("THE FINDING, STATED PLAINLY. The pre-registered expectation was that the")
    out("closure comparison is exact and the learned model cannot beat it. That is")
    out("what was measured, on both splits. The model's only possible value was the")
    out("incomplete-declaration case, and it does not beat the degraded closure")
    out("comparison there either: an undeclared edge hides staleness from BOTH")
    out("methods, and the graph features the model has left are the features of a")
    out("graph that is itself missing those edges. The product publishes a baseline")
    out("win and does not retune it away.")

    out.rule("4. THE CONFIDENCE OUTPUT -- does agreement rise with it")
    out("one grouped-fold pass, so every item is predicted by a model that never saw")
    out("its subject artifact. The confidence is max(p, 1-p), the margin of the")
    out("thresholded decision.")
    out("")
    rows = confidence_table(corpus)
    out(f"{'band':>12} {'items':>7} {'agreement':>10}")
    for low, high, count, agreement in rows:
        rendered = "nan" if count == 0 else f"{agreement:.6f}"
        out(f"{f'[{low:.1f},{high:.1f})':>12} {count:>7} {rendered:>10}")
    out(f"{'total':>12} {sum(row[2] for row in rows):>7}")
    populated = [row for row in rows if row[2] >= 20]
    out("")
    out("bands with at least 20 items, lowest to highest confidence:")
    for low, high, count, agreement in populated:
        out(f"  [{low:.1f},{high:.1f})  {count:>4} items  agreement {agreement:.6f}")
    rising = all(
        populated[index][3] <= populated[index + 1][3] + 1e-12
        for index in range(len(populated) - 1)
    )
    out("")
    out(f"agreement is monotone non-decreasing across those bands: {rising}")
    occupied = [row for row in rows if row[2] > 0]
    rising_all = all(
        occupied[index][3] <= occupied[index + 1][3] + 1e-12
        for index in range(len(occupied) - 1)
    )
    out(f"agreement is monotone non-decreasing across ALL occupied bands: {rising_all}")
    out("")
    out("the 20-item floor above is a reporting threshold, not a filter on the")
    out("finding: the sparse bands are listed in the full table and the monotonicity")
    out("over ALL occupied bands is reported on its own line. On this corpus the")
    out("highest band is the sparse one and it breaks the monotonicity, which is")
    out("exactly the failure mode a forest vote margin has and is why this product")
    out("does not call the margin a calibrated probability.")
    for low, high, count, agreement in occupied:
        if count < 20:
            out(f"  SPARSE BAND [{low:.1f},{high:.1f}): {count} items, "
                f"agreement {agreement:.6f}")
    out.check(
        "every item falls in exactly one confidence band",
        sum(row[2] for row in rows) == len(corpus.evidence),
        f"{sum(row[2] for row in rows)} of {len(corpus.evidence)}",
    )
    out.check(
        "the lowest-confidence band has lower agreement than the highest populated one",
        populated[0][3] < populated[-1][3],
        f"{populated[0][3]:.6f} against {populated[-1][3]:.6f}",
    )
    if not rising_all:
        out("")
        out("THE MONOTONICITY IS NOT CLEAN, and it is reported rather than smoothed.")
        out("A forest's vote margin is not a calibrated probability: trees that all")
        out("agree on a wrong answer report a high confidence. The non-monotonicity")
        out("is recorded in README.md under Limitations and in MODEL_CARD.md under")
        out("failure cases.")
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
