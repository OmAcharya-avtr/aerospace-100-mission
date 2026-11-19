"""Validate both classical baselines against the by-construction ground truth.

THIS RUNS BEFORE ANY LEARNED MODEL IS SCORED. The build guide of this mission
requires that order, because an under-strength baseline inflates an apparent
learned win: one product of this mission measured a 50 per cent inflation from
exactly that, and another found that the baseline its specification NAMED was
the weakest of three candidates. So baseline 1 is implemented at full strength
AND in its strongest honest variant, and both are published.

WHAT "EXACTLY" MEANS HERE, AND WHAT IT IS CONDITIONAL ON
--------------------------------------------------------
The claim is that the content-hash comparison over the dependency closure
agrees with the by-construction ground truth on every evidence item. The
comparison is over 64-character hexadecimal digests and integer counts, so
there is no rounding in it and "agreeing to within rounding" does not arise;
the only floating-point quantity reported is the stale fraction itself, which
is one division of two exact integers and is compared to the ground truth's
same division. The agreement is therefore exact in exact arithmetic, and the
two fractions are additionally compared to a tolerance that follows the
conditioning of that division.

It is conditional on three things, each measured rather than assumed:

1. **Complete dependency declarations.** Section 4 measures what happens when
   30 per cent of the true edges are withheld: the comparison is no longer
   exact, and its error is one-directional.
2. **No digest collision.** The pairwise union bound for the corpus actually
   used is printed by ``validate_corpus_pin.py``.
3. **The closure depth limit not binding.** The deepest closure in the corpus
   is printed by ``validate_corpus_pin.py`` against the default limit.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from evidencestale.baselines import (  # noqa: E402
    baseline_closure_hash,
    baseline_timestamp,
    baseline_timestamp_depth1,
    committed_labels,
    ground_truth,
)
from evidencestale.bootstrap import bootstrap_stale_fraction  # noqa: E402
from evidencestale.corpus import CYCLIC_CORPUS, MAIN_CORPUS, load_corpus  # noqa: E402
from evidencestale.metrics import error_directions  # noqa: E402
from evidencestale.report import build_report  # noqa: E402

#: Tolerance on a comparison of two stale fractions. Each is one division of
#: two exact integers, relative condition number 1, so four units in the last
#: place at the magnitude compared is the conditioning-derived slack.
FRACTION_SLACK = 4.0 * sys.float_info.epsilon


def main() -> int:
    """Validate baseline 1 and baseline 2 against baseline 3."""
    out = _bootstrap.Tee("validate_baselines_output.txt")
    out("Baselines against the by-construction ground truth -- P098 EvidenceStale")
    out("")
    out("baseline 1  timestamp comparison, the thing being refuted")
    out("baseline 1b timestamp comparison over the subject and its DIRECT dependencies")
    out("            -- the strongest honest form of the same idea")
    out("baseline 2  SHA-256 comparison over the dependency CLOSURE")
    out("baseline 3  the replayed change history, which is the truth")
    out("")
    out(f"fast path (EVIDENCESTALE_FAST=1): {_bootstrap.FAST}")
    out(f"bootstrap resamples: {_bootstrap.RESAMPLES}")
    out(f"confidence level: {_bootstrap.CONFIDENCE}")
    out(f"seed: {_bootstrap.SEED}")
    out(f"fraction comparison slack: {FRACTION_SLACK:.6e}")

    corpus = load_corpus(ROOT / MAIN_CORPUS)
    items = corpus.evidence

    out.rule("1. THE HEADLINE OUTPUT -- stale fraction of the assurance case")
    report = build_report(
        corpus,
        "complete",
        resamples=_bootstrap.RESAMPLES,
        confidence=_bootstrap.CONFIDENCE,
        seed=_bootstrap.SEED,
    )
    truth = report.truth
    out(f"corpus                              : {corpus.name}")
    out(f"evidence items                      : {truth.total}")
    out(f"truly stale items                   : {truth.stale_count}")
    out(f"STALE FRACTION (ground truth)       : {truth.stale_fraction:.6f}")
    out(f"STALE FRACTION (closure comparison) : {report.closure_hash.stale_fraction:.6f}")
    out(f"goal-weighted stale fraction        : {report.goal_stale_fraction:.6f}")
    out(f"GSN leaf goals                      : {report.goal_count}")
    out("")
    out("THE RESAMPLING UNIT IS DECLARED, AND BOTH UNITS ARE REPORTED.")
    out("Evidence items that share a subject artifact go stale together, so the")
    out("cluster bootstrap over subject artifacts is the primary interval and the")
    out("item bootstrap is reported beside it to show what the dependence costs.")
    for interval in (report.cluster_interval, report.item_interval):
        out("")
        out(f"  unit        : {interval.unit}")
        out(f"  units        : {interval.n_units}")
        out(f"  resamples    : {interval.resamples}")
        out(f"  point        : {interval.point:.6f}")
        out(f"  interval     : [{interval.low:.6f}, {interval.high:.6f}]")
        out(f"  half width   : {interval.half_width:.6f}")
    out("")
    out.check(
        "the cluster interval contains the point estimate",
        report.cluster_interval.contains(report.cluster_interval.point),
    )
    out.check(
        "the item interval contains the point estimate",
        report.item_interval.contains(report.item_interval.point),
    )
    out.check(
        "the cluster interval is not narrower than the item interval on this corpus",
        report.cluster_interval.half_width >= report.item_interval.half_width,
        f"cluster {report.cluster_interval.half_width:.6f} against item "
        f"{report.item_interval.half_width:.6f}",
    )

    out.rule("2. BASELINE 1 -- both error directions, reported separately")
    out("the two directions are different failures with different consequences for")
    out("an audit, and they are never summed into one accuracy figure.")
    for method in ("baseline1_timestamp", "baseline1b_timestamp_depth1"):
        errors = report.errors(method)
        verdicts = (
            report.timestamp if method == "baseline1_timestamp" else report.timestamp_depth1
        )
        out("")
        out(f"{method}")
        out(f"  stale fraction reported   : {verdicts.stale_fraction:.6f}")
        out(f"  true positives            : {errors.true_positive}")
        out(f"  MISSED STALENESS (FN)     : {errors.false_negative} of "
            f"{errors.actually_stale} truly stale")
        out(f"  missed-staleness rate     : {errors.missed_staleness_rate:.6f}")
        out(f"  FALSE ALARMS (FP)         : {errors.false_positive} of "
            f"{errors.actually_current} truly current")
        out(f"  false-alarm rate          : {errors.false_alarm_rate:.6f}")
        out(f"  agreement with the truth  : {errors.agreement:.6f}")
        out.check(
            f"{method} makes errors in BOTH directions, so neither can be ignored",
            errors.false_negative > 0 and errors.false_positive > 0,
            f"FN {errors.false_negative}, FP {errors.false_positive}",
        )
    plain = report.errors("baseline1_timestamp")
    stronger = report.errors("baseline1b_timestamp_depth1")
    out("")
    out("reading one level of the declared graph trades missed staleness for false")
    out("alarms. It does not remove either, because a change at depth 2 or more is")
    out("invisible to a modification-revision comparison however many levels it")
    out("reads short of the whole closure.")
    out(f"  missed staleness : {plain.false_negative} -> {stronger.false_negative}")
    out(f"  false alarms     : {plain.false_positive} -> {stronger.false_positive}")
    out.check(
        "the depth-1 variant reduces missed staleness",
        stronger.false_negative < plain.false_negative,
    )
    out.check(
        "the depth-1 variant increases false alarms",
        stronger.false_positive > plain.false_positive,
    )
    out.check(
        "the depth-1 variant still misses some staleness",
        stronger.false_negative > 0,
        f"{stronger.false_negative} still missed",
    )

    out.rule("3. BASELINE 2 -- the exactness claim, proved against the ground truth")
    closure_errors = report.errors("baseline2_closure_hash")
    out(f"items compared                   : {closure_errors.total}")
    out(f"missed staleness                 : {closure_errors.false_negative}")
    out(f"false alarms                     : {closure_errors.false_positive}")
    out(f"agreement                        : {closure_errors.agreement:.6f}")
    out(f"stale fraction, closure          : {report.closure_hash.stale_fraction:.17g}")
    out(f"stale fraction, ground truth     : {truth.stale_fraction:.17g}")
    difference = abs(report.closure_hash.stale_fraction - truth.stale_fraction)
    out(f"absolute difference              : {difference:.6e}")
    out.check(
        "the closure comparison makes no missed-staleness error on any item",
        closure_errors.false_negative == 0,
    )
    out.check(
        "the closure comparison makes no false-alarm error on any item",
        closure_errors.false_positive == 0,
    )
    out.check(
        "the closure comparison agrees with the ground truth item by item",
        report.closure_hash.as_array() == truth.as_array(),
    )
    out.check(
        "the two stale fractions agree within the conditioning-derived slack",
        difference <= FRACTION_SLACK * max(1.0, truth.stale_fraction),
        f"{difference:.3e} against {FRACTION_SLACK:.3e}",
    )
    out("")
    out("the same claim on the CYCLIC corpus, which is not a directed acyclic graph.")
    cyclic = load_corpus(ROOT / CYCLIC_CORPUS)
    cyclic_truth = ground_truth(
        cyclic.evidence, cyclic.history, cyclic.true_graph, cyclic.evaluation_time
    )
    cyclic_closure = baseline_closure_hash(
        cyclic.evidence, cyclic.history, cyclic.true_graph, cyclic.evaluation_time
    )
    cyclic_errors = error_directions(cyclic_closure.as_array(), cyclic_truth.as_array())
    out(f"  artifacts                 : {len(cyclic.true_graph.nodes)}")
    out(f"  acyclic                   : {cyclic.true_graph.is_acyclic()}")
    out(f"  cycles                    : {cyclic.true_graph.cycles()}")
    out(f"  evidence items            : {cyclic_truth.total}")
    out(f"  stale fraction, truth     : {cyclic_truth.stale_fraction:.6f}")
    out(f"  stale fraction, closure   : {cyclic_closure.stale_fraction:.6f}")
    out(f"  missed staleness          : {cyclic_errors.false_negative}")
    out(f"  false alarms              : {cyclic_errors.false_positive}")
    out.check(
        "the closure comparison is exact on a graph containing a cycle",
        cyclic_closure.as_array() == cyclic_truth.as_array(),
    )
    out("")
    out("THE EXACTNESS IS NOT A TAUTOLOGY, AND IT IS NOT INDEPENDENT EITHER. Said")
    out("plainly: the closure comparison and the ground truth implement the SAME")
    out("definition of staleness. They differ in representation -- one compares")
    out("SHA-256 digests of contents reconstructed from the event log at two")
    out("revisions, the other compares the content values themselves, and the")
    out("committed labels were written by the generator before this package read")
    out("them. So the agreement tests the closure search, the replay and the")
    out("hashing against each other and against the generator; it does not test")
    out("the definition, and no measurement could. The independent check on the")
    out("definition is the bracket in validate_bound_bracket.py, whose bound comes")
    out("from counting arguments this implementation does not use.")

    out.rule("4. WHAT THE EXACTNESS IS CONDITIONAL ON -- incomplete declarations")
    out("30 per cent of the true dependency edges are withheld from the declared")
    out("set, which is what an incomplete dependency declaration looks like. The")
    out("ground truth is still computed over the TRUE graph.")
    out("")
    partial = baseline_closure_hash(
        items, corpus.history, corpus.declared_graph, corpus.evaluation_time
    )
    partial_errors = error_directions(partial.as_array(), truth.as_array())
    out(f"true edges                 : {len(corpus.true_graph.edges)}")
    out(f"declared edges             : {len(corpus.declared_graph.edges)}")
    out(f"hidden edges               : {corpus.hidden_edge_count()}")
    out(f"stale fraction reported    : {partial.stale_fraction:.6f}")
    out(f"stale fraction, truth      : {truth.stale_fraction:.6f}")
    out(f"MISSED STALENESS           : {partial_errors.false_negative} of "
        f"{partial_errors.actually_stale}")
    out(f"missed-staleness rate      : {partial_errors.missed_staleness_rate:.6f}")
    out(f"FALSE ALARMS               : {partial_errors.false_positive}")
    out(f"false-alarm rate           : {partial_errors.false_alarm_rate:.6f}")
    out(f"agreement                  : {partial_errors.agreement:.6f}")
    out.check(
        "an incomplete declaration breaks the exactness",
        partial_errors.false_negative > 0,
        f"{partial_errors.false_negative} items missed",
    )
    out.check(
        "an undeclared edge can only HIDE a change, never invent one, so the "
        "error is one-directional",
        partial_errors.false_positive == 0,
        f"{partial_errors.false_positive} false alarms",
    )
    out.check(
        "the reported stale fraction under-states the truth when edges are hidden",
        partial.stale_fraction < truth.stale_fraction,
        f"{partial.stale_fraction:.6f} < {truth.stale_fraction:.6f}",
    )

    out.rule("5. THE COMMITTED LABELS, AND THE AGGREGATE-MATCHES-ITEMS TRAP")
    committed = committed_labels(items)
    out(f"committed stale items : {committed.stale_count}")
    out(f"replayed stale items  : {truth.stale_count}")
    out.check(
        "the committed by-construction labels agree with the replay item by item",
        committed.as_array() == truth.as_array(),
    )
    out("")
    out("An aggregate stale fraction can be right while every item is wrong. On the")
    out("hand example in tests/test_baselines.py the timestamp baseline reports 0.5")
    out("against a true 0.5 with an agreement of 0.0. On this corpus:")
    timestamp = baseline_timestamp(
        items, corpus.history, corpus.true_graph, corpus.evaluation_time
    )
    depth1 = baseline_timestamp_depth1(
        items, corpus.history, corpus.true_graph, corpus.evaluation_time
    )
    for name, verdicts in (
        ("baseline1_timestamp", timestamp),
        ("baseline1b_timestamp_depth1", depth1),
    ):
        errors = error_directions(verdicts.as_array(), truth.as_array())
        gap = verdicts.stale_fraction - truth.stale_fraction
        out(
            f"  {name:<28} fraction {verdicts.stale_fraction:.6f}  "
            f"error {gap:+.6f}  agreement {errors.agreement:.6f}"
        )
    out("")
    out("so the aggregate error of the timestamp baseline understates how wrong it")
    out("is per item, and the two error directions partly cancel in the aggregate.")
    out("That cancellation is the reason this product reports both directions.")

    out.rule("6. THE BOOTSTRAP UNIT MUST BE PASSED -- no default is accepted")
    try:
        bootstrap_stale_fraction(truth.as_array(), corpus.subjects(), unit="block")
        refused = False
        message = ""
    except ValueError as error:
        refused = True
        message = str(error).splitlines()[0]
    out(f"calling the bootstrap with unit='block' refused: {refused}")
    out(f"message: {message[:120]}")
    out.check(
        "an unknown resampling unit is refused rather than silently defaulted",
        refused,
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
