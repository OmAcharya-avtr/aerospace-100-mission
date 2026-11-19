"""Negative controls: inject known staleness at a known rate and assert each fired.

EVERY CONTROL HERE IS ANCHORED ON THE TRUE VALUE, NOT ON AN OUTPUT. The
injection is applied to the change history, the ground truth is RECOMPUTED over
the true dependency graph from the modified history, and the control's
criterion is an exact integer count of items whose true staleness changed.
Nothing is compared against a previous reading of a detector, and no control's
criterion is a confidence interval, so no control can be satisfied by a lucky
seed. An earlier product of this mission shipped two of eleven controls
silently too weak by offsetting an output instead of the truth; that is the
failure this arrangement exists to avoid, and it is the worse failure of the
two because a control that cannot fire reads as evidence.

EVERY CONTROL IS ASSERTED TO BEHAVE AS DECLARED, in both directions: a control
that must fire and did not is a failure, and so is one that must stay silent
and did not.

INJECTION REVISIONS ARE 2001 AND ABOVE, and the audit revision is 2200. Every
evidence item in the corpus is produced in revisions 1..600 and every committed
event lies in revisions 1..2000, so an injected event cannot collide with a
committed one and cannot precede any item's production. The unmodified
baselines at revision 2200 therefore agree with the published figures at
revision 2000, which is checked below rather than assumed.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from evidencestale.baselines import (  # noqa: E402
    baseline_closure_hash,
    baseline_timestamp,
    ground_truth,
)
from evidencestale.corpus import MAIN_CORPUS, load_corpus  # noqa: E402
from evidencestale.graph import DependencyGraph  # noqa: E402
from evidencestale.metrics import error_directions  # noqa: E402
from evidencestale.model import ChangeEvent  # noqa: E402
from evidencestale.replay import ChangeHistory  # noqa: E402

#: Audit revision used by every control.
AUDIT = 2200

#: First revision available for an injected event.
FIRST_INJECTION = 2001

#: Declared injection rates, as a count of artifacts out of 240.
DEEP_CHANGE_COUNT = 24
TOUCH_ONLY_COUNT = 30
REVERT_COUNT = 12
HIDDEN_EDGE_COUNT = 40

#: Slack on a comparison of two stale fractions: one division of exact
#: integers, relative condition number 1, four units in the last place.
FRACTION_SLACK = 4.0 * sys.float_info.epsilon


def rebuilt(corpus, extra: list[ChangeEvent]) -> ChangeHistory:
    """The corpus history with ``extra`` events appended."""
    return ChangeHistory(
        corpus.history.artifacts,
        corpus.history.initial,
        [*corpus.history.events, *extra],
    )


def main() -> int:
    """Run every negative control and assert each behaved as declared."""
    out = _bootstrap.Tee("validate_negative_controls_output.txt")
    out("Negative controls -- P098 EvidenceStale")
    out("")
    out("each control is anchored on the TRUE value: the injection is applied to the")
    out("change history and the ground truth is RECOMPUTED over the true dependency")
    out("graph. Each control's criterion is an exact integer count, so no control")
    out("can be satisfied by a lucky seed.")
    out("")
    out(f"fast path (EVIDENCESTALE_FAST=1): {_bootstrap.FAST}")
    out(f"seed: {_bootstrap.SEED}")
    out(f"audit revision: {AUDIT}")
    out(f"first injection revision: {FIRST_INJECTION}")

    corpus = load_corpus(ROOT / MAIN_CORPUS)
    items = corpus.evidence
    rng = np.random.default_rng(_bootstrap.SEED)
    artifacts = list(corpus.true_graph.nodes)
    leaves = [
        name for name in artifacts if not corpus.true_graph.dependencies()[name]
    ]
    subjects_with_dependents = [
        name for name in artifacts if corpus.true_graph.dependents()[name]
    ]

    out.rule("0. THE UNMODIFIED BASELINE AT REVISION 2200")
    out("no committed event lies after revision 2000, so auditing at 2200 must give")
    out("the published figures. Checked rather than assumed, because every control")
    out("below is a difference against this row.")
    base_truth = ground_truth(items, corpus.history, corpus.true_graph, AUDIT)
    base_closure = baseline_closure_hash(
        items, corpus.history, corpus.true_graph, AUDIT
    )
    base_timestamp = baseline_timestamp(
        items, corpus.history, corpus.true_graph, AUDIT
    )
    truth_at_2000 = ground_truth(
        items, corpus.history, corpus.true_graph, corpus.evaluation_time
    )
    out(f"truly stale at revision 2000 : {truth_at_2000.stale_count}")
    out(f"truly stale at revision 2200 : {base_truth.stale_count}")
    out(f"stale fraction at 2200       : {base_truth.stale_fraction:.6f}")
    out(f"closure comparison at 2200   : {base_closure.stale_count}")
    out(f"timestamp baseline at 2200   : {base_timestamp.stale_count}")
    out.check(
        "the ground truth at revision 2200 equals the ground truth at revision 2000",
        base_truth.as_array() == truth_at_2000.as_array(),
    )
    out.check(
        "the closure comparison is exact on the unmodified corpus at revision 2200",
        base_closure.as_array() == base_truth.as_array(),
    )

    out.rule("1. CONTROL -- an injected change at DEPTH 2, at a declared rate")
    out(f"{DEEP_CHANGE_COUNT} leaf artifacts (no dependencies of their own) are")
    out("written with new content at revisions 2001 and above. A leaf is reached")
    out("from an evidence subject only through the dependency graph, so this is")
    out("staleness that arrives through a dependency and never touches the subject.")
    chosen = sorted(
        str(name)
        for name in rng.choice(leaves, size=DEEP_CHANGE_COUNT, replace=False)
    )
    injected = [
        ChangeEvent(
            time=FIRST_INJECTION + index,
            artifact=name,
            kind="write",
            content=f"{name}:injected",
        )
        for index, name in enumerate(chosen)
    ]
    history = rebuilt(corpus, injected)
    new_truth = ground_truth(items, history, corpus.true_graph, AUDIT)
    new_closure = baseline_closure_hash(items, history, corpus.true_graph, AUDIT)
    new_timestamp = baseline_timestamp(items, history, corpus.true_graph, AUDIT)
    true_increment = new_truth.stale_count - base_truth.stale_count
    closure_increment = new_closure.stale_count - base_closure.stale_count
    timestamp_increment = new_timestamp.stale_count - base_timestamp.stale_count
    newly_true = {
        item.identifier
        for item in items
        if new_truth.verdicts[item.identifier] and not base_truth.verdicts[item.identifier]
    }
    newly_timestamp = {
        identifier
        for identifier in newly_true
        if new_timestamp.verdicts[identifier] and not base_timestamp.verdicts[identifier]
    }
    # Every artifact in this corpus is the subject of two evidence items, so a
    # written leaf is also somebody's subject and the timestamp baseline CAN
    # see those. The structural failure is the items made stale ONLY through a
    # dependency: their subject's modification revision is untouched by the
    # injection, so the timestamp baseline cannot see them however the audit is
    # timed. Both groups are counted, because reporting only the second would
    # overstate the failure.
    chosen_set = set(chosen)
    through_dependency = {
        item.identifier
        for item in items
        if item.identifier in newly_true and item.subject not in chosen_set
    }
    direct = newly_true - through_dependency
    newly_timestamp_through_dependency = newly_timestamp & through_dependency
    out("")
    out(f"artifacts written          : {len(chosen)} of {len(artifacts)} "
        f"({len(chosen) / len(artifacts):.6f})")
    out(f"TRUE newly stale items     : {true_increment}")
    out(f"  of those, subject written directly        : {len(direct)}")
    out(f"  of those, stale ONLY through a dependency : {len(through_dependency)}")
    out(f"closure comparison increase: {closure_increment}")
    out(f"timestamp baseline increase: {timestamp_increment}")
    out(f"newly stale items the timestamp baseline newly flags: {len(newly_timestamp)}")
    out(f"  of the depth-2-only group, newly flagged: "
        f"{len(newly_timestamp_through_dependency)}")
    out(f"true stale fraction before : {base_truth.stale_fraction:.6f}")
    out(f"true stale fraction after  : {new_truth.stale_fraction:.6f}")
    out.control(
        "an injected depth-2 change makes a positive known number of items stale",
        true_increment > 0,
        must_fire=True,
        detail=f"{true_increment} items",
    )
    out.control(
        "the closure comparison recovers the injected staleness exactly",
        closure_increment == true_increment
        and new_closure.as_array() == new_truth.as_array(),
        must_fire=True,
        detail=f"recovered {closure_increment} of {true_increment}",
    )
    out.control(
        "the injection makes a positive number of items stale ONLY through a "
        "dependency",
        len(through_dependency) > 0,
        must_fire=True,
        detail=f"{len(through_dependency)} items",
    )
    out.control(
        "the timestamp baseline newly flags NONE of the items made stale only "
        "through a dependency",
        len(newly_timestamp_through_dependency) == 0,
        must_fire=True,
        detail=f"{len(newly_timestamp_through_dependency)} of "
        f"{len(through_dependency)} newly flagged",
    )

    out.rule("2. CONTROL -- injected TOUCH-ONLY events, at a declared rate")
    out(f"{TOUCH_ONLY_COUNT} artifacts that are evidence subjects are TOUCHED at")
    out("revisions 2001 and above. A touch bumps the modification revision and")
    out("leaves the content identical, so by construction the true staleness of")
    out("every item is unchanged. The closure comparison must stay silent and the")
    out("timestamp baseline must raise new false alarms.")
    touched = sorted(
        str(name)
        for name in rng.choice(
            subjects_with_dependents, size=TOUCH_ONLY_COUNT, replace=False
        )
    )
    touch_events = [
        ChangeEvent(time=FIRST_INJECTION + 100 + index, artifact=name, kind="touch")
        for index, name in enumerate(touched)
    ]
    history = rebuilt(corpus, touch_events)
    touch_truth = ground_truth(items, history, corpus.true_graph, AUDIT)
    touch_closure = baseline_closure_hash(items, history, corpus.true_graph, AUDIT)
    touch_timestamp = baseline_timestamp(items, history, corpus.true_graph, AUDIT)
    base_errors = error_directions(base_timestamp.as_array(), base_truth.as_array())
    touch_errors = error_directions(
        touch_timestamp.as_array(), touch_truth.as_array()
    )
    out("")
    out(f"artifacts touched            : {len(touched)}")
    out(f"TRUE stale count before      : {base_truth.stale_count}")
    out(f"TRUE stale count after       : {touch_truth.stale_count}")
    out(f"closure comparison before    : {base_closure.stale_count}")
    out(f"closure comparison after     : {touch_closure.stale_count}")
    out(f"timestamp false alarms before: {base_errors.false_positive}")
    out(f"timestamp false alarms after : {touch_errors.false_positive}")
    out.control(
        "a touch-only injection changes no item's true staleness",
        touch_truth.as_array() != base_truth.as_array(),
        must_fire=False,
        detail=f"{touch_truth.stale_count} against {base_truth.stale_count}",
    )
    out.control(
        "the closure comparison is unmoved by a touch-only injection",
        touch_closure.as_array() != base_closure.as_array(),
        must_fire=False,
        detail=f"{touch_closure.stale_count} against {base_closure.stale_count}",
    )
    out.control(
        "a touch-only injection raises the timestamp baseline's false alarms",
        touch_errors.false_positive > base_errors.false_positive,
        must_fire=True,
        detail=f"{base_errors.false_positive} -> {touch_errors.false_positive}",
    )

    out.rule("3. CONTROL -- an injected WRITE AND REVERT, at a declared rate")
    out(f"{REVERT_COUNT} leaf artifacts are written to a new content and then")
    out("written back to the content they held, both at revisions 2001 and above.")
    out("The value at the audit revision equals the value at every item's")
    out("production revision, so by construction nothing is stale that was not")
    out("already. A method that counted EVENTS rather than comparing VALUES would")
    out("flag all of them.")
    reverted = sorted(
        str(name) for name in rng.choice(leaves, size=REVERT_COUNT, replace=False)
    )
    contents_now = corpus.history.contents_at(corpus.evaluation_time)
    revert_events: list[ChangeEvent] = []
    for index, name in enumerate(reverted):
        revert_events.append(
            ChangeEvent(
                time=FIRST_INJECTION + 200 + 2 * index,
                artifact=name,
                kind="write",
                content=f"{name}:transient",
            )
        )
        revert_events.append(
            ChangeEvent(
                time=FIRST_INJECTION + 201 + 2 * index,
                artifact=name,
                kind="write",
                content=contents_now[name],
            )
        )
    history = rebuilt(corpus, revert_events)
    revert_truth = ground_truth(items, history, corpus.true_graph, AUDIT)
    revert_closure = baseline_closure_hash(items, history, corpus.true_graph, AUDIT)
    revert_timestamp = baseline_timestamp(items, history, corpus.true_graph, AUDIT)
    revert_errors = error_directions(
        revert_timestamp.as_array(), revert_truth.as_array()
    )
    event_counting_would_flag = sum(
        1
        for item in items
        if corpus.true_graph.closure(item.subject) & set(reverted)
    )
    out("")
    out(f"artifacts written and reverted     : {len(reverted)}")
    out(f"events injected                    : {len(revert_events)}")
    out(f"TRUE stale count before            : {base_truth.stale_count}")
    out(f"TRUE stale count after             : {revert_truth.stale_count}")
    out(f"closure comparison after           : {revert_closure.stale_count}")
    out(f"items an EVENT-counting method would flag: {event_counting_would_flag}")
    out(f"timestamp false alarms after       : {revert_errors.false_positive}")
    out.control(
        "a write-and-revert injection changes no item's true staleness",
        revert_truth.as_array() != base_truth.as_array(),
        must_fire=False,
        detail=f"{revert_truth.stale_count} against {base_truth.stale_count}",
    )
    out.control(
        "the closure comparison is unmoved by a write-and-revert injection",
        revert_closure.as_array() != base_closure.as_array(),
        must_fire=False,
    )
    out.control(
        "an event-counting method would have flagged a positive number of items "
        "that are not stale",
        event_counting_would_flag > 0,
        must_fire=True,
        detail=f"{event_counting_would_flag} items",
    )

    out.rule("4. CONTROL -- a declared number of edges HIDDEN from the declaration")
    out(f"{HIDDEN_EDGE_COUNT} true dependency edges are withheld from the graph the")
    out("closure comparison reads, on top of the corpus's own declared subset being")
    out("complete here. The ground truth is unchanged because it uses the true")
    out("graph, so the closure comparison must start missing staleness.")
    true_edges = list(corpus.true_graph.edges)
    hidden_positions = sorted(
        int(position)
        for position in rng.choice(
            len(true_edges), size=HIDDEN_EDGE_COUNT, replace=False
        )
    )
    hidden = {true_edges[position] for position in hidden_positions}
    crippled = DependencyGraph.build(
        corpus.true_graph.nodes, [edge for edge in true_edges if edge not in hidden]
    )
    crippled_closure = baseline_closure_hash(
        items, corpus.history, crippled, AUDIT
    )
    crippled_errors = error_directions(
        crippled_closure.as_array(), base_truth.as_array()
    )
    out("")
    out(f"true edges                 : {len(true_edges)}")
    out(f"edges hidden               : {len(hidden)}")
    out(f"TRUE stale count           : {base_truth.stale_count}")
    out(f"closure comparison reports : {crippled_closure.stale_count}")
    out(f"MISSED STALENESS           : {crippled_errors.false_negative}")
    out(f"FALSE ALARMS               : {crippled_errors.false_positive}")
    out.control(
        "hiding declared edges makes the closure comparison miss staleness",
        crippled_errors.false_negative > 0,
        must_fire=True,
        detail=f"{crippled_errors.false_negative} missed",
    )
    out.control(
        "hiding declared edges never makes the closure comparison raise a false alarm",
        crippled_errors.false_positive > 0,
        must_fire=False,
        detail=f"{crippled_errors.false_positive} false alarms",
    )

    out.rule("5. CONTROL -- the recovered RATE against the injected rate")
    out("the deep-change injection of section 1 is re-applied and the increase in")
    out("the measured stale FRACTION is compared against the increase in the TRUE")
    out("stale fraction. Both are one division of exact integers, so the comparison")
    out("carries a slack of four units in the last place and nothing more.")
    history = rebuilt(corpus, injected)
    rate_truth = ground_truth(items, history, corpus.true_graph, AUDIT)
    rate_closure = baseline_closure_hash(items, history, corpus.true_graph, AUDIT)
    true_delta = rate_truth.stale_fraction - base_truth.stale_fraction
    measured_delta = rate_closure.stale_fraction - base_closure.stale_fraction
    difference = abs(true_delta - measured_delta)
    out("")
    out(f"TRUE increase in the stale fraction     : {true_delta:.17g}")
    out(f"MEASURED increase in the stale fraction : {measured_delta:.17g}")
    out(f"absolute difference                     : {difference:.6e}")
    out(f"slack                                   : {FRACTION_SLACK:.6e}")
    out.control(
        "the injected rate is a positive change in the true stale fraction",
        true_delta > 0.0,
        must_fire=True,
        detail=f"{true_delta:+.6f}",
    )
    out.control(
        "the recovered rate differs from the injected rate by more than the slack",
        difference > FRACTION_SLACK,
        must_fire=False,
        detail=f"{difference:.3e} against {FRACTION_SLACK:.3e}",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
