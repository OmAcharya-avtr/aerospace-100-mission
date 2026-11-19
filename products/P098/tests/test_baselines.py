"""The three baselines, with hand-calculated known answers.

THE HAND EXAMPLE, worked once here and reused through the file.

Graph:   top -> middle -> bottom   (so closure(top) = {top, middle, bottom})
History: revision 0 lays down top:v0, middle:v0, bottom:v0
         revision 4  write  bottom := bottom:v1   (a real content change)
         revision 6  touch  top                   (no content change)
         revision 8  write  bottom := bottom:v0   (a revert to the old value)
Evidence: e0 about top produced at revision 1
          e1 about top produced at revision 7
Audit at revision 10.

Contents at revision 10: top:v0, middle:v0, bottom:v0.
Modification revisions at 10: top 6, middle 0, bottom 8.

GROUND TRUTH
  e0: closure contents at 1 were (v0, v0, v0); at 10 they are (v0, v0, v0).
      Nothing differs, so e0 is NOT stale. The two writes to bottom cancel.
  e1: closure contents at 7 were (v0, v0, v1); at 10 they are (v0, v0, v0).
      bottom differs, so e1 IS stale, through a dependency at depth 2.
  truth = [0, 1], stale fraction 1/2 = 0.5

BASELINE 1, timestamp comparison on the subject only
  e0: modified(top) = 6 > produced 1  -> calls it stale.  Truth 0 -> FALSE ALARM
  e1: modified(top) = 6 > produced 7? no -> calls it current. Truth 1 -> MISSED
  baseline 1 = [1, 0], stale fraction 1/2 = 0.5
  confusion: TP 0, FN 1, FP 1, TN 0
  missed-staleness rate = 1/1 = 1.0 ; false-alarm rate = 1/1 = 1.0
  agreement = (0 + 0)/2 = 0.0

  So the aggregate stale fraction of baseline 1 EQUALS the truth, 0.5 against
  0.5, while every single item is wrong. That is the whole argument for
  reporting the two error directions separately and never the aggregate alone.

BASELINE 1b, strengthened to the subject and its direct dependencies
  neighbourhood of top = {top, middle}; max modification revision = max(6, 0) = 6
  e0: 6 > 1 -> stale (still a FALSE ALARM)
  e1: 6 > 7? no -> current (still MISSED, because bottom is at depth 2)

BASELINE 2, digest comparison over the closure
  e0: digests at 1 equal digests at 10 -> current. Correct.
  e1: digest of bottom differs -> stale. Correct.
  agreement = 2/2 = 1.0
"""

from __future__ import annotations

import pytest

from evidencestale.baselines import (
    baseline_closure_hash,
    baseline_timestamp,
    baseline_timestamp_depth1,
    closure_snapshot,
    committed_labels,
    ground_truth,
)
from evidencestale.graph import DependencyGraph
from evidencestale.hashing import content_hash
from evidencestale.metrics import error_directions
from evidencestale.model import EvidenceItem

AUDIT = 10


def test_ground_truth_known_answer(chain_graph, chain_history, chain_items) -> None:
    truth = ground_truth(chain_items, chain_history, chain_graph, AUDIT)
    assert truth.as_array() == [0, 1]
    assert truth.stale_count == 1
    assert truth.total == 2
    assert truth.stale_fraction == pytest.approx(0.5, rel=0.0, abs=0.0)


def test_baseline1_known_answer_and_both_error_directions(
    chain_graph, chain_history, chain_items
) -> None:
    truth = ground_truth(chain_items, chain_history, chain_graph, AUDIT)
    verdicts = baseline_timestamp(chain_items, chain_history, chain_graph, AUDIT)
    assert verdicts.as_array() == [1, 0]
    errors = error_directions(verdicts.as_array(), truth.as_array())
    assert (errors.true_positive, errors.false_negative) == (0, 1)
    assert (errors.false_positive, errors.true_negative) == (1, 0)
    assert errors.missed_staleness_rate == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert errors.false_alarm_rate == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert errors.agreement == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_baseline1_aggregate_matches_the_truth_while_every_item_is_wrong(
    chain_graph, chain_history, chain_items
) -> None:
    truth = ground_truth(chain_items, chain_history, chain_graph, AUDIT)
    verdicts = baseline_timestamp(chain_items, chain_history, chain_graph, AUDIT)
    assert verdicts.stale_fraction == pytest.approx(
        truth.stale_fraction, rel=0.0, abs=0.0
    )
    errors = error_directions(verdicts.as_array(), truth.as_array())
    assert errors.agreement == 0.0


def test_baseline1_depth1_is_stronger_but_still_misses_depth_two(
    chain_graph, chain_history, chain_items
) -> None:
    verdicts = baseline_timestamp_depth1(chain_items, chain_history, chain_graph, AUDIT)
    assert verdicts.as_array() == [1, 0]


def test_baseline1_depth1_does_see_a_depth_one_change() -> None:
    # A two-artifact graph: report -> source. A write to source at revision 4
    # is at depth 1 from report, which the depth-1 variant can see and the
    # subject-only variant cannot.
    from evidencestale.model import ChangeEvent
    from evidencestale.replay import ChangeHistory

    graph = DependencyGraph.build(["report", "source"], [("report", "source")])
    history = ChangeHistory(
        ["report", "source"],
        {"report": "r:v0", "source": "s:v0"},
        [ChangeEvent(4, "source", "write", "s:v1")],
    )
    items = (
        EvidenceItem(
            identifier="e", subject="report", produced_at=1, goal="G", solution="Sn"
        ),
    )
    assert baseline_timestamp(items, history, graph, 6).as_array() == [0]
    assert baseline_timestamp_depth1(items, history, graph, 6).as_array() == [1]
    assert ground_truth(items, history, graph, 6).as_array() == [1]


def test_baseline2_known_answer_is_exact_on_the_hand_example(
    chain_graph, chain_history, chain_items
) -> None:
    truth = ground_truth(chain_items, chain_history, chain_graph, AUDIT)
    verdicts = baseline_closure_hash(chain_items, chain_history, chain_graph, AUDIT)
    assert verdicts.as_array() == [0, 1]
    errors = error_directions(verdicts.as_array(), truth.as_array())
    assert errors.false_negative == 0
    assert errors.false_positive == 0
    assert errors.agreement == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_closure_snapshot_digests_are_the_published_literals(
    chain_graph, chain_history
) -> None:
    snapshot = closure_snapshot(chain_history, chain_graph, "top", 7)
    assert snapshot == {
        "bottom": content_hash("bottom:v1"),
        "middle": content_hash("middle:v0"),
        "top": content_hash("top:v0"),
    }
    assert snapshot["bottom"] == (
        "30a248148484e2c0d9854f6416531008f07605d1a47950e4020cbe31d6b1ed43"
    )


def test_stale_fraction_of_an_empty_item_set_is_zero_and_total_is_zero(
    chain_graph, chain_history
) -> None:
    verdicts = ground_truth((), chain_history, chain_graph, AUDIT)
    assert verdicts.total == 0
    assert verdicts.stale_fraction == 0.0


def test_evidence_about_an_undeclared_artifact_rejected(
    chain_graph, chain_history
) -> None:
    items = (
        EvidenceItem(
            identifier="e", subject="nope", produced_at=0, goal="G", solution="Sn"
        ),
    )
    with pytest.raises(ValueError, match="not a declared artifact"):
        baseline_timestamp(items, chain_history, chain_graph, AUDIT)


def test_duplicate_evidence_identifier_rejected(chain_graph, chain_history) -> None:
    items = (
        EvidenceItem(identifier="e", subject="top", produced_at=0, goal="G", solution="S1"),
        EvidenceItem(identifier="e", subject="top", produced_at=1, goal="G", solution="S2"),
    )
    with pytest.raises(ValueError, match="appears more than once"):
        ground_truth(items, chain_history, chain_graph, AUDIT)


def test_committed_labels_requires_every_item_to_carry_one(chain_items) -> None:
    with pytest.raises(ValueError, match="no by-construction label"):
        committed_labels(chain_items)


def test_baseline2_is_exact_on_the_committed_layered_corpus(layered_corpus) -> None:
    items = layered_corpus.evidence
    truth = ground_truth(
        items,
        layered_corpus.history,
        layered_corpus.true_graph,
        layered_corpus.evaluation_time,
    )
    verdicts = baseline_closure_hash(
        items,
        layered_corpus.history,
        layered_corpus.true_graph,
        layered_corpus.evaluation_time,
    )
    errors = error_directions(verdicts.as_array(), truth.as_array())
    assert errors.false_negative == 0
    assert errors.false_positive == 0
    assert errors.total == 480


def test_baseline2_is_exact_on_the_cyclic_corpus(cyclic_corpus) -> None:
    items = cyclic_corpus.evidence
    truth = ground_truth(
        items, cyclic_corpus.history, cyclic_corpus.true_graph, cyclic_corpus.evaluation_time
    )
    verdicts = baseline_closure_hash(
        items, cyclic_corpus.history, cyclic_corpus.true_graph, cyclic_corpus.evaluation_time
    )
    assert verdicts.as_array() == truth.as_array()
    assert cyclic_corpus.true_graph.is_acyclic() is False


def test_committed_labels_agree_with_the_replayed_ground_truth(layered_corpus) -> None:
    # The generator wrote these labels while it built the history; the replay
    # recomputes them from the committed event log. They are two computations
    # over two representations of the same definition.
    truth = ground_truth(
        layered_corpus.evidence,
        layered_corpus.history,
        layered_corpus.true_graph,
        layered_corpus.evaluation_time,
    )
    assert committed_labels(layered_corpus.evidence).as_array() == truth.as_array()


def test_partial_declarations_cause_missed_staleness_and_no_false_alarms(
    layered_corpus,
) -> None:
    items = layered_corpus.evidence
    truth = ground_truth(
        items,
        layered_corpus.history,
        layered_corpus.true_graph,
        layered_corpus.evaluation_time,
    )
    verdicts = baseline_closure_hash(
        items,
        layered_corpus.history,
        layered_corpus.declared_graph,
        layered_corpus.evaluation_time,
    )
    errors = error_directions(verdicts.as_array(), truth.as_array())
    # An undeclared edge can only hide a change, never invent one, so the error
    # is one-directional by construction.
    assert errors.false_positive == 0
    assert errors.false_negative > 0
    assert layered_corpus.hidden_edge_count() > 0
