"""Feature extraction: column meaning, hand-calculated values, and what is excluded."""

from __future__ import annotations

import numpy as np
import pytest

from evidencestale.features import FEATURE_NAMES, feature_matrix, leak_feature


def test_feature_names_are_the_published_columns() -> None:
    assert FEATURE_NAMES == (
        "age",
        "closure_size",
        "closure_depth",
        "direct_dependencies",
        "direct_dependents",
        "closure_leaf_count",
        "subject_writes_before",
        "subject_events_before",
        "closure_writes_before",
        "closure_max_writes_before",
        "closure_write_rate_before",
        "subject_touch_share_before",
    )


def test_feature_matrix_shape_and_dtype(chain_graph, chain_history, chain_items) -> None:
    matrix = feature_matrix(chain_items, chain_history, chain_graph, 10)
    assert matrix.shape == (2, len(FEATURE_NAMES))
    assert matrix.dtype == np.float64


def test_feature_values_known_answer_for_the_hand_example(
    chain_graph, chain_history, chain_items
) -> None:
    matrix = feature_matrix(chain_items, chain_history, chain_graph, 10)
    columns = {name: index for index, name in enumerate(FEATURE_NAMES)}

    # Item e0: subject top, produced at revision 1, audited at 10.
    #   age = 10 - 1 = 9
    #   closure(top) = {top, middle, bottom} -> size 3, depth 2
    #   direct dependencies of top = {middle} -> 1
    #   direct dependents of top = {} -> 0
    #   closure leaves (no dependencies of their own) = {bottom} -> 1
    #   events strictly before revision 1: none
    #     subject writes 0, subject events 0, closure writes 0, max 0
    #     rate = 0 / max(1, 1) = 0
    #     touch share = 0 (no events, defined as 0 rather than undefined)
    assert matrix[0, columns["age"]] == 9.0
    assert matrix[0, columns["closure_size"]] == 3.0
    assert matrix[0, columns["closure_depth"]] == 2.0
    assert matrix[0, columns["direct_dependencies"]] == 1.0
    assert matrix[0, columns["direct_dependents"]] == 0.0
    assert matrix[0, columns["closure_leaf_count"]] == 1.0
    assert matrix[0, columns["subject_writes_before"]] == 0.0
    assert matrix[0, columns["subject_events_before"]] == 0.0
    assert matrix[0, columns["closure_writes_before"]] == 0.0
    assert matrix[0, columns["closure_max_writes_before"]] == 0.0
    assert matrix[0, columns["closure_write_rate_before"]] == 0.0
    assert matrix[0, columns["subject_touch_share_before"]] == 0.0

    # Item e1: subject top, produced at revision 7, audited at 10.
    #   age = 10 - 7 = 3
    #   events strictly before revision 7: write bottom at 4, touch top at 6
    #     subject (top) writes 0, subject events 1 -> touch share = (1-0)/1 = 1
    #     closure writes = 1 (the write to bottom), max = 1
    #     rate = 1 / max(1, 7) = 1/7
    assert matrix[1, columns["age"]] == 3.0
    assert matrix[1, columns["subject_writes_before"]] == 0.0
    assert matrix[1, columns["subject_events_before"]] == 1.0
    assert matrix[1, columns["subject_touch_share_before"]] == 1.0
    assert matrix[1, columns["closure_writes_before"]] == 1.0
    assert matrix[1, columns["closure_max_writes_before"]] == 1.0
    assert matrix[1, columns["closure_write_rate_before"]] == pytest.approx(
        1.0 / 7.0, rel=1e-15, abs=0.0
    )


def test_no_feature_reads_anything_after_production(
    chain_graph, chain_history, chain_items
) -> None:
    """Only ``age`` depends on the audit revision; nothing else may.

    The audit revision is moved and every column except ``age`` must be
    unchanged. If a churn feature read events after production, this would
    catch it.
    """
    early = feature_matrix(chain_items, chain_history, chain_graph, 9)
    late = feature_matrix(chain_items, chain_history, chain_graph, 40)
    columns = {name: index for index, name in enumerate(FEATURE_NAMES)}
    age = columns["age"]
    others = [index for index in range(len(FEATURE_NAMES)) if index != age]
    assert np.array_equal(early[:, others], late[:, others])
    assert not np.array_equal(early[:, [age]], late[:, [age]])


def test_leak_feature_is_the_post_production_event_indicator(
    chain_graph, chain_history, chain_items
) -> None:
    # e0 produced at 1: events at 4, 6 and 8 all fall in (1, 10] -> 1
    # e1 produced at 7: the write at 8 falls in (7, 10] -> 1
    leak = leak_feature(chain_items, chain_history, chain_graph, 10)
    assert leak.shape == (2, 1)
    assert leak[0, 0] == 1.0
    assert leak[1, 0] == 1.0
    # Audited at revision 5, e0 has only the write at 4 in its window.
    shorter = leak_feature(chain_items[:1], chain_history, chain_graph, 5)
    assert shorter[0, 0] == 1.0


def test_leak_feature_is_zero_when_nothing_happened_in_the_window(
    chain_graph, chain_history
) -> None:
    from evidencestale.model import EvidenceItem

    items = (
        EvidenceItem(identifier="z", subject="top", produced_at=8, goal="G", solution="S"),
    )
    assert leak_feature(items, chain_history, chain_graph, 10)[0, 0] == 0.0


def test_empty_item_set_rejected(chain_graph, chain_history) -> None:
    with pytest.raises(ValueError, match="at least one evidence item"):
        feature_matrix((), chain_history, chain_graph, 10)


def test_features_on_the_committed_corpus_are_finite(layered_corpus) -> None:
    matrix = feature_matrix(
        layered_corpus.evidence,
        layered_corpus.history,
        layered_corpus.true_graph,
        layered_corpus.evaluation_time,
    )
    assert matrix.shape == (480, len(FEATURE_NAMES))
    assert np.all(np.isfinite(matrix))


def test_partial_declarations_give_smaller_or_equal_closures(layered_corpus) -> None:
    columns = {name: index for index, name in enumerate(FEATURE_NAMES)}
    complete = feature_matrix(
        layered_corpus.evidence,
        layered_corpus.history,
        layered_corpus.true_graph,
        layered_corpus.evaluation_time,
    )
    partial = feature_matrix(
        layered_corpus.evidence,
        layered_corpus.history,
        layered_corpus.declared_graph,
        layered_corpus.evaluation_time,
    )
    assert np.all(
        partial[:, columns["closure_size"]] <= complete[:, columns["closure_size"]]
    )
    assert partial[:, columns["closure_size"]].sum() < complete[
        :, columns["closure_size"]
    ].sum()
