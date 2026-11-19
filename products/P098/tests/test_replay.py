"""Replay of a change history, with hand-calculated known answers."""

from __future__ import annotations

import pytest

from evidencestale.graph import DependencyGraph
from evidencestale.model import ChangeEvent
from evidencestale.replay import (
    ChangeHistory,
    changed_sets_by_item,
    true_stale,
    true_stale_labels,
)


def test_contents_at_replays_writes_in_order(chain_history: ChangeHistory) -> None:
    # Revision 0: all v0. Revision 4 writes bottom:v1. Revision 8 writes it
    # back to bottom:v0.
    assert chain_history.contents_at(0)["bottom"] == "bottom:v0"
    assert chain_history.contents_at(3)["bottom"] == "bottom:v0"
    assert chain_history.contents_at(4)["bottom"] == "bottom:v1"
    assert chain_history.contents_at(7)["bottom"] == "bottom:v1"
    assert chain_history.contents_at(8)["bottom"] == "bottom:v0"


def test_modified_at_is_bumped_by_a_touch(chain_history: ChangeHistory) -> None:
    # top is only ever touched, at revision 6, so its content never changes and
    # its modification revision is 6 from then on.
    assert chain_history.modified_at(5)["top"] == 0
    assert chain_history.modified_at(6)["top"] == 6
    assert chain_history.modified_at(10) == {"top": 6, "middle": 0, "bottom": 8}
    assert chain_history.contents_at(10)["top"] == "top:v0"


def test_changed_between_is_value_based_so_a_revert_is_not_a_change(
    chain_history: ChangeHistory,
) -> None:
    # bottom: v0 at revision 1, v1 at revision 5, v0 again at revision 10.
    assert chain_history.changed_between(1, 5) == frozenset({"bottom"})
    assert chain_history.changed_between(1, 10) == frozenset()
    assert chain_history.changed_between(7, 10) == frozenset({"bottom"})


def test_changed_between_rejects_a_reversed_interval(
    chain_history: ChangeHistory,
) -> None:
    with pytest.raises(ValueError, match="end < "):
        chain_history.changed_between(10, 1)


def test_write_and_event_counts_are_strictly_before_the_revision(
    chain_history: ChangeHistory,
) -> None:
    # Events: write bottom at 4, touch top at 6, write bottom at 8.
    assert chain_history.write_count_before(4) == {"top": 0, "middle": 0, "bottom": 0}
    assert chain_history.write_count_before(5)["bottom"] == 1
    assert chain_history.write_count_before(9)["bottom"] == 2
    assert chain_history.event_count_before(7)["top"] == 1
    assert chain_history.write_count_before(7)["top"] == 0


def test_true_stale_sees_a_change_at_depth_two(
    chain_graph: DependencyGraph, chain_history: ChangeHistory
) -> None:
    # Evidence about top, produced at revision 7, audited at revision 10.
    # closure(top) = {top, middle, bottom}; bottom changed v1 -> v0 in (7, 10].
    assert true_stale(chain_history, chain_graph, "top", 7, 10) is True


def test_true_stale_ignores_a_touch_only_change(
    chain_graph: DependencyGraph, chain_history: ChangeHistory
) -> None:
    # Between revisions 5 and 7 the only event is the touch of top at 6, which
    # changes no content, so nothing is stale.
    assert true_stale(chain_history, chain_graph, "top", 5, 7) is False


def test_true_stale_ignores_a_revert(
    chain_graph: DependencyGraph, chain_history: ChangeHistory
) -> None:
    # Between revisions 1 and 10 bottom was written twice and ended where it
    # started, so the evidence still describes the artifact it described.
    assert true_stale(chain_history, chain_graph, "top", 1, 10) is False


def test_true_stale_rejects_an_evaluation_before_production(
    chain_graph: DependencyGraph, chain_history: ChangeHistory
) -> None:
    with pytest.raises(ValueError, match="precedes produced_at"):
        true_stale(chain_history, chain_graph, "top", 7, 5)


def test_true_stale_labels_matches_the_per_item_function(
    chain_graph: DependencyGraph, chain_history: ChangeHistory
) -> None:
    labels = true_stale_labels(
        chain_history, chain_graph, [("a", "top", 1), ("b", "top", 7)], 10
    )
    assert labels == {"a": False, "b": True}


def test_changed_sets_by_item_are_per_item_not_a_union(
    chain_history: ChangeHistory,
) -> None:
    sets = changed_sets_by_item(chain_history, [("a", 1), ("b", 7)], 10)
    assert sets["a"] == frozenset()
    assert sets["b"] == frozenset({"bottom"})


def test_missing_initial_content_rejected() -> None:
    with pytest.raises(ValueError, match="no initial content"):
        ChangeHistory(["a", "b"], {"a": "x"}, [])


def test_initial_content_for_an_undeclared_artifact_rejected() -> None:
    with pytest.raises(ValueError, match="undeclared artifacts"):
        ChangeHistory(["a"], {"a": "x", "b": "y"}, [])


def test_event_on_an_undeclared_artifact_rejected() -> None:
    with pytest.raises(ValueError, match="undeclared artifact"):
        ChangeHistory(["a"], {"a": "x"}, [ChangeEvent(1, "b", "touch")])


def test_two_events_on_one_artifact_at_one_revision_rejected() -> None:
    with pytest.raises(ValueError, match="same\n?\\s*revision|same revision"):
        ChangeHistory(
            ["a"],
            {"a": "x"},
            [
                ChangeEvent(3, "a", "write", "y"),
                ChangeEvent(3, "a", "touch"),
            ],
        )


def test_events_need_not_be_pre_sorted() -> None:
    history = ChangeHistory(
        ["a"],
        {"a": "v0"},
        [ChangeEvent(9, "a", "write", "v2"), ChangeEvent(4, "a", "write", "v1")],
    )
    assert [event.time for event in history.events] == [4, 9]
    assert history.contents_at(5)["a"] == "v1"
    assert history.contents_at(9)["a"] == "v2"
