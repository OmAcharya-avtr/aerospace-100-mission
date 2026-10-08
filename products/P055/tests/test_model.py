"""Tests for the data model and the GSN relationship table."""

from __future__ import annotations

import pytest
from conftest import document, edge, node

from assuregraph import PERMITTED_EDGES, EdgeKind, Evidence, Node, NodeKind


def test_six_core_gsn_element_types_exactly() -> None:
    # GSN Standard v3 Core GSN defines exactly six element types.
    assert {k.value for k in NodeKind} == {
        "goal",
        "strategy",
        "solution",
        "context",
        "assumption",
        "justification",
    }


def test_two_core_gsn_relationship_types_exactly() -> None:
    assert {k.value for k in EdgeKind} == {"supported_by", "in_context_of"}


@pytest.mark.parametrize(
    ("kind", "prefix"),
    [
        (NodeKind.GOAL, "G"),
        (NodeKind.STRATEGY, "S"),
        (NodeKind.SOLUTION, "Sn"),
        (NodeKind.CONTEXT, "C"),
        (NodeKind.ASSUMPTION, "A"),
        (NodeKind.JUSTIFICATION, "J"),
    ],
)
def test_conventional_prefixes(kind: NodeKind, prefix: str) -> None:
    assert kind.conventional_prefix == prefix


def test_only_goal_and_strategy_may_source_an_edge() -> None:
    sources = {source for source, _ in PERMITTED_EDGES}
    assert sources == {NodeKind.GOAL, NodeKind.STRATEGY}


def test_supported_by_targets() -> None:
    expected = {NodeKind.GOAL, NodeKind.STRATEGY, NodeKind.SOLUTION}
    assert PERMITTED_EDGES[(NodeKind.GOAL, EdgeKind.SUPPORTED_BY)] == expected
    assert PERMITTED_EDGES[(NodeKind.STRATEGY, EdgeKind.SUPPORTED_BY)] == expected


def test_in_context_of_targets() -> None:
    expected = {NodeKind.CONTEXT, NodeKind.ASSUMPTION, NodeKind.JUSTIFICATION}
    assert PERMITTED_EDGES[(NodeKind.GOAL, EdgeKind.IN_CONTEXT_OF)] == expected


def test_nodes_and_edges_are_frozen() -> None:
    n = Node(node_id="G1", kind=NodeKind.GOAL, statement="s")
    with pytest.raises(AttributeError):
        n.statement = "other"  # type: ignore[misc]


def test_evidence_defaults_to_no_digest() -> None:
    assert Evidence(path="a.txt").sha256 is None


def test_nodes_of_kind_preserves_document_order(make_case) -> None:
    doc = document(
        [
            node("G1", "goal"),
            node("G2", "goal"),
            node("S1", "strategy"),
            node("G3", "goal"),
        ],
        [edge("G1", "G2"), edge("G1", "S1"), edge("S1", "G3")],
    )
    case = make_case(doc)
    assert [n.node_id for n in case.nodes_of_kind(NodeKind.GOAL)] == ["G1", "G2", "G3"]


def test_resolved_top_goals_infers_roots_when_undeclared(make_case) -> None:
    # G1 and G9 have no incoming SupportedBy; G2 does. Expected roots: G1, G9.
    doc = document(
        [node("G1", "goal"), node("G2", "goal"), node("G9", "goal")],
        [edge("G1", "G2")],
    )
    assert make_case(doc).resolved_top_goals() == ("G1", "G9")


def test_resolved_top_goals_uses_declaration_verbatim(make_case) -> None:
    doc = document(
        [node("G1", "goal"), node("G2", "goal"), node("G9", "goal")],
        [edge("G1", "G2")],
        top_goals=["G1"],
    )
    # G9 is NOT inferred when a declaration is present: that is the point of
    # declaring, and it is what makes an orphan detectable.
    assert make_case(doc).resolved_top_goals() == ("G1",)


def test_edges_of_kind_filters(make_case) -> None:
    doc = document(
        [node("G1", "goal"), node("C1", "context"), node("Sn1", "solution",
                                                         evidence={"path": "x.txt"})],
        [edge("G1", "C1", "in_context_of"), edge("G1", "Sn1")],
    )
    case = make_case(doc)
    assert len(case.edges_of_kind(EdgeKind.IN_CONTEXT_OF)) == 1
    assert len(case.edges_of_kind(EdgeKind.SUPPORTED_BY)) == 1
