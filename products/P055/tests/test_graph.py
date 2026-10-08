"""Tests for the graph primitives, with hand-worked expected answers."""

from __future__ import annotations

import pytest
from conftest import document, edge, node

from assuregraph import (
    EdgeKind,
    adjacency,
    find_cycles,
    node_depths,
    reachable_from,
    strongly_connected_components,
)


def _chain(length: int) -> dict:
    """G1 -> G2 -> ... -> G<length>, all SupportedBy."""
    nodes = [node(f"G{i}", "goal") for i in range(1, length + 1)]
    edges = [edge(f"G{i}", f"G{i + 1}") for i in range(1, length)]
    return document(nodes, edges, top_goals=["G1"])


def test_adjacency_covers_every_node(make_case) -> None:
    case = make_case(_chain(4))
    succ = adjacency(case)
    assert set(succ) == {"G1", "G2", "G3", "G4"}
    assert succ["G4"] == []


def test_adjacency_filters_by_kind(make_case) -> None:
    doc = document(
        [node("G1", "goal"), node("G2", "goal"), node("C1", "context")],
        [edge("G1", "G2"), edge("G1", "C1", "in_context_of")],
    )
    case = make_case(doc)
    assert adjacency(case, [EdgeKind.SUPPORTED_BY])["G1"] == ["G2"]
    assert adjacency(case, [EdgeKind.IN_CONTEXT_OF])["G1"] == ["C1"]


def test_chain_has_one_component_per_node(make_case) -> None:
    # A 5-chain is acyclic, so every strongly connected component is a singleton.
    case = make_case(_chain(5))
    components = strongly_connected_components(case)
    assert len(components) == 5
    assert all(len(c) == 1 for c in components)


def test_acyclic_case_reports_no_cycle(make_case) -> None:
    assert find_cycles(make_case(_chain(6))) == []


def test_four_node_cycle_is_reported_as_one_closed_walk(make_case) -> None:
    # G1 -> S1 -> G2 -> S2 -> G1. Expected: one walk, first == last, four distinct
    # members, and every member of the cycle present.
    doc = document(
        [node("G1", "goal"), node("S1", "strategy"), node("G2", "goal"), node("S2", "strategy")],
        [edge("G1", "S1"), edge("S1", "G2"), edge("G2", "S2"), edge("S2", "G1")],
        top_goals=["G1"],
    )
    cycles = find_cycles(make_case(doc))
    assert len(cycles) == 1
    walk = cycles[0]
    assert walk[0] == walk[-1]
    assert set(walk) == {"G1", "S1", "G2", "S2"}


def test_two_disjoint_cycles_report_two_regions(make_case) -> None:
    doc = document(
        [
            node("G1", "goal"),
            node("S1", "strategy"),
            node("G2", "goal"),
            node("S2", "strategy"),
        ],
        [edge("G1", "S1"), edge("S1", "G1"), edge("G2", "S2"), edge("S2", "G2")],
        top_goals=["G1", "G2"],
    )
    assert len(find_cycles(make_case(doc))) == 2


def test_reachability_on_a_chain(make_case) -> None:
    case = make_case(_chain(4))
    assert reachable_from(case, ["G2"]) == {"G2", "G3", "G4"}


def test_reachability_includes_the_roots(make_case) -> None:
    case = make_case(_chain(3))
    assert "G1" in reachable_from(case, ["G1"])


def test_reachability_ignores_an_unknown_root(make_case) -> None:
    case = make_case(_chain(3))
    assert reachable_from(case, ["G1", "nope"]) == {"G1", "G2", "G3"}


def test_reachability_terminates_on_a_cycle(make_case) -> None:
    doc = document(
        [node("G1", "goal"), node("S1", "strategy")],
        [edge("G1", "S1"), edge("S1", "G1")],
        top_goals=["G1"],
    )
    assert reachable_from(make_case(doc), ["G1"]) == {"G1", "S1"}


def test_node_depths_on_a_chain(make_case) -> None:
    # G1 at 0, G2 at 1, G3 at 2, G4 at 3.
    case = make_case(_chain(4))
    assert node_depths(case) == {"G1": 0, "G2": 1, "G3": 2, "G4": 3}


def test_node_depths_takes_the_shortest_path(make_case) -> None:
    # G1 -> G2 -> G3 and G1 -> G3 directly: G3 is at depth 1, not 2.
    doc = document(
        [node("G1", "goal"), node("G2", "goal"), node("G3", "goal")],
        [edge("G1", "G2"), edge("G2", "G3"), edge("G1", "G3")],
        top_goals=["G1"],
    )
    assert node_depths(make_case(doc))["G3"] == 1


def test_node_depths_omits_unreachable_nodes(make_case) -> None:
    doc = document(
        [node("G1", "goal"), node("G2", "goal"), node("G9", "goal")],
        [edge("G1", "G2")],
        top_goals=["G1"],
    )
    assert "G9" not in node_depths(make_case(doc))


def test_deep_chain_does_not_exhaust_recursion(make_case) -> None:
    # 4000 nodes is far beyond the default recursion limit of 1000; the
    # iterative formulations must handle it. This is a regression guard.
    case = make_case(_chain(4000))
    assert len(strongly_connected_components(case)) == 4000
    assert find_cycles(case) == []
    assert node_depths(case)["G4000"] == 3999


def test_strongly_connected_components_partition_the_nodes(make_case) -> None:
    doc = document(
        [node("G1", "goal"), node("S1", "strategy"), node("G2", "goal"), node("G3", "goal")],
        [edge("G1", "S1"), edge("S1", "G2"), edge("G2", "S1"), edge("G1", "G3")],
        top_goals=["G1"],
    )
    case = make_case(doc)
    members = [n for c in strongly_connected_components(case) for n in c]
    assert sorted(members) == ["G1", "G2", "G3", "S1"]
    assert len(members) == len(set(members))


def test_components_are_sorted_internally(make_case) -> None:
    doc = document(
        [node("S1", "strategy"), node("G2", "goal"), node("G1", "goal")],
        [edge("G1", "S1"), edge("S1", "G2"), edge("G2", "S1")],
        top_goals=["G1"],
    )
    nontrivial = [c for c in strongly_connected_components(make_case(doc)) if len(c) > 1]
    assert nontrivial == [["G2", "S1"]]


def test_find_cycles_respects_the_kind_filter(make_case) -> None:
    doc = document(
        [node("G1", "goal"), node("S1", "strategy")],
        [edge("G1", "S1"), edge("S1", "G1")],
        top_goals=["G1"],
    )
    case = make_case(doc)
    assert len(find_cycles(case, [EdgeKind.SUPPORTED_BY])) == 1
    assert find_cycles(case, [EdgeKind.IN_CONTEXT_OF]) == []


@pytest.mark.parametrize("length", [1, 2, 3, 10, 50])
def test_chain_depth_equals_length_minus_one(make_case, length: int) -> None:
    case = make_case(_chain(length))
    assert max(node_depths(case).values()) == length - 1
