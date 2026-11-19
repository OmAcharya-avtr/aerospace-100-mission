"""Closure, impact, cycles, depth limits and the graph's input validation."""

from __future__ import annotations

import pytest

from evidencestale.graph import DependencyGraph


def test_closure_of_a_chain_is_the_whole_chain(chain_graph: DependencyGraph) -> None:
    assert chain_graph.closure("top") == frozenset({"top", "middle", "bottom"})
    assert chain_graph.closure("middle") == frozenset({"middle", "bottom"})
    assert chain_graph.closure("bottom") == frozenset({"bottom"})


def test_closure_depth_of_a_chain_counts_levels(chain_graph: DependencyGraph) -> None:
    # top -> middle -> bottom: two breadth-first levels from top, one from
    # middle, none from bottom.
    assert chain_graph.closure_depth("top") == 2
    assert chain_graph.closure_depth("middle") == 1
    assert chain_graph.closure_depth("bottom") == 0


def test_impact_is_the_reverse_of_closure(chain_graph: DependencyGraph) -> None:
    # bottom is depended on by middle, which is depended on by top.
    assert chain_graph.impacted(["bottom"]) == frozenset({"top", "middle", "bottom"})
    assert chain_graph.impacted(["top"]) == frozenset({"top"})


def test_impact_and_closure_are_consistent_on_a_hand_graph() -> None:
    graph = DependencyGraph.build(
        ["r", "s", "t", "u"], [("r", "s"), ("s", "t"), ("u", "t")]
    )
    # Every artifact whose closure contains t: r (via s), s, u, and t itself.
    reachers = {name for name in graph.nodes if "t" in graph.closure(name)}
    assert reachers == set(graph.impacted(["t"]))


def test_cycle_terminates_and_closure_is_the_whole_component() -> None:
    graph = DependencyGraph.build(
        ["x", "y", "z", "w"], [("x", "y"), ("y", "z"), ("z", "x"), ("w", "x")]
    )
    for member in ("x", "y", "z"):
        assert graph.closure(member) == frozenset({"x", "y", "z"})
    assert graph.closure("w") == frozenset({"w", "x", "y", "z"})
    assert graph.is_acyclic() is False
    assert graph.cycles() == (("x", "y", "z"),)


def test_self_loop_is_reported_as_a_cycle_and_is_a_closure_no_op() -> None:
    graph = DependencyGraph.build(["p", "q"], [("p", "p"), ("p", "q")])
    assert graph.closure("p") == frozenset({"p", "q"})
    assert graph.cycles() == (("p",),)
    assert graph.is_acyclic() is False


def test_acyclic_graph_reports_no_cycle(chain_graph: DependencyGraph) -> None:
    assert chain_graph.is_acyclic() is True
    assert chain_graph.cycles() == ()


def test_max_out_degree_of_impact_graph_counts_dependents() -> None:
    # c is depended on by a and b, so the impact graph's out-degree at c is 2.
    graph = DependencyGraph.build(["a", "b", "c"], [("a", "c"), ("b", "c")])
    assert graph.max_out_degree_of_impact_graph() == 2
    assert graph.dependents()["c"] == ("a", "b")
    assert graph.dependencies()["a"] == ("c",)


def test_empty_graph_has_zero_max_out_degree() -> None:
    graph = DependencyGraph.build([], [])
    assert graph.max_out_degree_of_impact_graph() == 0


def test_duplicate_nodes_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate identifiers"):
        DependencyGraph.build(["a", "a"], [])


def test_edge_to_undeclared_node_rejected() -> None:
    with pytest.raises(ValueError, match="undeclared artifact as its dependency"):
        DependencyGraph.build(["a"], [("a", "b")])


def test_edge_from_undeclared_node_rejected() -> None:
    with pytest.raises(ValueError, match="undeclared artifact as its dependent"):
        DependencyGraph.build(["a"], [("b", "a")])


def test_malformed_edge_rejected() -> None:
    with pytest.raises(ValueError, match="dependent, dependency"):
        DependencyGraph.build(["a", "b"], [("a", "b", "c")])


def test_closure_of_unknown_artifact_names_the_count() -> None:
    graph = DependencyGraph.build(["a"], [])
    with pytest.raises(ValueError, match="not a declared artifact"):
        graph.closure("zzz")


def test_closure_depth_of_unknown_artifact_rejected() -> None:
    graph = DependencyGraph.build(["a"], [])
    with pytest.raises(ValueError, match="not a declared artifact"):
        graph.closure_depth("zzz")


def test_impacted_with_unknown_seed_rejected() -> None:
    graph = DependencyGraph.build(["a"], [])
    with pytest.raises(ValueError, match="not a declared artifact"):
        graph.impacted(["zzz"])


def test_depth_limit_must_be_a_positive_int(chain_graph: DependencyGraph) -> None:
    with pytest.raises(ValueError, match="positive int"):
        chain_graph.closure("top", depth_limit=0)
    with pytest.raises(TypeError, match="positive int"):
        chain_graph.closure("top", depth_limit=2.5)  # type: ignore[arg-type]


def test_depth_limit_binding_raises_with_an_actionable_message(
    chain_graph: DependencyGraph,
) -> None:
    # The chain needs 2 levels; a limit of 1 leaves the frontier non-empty.
    with pytest.raises(ValueError, match="did not reach a fixed point"):
        chain_graph.closure("top", depth_limit=1)


def test_depth_limit_exactly_sufficient_does_not_raise(
    chain_graph: DependencyGraph,
) -> None:
    assert chain_graph.closure("top", depth_limit=2) == frozenset(
        {"top", "middle", "bottom"}
    )


def test_all_closures_covers_every_node(chain_graph: DependencyGraph) -> None:
    closures = chain_graph.all_closures()
    assert set(closures) == set(chain_graph.nodes)
    assert sum(len(value) for value in closures.values()) == 6  # 3 + 2 + 1


def test_duplicate_edges_do_not_change_the_closure() -> None:
    once = DependencyGraph.build(["a", "b"], [("a", "b")])
    twice = DependencyGraph.build(["a", "b"], [("a", "b"), ("a", "b")])
    assert once.closure("a") == twice.closure("a")
    assert once.dependencies()["a"] == twice.dependencies()["a"]


def test_cyclic_corpus_is_cyclic_and_its_closures_contain_the_component(
    cyclic_corpus,
) -> None:
    graph = cyclic_corpus.true_graph
    assert graph.is_acyclic() is False
    cycle = graph.cycles()
    assert ("c00", "c01", "c02") in cycle
    assert graph.closure("c00") == frozenset({"c00", "c01", "c02"})


def test_layered_corpus_is_acyclic_and_respects_its_declared_structure(
    layered_corpus,
) -> None:
    graph = layered_corpus.true_graph
    assert graph.is_acyclic() is True
    assert graph.max_out_degree_of_impact_graph() <= layered_corpus.declared_max_dependents
    deepest = max(graph.closure_depth(name) for name in graph.nodes)
    assert deepest <= layered_corpus.declared_max_impact_depth
