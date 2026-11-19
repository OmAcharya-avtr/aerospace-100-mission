"""Property-based tests of the closure identities, with Hypothesis.

The identities tested here are algebraic facts about reachability, so they are
the right target for property-based testing:

- a node is in its own closure (reflexivity);
- adding an edge never shrinks any closure (monotonicity in edge addition);
- the closure is transitively closed (idempotence);
- ``impacted(C)`` is exactly the set of nodes whose closure meets ``C``, which
  ties the forward and reverse searches together;
- the closure never exceeds the node count, and the depth never exceeds
  ``n - 1`` in a graph of ``n`` nodes.

Every counterexample Hypothesis has found against these is pinned as a named
regression test in ``tests/test_regressions.py`` with literal inputs, because
``.hypothesis/`` is gitignored and a falsifying example is otherwise lost on a
fresh clone.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from evidencestale.graph import DependencyGraph

#: Node counts kept small: the identities do not depend on size and the suite
#: shares two cores with four other builds.
NODES = st.integers(min_value=1, max_value=7)

PROFILE = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@st.composite
def graphs(draw: st.DrawFn) -> DependencyGraph:
    """A random graph on ``n`` nodes with between 0 and 12 edges."""
    count = draw(NODES)
    names = [f"n{index}" for index in range(count)]
    edges = draw(
        st.lists(
            st.tuples(st.sampled_from(names), st.sampled_from(names)),
            min_size=0,
            max_size=12,
        )
    )
    return DependencyGraph.build(names, edges)


@PROFILE
@given(graphs())
def test_node_is_in_its_own_closure(graph: DependencyGraph) -> None:
    for name in graph.nodes:
        assert name in graph.closure(name)


@PROFILE
@given(graphs())
def test_closure_is_bounded_by_the_node_count(graph: DependencyGraph) -> None:
    for name in graph.nodes:
        assert 1 <= len(graph.closure(name)) <= len(graph.nodes)


@PROFILE
@given(graphs())
def test_closure_depth_is_bounded_by_node_count_minus_one(
    graph: DependencyGraph,
) -> None:
    for name in graph.nodes:
        assert 0 <= graph.closure_depth(name) <= max(0, len(graph.nodes) - 1)


@PROFILE
@given(graphs(), st.integers(min_value=0, max_value=6), st.integers(min_value=0, max_value=6))
def test_adding_an_edge_never_shrinks_a_closure(
    graph: DependencyGraph, source: int, target: int
) -> None:
    """Monotonicity in edge addition."""
    count = len(graph.nodes)
    extra = (f"n{source % count}", f"n{target % count}")
    bigger = DependencyGraph.build(graph.nodes, (*graph.edges, extra))
    for name in graph.nodes:
        assert graph.closure(name) <= bigger.closure(name)


@PROFILE
@given(graphs())
def test_closure_is_transitively_closed(graph: DependencyGraph) -> None:
    """Idempotence: the closure of a member is inside the closure."""
    for name in graph.nodes:
        closure = graph.closure(name)
        for member in closure:
            assert graph.closure(member) <= closure


@PROFILE
@given(graphs(), st.integers(min_value=0, max_value=6))
def test_impacted_is_exactly_the_nodes_whose_closure_meets_the_seed(
    graph: DependencyGraph, seed: int
) -> None:
    """The forward and reverse searches agree, which neither can check alone."""
    count = len(graph.nodes)
    target = f"n{seed % count}"
    forward = {name for name in graph.nodes if target in graph.closure(name)}
    assert forward == set(graph.impacted([target]))


@PROFILE
@given(graphs())
def test_impacted_is_monotone_in_the_seed_set(graph: DependencyGraph) -> None:
    names = list(graph.nodes)
    first = graph.impacted(names[:1])
    whole = graph.impacted(names)
    assert first <= whole


@PROFILE
@given(graphs())
def test_cycle_members_have_equal_closures(graph: DependencyGraph) -> None:
    for component in graph.cycles():
        if len(component) < 2:
            continue
        closures = {graph.closure(member) for member in component}
        assert len(closures) == 1


@PROFILE
@given(graphs(), st.integers(min_value=1, max_value=12))
def test_a_depth_limit_at_least_the_measured_depth_is_not_binding(
    graph: DependencyGraph, limit: int
) -> None:
    """``closure_depth`` and ``closure`` must agree on when the limit binds.

    This is the identity that caught the off-by-one pinned in
    ``tests/test_regressions.py::test_regression_depth_limit_equal_to_the_closure_depth_is_sufficient``:
    an earlier version of ``closure`` performed ``depth_limit`` expansions
    rather than ``depth_limit + 1``, so a closure of depth ``d`` raised at
    ``depth_limit == d`` while ``closure_depth`` reported ``d``.
    """
    for name in graph.nodes:
        depth = graph.closure_depth(name)
        if limit >= depth:
            assert graph.closure(name, depth_limit=limit) == graph.closure(name)
