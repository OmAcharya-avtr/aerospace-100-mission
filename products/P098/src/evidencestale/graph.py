"""Dependency graph and the dependency-closure reachability computation.

ORIENTATION, STATED ONCE AND RELIED ON EVERYWHERE
-------------------------------------------------
An edge ``(dependent, dependency)`` means "``dependent`` is influenced by
``dependency``". A test report depends on the source files and the
configuration it ran against; an analysis depends on the model it analysed.

Two reachability sets follow, and they are not the same set:

- :meth:`DependencyGraph.closure` of an artifact ``a`` -- everything ``a``
  depends on, transitively, including ``a`` itself. Evidence about ``a`` is
  stale when anything in ``closure(a)`` has changed.
- :meth:`DependencyGraph.impacted` of a set ``C`` -- every artifact that can
  reach some member of ``C``, that is every artifact whose closure meets ``C``.
  This is reachability in the **reversed** graph, in which the out-degree of a
  node is its number of direct **dependents**. ``src/evidencestale/bounds.py``
  bounds the stale fraction using the maximum out-degree of that reversed
  graph, so the orientation matters to the bound and is asserted in the tests.

COMPLEXITY
----------
:meth:`closure` is an iterative depth-first search with an explicit visited
set: ``O(V_r + E_r)`` time and ``O(V_r)`` additional space, where ``V_r`` and
``E_r`` count only the nodes and edges reachable from the start node. Computing
the closure of every node with :meth:`all_closures` is ``O(V * (V + E))`` in
the worst case; this product's corpus has ``V = 240`` and ``E = 240``, so the
worst case is immaterial here and the bound is stated rather than optimised.
The reference for depth-first reachability and its linear-time bound is
T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, *Introduction to
Algorithms*, 4th edition, MIT Press, 2022, chapter on elementary graph
algorithms.

CYCLES ARE NOT ASSUMED AWAY
---------------------------
An assurance-case dependency graph can contain a cycle through a shared
artifact: a generated interface header that a tool reads and also writes, or a
pair of models each calibrated against the other's output. This product does
**not** assume a directed acyclic graph. The search carries a visited set, so a
cycle terminates, and the closure of any node on a cycle contains the whole
strongly connected component of that node -- which is the correct answer, since
each member really is reachable from each other member. :meth:`is_acyclic` and
:meth:`cycles` report the condition rather than requiring it; the
strongly-connected-component view follows R. E. Tarjan, "Depth-first search and
linear graph algorithms", *SIAM Journal on Computing*, 1(2):146--160, 1972,
although only the reachability consequence is used here.

``depth_limit`` is a guard on malformed input, not part of the definition.
It bounds the DEPTH of the closure, so a closure whose fixed point is reached
after exactly ``depth_limit`` breadth-first levels is accepted and one needing
a further level raises; the search therefore performs ``depth_limit + 1``
expansions, the last of which must discover nothing.
:meth:`closure` reaches a fixed point; if the breadth-first frontier is still
non-empty after ``depth_limit`` levels the method raises, because a closure
that deep in a graph of this size means the declared edges are wrong. Whether
the limit was ever reached is reported, because the exactness claim in
``validation/VALIDATION.md`` is conditional on it not binding.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

#: Default guard on closure depth. Not part of the definition of a closure: the
#: search runs to a fixed point and this only bounds malformed input.
DEFAULT_DEPTH_LIMIT = 64


@dataclass(frozen=True)
class DependencyGraph:
    """A directed graph of artifacts and declared dependency edges.

    Parameters
    ----------
    nodes:
        Artifact identifiers. Duplicates are a ``ValueError``.
    edges:
        ``(dependent, dependency)`` pairs. An edge naming an undeclared node is
        a ``ValueError``; a self-loop is permitted and is a no-op for the
        closure, since a node is always in its own closure.
    """

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if len(set(self.nodes)) != len(self.nodes):
            duplicates = sorted({name for name in self.nodes if self.nodes.count(name) > 1})
            raise ValueError(
                f"DependencyGraph.nodes contains {len(self.nodes) - len(set(self.nodes))} "
                f"duplicate identifiers: {duplicates[:8]}. Artifact identifiers must be "
                f"unique; de-duplicate the artifact list before constructing the graph."
            )
        known = set(self.nodes)
        for edge in self.edges:
            if len(edge) != 2:
                raise ValueError(
                    f"each edge must be a (dependent, dependency) pair, got {edge!r}"
                )
            for end, label in ((edge[0], "dependent"), (edge[1], "dependency")):
                if end not in known:
                    raise ValueError(
                        f"edge {edge!r} names an undeclared artifact as its {label}: "
                        f"{end!r} is not in nodes. Every edge endpoint must be a "
                        f"declared artifact; add it to nodes or remove the edge."
                    )

    @classmethod
    def build(
        cls, nodes: Iterable[str], edges: Iterable[Sequence[str]]
    ) -> DependencyGraph:
        """Construct from any iterables, normalising edges to tuples."""
        normalised: list[tuple[str, str]] = []
        for edge in edges:
            pair = tuple(edge)
            if len(pair) != 2:
                raise ValueError(
                    f"each edge must be a (dependent, dependency) pair, got {edge!r}"
                )
            normalised.append((pair[0], pair[1]))
        return cls(nodes=tuple(nodes), edges=tuple(normalised))

    # -- adjacency ---------------------------------------------------------

    def dependencies(self) -> Mapping[str, tuple[str, ...]]:
        """``node -> its direct dependencies``, every node present."""
        out: dict[str, list[str]] = {name: [] for name in self.nodes}
        for dependent, dependency in self.edges:
            out[dependent].append(dependency)
        return {name: tuple(sorted(set(values))) for name, values in out.items()}

    def dependents(self) -> Mapping[str, tuple[str, ...]]:
        """``node -> its direct dependents``, every node present.

        This is the adjacency of the reversed graph, whose out-degree is the
        quantity ``bounds.stale_fraction_bracket`` needs.
        """
        out: dict[str, list[str]] = {name: [] for name in self.nodes}
        for dependent, dependency in self.edges:
            out[dependency].append(dependent)
        return {name: tuple(sorted(set(values))) for name, values in out.items()}

    def max_out_degree_of_impact_graph(self) -> int:
        """Maximum number of direct dependents of any artifact.

        This is the maximum out-degree of the reversed ("impact") graph, and it
        is the only structural statistic the independent bound in
        ``bounds.py`` consumes. Counting, not reachability.
        """
        dependents = self.dependents()
        return max((len(values) for values in dependents.values()), default=0)

    # -- reachability ------------------------------------------------------

    def closure(
        self, node: str, depth_limit: int = DEFAULT_DEPTH_LIMIT
    ) -> frozenset[str]:
        """Everything ``node`` depends on, transitively, including ``node``.

        Iterative breadth-first search with a visited set, so a cycle
        terminates. ``O(V_r + E_r)`` in the reachable subgraph.

        Raises
        ------
        ValueError
            If ``node`` is not a declared artifact, if ``depth_limit`` is not a
            positive integer, or if the frontier is still non-empty after
            ``depth_limit`` levels.
        """
        if node not in set(self.nodes):
            raise ValueError(
                f"closure() was asked for artifact {node!r}, which is not a declared "
                f"artifact. Declared artifacts number {len(self.nodes)}; check the "
                f"identifier spelling or add the artifact to the graph."
            )
        if isinstance(depth_limit, bool) or not isinstance(depth_limit, int):
            raise TypeError(
                f"depth_limit must be a positive int, got {type(depth_limit).__name__}"
            )
        if depth_limit < 1:
            raise ValueError(f"depth_limit must be a positive int, got {depth_limit}")
        dependencies = self.dependencies()
        visited = {node}
        frontier = [node]
        for _level in range(depth_limit + 1):
            if not frontier:
                return frozenset(visited)
            nxt: list[str] = []
            for current in frontier:
                for neighbour in dependencies[current]:
                    if neighbour not in visited:
                        visited.add(neighbour)
                        nxt.append(neighbour)
            frontier = nxt
        if frontier:
            raise ValueError(
                f"closure({node!r}) did not reach a fixed point within depth_limit="
                f"{depth_limit}; {len(frontier)} artifacts were still being discovered "
                f"at that depth, which in a graph of {len(self.nodes)} artifacts means "
                f"the declared edges are malformed. Inspect the declared dependency "
                f"edges, or raise depth_limit deliberately and record that you did."
            )
        return frozenset(visited)

    def closure_depth(self, node: str, depth_limit: int = DEFAULT_DEPTH_LIMIT) -> int:
        """Number of breadth-first levels needed to reach the closure's fixed point.

        ``0`` for an artifact with no dependencies. Reported because the
        exactness claim is conditional on ``depth_limit`` not binding.
        """
        if node not in set(self.nodes):
            raise ValueError(
                f"closure_depth() was asked for artifact {node!r}, which is not a "
                f"declared artifact."
            )
        dependencies = self.dependencies()
        visited = {node}
        frontier = [node]
        depth = 0
        for _level in range(depth_limit + 1):
            nxt = [
                neighbour
                for current in frontier
                for neighbour in dependencies[current]
                if neighbour not in visited
            ]
            for neighbour in nxt:
                visited.add(neighbour)
            if not nxt:
                return depth
            depth += 1
            frontier = nxt
        raise ValueError(
            f"closure_depth({node!r}) exceeded depth_limit={depth_limit}; the declared "
            f"dependency edges are malformed."
        )

    def all_closures(
        self, depth_limit: int = DEFAULT_DEPTH_LIMIT
    ) -> Mapping[str, frozenset[str]]:
        """Closure of every artifact. ``O(V * (V + E))`` worst case."""
        return {name: self.closure(name, depth_limit=depth_limit) for name in self.nodes}

    def impacted(
        self, changed: Iterable[str], depth_limit: int = DEFAULT_DEPTH_LIMIT
    ) -> frozenset[str]:
        """Every artifact whose closure meets ``changed``.

        Reachability in the reversed graph, so the same ``O(V_r + E_r)`` bound
        applies with the reversed adjacency.
        """
        known = set(self.nodes)
        seeds = list(changed)
        for name in seeds:
            if name not in known:
                raise ValueError(
                    f"impacted() was given {name!r}, which is not a declared artifact."
                )
        dependents = self.dependents()
        visited = set(seeds)
        frontier = list(seeds)
        for _level in range(depth_limit + 1):
            if not frontier:
                return frozenset(visited)
            nxt: list[str] = []
            for current in frontier:
                for neighbour in dependents[current]:
                    if neighbour not in visited:
                        visited.add(neighbour)
                        nxt.append(neighbour)
            frontier = nxt
        if frontier:
            raise ValueError(
                f"impacted() did not reach a fixed point within depth_limit="
                f"{depth_limit}; the declared dependency edges are malformed."
            )
        return frozenset(visited)

    # -- cycle reporting ---------------------------------------------------

    def cycles(self) -> tuple[tuple[str, ...], ...]:
        """Strongly connected components of size > 1, plus every self-loop.

        Reported, not forbidden. Each component is returned sorted, and the
        components are returned sorted, so the output is deterministic.
        """
        dependencies = self.dependencies()
        closures = {name: self.closure(name) for name in self.nodes}
        components: dict[str, set[str]] = {}
        for name in self.nodes:
            mutual = {
                other
                for other in closures[name]
                if other != name and name in closures[other]
            }
            if mutual:
                members = frozenset(mutual | {name})
                components[",".join(sorted(members))] = set(members)
        out = [tuple(sorted(members)) for members in components.values()]
        for name in self.nodes:
            if name in dependencies[name]:
                out.append((name,))
        return tuple(sorted(set(out)))

    def is_acyclic(self) -> bool:
        """Whether the declared graph has no cycle. Reported, never assumed."""
        return not self.cycles()
