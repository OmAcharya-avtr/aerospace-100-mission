"""Graph primitives over a parsed assurance case.

All algorithms here are standard and iterative, so a deep or wide case cannot
exhaust the Python recursion limit:

* strongly connected components by Tarjan's algorithm (Tarjan 1972, "Depth-first
  search and linear graph algorithms", SIAM Journal on Computing 1(2)), in an
  explicit-stack formulation. Complexity O(V + E).
* reachability and depth by breadth-first search. Complexity O(V + E).

A GSN argument is a directed graph and, when well formed, a directed acyclic
graph; nothing in the GSN Community Standard forbids a Goal from being supported
by more than one parent, so the structure is a DAG and not a tree. These
functions make no safety claim and interpret no statement text.

Units: every quantity returned here is a count of nodes or a hop distance in
edges. There are no physical units in this module.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from .model import AssuranceCase, EdgeKind

__all__ = [
    "adjacency",
    "find_cycles",
    "node_depths",
    "reachable_from",
    "strongly_connected_components",
]


def adjacency(
    case: AssuranceCase, kinds: Iterable[EdgeKind] | None = None
) -> dict[str, list[str]]:
    """Successor lists keyed by node id.

    Args:
        case: The parsed case.
        kinds: Restrict to these relationship types. ``None`` means both
            ``SupportedBy`` and ``InContextOf``.

    Returns:
        A mapping with an entry for every node in the case, in document order;
        nodes with no successors map to an empty list. Successor order follows
        document order of the edges.
    """
    selected = frozenset(kinds) if kinds is not None else frozenset(EdgeKind)
    out: dict[str, list[str]] = {node_id: [] for node_id in case.nodes}
    for edge in case.edges:
        if edge.kind in selected:
            out[edge.source].append(edge.target)
    return out


def strongly_connected_components(
    case: AssuranceCase, kinds: Iterable[EdgeKind] | None = None
) -> list[list[str]]:
    """Strongly connected components, by Tarjan's algorithm (Tarjan 1972).

    Returns:
        One list of node ids per component. Components are returned in the order
        Tarjan's algorithm completes them; the node ids inside each component
        are sorted so the output is deterministic across runs. Every node of the
        case appears in exactly one component.
    """
    succ = adjacency(case, kinds)
    index_of: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[list[str]] = []
    counter = 0

    for root in succ:
        if root in index_of:
            continue
        # Explicit DFS stack of (node, iterator position into its successors).
        work: list[tuple[str, int]] = [(root, 0)]
        index_of[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, position = work[-1]
            neighbours = succ[node]
            if position < len(neighbours):
                work[-1] = (node, position + 1)
                nxt = neighbours[position]
                if nxt not in index_of:
                    index_of[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, 0))
                elif nxt in on_stack:
                    low[node] = min(low[node], index_of[nxt])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index_of[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(sorted(component))
    return components


def _one_cycle_within(component: set[str], succ: dict[str, list[str]]) -> list[str]:
    """Return one elementary cycle inside a strongly connected ``component``.

    The component must have at least two members, or a member with a self-loop.
    The returned list is the cycle's node ids in traversal order, with the entry
    node repeated at the end so the cycle reads as a closed walk.
    """
    start = min(component)
    path: list[str] = [start]
    position: list[int] = [0]
    on_path = {start}
    while path:
        node = path[-1]
        neighbours = [n for n in succ[node] if n in component]
        if position[-1] < len(neighbours):
            nxt = neighbours[position[-1]]
            position[-1] += 1
            if nxt == start:
                return [*path, start]
            if nxt not in on_path:
                path.append(nxt)
                position.append(0)
                on_path.add(nxt)
            continue
        on_path.discard(path.pop())
        position.pop()
    # Unreachable for a genuine strongly connected component of size >= 2.
    return sorted(component)


def find_cycles(
    case: AssuranceCase, kinds: Iterable[EdgeKind] | None = None
) -> list[list[str]]:
    """Return one closed walk per cyclic region of the case.

    A cyclic region is a strongly connected component with more than one member.
    Self-loops cannot occur: :mod:`assuregraph.parse` rejects an edge whose
    source equals its target, so a document containing one never becomes an
    :class:`~assuregraph.model.AssuranceCase`.

    This reports **one** cycle per component, not every cycle. Enumerating all
    elementary cycles is exponential in the worst case (Johnson 1975), and one
    witness per region is what a reviewer needs in order to break it. The number
    of regions is exact; the number of distinct cycles is not reported at all.

    Returns:
        A list of closed walks, each a list of node ids whose first and last
        entries are the same node. Empty when the case is acyclic.
    """
    succ = adjacency(case, kinds)
    cycles: list[list[str]] = []
    for component in strongly_connected_components(case, kinds):
        if len(component) > 1:
            cycles.append(_one_cycle_within(set(component), succ))
    return cycles


def reachable_from(
    case: AssuranceCase,
    roots: Iterable[str],
    kinds: Iterable[EdgeKind] | None = None,
) -> set[str]:
    """Node ids reachable from ``roots`` by forward breadth-first search.

    ``roots`` themselves are included when they exist in the case. A root that
    is not a node of the case is ignored rather than raising, so a caller can
    pass a candidate set freely.
    """
    succ = adjacency(case, kinds)
    seen: set[str] = set()
    queue: deque[str] = deque(r for r in roots if r in succ)
    seen.update(queue)
    while queue:
        node = queue.popleft()
        for nxt in succ[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def node_depths(
    case: AssuranceCase,
    roots: Iterable[str] | None = None,
    kinds: Iterable[EdgeKind] | None = None,
) -> dict[str, int]:
    """Shortest hop distance from ``roots`` to every reachable node.

    Args:
        case: The parsed case.
        roots: Starting nodes. ``None`` uses
            :meth:`~assuregraph.model.AssuranceCase.resolved_top_goals`.
        kinds: Relationship types to traverse. ``None`` means both.

    Returns:
        Depth in edges, 0 at a root. Unreachable nodes are absent from the
        mapping; a caller that needs them should consult
        :func:`reachable_from`.
    """
    succ = adjacency(case, kinds)
    start = list(case.resolved_top_goals() if roots is None else roots)
    depths: dict[str, int] = {}
    queue: deque[str] = deque()
    for node_id in start:
        if node_id in succ and node_id not in depths:
            depths[node_id] = 0
            queue.append(node_id)
    while queue:
        node = queue.popleft()
        for nxt in succ[node]:
            if nxt not in depths:
                depths[nxt] = depths[node] + 1
                queue.append(nxt)
    return depths
