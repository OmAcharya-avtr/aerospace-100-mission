"""The Boolean structure function of a declared diagram, and its cut sets.

The structure function
----------------------

For a coherent system the structure function ``phi: {0,1}^n -> {0,1}`` maps the
state of each basic event to the state of the system, is monotone
non-decreasing in every argument, and satisfies ``phi(0) = 0`` and
``phi(1) = 1``. Barlow, R. E. and Proschan, F. (1975), *Statistical Theory of
Reliability and Life Testing*, Holt, Rinehart and Winston, chapter 1, is the
standard treatment; the definition and the pivotal decomposition used in
:mod:`reliablock.bridge` are from it. No numerical value in this repository is
taken from that book.

Here the basic events are the diagram's leaf blocks plus, for a
:class:`~reliablock.diagram.Standby` node, a synthetic basic event for the
changeover switch. The switch is a real basic event -- it has a declared
failure probability and it appears in a cut set of its own with the primary --
and making it synthetic rather than a declared block keeps the declaration
readable while keeping the cut sets honest.

Minimal cut sets
----------------

A cut set is a set of basic events whose simultaneous failure fails the system;
it is minimal if no proper subset is a cut set. The enumeration here is
structural and recursive, not a search over states:

- ``series``: the union of the children's cut sets. Any child's cut fails the
  chain.
- ``parallel``: one cut from each child, unioned. All children must fail.
- ``k_of_n``: one cut from each of every ``(n - k + 1)``-subset of children.
  Failing ``n - k + 1`` children leaves fewer than ``k`` surviving.
- ``standby``: the primary's cuts crossed with (the switch) and with (one cut
  from each standby unit).
- ``shared_load``: one cut from each unit, as for parallel. The *Boolean*
  structure is parallel; the *quantification* is not, and that is stated in
  every report rather than left to be discovered.
- ``network``: the minimal ``s``-``t`` edge cuts, obtained by dualising the
  minimal path sets.

The result is minimalised at every level, so the count returned is the count of
*minimal* cut sets. :func:`enumerate_states` is the independent check: it walks
all ``2**n`` states of a small diagram and derives the minimal cut sets from
the structure function directly, with no shared code path. The two are compared
in ``validation/validate_cutsets.py``.

Cost: the structural enumeration is exponential in the worst case, because the
number of minimal cut sets is. :data:`MAX_ENUMERATION_EVENTS` caps the
exhaustive route at a size that runs in a bounded time rather than failing
slowly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations, product
from typing import Any

from .diagram import (
    Block,
    Diagram,
    KOfN,
    Network,
    Node,
    Parallel,
    Series,
    SharedLoad,
    Standby,
)

__all__ = [
    "MAX_ENUMERATION_EVENTS",
    "basic_events",
    "enumerate_states",
    "minimal_cut_sets",
    "minimal_path_sets",
    "structure_function",
    "switch_event_name",
]

#: Above this many basic events the exhaustive ``2**n`` route is refused rather
#: than run. 22 events is 4194304 states, which evaluates in a few seconds.
MAX_ENUMERATION_EVENTS = 22


def switch_event_name(node: Standby, index: int) -> str:
    """Name of the synthetic basic event for a standby node's changeover switch."""
    label = node.label or f"standby#{index}"
    return f"{label}::switch"


def basic_events(diagram: Diagram) -> tuple[str, ...]:
    """Every basic event of ``diagram``: leaf blocks plus standby switches.

    Order is the diagram's first-encountered order for blocks, with each
    standby switch appended immediately after the node it belongs to, so that
    the ordering is a property of the declaration and not of a dict's
    insertion history.
    """
    events: list[str] = []
    for index, item in enumerate(diagram.walk()):
        if isinstance(item, Block):
            if item.name not in events:
                events.append(item.name)
        elif isinstance(item, Standby):
            name = switch_event_name(item, index)
            if name not in events:
                events.append(name)
    return tuple(events)


def structure_function(diagram: Diagram, working: Iterable[str]) -> bool:
    """Evaluate the structure function: does the system succeed?

    Parameters
    ----------
    diagram
        The declared diagram.
    working
        Names of the basic events that are *working*. Every basic event not
        listed is failed.

    Returns
    -------
    bool
        True if the system succeeds in that state.
    """
    alive = set(working)
    return _evaluate(diagram.root, alive, _switch_names(diagram))


def _switch_names(diagram: Diagram) -> dict[int, str]:
    """Map ``id(node)`` to the switch event name, for every standby node."""
    out: dict[int, str] = {}
    for index, item in enumerate(diagram.walk()):
        if isinstance(item, Standby):
            out[id(item)] = switch_event_name(item, index)
    return out


def _evaluate(item: Node | Block, alive: set[str], switches: dict[int, str]) -> bool:
    """Recursive structure-function evaluation."""
    if isinstance(item, Block):
        return item.name in alive
    if isinstance(item, Series):
        return all(_evaluate(child, alive, switches) for child in item.items)
    if isinstance(item, Parallel):
        return any(_evaluate(child, alive, switches) for child in item.items)
    if isinstance(item, KOfN):
        return sum(_evaluate(child, alive, switches) for child in item.items) >= item.k
    if isinstance(item, SharedLoad):
        return any(_evaluate(child, alive, switches) for child in item.units)
    if isinstance(item, Standby):
        if item.primary is not None and _evaluate(item.primary, alive, switches):
            return True
        if switches[id(item)] not in alive:
            return False
        return any(_evaluate(child, alive, switches) for child in item.standbys)
    if isinstance(item, Network):
        return _network_connected(item, alive)
    raise TypeError(f"cannot evaluate {type(item).__name__}")  # pragma: no cover


def _network_connected(node: Network, alive: set[str]) -> bool:
    """True if surviving edges connect ``node.source`` to ``node.sink``."""
    adjacency: dict[str, list[str]] = {}
    for a, b, block in node.edges:
        if block.name in alive:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
    seen = {node.source}
    stack = [node.source]
    while stack:
        current = stack.pop()
        if current == node.sink:
            return True
        for neighbour in adjacency.get(current, ()):
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return node.sink in seen


# ----------------------------------------------------------------- minimalising


def _minimalise(sets: Iterable[frozenset[str]]) -> tuple[frozenset[str], ...]:
    """Drop every set that is a proper superset of another, and deduplicate.

    Returned in order of increasing size then sorted contents, so the result is
    a function of the input set and not of iteration order. Two runs on the
    same diagram return the same tuple, which is what makes a committed cut-set
    listing diffable.
    """
    unique = sorted(set(sets), key=lambda s: (len(s), sorted(s)))
    kept: list[frozenset[str]] = []
    for candidate in unique:
        if not any(existing <= candidate for existing in kept):
            kept.append(candidate)
    return tuple(kept)


def _cross(groups: Sequence[tuple[frozenset[str], ...]]) -> list[frozenset[str]]:
    """One set from each group, unioned, for every combination."""
    out: list[frozenset[str]] = []
    for choice in product(*groups):
        union: frozenset[str] = frozenset()
        for part in choice:
            union = union | part
        out.append(union)
    return out


# ------------------------------------------------------------------- cut sets


def minimal_cut_sets(diagram: Diagram) -> tuple[frozenset[str], ...]:
    """Minimal cut sets of ``diagram``, each a frozenset of basic-event names.

    A cut set fails the system when every event in it has failed. Minimal means
    no proper subset is also a cut set, which is checked structurally at every
    level of the recursion rather than only at the end.

    Returned in order of increasing size, then lexicographically, so the result
    is reproducible and diffable.
    """
    return _minimalise(_cuts(diagram.root, _switch_names(diagram)))


def _cuts(item: Node | Block, switches: dict[int, str]) -> tuple[frozenset[str], ...]:
    """Recursive minimal-cut-set enumeration."""
    if isinstance(item, Block):
        return (frozenset({item.name}),)
    if isinstance(item, Series):
        collected: list[frozenset[str]] = []
        for child in item.items:
            collected.extend(_cuts(child, switches))
        return _minimalise(collected)
    if isinstance(item, Parallel):
        return _minimalise(_cross([_cuts(child, switches) for child in item.items]))
    if isinstance(item, SharedLoad):
        return _minimalise(_cross([_cuts(child, switches) for child in item.units]))
    if isinstance(item, KOfN):
        n = len(item.items)
        needed = n - item.k + 1
        collected = []
        for subset in combinations(range(n), needed):
            collected.extend(_cross([_cuts(item.items[i], switches) for i in subset]))
        return _minimalise(collected)
    if isinstance(item, Standby):
        assert item.primary is not None  # guaranteed by Diagram.validate
        primary_cuts = _cuts(item.primary, switches)
        switch_cut = (frozenset({switches[id(item)]}),)
        standby_all = _cross([_cuts(child, switches) for child in item.standbys])
        # The backup branch fails if the switch fails OR every standby unit
        # fails. Those are the branch's own cut sets; cross them with the
        # primary's, because the primary must fail first for any of it to
        # matter.
        branch = _minimalise(list(switch_cut) + standby_all)
        return _minimalise(_cross([primary_cuts, branch]))
    if isinstance(item, Network):
        return _minimalise(_network_cuts(item))
    raise TypeError(f"cannot enumerate cut sets of {type(item).__name__}")  # pragma: no cover


def _network_cuts(node: Network) -> list[frozenset[str]]:
    """Minimal edge cuts of a two-terminal network, by dualising its paths.

    A set of edges is a cut iff it intersects every minimal path. The minimal
    such sets are the minimal transversals (the hypergraph dual) of the minimal
    path sets, built here by incremental crossing with minimalisation at each
    step so that the intermediate size stays bounded.
    """
    paths = _network_paths(node)
    transversals: list[frozenset[str]] = [frozenset()]
    for path in paths:
        grown = [t | {event} for t in transversals for event in sorted(path)]
        transversals = list(_minimalise(grown))
    return transversals


def _network_paths(node: Network) -> list[frozenset[str]]:
    """Minimal ``s``-``t`` path sets of a two-terminal network, as edge-name sets.

    Depth-first enumeration of simple paths over vertices. A path that revisits
    a vertex carries a redundant edge and is never minimal, so vertex-simple
    enumeration is sufficient and complete for minimal path sets.
    """
    adjacency: dict[str, list[tuple[str, str]]] = {}
    for a, b, block in node.edges:
        adjacency.setdefault(a, []).append((b, block.name))
        adjacency.setdefault(b, []).append((a, block.name))

    found: list[frozenset[str]] = []

    def walk(vertex: str, visited: frozenset[str], used: frozenset[str]) -> None:
        if vertex == node.sink:
            found.append(used)
            return
        for neighbour, event in sorted(adjacency.get(vertex, ())):
            if neighbour in visited:
                continue
            walk(neighbour, visited | {neighbour}, used | {event})

    walk(node.source, frozenset({node.source}), frozenset())
    return list(_minimalise(found))


def minimal_path_sets(diagram: Diagram) -> tuple[frozenset[str], ...]:
    """Minimal path sets of ``diagram``: minimal sets whose survival succeeds.

    Used by the Esary-Proschan min-path bound in
    :mod:`reliablock.bounds`, and as the dual cross-check on the cut sets.
    """
    return _minimalise(_paths(diagram.root, _switch_names(diagram)))


def _paths(item: Node | Block, switches: dict[int, str]) -> tuple[frozenset[str], ...]:
    """Recursive minimal-path-set enumeration: the dual of :func:`_cuts`."""
    if isinstance(item, Block):
        return (frozenset({item.name}),)
    if isinstance(item, Series):
        return _minimalise(_cross([_paths(child, switches) for child in item.items]))
    if isinstance(item, Parallel):
        collected: list[frozenset[str]] = []
        for child in item.items:
            collected.extend(_paths(child, switches))
        return _minimalise(collected)
    if isinstance(item, SharedLoad):
        collected = []
        for child in item.units:
            collected.extend(_paths(child, switches))
        return _minimalise(collected)
    if isinstance(item, KOfN):
        n = len(item.items)
        collected = []
        for subset in combinations(range(n), item.k):
            collected.extend(_cross([_paths(item.items[i], switches) for i in subset]))
        return _minimalise(collected)
    if isinstance(item, Standby):
        assert item.primary is not None
        collected = list(_paths(item.primary, switches))
        switch = frozenset({switches[id(item)]})
        for child in item.standbys:
            for path in _paths(child, switches):
                collected.append(path | switch)
        return _minimalise(collected)
    if isinstance(item, Network):
        return _minimalise(_network_paths(item))
    raise TypeError(f"cannot enumerate path sets of {type(item).__name__}")  # pragma: no cover


# ------------------------------------------------------- exhaustive enumeration


def enumerate_states(diagram: Diagram) -> dict[str, Any]:
    """Walk all ``2**n`` states and derive the cut sets from the structure function.

    This is the independent check on :func:`minimal_cut_sets`. It shares no
    code path with the structural enumeration: it evaluates
    :func:`structure_function` on every state, collects the failed states,
    and keeps those that are minimal in the sense that restoring any single
    failed event makes the system succeed.

    Returns a mapping with keys ``events``, ``states``, ``failed_states``,
    ``minimal_cut_sets`` and ``n_minimal``.

    Raises ``ValueError`` above :data:`MAX_ENUMERATION_EVENTS` basic events
    rather than running for an unbounded time.
    """
    events = basic_events(diagram)
    n = len(events)
    if n > MAX_ENUMERATION_EVENTS:
        raise ValueError(
            f"exhaustive state enumeration refused for {n} basic events: "
            f"2**{n} states exceeds the declared cap of 2**{MAX_ENUMERATION_EVENTS}. "
            "This routine exists as an independent check on small diagrams; "
            "use minimal_cut_sets() for a large one."
        )
    failed: list[frozenset[str]] = []
    for mask in range(1 << n):
        # bit set means FAILED, so that the collected object is a cut set.
        down = frozenset(events[i] for i in range(n) if mask >> i & 1)
        alive = [e for e in events if e not in down]
        if not structure_function(diagram, alive):
            failed.append(down)
    failed_lookup = set(failed)
    minimal = [
        cut
        for cut in failed
        if cut and all((cut - {event}) not in failed_lookup for event in sorted(cut))
    ]
    return {
        "events": events,
        "states": 1 << n,
        "failed_states": len(failed),
        "minimal_cut_sets": _minimalise(minimal),
        "n_minimal": len(_minimalise(minimal)),
    }
