"""Routing over the time-expanded contact graph.

Problem
-------
Given a time-expanded graph (see :mod:`constellink.graph`), a source node, a
destination node, a release slot and a message size, find the route that
delivers the message with the smallest total latency.  Latency is the sum of
edge costs: one slot duration per hop plus the one-way propagation delay of
each transmission.  An edge is usable only if its capacity is at least the
message size, which is the *store-and-forward, whole-message* model -- the
message is not fragmented across slots.  Fragmentation is the maximum-flow
problem instead, handled by :mod:`constellink.flow`.

Algorithms
----------
* :func:`shortest_route` -- Dijkstra with a binary heap over the time-expanded
  nodes (Dijkstra 1959, "A note on two problems in connexion with graphs",
  Numerische Mathematik 1, 269-271).  All edge costs are strictly positive, so
  Dijkstra's optimality conditions hold.  Complexity ``O(E log V)`` on the
  time-expanded graph.
* :func:`enumerate_routes` -- exhaustive depth-first enumeration of every
  feasible route.  The time-expanded graph is a DAG layered by slot, so the
  enumeration terminates without a visited set; it is exponential in the
  number of slots and exists only as an independent reference for small
  instances (see ``validation/validate_routing.py``).

Both return ``None`` when no route exists, which is the correct answer for an
empty contact graph or a partitioned constellation -- not an error.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from .graph import TeEdge, TimeExpandedGraph

__all__ = ["Route", "shortest_route", "enumerate_routes", "reachable_nodes"]


@dataclass(frozen=True)
class Route:
    """A delivered route through the time-expanded graph.

    Attributes
    ----------
    hops : the :class:`TeEdge` objects in order.
    total_latency_s : sum of edge costs [s].
    arrival_slot : slot index at which the message is available at the
        destination.
    """

    hops: tuple[TeEdge, ...]
    total_latency_s: float
    arrival_slot: int

    @property
    def n_transmissions(self) -> int:
        """Number of transmit hops (hold edges excluded)."""
        return sum(1 for e in self.hops if e.kind == "tx")

    @property
    def node_sequence(self) -> tuple[str, ...]:
        """Distinct node sequence traversed, holds collapsed."""
        if not self.hops:
            return ()
        seq = [self.hops[0].src[0]]
        for e in self.hops:
            if e.dst[0] != seq[-1]:
                seq.append(e.dst[0])
        return tuple(seq)


def _feasible(e: TeEdge, message_bits: float) -> bool:
    return e.capacity_bits + 1e-9 >= message_bits


def shortest_route(teg: TimeExpandedGraph, source: str, destination: str,
                   release_slot: int = 0, message_bits: float = 0.0,
                   ) -> Route | None:
    """Minimum-latency route by Dijkstra; ``None`` if the message cannot arrive.

    Parameters
    ----------
    teg : the time-expanded graph.
    source, destination : node names, both in ``teg.nodes`` and distinct.
    release_slot : slot at which the message becomes available at ``source``,
        in ``[0, teg.n_slots]``.
    message_bits : message size [bits], >= 0.  Edges with less capacity are
        not traversed.
    """
    _check_endpoints(teg, source, destination, release_slot, message_bits)
    start = (source, release_slot)
    dist: dict[tuple[str, int], float] = {start: 0.0}
    prev: dict[tuple[str, int], tuple[tuple[str, int], TeEdge]] = {}
    heap: list[tuple[float, int, tuple[str, int]]] = [(0.0, 0, start)]
    counter = 1
    best: tuple[str, int] | None = None
    while heap:
        d, _, u = heapq.heappop(heap)
        if d > dist.get(u, math.inf) + 1e-15:
            continue
        if u[0] == destination:
            best = u
            break
        for e in teg.edges_out.get(u, ()):
            if not _feasible(e, message_bits):
                continue
            nd = d + e.cost_s
            if nd < dist.get(e.dst, math.inf) - 1e-15:
                dist[e.dst] = nd
                prev[e.dst] = (u, e)
                heapq.heappush(heap, (nd, counter, e.dst))
                counter += 1
    if best is None:
        return None
    hops: list[TeEdge] = []
    cur = best
    while cur != start:
        parent, edge = prev[cur]
        hops.append(edge)
        cur = parent
    hops.reverse()
    return Route(hops=tuple(hops), total_latency_s=dist[best], arrival_slot=best[1])


def enumerate_routes(teg: TimeExpandedGraph, source: str, destination: str,
                     release_slot: int = 0, message_bits: float = 0.0,
                     max_routes: int = 2_000_000) -> list[Route]:
    """Every feasible route, by exhaustive DFS.  Reference implementation only.

    Raises ``RuntimeError`` if more than ``max_routes`` routes are found, so a
    mis-sized instance fails loudly instead of running forever.
    """
    _check_endpoints(teg, source, destination, release_slot, message_bits)
    out: list[Route] = []

    def walk(node: tuple[str, int], cost: float, path: list[TeEdge]) -> None:
        if node[0] == destination and path:
            out.append(Route(hops=tuple(path), total_latency_s=cost,
                             arrival_slot=node[1]))
            if len(out) > max_routes:
                raise RuntimeError(
                    f"enumerate_routes exceeded max_routes={max_routes}; the instance is "
                    f"too large for exhaustive enumeration")
            return
        for e in teg.edges_out.get(node, ()):
            if not _feasible(e, message_bits):
                continue
            path.append(e)
            walk(e.dst, cost + e.cost_s, path)
            path.pop()

    walk((source, release_slot), 0.0, [])
    out.sort(key=lambda r: (r.total_latency_s, r.arrival_slot))
    return out


def reachable_nodes(teg: TimeExpandedGraph, source: str, release_slot: int = 0,
                    message_bits: float = 0.0) -> set[str]:
    """Node names reachable from ``(source, release_slot)`` within the horizon.

    Useful for partition diagnosis: the complement is the unreachable set.
    """
    if source not in teg.nodes:
        raise KeyError(f"unknown source node '{source}'")
    if not 0 <= release_slot <= teg.n_slots:
        raise ValueError(f"release_slot must be in [0, {teg.n_slots}], got {release_slot}")
    seen: set[tuple[str, int]] = set()
    stack = [(source, release_slot)]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        for e in teg.edges_out.get(u, ()):
            if _feasible(e, message_bits) and e.dst not in seen:
                stack.append(e.dst)
    return {n for n, _ in seen}


def _check_endpoints(teg: TimeExpandedGraph, source: str, destination: str,
                     release_slot: int, message_bits: float) -> None:
    if source not in teg.nodes:
        raise KeyError(f"unknown source node '{source}'")
    if destination not in teg.nodes:
        raise KeyError(f"unknown destination node '{destination}'")
    if source == destination:
        raise ValueError("source and destination must differ")
    if not 0 <= release_slot <= teg.n_slots:
        raise ValueError(f"release_slot must be in [0, {teg.n_slots}], got {release_slot}")
    if message_bits < 0.0:
        raise ValueError(f"message_bits must be >= 0, got {message_bits}")
