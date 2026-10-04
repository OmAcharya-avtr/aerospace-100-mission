"""The time-varying contact graph and its time-expanded unrolling.

Contact graph
-------------
:class:`ContactGraph` holds a node set and a list of :class:`ContactWindow`
objects.  It answers two questions: which links are open in a given time slot,
and is the graph connected over a slot or over the whole horizon.  This is the
*time-varying graph* (also called an evolving or temporal graph) of the
constellation; the formalism and the terminology used here follow Casteigts,
Flocchini, Quattrociocchi & Santoro 2012, "Time-varying graphs and dynamic
networks", Int. J. Parallel, Emergent and Distributed Systems 27(5), 387-408.

Time-expanded graph
-------------------
:class:`TimeExpandedGraph` unrolls the horizon into ``n_slots`` slots of equal
duration ``slot_s``.  Nodes are pairs ``(node, slot)``; edges are:

* *hold* edges ``(v, k) -> (v, k + 1)``, cost ``slot_s`` seconds, capacity
  ``store_capacity_bits`` -- a message waiting in an onboard buffer;
* *transmit* edges ``(u, k) -> (v, k + 1)``, cost
  ``slot_s + range/c`` seconds, capacity ``rate * slot_s`` bits -- a
  transmission occupying slot ``k``, arriving at the start of slot ``k + 1``.

A contact is treated as available in slot ``k`` when the window covers at
least ``min_overlap_fraction`` of the slot (default 1.0, i.e. the whole slot).
Requiring full coverage is the conservative choice: a partially covered slot
cannot be assumed to carry a full slot's worth of bits.  The fraction is
exposed so the conservatism can be relaxed deliberately, and the effective
capacity is scaled by the realised overlap fraction when it is.

The resulting structure is a directed acyclic graph layered by slot, which is
why shortest-path and maximum-flow problems on it are both tractable and
exhaustively checkable on small instances.  The construction is the standard
time-expanded network of Ford & Fulkerson 1958 ("Constructing maximal dynamic
flows from static flows", Operations Research 6(3), 419-433).

Cost of the unrolling: ``O(n_slots * n_links)`` edges.  A 24-satellite Walker
shell over 3 hours in 60 s slots with a few hundred windows yields a graph of
order 1e4 edges, which both the Dijkstra and the ILP paths handle in seconds
(see ``validation/validate_benchmark.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from .contacts import ContactWindow
from .frames import SPEED_OF_LIGHT_KM_S, to_utc

__all__ = ["ContactGraph", "TimeExpandedGraph", "TeEdge"]


@dataclass
class ContactGraph:
    """A time-varying contact graph over a fixed horizon.

    Attributes
    ----------
    nodes : sorted unique node names.
    windows : the contact windows.
    t0, t1 : horizon edges (UTC).
    """

    windows: list[ContactWindow]
    t0: datetime
    t1: datetime
    nodes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.t0 = to_utc(self.t0)
        self.t1 = to_utc(self.t1)
        if self.t1 <= self.t0:
            raise ValueError(f"t1 ({self.t1}) must be after t0 ({self.t0})")
        seen = set(self.nodes)
        for w in self.windows:
            seen.add(w.node_a)
            seen.add(w.node_b)
        self.nodes = sorted(seen)

    @property
    def is_empty(self) -> bool:
        """True when there are no contact windows at all."""
        return len(self.windows) == 0

    @property
    def horizon_s(self) -> float:
        """Horizon duration [s]."""
        return (self.t1 - self.t0).total_seconds()

    def windows_in_slot(self, t_start: datetime, t_end: datetime,
                        min_overlap_fraction: float = 1.0,
                        ) -> list[tuple[ContactWindow, float]]:
        """Windows covering ``[t_start, t_end)``, with their overlap fraction.

        Only windows whose overlap fraction is at least
        ``min_overlap_fraction`` are returned.
        """
        if not 0.0 < min_overlap_fraction <= 1.0:
            raise ValueError(
                f"min_overlap_fraction must be in (0, 1], got {min_overlap_fraction}")
        slot_s = (t_end - t_start).total_seconds()
        if slot_s <= 0.0:
            raise ValueError("slot end must be after slot start")
        out = []
        for w in self.windows:
            lo = max(to_utc(w.t_open), t_start)
            hi = min(to_utc(w.t_close), t_end)
            overlap = (hi - lo).total_seconds()
            if overlap <= 0.0:
                continue
            frac = overlap / slot_s
            if frac + 1e-12 >= min_overlap_fraction:
                out.append((w, min(frac, 1.0)))
        return out

    def adjacency_at(self, t_start: datetime, t_end: datetime,
                     min_overlap_fraction: float = 1.0) -> dict[str, set[str]]:
        """Undirected adjacency of the slot ``[t_start, t_end)``."""
        adj: dict[str, set[str]] = {n: set() for n in self.nodes}
        for w, _ in self.windows_in_slot(t_start, t_end, min_overlap_fraction):
            adj[w.node_a].add(w.node_b)
            adj[w.node_b].add(w.node_a)
        return adj

    def union_adjacency(self) -> dict[str, set[str]]:
        """Adjacency of the union graph: an edge if the link is ever open."""
        adj: dict[str, set[str]] = {n: set() for n in self.nodes}
        for w in self.windows:
            adj[w.node_a].add(w.node_b)
            adj[w.node_b].add(w.node_a)
        return adj

    def components(self, adjacency: dict[str, set[str]] | None = None) -> list[list[str]]:
        """Connected components of ``adjacency`` (union graph by default).

        Returned as sorted lists, ordered by descending size then by first
        member, so the output is deterministic.
        """
        adj = self.union_adjacency() if adjacency is None else adjacency
        unseen = set(adj)
        comps: list[list[str]] = []
        while unseen:
            root = min(unseen)
            stack = [root]
            comp = set()
            while stack:
                v = stack.pop()
                if v in comp:
                    continue
                comp.add(v)
                stack.extend(adj.get(v, set()) - comp)
            comps.append(sorted(comp))
            unseen -= comp
        comps.sort(key=lambda c: (-len(c), c[0]))
        return comps

    def is_partitioned(self, adjacency: dict[str, set[str]] | None = None) -> bool:
        """True when the graph has more than one connected component."""
        return len(self.components(adjacency)) > 1

    def drop_node_after(self, node: str, t_loss: datetime) -> ContactGraph:
        """Return a copy in which ``node`` stops participating at ``t_loss``.

        Models a satellite loss mid-horizon: windows touching ``node`` are
        truncated at ``t_loss`` and dropped if they start at or after it.  The
        node stays in the node set (it existed), so routing reports it as
        unreachable rather than unknown.
        """
        t_loss = to_utc(t_loss)
        if not (self.t0 <= t_loss <= self.t1):
            raise ValueError(
                f"t_loss {t_loss.isoformat()} is outside the horizon "
                f"[{self.t0.isoformat()}, {self.t1.isoformat()}]")
        if node not in self.nodes:
            raise KeyError(f"unknown node '{node}'")
        kept: list[ContactWindow] = []
        for w in self.windows:
            if node not in (w.node_a, w.node_b):
                kept.append(w)
                continue
            if to_utc(w.t_open) >= t_loss:
                continue
            new_close = min(to_utc(w.t_close), t_loss)
            if new_close <= to_utc(w.t_open):
                continue
            kept.append(ContactWindow(
                node_a=w.node_a, node_b=w.node_b, kind=w.kind,
                t_open=w.t_open, t_close=new_close,
                min_range_km=w.min_range_km, max_range_km=w.max_range_km,
                max_elevation_deg=w.max_elevation_deg, grid_step_s=w.grid_step_s,
                clipped_start=w.clipped_start, clipped_end=False))
        return ContactGraph(windows=kept, t0=self.t0, t1=self.t1, nodes=list(self.nodes))


@dataclass(frozen=True)
class TeEdge:
    """One edge of a time-expanded graph.

    Attributes
    ----------
    src, dst : ``(node, slot)`` pairs.
    cost_s : edge latency [s], > 0.
    capacity_bits : bits the edge can carry in its slot, >= 0.
    kind : ``"hold"`` or ``"tx"``.
    link : the originating link as ``(node_a, node_b)``, or ``None`` for a hold.
    """

    src: tuple[str, int]
    dst: tuple[str, int]
    cost_s: float
    capacity_bits: float
    kind: str
    link: tuple[str, str] | None = None


@dataclass
class TimeExpandedGraph:
    """Slot-layered unrolling of a :class:`ContactGraph`.

    Build with :meth:`from_contact_graph`.  ``edges_out`` maps a
    ``(node, slot)`` pair to its outgoing :class:`TeEdge` list.
    """

    nodes: list[str]
    n_slots: int
    slot_s: float
    t0: datetime
    edges: list[TeEdge]
    edges_out: dict[tuple[str, int], list[TeEdge]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {self.n_slots}")
        if self.slot_s <= 0.0:
            raise ValueError(f"slot_s must be > 0, got {self.slot_s}")
        self.edges_out = {}
        for e in self.edges:
            self.edges_out.setdefault(e.src, []).append(e)

    def slot_start(self, k: int) -> datetime:
        """UTC start of slot ``k``."""
        if not 0 <= k <= self.n_slots:
            raise IndexError(f"slot {k} outside [0, {self.n_slots}]")
        return self.t0 + timedelta(seconds=self.slot_s * k)

    @property
    def n_te_nodes(self) -> int:
        """Number of time-expanded nodes, ``len(nodes) * (n_slots + 1)``."""
        return len(self.nodes) * (self.n_slots + 1)

    @classmethod
    def from_contact_graph(cls, cg: ContactGraph, slot_s: float,
                           rate_fn=None,
                           store_capacity_bits: float = float("inf"),
                           min_overlap_fraction: float = 1.0,
                           default_rate_bps: float = 1.0e6,
                           ) -> TimeExpandedGraph:
        """Unroll ``cg`` into slots of ``slot_s`` seconds.

        Parameters
        ----------
        cg : the contact graph.
        slot_s : slot duration [s], > 0.  Must divide the horizon to within one
            slot; a trailing partial slot is dropped (and so is any contact
            that only exists inside it).
        rate_fn : optional ``f(window, slot_mid_utc) -> bit/s`` giving the link
            rate.  When ``None``, ``default_rate_bps`` is used for every link.
        store_capacity_bits : onboard buffer capacity per hold edge [bits].
        min_overlap_fraction : see :meth:`ContactGraph.windows_in_slot`.
        default_rate_bps : rate used when ``rate_fn`` is None [bit/s], > 0.

        Transmit edges are created in both directions (links are symmetric
        here; a one-way link would be modelled by a ``rate_fn`` returning 0).
        """
        if slot_s <= 0.0:
            raise ValueError(f"slot_s must be > 0, got {slot_s}")
        if default_rate_bps <= 0.0:
            raise ValueError(f"default_rate_bps must be > 0, got {default_rate_bps}")
        n_slots = int(np.floor(cg.horizon_s / slot_s + 1e-9))
        if n_slots < 1:
            raise ValueError(
                f"slot_s {slot_s} s exceeds the horizon {cg.horizon_s} s: no slot fits")
        edges: list[TeEdge] = []
        for k in range(n_slots):
            t_a = cg.t0 + timedelta(seconds=slot_s * k)
            t_b = t_a + timedelta(seconds=slot_s)
            t_mid = t_a + timedelta(seconds=slot_s / 2.0)
            for n in cg.nodes:
                edges.append(TeEdge(src=(n, k), dst=(n, k + 1), cost_s=slot_s,
                                    capacity_bits=store_capacity_bits, kind="hold"))
            for w, frac in cg.windows_in_slot(t_a, t_b, min_overlap_fraction):
                rate = default_rate_bps if rate_fn is None else float(rate_fn(w, t_mid))
                if rate < 0.0:
                    raise ValueError(f"rate_fn returned a negative rate for {w.node_a}-"
                                     f"{w.node_b}: {rate}")
                cap = rate * slot_s * frac
                prop_s = w.min_range_km / SPEED_OF_LIGHT_KM_S
                cost = slot_s + prop_s
                edges.append(TeEdge(src=(w.node_a, k), dst=(w.node_b, k + 1),
                                    cost_s=cost, capacity_bits=cap, kind="tx",
                                    link=(w.node_a, w.node_b)))
                edges.append(TeEdge(src=(w.node_b, k), dst=(w.node_a, k + 1),
                                    cost_s=cost, capacity_bits=cap, kind="tx",
                                    link=(w.node_a, w.node_b)))
        return cls(nodes=list(cg.nodes), n_slots=n_slots, slot_s=slot_s, t0=cg.t0,
                   edges=edges)
