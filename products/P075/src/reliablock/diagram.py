"""The declared reliability block diagram, as a checkable artifact.

What makes this a product rather than a formula
-----------------------------------------------

A reliability number is almost always quoted without the diagram it came from,
and the diagram is where the assumptions live. This module makes the diagram a
declared object -- a YAML file or a constructed tree -- that can be checked,
diffed, reviewed and attached to the number. The checks are in
:func:`Diagram.validate` and they are deliberately strict: a declaration that
does not say what it means is rejected rather than defaulted.

What is rejected, and why
-------------------------

================================  ============================================
rejected                          because
================================  ============================================
unknown node or distribution key  a typo in ``rate_per_hour`` that fell through
                                  to a default would report a reliability of
                                  1.0 and no complaint
a block referenced but not        the arithmetic would silently use whatever
declared                          the loader left behind
a duplicate block name            two different items would share one
                                  failure probability and one importance
                                  measure, and the cut sets would be wrong
a parallel or k-of-n group with   redundancy over one item is not redundancy,
one child                         and the declaration almost certainly meant
                                  something else
``k`` outside ``1..n``            ``k = 0`` is certain success and ``k > n``
                                  certain failure; neither is a diagram
a standby group with no switch    the switching reliability is the term this
reliability declared              package exists to make explicit, so it is
                                  required and not defaulted to 1.0
a common-cause beta on anything   a beta factor is accepted on ``parallel``
but ``parallel`` or ``k_of_n``    and ``k_of_n`` only. A ``standby`` or
with ``k < n``                    ``shared_load`` node is already a
                                  dependent-failure model, and compounding it
                                  with a beta factor needs a combined Markov
                                  model this package does not have. A
                                  ``k_of_n`` node with ``k = n`` is a series
                                  group, not a redundant one: a beta factor on
                                  it has no meaning and makes the arithmetic
                                  look better, measured at a ratio of 0.918939
                                  for ``k = n = 6`` at ``beta = 0.1``
a common-cause beta outside       beta is a fraction of the total failure
``[0, 1]``                        rate and nothing else
a mission profile that ages a     a per-actuation probability is not a rate and
one-shot block                    cannot be multiplied by a duty cycle
================================  ============================================

Structure types
---------------

``series``
    Succeeds only if every child succeeds.
``parallel``
    Succeeds if any child succeeds. ``k_of_n`` with ``k = 1``.
``k_of_n``
    Succeeds if at least ``k`` of ``n`` children succeed.
``standby``
    One primary and one or more standby units, with a declared switching
    reliability. Cold standby by default: a standby unit accrues no hazard
    until it is called upon. Warm standby accrues hazard at a declared
    fraction of its operating rate.
``shared_load``
    Two units sharing a load, each at a declared fraction of its full-load
    hazard rate while both survive and at the full rate once one has failed.
``network``
    An explicit two-terminal edge list, for a structure that is not
    series-parallel. The bridge network is the canonical case and the reason
    this node type exists: no nesting of series and parallel reproduces it.

Common cause is declared on a redundant node as a beta factor, not as a
structure type: see :mod:`reliablock.commoncause`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .distributions import Exponential, FailureDistribution, distribution_from_mapping

__all__ = [
    "Block",
    "Diagram",
    "KOfN",
    "Network",
    "Node",
    "Parallel",
    "Series",
    "SharedLoad",
    "Standby",
    "load_diagram",
    "node_from_mapping",
]

#: Node kinds whose success is a monotone Boolean function of its children's
#: success, quantified by independent-event arithmetic. Cut sets, exhaustive
#: state enumeration and importance measures are exact for a diagram built
#: only from these.
BOOLEAN_KINDS = frozenset({"series", "parallel", "k_of_n", "network"})

#: Node kinds whose Boolean structure function is exact but whose
#: quantification is *not* the independent-event arithmetic over that function,
#: because the units' failure times are dependent by construction. Reported
#: explicitly rather than left for the reader to discover.
DYNAMIC_KINDS = frozenset({"standby", "shared_load"})


@dataclass(frozen=True)
class Block:
    """A leaf item with a failure distribution.

    ``name`` is the identifier used in cut sets and importance tables.
    ``beta_group`` is set by :mod:`reliablock.commoncause` when a block has
    been split into its independent and common-cause parts; a declared diagram
    leaves it ``None``.
    """

    name: str
    distribution: FailureDistribution
    description: str = ""
    beta_group: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"block name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.distribution, FailureDistribution):
            raise TypeError(
                f"block {self.name!r} needs a FailureDistribution, "
                f"got {type(self.distribution).__name__}"
            )

    def to_mapping(self) -> dict[str, Any]:
        """Round-trippable declaration of this block."""
        body: dict[str, Any] = {
            "kind": "block",
            "name": self.name,
            "distribution": self.distribution.to_mapping(),
        }
        if self.description:
            body["description"] = self.description
        return body


@dataclass(frozen=True)
class Node:
    """Base class for a structure node. ``kind`` is the declared type name."""

    kind: str = ""

    @property
    def children(self) -> tuple[Node | Block, ...]:
        """Child nodes and blocks, in declaration order."""
        return ()

    @property
    def beta(self) -> float:
        """Declared common-cause beta factor; 0.0 for a node that has none."""
        return 0.0

    def to_mapping(self) -> dict[str, Any]:
        """Round-trippable declaration of this node."""
        raise NotImplementedError


@dataclass(frozen=True)
class Series(Node):
    """Succeeds only if every child succeeds."""

    kind: str = "series"
    items: tuple[Node | Block, ...] = ()
    label: str = ""

    @property
    def children(self) -> tuple[Node | Block, ...]:
        return self.items

    def to_mapping(self) -> dict[str, Any]:
        body: dict[str, Any] = {"kind": "series", "items": [c.to_mapping() for c in self.items]}
        if self.label:
            body["label"] = self.label
        return body


@dataclass(frozen=True)
class Parallel(Node):
    """Succeeds if at least one child succeeds. ``beta`` is the common-cause factor."""

    kind: str = "parallel"
    items: tuple[Node | Block, ...] = ()
    beta_factor: float = 0.0
    label: str = ""

    @property
    def children(self) -> tuple[Node | Block, ...]:
        return self.items

    @property
    def beta(self) -> float:
        return self.beta_factor

    def to_mapping(self) -> dict[str, Any]:
        body: dict[str, Any] = {"kind": "parallel", "items": [c.to_mapping() for c in self.items]}
        if self.beta_factor:
            body["beta_factor"] = self.beta_factor
        if self.label:
            body["label"] = self.label
        return body


@dataclass(frozen=True)
class KOfN(Node):
    """Succeeds if at least ``k`` of the ``n`` children succeed."""

    kind: str = "k_of_n"
    k: int = 1
    items: tuple[Node | Block, ...] = ()
    beta_factor: float = 0.0
    label: str = ""

    @property
    def children(self) -> tuple[Node | Block, ...]:
        return self.items

    @property
    def beta(self) -> float:
        return self.beta_factor

    def to_mapping(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "kind": "k_of_n",
            "k": self.k,
            "items": [c.to_mapping() for c in self.items],
        }
        if self.beta_factor:
            body["beta_factor"] = self.beta_factor
        if self.label:
            body["label"] = self.label
        return body


@dataclass(frozen=True)
class Standby(Node):
    """A primary with one or more standby units and a declared switch reliability.

    ``switch_reliability`` is the probability that the changeover succeeds when
    called upon, dimensionless, in ``[0, 1]``. It is required: an undeclared
    switching reliability defaulting to 1.0 is exactly the implicit assumption
    this package exists to surface.

    ``standby_hazard_fraction`` is the fraction of its operating hazard rate
    that a standby unit accrues while waiting. ``0.0`` is cold standby,
    ``1.0`` is hot standby (which is parallel redundancy with a switch),
    anything between is warm standby.
    """

    kind: str = "standby"
    primary: Node | Block | None = None
    standbys: tuple[Node | Block, ...] = ()
    switch_reliability: float = 1.0
    standby_hazard_fraction: float = 0.0
    label: str = ""

    @property
    def children(self) -> tuple[Node | Block, ...]:
        head: tuple[Node | Block, ...] = () if self.primary is None else (self.primary,)
        return head + self.standbys

    def to_mapping(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "kind": "standby",
            "primary": None if self.primary is None else self.primary.to_mapping(),
            "standbys": [c.to_mapping() for c in self.standbys],
            "switch_reliability": self.switch_reliability,
            "standby_hazard_fraction": self.standby_hazard_fraction,
        }
        if self.label:
            body["label"] = self.label
        return body


@dataclass(frozen=True)
class SharedLoad(Node):
    """Two units sharing a load, each at a reduced hazard rate while both survive.

    Each unit's declared distribution gives its **full-load** hazard rate and
    must be :class:`~reliablock.distributions.Exponential`: the closed form
    implemented in :mod:`reliablock.quantify` is the four-state Markov solution
    and a non-exponential unit has no closed form in it. That is a limitation,
    stated rather than silently approximated.

    ``shared_rate_factor`` is the fraction of its full-load rate that a unit
    carries while both units are alive and the load is shared, dimensionless,
    in ``(0, 1]``. ``1.0`` means sharing buys nothing and the node is then
    exactly a parallel pair -- which is asserted as a property test.

    There is one source of truth for every rate: the units' distributions. A
    node that also declared its own absolute rates would let the two disagree,
    and a declaration that can contradict itself is not a checkable artifact.
    """

    kind: str = "shared_load"
    units: tuple[Block, ...] = ()
    shared_rate_factor: float = 1.0
    label: str = ""

    @property
    def children(self) -> tuple[Node | Block, ...]:
        return self.units

    def to_mapping(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "kind": "shared_load",
            "units": [c.to_mapping() for c in self.units],
            "shared_rate_factor": self.shared_rate_factor,
        }
        if self.label:
            body["label"] = self.label
        return body


@dataclass(frozen=True)
class Network(Node):
    """An explicit two-terminal network: edges carry the blocks.

    ``edges`` is a sequence of ``(source_node, sink_node, block)``. The system
    succeeds if a path of surviving edges connects ``source`` to ``sink``.

    This node type exists because the bridge network is not series-parallel:
    no nesting of :class:`Series` and :class:`Parallel` reproduces it, and a
    package that only nests those two cannot express it at all. The structure
    function is evaluated by connectivity, and the closed form for the
    five-edge bridge is derived in :mod:`reliablock.bridge`.
    """

    kind: str = "network"
    edges: tuple[tuple[str, str, Block], ...] = ()
    source: str = "s"
    sink: str = "t"
    label: str = ""

    @property
    def children(self) -> tuple[Node | Block, ...]:
        return tuple(block for _, _, block in self.edges)

    def to_mapping(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "kind": "network",
            "source": self.source,
            "sink": self.sink,
            "edges": [
                {"from": a, "to": b, "block": block.to_mapping()} for a, b, block in self.edges
            ],
        }
        if self.label:
            body["label"] = self.label
        return body


@dataclass
class Diagram:
    """A declared reliability block diagram with a name and a root node."""

    name: str
    root: Node | Block
    description: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"diagram name must be a non-empty string, got {self.name!r}")
        self.validate()

    # ---------------------------------------------------------------- walking

    def walk(self) -> Iterator[Node | Block]:
        """Yield every node and block, root first, in declaration order."""
        stack: list[Node | Block] = [self.root]
        while stack:
            item = stack.pop()
            yield item
            if isinstance(item, Node):
                stack.extend(reversed(item.children))

    def blocks(self) -> tuple[Block, ...]:
        """Every leaf block, in first-encountered order, each exactly once."""
        seen: dict[str, Block] = {}
        for item in self.walk():
            if isinstance(item, Block) and item.name not in seen:
                seen[item.name] = item
        return tuple(seen.values())

    def block_names(self) -> tuple[str, ...]:
        """Names of every leaf block, in first-encountered order."""
        return tuple(b.name for b in self.blocks())

    def nodes_of_kind(self, kind: str) -> tuple[Node, ...]:
        """Every node whose declared kind is ``kind``."""
        return tuple(n for n in self.walk() if isinstance(n, Node) and n.kind == kind)

    @property
    def is_boolean(self) -> bool:
        """True when every node is a Boolean kind, so the structure function quantifies.

        A diagram containing a ``standby`` or ``shared_load`` node has an exact
        Boolean structure function -- and therefore exact minimal cut sets --
        but its quantification is not the independent-event arithmetic over
        that function, because the units' failure times are dependent by
        construction. Importance measures and the exhaustive-state cross-check
        require this to be True, and say so when it is not.
        """
        return all(not isinstance(n, Node) or n.kind in BOOLEAN_KINDS for n in self.walk())

    @property
    def declared_betas(self) -> dict[str, float]:
        """Label-to-beta for every node carrying a non-zero common-cause factor."""
        out: dict[str, float] = {}
        for index, item in enumerate(self.walk()):
            if isinstance(item, Node) and item.beta > 0.0:
                out[item.label or f"{item.kind}#{index}"] = item.beta
        return out

    # ------------------------------------------------------------- validation

    def validate(self) -> None:
        """Raise ``ValueError`` on any declaration this package will not quantify.

        Called by ``__post_init__``, so a ``Diagram`` object cannot exist in an
        invalid state. The table in this module's docstring lists every
        rejection and the reason for it.
        """
        names: list[str] = []
        for item in self.walk():
            if isinstance(item, Block):
                names.append(item.name)
                continue
            self._validate_node(item)
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"diagram {self.name!r} declares duplicate block names {duplicates}. "
                "Two different items sharing one name would share one failure "
                "probability and one importance measure, and the cut sets would "
                "be wrong. Rename them, or declare one block and reference it once."
            )
        if not names:
            raise ValueError(f"diagram {self.name!r} declares no blocks")

    @staticmethod
    def _validate_node(node: Node) -> None:
        """Validate one structure node. Raises ``ValueError`` with the reason."""
        where = node.label or node.kind
        if isinstance(node, Series):
            if not node.items:
                raise ValueError(f"series node {where!r} has no items")
        elif isinstance(node, Parallel):
            if len(node.items) < 2:
                raise ValueError(
                    f"parallel node {where!r} has {len(node.items)} item(s); "
                    "redundancy over fewer than two items is not redundancy. "
                    "Declare the item directly if that is what was meant."
                )
            _check_beta(node.beta_factor, where)
        elif isinstance(node, KOfN):
            n = len(node.items)
            if n < 2:
                raise ValueError(f"k_of_n node {where!r} has {n} item(s); at least 2 are needed")
            if not isinstance(node.k, int) or isinstance(node.k, bool):
                raise ValueError(f"k_of_n node {where!r} has non-integer k={node.k!r}")
            if not 1 <= node.k <= n:
                raise ValueError(
                    f"k_of_n node {where!r} declares k={node.k} with n={n}. "
                    "k must lie in 1..n: k=0 is certain success and k>n certain "
                    "failure, and neither is a diagram."
                )
            if node.beta_factor > 0.0 and node.k == n:
                raise ValueError(
                    f"k_of_n node {where!r} declares beta_factor="
                    f"{node.beta_factor:g} with k = n = {n}, which is a series "
                    "group and not a redundant one. A beta factor describes the "
                    "correlation BETWEEN redundant items; on a group that fails "
                    "when any single member fails it has no meaning, and the "
                    "arithmetic makes the group look BETTER -- measured at a "
                    "ratio of 0.918939 for k = n = 6 at beta = 0.1 in "
                    "validation/validate_commoncause_output.txt -- because a "
                    "shared cause that fails all n members at once is no worse "
                    "than any one of them failing alone. Declare the group as "
                    "series, or declare the beta on the redundant group that "
                    "actually shares the cause."
                )
            _check_beta(node.beta_factor, where)
        elif isinstance(node, Standby):
            if node.primary is None:
                raise ValueError(f"standby node {where!r} has no primary")
            if not node.standbys:
                raise ValueError(f"standby node {where!r} has no standby units")
            if not 0.0 <= float(node.switch_reliability) <= 1.0:
                raise ValueError(
                    f"standby node {where!r} declares switch_reliability="
                    f"{node.switch_reliability!r}, which is not a probability. "
                    "The switching reliability is the term this package exists "
                    "to make explicit; it is required and never defaulted."
                )
            if not 0.0 <= float(node.standby_hazard_fraction) <= 1.0:
                raise ValueError(
                    f"standby node {where!r} declares standby_hazard_fraction="
                    f"{node.standby_hazard_fraction!r}; it must lie in [0, 1], "
                    "0 being cold standby and 1 hot."
                )
        elif isinstance(node, SharedLoad):
            if len(node.units) != 2:
                raise ValueError(
                    f"shared_load node {where!r} declares {len(node.units)} units; "
                    "the two-state shared-load model implemented here takes "
                    "exactly 2. More units need a full Markov model, which this "
                    "package does not have."
                )
            factor = float(node.shared_rate_factor)
            if not 0.0 < factor <= 1.0:
                raise ValueError(
                    f"shared_load node {where!r} declares shared_rate_factor="
                    f"{node.shared_rate_factor!r}. It is the fraction of its "
                    "full-load hazard rate that a unit carries while sharing, so "
                    "it lies in (0, 1]. A factor above 1 would mean sharing a "
                    "load makes a unit fail faster than carrying all of it."
                )
            for unit in node.units:
                if not isinstance(unit, Block) or not isinstance(unit.distribution, Exponential):
                    name = getattr(unit, "name", unit)
                    raise ValueError(
                        f"shared_load node {where!r} unit {name!r} must be a block "
                        "with an exponential distribution. The closed form in "
                        "reliablock.quantify is the four-state Markov solution for "
                        "constant rates; there is no closed form in it for a "
                        "non-exponential unit."
                    )
        elif isinstance(node, Network):
            _validate_network(node, where)
        else:  # pragma: no cover - unreachable for the declared kinds
            raise ValueError(f"unknown node type {type(node).__name__}")

    # ----------------------------------------------------------- round-tripping

    def to_mapping(self) -> dict[str, Any]:
        """Round-trippable declaration of the whole diagram."""
        body: dict[str, Any] = {"name": self.name, "root": self.root.to_mapping()}
        if self.description:
            body["description"] = self.description
        if self.notes:
            body["notes"] = list(self.notes)
        return body

    def to_yaml(self) -> str:
        """The declaration as YAML, sorted so that two diagrams diff cleanly."""
        return yaml.safe_dump(self.to_mapping(), sort_keys=True, default_flow_style=False)


def _check_beta(beta: float, where: str) -> None:
    """Raise unless ``beta`` is a fraction in ``[0, 1]``."""
    value = float(beta)
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"node {where!r} declares beta_factor={beta!r}. A beta factor is the "
            "fraction of a component's total failure rate attributed to a cause "
            "shared with the rest of its redundant group, so it lies in [0, 1]."
        )


def _validate_network(node: Network, where: str) -> None:
    """Raise unless the network is a usable two-terminal structure."""
    if not node.edges:
        raise ValueError(f"network node {where!r} has no edges")
    vertices = {node.source, node.sink}
    for a, b, block in node.edges:
        if not isinstance(a, str) or not isinstance(b, str):
            raise ValueError(f"network node {where!r} has a non-string vertex in an edge")
        if a == b:
            raise ValueError(
                f"network node {where!r} declares a self-loop at {a!r} carrying "
                f"block {block.name!r}. A self-loop cannot be on any path and "
                "its reliability would never enter the answer."
            )
        vertices.update((a, b))
    if node.source == node.sink:
        raise ValueError(f"network node {where!r} has source == sink == {node.source!r}")
    everything = {block.name for _, _, block in node.edges}
    from_source = _reachable(node, everything, node.source)
    if node.sink not in from_source:
        raise ValueError(
            f"network node {where!r}: no path from {node.source!r} to {node.sink!r} "
            "even with every edge working. The declared network can never succeed."
        )
    # An edge lies on some source-sink path exactly when, with that edge
    # removed, one of its endpoints is still reachable from the source and the
    # other from the sink. Reachability WITH the edge present is not the test:
    # in a connected undirected graph every vertex is reachable from every
    # other, so that test passes for a dangling stub and lets through an edge
    # whose reliability can never enter the answer.
    unused: list[str] = []
    for a, b, block in node.edges:
        others = everything - {block.name}
        from_s = _reachable(node, others, node.source)
        from_t = _reachable(node, others, node.sink)
        on_a_path = (a in from_s and b in from_t) or (b in from_s and a in from_t)
        if not on_a_path:
            unused.append(block.name)
    unused = sorted(set(unused))
    if unused:
        raise ValueError(
            f"network node {where!r} declares blocks {unused} on edges that lie on no "
            f"path from {node.source!r} to {node.sink!r}. Their reliability would "
            "never enter the answer."
        )


def _reachable(node: Network, working: set[str], start: str) -> set[str]:
    """Vertices reachable from ``start`` using only edges whose block is in ``working``.

    The network is treated as undirected: a reliability block diagram's edges
    carry items, not flow direction, and a path is a connection.
    """
    adjacency: dict[str, list[str]] = {}
    for a, b, block in node.edges:
        if block.name in working:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbour in adjacency.get(current, ()):
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen


# ------------------------------------------------------------------- loading

_NODE_KEYS: dict[str, set[str]] = {
    "block": {"kind", "name", "distribution", "description"},
    "series": {"kind", "items", "label"},
    "parallel": {"kind", "items", "beta_factor", "label"},
    "k_of_n": {"kind", "k", "items", "beta_factor", "label"},
    "standby": {
        "kind",
        "primary",
        "standbys",
        "switch_reliability",
        "standby_hazard_fraction",
        "label",
    },
    "shared_load": {"kind", "units", "shared_rate_factor", "label"},
    "network": {"kind", "edges", "source", "sink", "label"},
}


def node_from_mapping(mapping: dict[str, Any]) -> Node | Block:
    """Build a node or block from a declared mapping, rejecting unknown keys.

    Every key is checked against the node kind's accepted set. A declaration
    is an artifact a reader has to be able to check, so a key this loader does
    not understand is an error: it is far more likely to be a misspelling of a
    key that matters than a harmless annotation.
    """
    if not isinstance(mapping, dict):
        raise TypeError(f"node declaration must be a mapping, got {type(mapping).__name__}")
    kind = mapping.get("kind")
    if kind is None:
        raise ValueError(f"node declaration needs a 'kind' key; got keys {sorted(mapping)}")
    if kind not in _NODE_KEYS:
        raise ValueError(f"unknown node kind {kind!r}; expected one of {sorted(_NODE_KEYS)}")
    unknown = set(mapping) - _NODE_KEYS[kind]
    if unknown:
        raise ValueError(
            f"node kind {kind!r} does not accept {sorted(unknown)}; "
            f"accepted keys are {sorted(_NODE_KEYS[kind])}"
        )

    if kind == "block":
        if "name" not in mapping or "distribution" not in mapping:
            raise ValueError("block declaration requires 'name' and 'distribution'")
        return Block(
            name=mapping["name"],
            distribution=distribution_from_mapping(mapping["distribution"]),
            description=mapping.get("description", ""),
        )
    if kind == "series":
        return Series(items=_items(mapping, "items"), label=mapping.get("label", ""))
    if kind == "parallel":
        return Parallel(
            items=_items(mapping, "items"),
            beta_factor=float(mapping.get("beta_factor", 0.0)),
            label=mapping.get("label", ""),
        )
    if kind == "k_of_n":
        if "k" not in mapping:
            raise ValueError("k_of_n declaration requires 'k'")
        return KOfN(
            k=int(mapping["k"]),
            items=_items(mapping, "items"),
            beta_factor=float(mapping.get("beta_factor", 0.0)),
            label=mapping.get("label", ""),
        )
    if kind == "standby":
        if "switch_reliability" not in mapping:
            raise ValueError(
                "standby declaration requires 'switch_reliability'. An undeclared "
                "switching reliability defaulting to 1.0 is exactly the implicit "
                "assumption this package exists to surface."
            )
        if "primary" not in mapping:
            raise ValueError("standby declaration requires 'primary'")
        return Standby(
            primary=node_from_mapping(mapping["primary"]),
            standbys=_items(mapping, "standbys"),
            switch_reliability=float(mapping["switch_reliability"]),
            standby_hazard_fraction=float(mapping.get("standby_hazard_fraction", 0.0)),
            label=mapping.get("label", ""),
        )
    if kind == "shared_load":
        for required in ("units", "shared_rate_factor"):
            if required not in mapping:
                raise ValueError(f"shared_load declaration requires {required!r}")
        units = _items(mapping, "units")
        for unit in units:
            if not isinstance(unit, Block):
                raise ValueError("shared_load units must be blocks, not nested structures")
        return SharedLoad(
            units=tuple(units),  # type: ignore[arg-type]
            shared_rate_factor=float(mapping["shared_rate_factor"]),
            label=mapping.get("label", ""),
        )
    # network
    if "edges" not in mapping:
        raise ValueError("network declaration requires 'edges'")
    edges: list[tuple[str, str, Block]] = []
    for entry in mapping["edges"]:
        if not isinstance(entry, dict) or set(entry) - {"from", "to", "block"}:
            raise ValueError(
                f"network edge must be a mapping with keys from/to/block, got {entry!r}"
            )
        for required in ("from", "to", "block"):
            if required not in entry:
                raise ValueError(f"network edge requires {required!r}, got keys {sorted(entry)}")
        block = node_from_mapping(entry["block"])
        if not isinstance(block, Block):
            raise ValueError("a network edge carries a block, not a nested structure")
        edges.append((str(entry["from"]), str(entry["to"]), block))
    return Network(
        edges=tuple(edges),
        source=str(mapping.get("source", "s")),
        sink=str(mapping.get("sink", "t")),
        label=mapping.get("label", ""),
    )


def _items(mapping: dict[str, Any], key: str) -> tuple[Node | Block, ...]:
    """Build the child sequence under ``key``, rejecting a non-sequence."""
    raw = mapping.get(key, ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{mapping.get('kind')!r} node's {key!r} must be a list, got {raw!r}")
    return tuple(node_from_mapping(entry) for entry in raw)


def load_diagram(path: str | Path) -> Diagram:
    """Load a declared diagram from a YAML file.

    The path is used as given, so a caller passing a relative path gets a
    relative path quoted back in any error message. Every validation script and
    example in this repository changes to the repository root first, so a
    relative path means the same thing wherever the script was invoked from.
    """
    text = Path(path).read_text()
    body = yaml.safe_load(text)
    if not isinstance(body, dict):
        raise ValueError(f"{path}: a diagram declaration must be a mapping at the top level")
    unknown = set(body) - {"name", "root", "description", "notes"}
    if unknown:
        raise ValueError(
            f"{path}: diagram does not accept {sorted(unknown)}; "
            "accepted keys are ['description', 'name', 'notes', 'root']"
        )
    if "name" not in body or "root" not in body:
        raise ValueError(f"{path}: diagram declaration requires 'name' and 'root'")
    return Diagram(
        name=str(body["name"]),
        root=node_from_mapping(body["root"]),
        description=str(body.get("description", "")),
        notes=tuple(str(n) for n in body.get("notes", ())),
    )
