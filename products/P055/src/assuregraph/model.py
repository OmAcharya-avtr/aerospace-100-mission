"""Data model for a Goal Structuring Notation (GSN) assurance case.

Naming follows the **GSN Community Standard Version 3** (SCSC-141C, SCSC
Assurance Case Working Group, May 2021), Core GSN: the six element types are
Goal, Strategy, Solution, Context, Assumption and Justification, and the two
relationship types are ``SupportedBy`` and ``InContextOf``. A Goal or Strategy
may additionally carry the *Undeveloped* decorator, which in the standard
declares that the element is deliberately not yet elaborated.

Scope of this module: it holds the structure of a case. It carries no notion of
whether any argument in the case is sound, and nothing here is evidence that a
system is safe. See the package docstring.

Units: none of the fields in this module is a physical quantity. The only
numeric quantities anywhere in the package are counts of nodes, counts of
findings, byte lengths of evidence artifacts and wall-clock seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "PERMITTED_EDGES",
    "AssuranceCase",
    "Edge",
    "EdgeKind",
    "Evidence",
    "Node",
    "NodeKind",
]


class NodeKind(Enum):
    """The six Core GSN element types (GSN Standard v3, Core GSN).

    The value of each member is the lowercase key used in the YAML case format.
    ``conventional_prefix`` is the identifier prefix the standard uses in its
    own examples; assuregraph does *not* enforce it, because real cases use
    project-specific identifier schemes.
    """

    GOAL = "goal"
    STRATEGY = "strategy"
    SOLUTION = "solution"
    CONTEXT = "context"
    ASSUMPTION = "assumption"
    JUSTIFICATION = "justification"

    @property
    def conventional_prefix(self) -> str:
        """The GSN Standard v3 identifier prefix for this element type."""
        return {
            NodeKind.GOAL: "G",
            NodeKind.STRATEGY: "S",
            NodeKind.SOLUTION: "Sn",
            NodeKind.CONTEXT: "C",
            NodeKind.ASSUMPTION: "A",
            NodeKind.JUSTIFICATION: "J",
        }[self]


class EdgeKind(Enum):
    """The two Core GSN relationship types (GSN Standard v3, Core GSN).

    ``SUPPORTED_BY`` is the inferential or evidential relationship (drawn with a
    solid line and filled arrowhead in the standard). ``IN_CONTEXT_OF`` is the
    contextual relationship (solid line, hollow arrowhead).
    """

    SUPPORTED_BY = "supported_by"
    IN_CONTEXT_OF = "in_context_of"


#: Which (source kind, edge kind) -> permitted target kinds, per GSN Standard
#: v3 Core GSN. A ``SupportedBy`` relationship may originate only from a Goal or
#: a Strategy and may terminate only on a Goal, a Strategy or a Solution. An
#: ``InContextOf`` relationship may originate only from a Goal or a Strategy and
#: may terminate only on a Context, an Assumption or a Justification. Edges that
#: violate this table are rejected at parse time, not reported as findings:
#: a document containing one is not a GSN case.
PERMITTED_EDGES: dict[tuple[NodeKind, EdgeKind], frozenset[NodeKind]] = {
    (NodeKind.GOAL, EdgeKind.SUPPORTED_BY): frozenset(
        {NodeKind.GOAL, NodeKind.STRATEGY, NodeKind.SOLUTION}
    ),
    (NodeKind.STRATEGY, EdgeKind.SUPPORTED_BY): frozenset(
        {NodeKind.GOAL, NodeKind.STRATEGY, NodeKind.SOLUTION}
    ),
    (NodeKind.GOAL, EdgeKind.IN_CONTEXT_OF): frozenset(
        {NodeKind.CONTEXT, NodeKind.ASSUMPTION, NodeKind.JUSTIFICATION}
    ),
    (NodeKind.STRATEGY, EdgeKind.IN_CONTEXT_OF): frozenset(
        {NodeKind.CONTEXT, NodeKind.ASSUMPTION, NodeKind.JUSTIFICATION}
    ),
}


@dataclass(frozen=True)
class Evidence:
    """An evidence artifact cited by a Solution node.

    Attributes:
        path: Path to the artifact, relative to the case file's directory unless
            absolute. Not resolved at construction time.
        sha256: The SHA-256 hex digest of the artifact's bytes *as recorded when
            the claim was written*, or ``None`` if the author recorded no digest.
            A recorded digest that no longer matches the file on disk is what
            this package calls staleness. The digest says the bytes changed; it
            says nothing about whether the change matters.
        recorded: Free-text date or revision the author recorded alongside the
            digest. Never parsed, never compared, carried through to reports.
    """

    path: str
    sha256: str | None = None
    recorded: str | None = None


@dataclass(frozen=True)
class Node:
    """One GSN element.

    Attributes:
        node_id: Unique identifier within the case.
        kind: One of the six Core GSN element types.
        statement: The element's text, as it would appear in the drawn node.
        undeveloped: The GSN *Undeveloped* decorator. Only meaningful on a Goal
            or a Strategy. When true, the element's lack of support is a
            declaration by the author, reported separately from an undeclared
            gap.
        discharged: Only meaningful on an Assumption. True asserts that the
            assumption has been discharged. ``discharged_by`` names the node
            that discharges it. assuregraph checks that the reference resolves;
            it cannot check that the discharge is valid.
        discharged_by: Node id that discharges this Assumption, or ``None``.
        evidence: The artifact a Solution cites, or ``None``.
    """

    node_id: str
    kind: NodeKind
    statement: str
    undeveloped: bool = False
    discharged: bool = False
    discharged_by: str | None = None
    evidence: Evidence | None = None


@dataclass(frozen=True)
class Edge:
    """One GSN relationship, directed from ``source`` to ``target``."""

    source: str
    target: str
    kind: EdgeKind


@dataclass(frozen=True)
class AssuranceCase:
    """A parsed assurance case: nodes, edges and the directory they resolve against.

    Instances are produced by :func:`assuregraph.parse.load_case` or
    :func:`assuregraph.parse.parse_case`, which are the only places the schema
    is enforced. Constructing one directly bypasses schema validation.

    Attributes:
        name: Case name, from the document's ``name`` key.
        nodes: Mapping from node id to :class:`Node`, in document order.
        edges: Relationships, in document order.
        base_dir: Directory that relative evidence paths resolve against. This
            is the case file's own directory for :func:`load_case`.
        top_goals: Node ids the document declares as the top of the argument.
            Empty when the document declares none, in which case the roots are
            inferred as the Goals with no incoming ``SupportedBy`` edge.
    """

    name: str
    nodes: dict[str, Node]
    edges: tuple[Edge, ...]
    base_dir: str = "."
    top_goals: tuple[str, ...] = field(default_factory=tuple)

    def nodes_of_kind(self, kind: NodeKind) -> list[Node]:
        """Return every node of ``kind``, in document order."""
        return [n for n in self.nodes.values() if n.kind == kind]

    def edges_of_kind(self, kind: EdgeKind) -> list[Edge]:
        """Return every edge of ``kind``, in document order."""
        return [e for e in self.edges if e.kind == kind]

    def resolved_top_goals(self) -> tuple[str, ...]:
        """Declared top goals, or the inferred roots when none were declared.

        A root is a Goal with no incoming ``SupportedBy`` edge. When the
        document declares ``top_goals`` explicitly, that declaration is returned
        unchanged and no inference is performed.
        """
        if self.top_goals:
            return self.top_goals
        supported = {e.target for e in self.edges if e.kind == EdgeKind.SUPPORTED_BY}
        return tuple(
            n.node_id
            for n in self.nodes.values()
            if n.kind == NodeKind.GOAL and n.node_id not in supported
        )
