"""Assurance-case, artifact and evidence data types, with input validation.

VOCABULARY AND ITS SOURCE
-------------------------
The claim / argument / evidence vocabulary used here is the vocabulary of the
Goal Structuring Notation (GSN) and of ISO/IEC/IEEE 15026-2.

- Assurance Case Working Group, *Goal Structuring Notation Community
  Standard*, Version 3, Safety-Critical Systems Club, 2021. Node types
  ``Goal``, ``Strategy``, ``Solution``, ``Context``, ``Assumption``,
  ``Justification``; link types ``SupportedBy`` and ``InContextOf``.
- T. P. Kelly and J. A. McDermid, "The Goal Structuring Notation -- A Safety
  Argument Notation", *Proceedings of the DSN 2004 Workshop on Assurance
  Cases*, 2004, for the notation's original published description.
- ISO/IEC/IEEE 15026-2:2022, *Systems and software engineering -- Systems and
  software assurance -- Part 2: Assurance case*, for the claim / argument /
  evidence decomposition and the requirement that an assurance case record the
  evidence supporting each claim.
- T. P. Kelly and J. A. McDermid, "A Systematic Approach to Safety Case
  Maintenance", *Reliability Engineering and System Safety*, 71(3):271--284,
  2001, for the problem this product measures: that a change to the system
  invalidates parts of an existing safety argument, and that the affected parts
  must be identified rather than assumed.

WHAT IS MODELLED AND WHAT IS NOT is stated in ``README.md`` under Limitations
and in ``validation/VALIDATION.md``. In short: the node and link types above,
the restriction that ``Solution`` is a leaf, and the restriction that
``SupportedBy`` and ``InContextOf`` may only originate at a ``Goal`` or a
``Strategy``. Everything else in the standards -- modules, contracts, away
goals, undeveloped entities, public/private visibility, confidence arguments,
the pattern language, and every process requirement -- is NOT modelled.

UNITS
-----
Times are integer logical revision counts on one monotonically increasing axis,
not seconds. The replayed change history of this product is a sequence of
revisions, so the natural unit is the revision index. Nothing here converts to
SI time, and nothing here assumes a uniform spacing between revisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

#: GSN node types this product models. Everything else in the standard is out
#: of scope and is listed as such in README.md under Limitations.
NODE_TYPES: tuple[str, ...] = (
    "Goal",
    "Strategy",
    "Solution",
    "Context",
    "Assumption",
    "Justification",
)

#: GSN link types this product models.
LINK_TYPES: tuple[str, ...] = ("SupportedBy", "InContextOf")

#: ``SupportedBy`` is permitted from these node types only (GSN Community
#: Standard v3: a SupportedBy link originates at a Goal or a Strategy).
SUPPORTED_BY_SOURCES: tuple[str, ...] = ("Goal", "Strategy")

#: ``SupportedBy`` may terminate at these node types only.
SUPPORTED_BY_TARGETS: tuple[str, ...] = ("Goal", "Strategy", "Solution")

#: ``InContextOf`` may terminate at these node types only.
IN_CONTEXT_OF_TARGETS: tuple[str, ...] = ("Context", "Assumption", "Justification")

#: Change-event kinds. ``write`` replaces an artifact's content; ``touch``
#: updates its modification revision and leaves the content identical. The
#: second kind exists because it is the case baseline 1 gets wrong in the
#: false-positive direction, and a corpus without it could not measure that.
EVENT_KINDS: tuple[str, ...] = ("write", "touch")

EventKind = Literal["write", "touch"]


def _require_identifier(value: object, what: str) -> str:
    """Return ``value`` as a non-empty identifier string or raise ``ValueError``."""
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string, got {type(value).__name__}")
    if not value or value.strip() != value:
        raise ValueError(
            f"{what} must be a non-empty string with no leading or trailing "
            f"whitespace, got {value!r}"
        )
    return value


def _require_revision(value: object, what: str) -> int:
    """Return ``value`` as a non-negative integer revision or raise."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} must be an int revision index, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{what} must be a non-negative revision index, got {value}")
    return value


@dataclass(frozen=True)
class ChangeEvent:
    """One entry in a replayed change history.

    Attributes
    ----------
    time:
        Revision index, non-negative integer, on the single logical axis.
    artifact:
        Identifier of the artifact the event applies to.
    kind:
        ``"write"`` (content replaced) or ``"touch"`` (modification revision
        bumped, content identical).
    content:
        The new content for a ``write``; must be ``None`` for a ``touch``.
    """

    time: int
    artifact: str
    kind: EventKind
    content: str | None = None

    def __post_init__(self) -> None:
        _require_revision(self.time, "ChangeEvent.time")
        _require_identifier(self.artifact, "ChangeEvent.artifact")
        if self.kind not in EVENT_KINDS:
            raise ValueError(
                f"ChangeEvent.kind must be one of {EVENT_KINDS}, got {self.kind!r}. "
                f"Use 'write' when the content changes and 'touch' when only the "
                f"modification revision changes."
            )
        if self.kind == "write" and self.content is None:
            raise ValueError(
                f"ChangeEvent(kind='write') for artifact {self.artifact!r} at time "
                f"{self.time} must carry content; a write with no content cannot be "
                f"replayed. Pass kind='touch' if the content is unchanged."
            )
        if self.kind == "touch" and self.content is not None:
            raise ValueError(
                f"ChangeEvent(kind='touch') for artifact {self.artifact!r} at time "
                f"{self.time} must not carry content; a touch leaves the content "
                f"identical by definition. Pass kind='write' if the content changes."
            )


@dataclass(frozen=True)
class EvidenceItem:
    """One evidence item: a GSN ``Solution`` node and the artifact it rests on.

    Attributes
    ----------
    identifier:
        Unique identifier of the evidence item.
    subject:
        Identifier of the artifact the evidence is *about*. This is the subject
        whose change makes the evidence stale.
    produced_at:
        Revision index at which the evidence was produced.
    goal:
        Identifier of the GSN ``Goal`` the evidence ultimately supports.
    solution:
        Identifier of the GSN ``Solution`` node that cites this evidence.
    true_stale:
        The by-construction ground-truth staleness label, when the corpus
        carries one. ``None`` for evidence with no known truth.
    """

    identifier: str
    subject: str
    produced_at: int
    goal: str
    solution: str
    true_stale: bool | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.identifier, "EvidenceItem.identifier")
        _require_identifier(self.subject, "EvidenceItem.subject")
        _require_identifier(self.goal, "EvidenceItem.goal")
        _require_identifier(self.solution, "EvidenceItem.solution")
        _require_revision(self.produced_at, "EvidenceItem.produced_at")
        if self.true_stale is not None and not isinstance(self.true_stale, bool):
            raise TypeError(
                f"EvidenceItem.true_stale must be a bool or None, got "
                f"{type(self.true_stale).__name__}"
            )


@dataclass(frozen=True)
class AssuranceCase:
    """A GSN argument structure, reduced to the node and link types modelled.

    ``nodes`` maps a node identifier to one of :data:`NODE_TYPES`. ``links`` is
    a tuple of ``(link_type, source, target)``.

    Validation enforces the GSN restrictions this product relies on, and no
    others: a ``SupportedBy`` link originates at a ``Goal`` or ``Strategy`` and
    terminates at a ``Goal``, ``Strategy`` or ``Solution``; an ``InContextOf``
    link originates at a ``Goal`` or ``Strategy`` and terminates at a
    ``Context``, ``Assumption`` or ``Justification``; and a ``Solution`` is a
    leaf, which is what makes it the carrier of evidence.
    """

    nodes: dict[str, str] = field(default_factory=dict)
    links: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        for identifier, kind in self.nodes.items():
            _require_identifier(identifier, "AssuranceCase node identifier")
            if kind not in NODE_TYPES:
                raise ValueError(
                    f"node {identifier!r} has type {kind!r}, which is not one of the "
                    f"GSN node types this product models, {NODE_TYPES}. Modules, "
                    f"away goals, contracts and undeveloped entities are out of scope."
                )
        for link in self.links:
            if len(link) != 3:
                raise ValueError(
                    f"each link must be a (link_type, source, target) triple, got {link!r}"
                )
            link_type, source, target = link
            if link_type not in LINK_TYPES:
                raise ValueError(
                    f"link type {link_type!r} is not one of {LINK_TYPES}; GSN link "
                    f"types outside this set are not modelled"
                )
            for end, label in ((source, "source"), (target, "target")):
                if end not in self.nodes:
                    raise ValueError(
                        f"link {link!r} names an unknown node as its {label}: "
                        f"{end!r} is not in nodes. Declare every node before linking it."
                    )
            if self.nodes[source] not in SUPPORTED_BY_SOURCES:
                raise ValueError(
                    f"link {link!r} originates at a {self.nodes[source]!r} node; a GSN "
                    f"{link_type} link originates at one of {SUPPORTED_BY_SOURCES}"
                )
            allowed = (
                SUPPORTED_BY_TARGETS if link_type == "SupportedBy" else IN_CONTEXT_OF_TARGETS
            )
            if self.nodes[target] not in allowed:
                raise ValueError(
                    f"link {link!r} terminates at a {self.nodes[target]!r} node; a GSN "
                    f"{link_type} link terminates at one of {allowed}"
                )
            if self.nodes[source] == "Solution":
                raise ValueError(
                    f"link {link!r} originates at a Solution node; in GSN a Solution is "
                    f"a leaf and carries evidence rather than supporting further nodes"
                )

    def solutions(self) -> tuple[str, ...]:
        """Identifiers of every ``Solution`` node, sorted."""
        return tuple(sorted(name for name, kind in self.nodes.items() if kind == "Solution"))

    def leaf_goals(self) -> tuple[str, ...]:
        """Goals whose ``SupportedBy`` children are all ``Solution`` nodes, sorted.

        A leaf goal is the finest-grained claim that rests directly on evidence,
        so the goal-weighted stale fraction is reported over these.
        """
        children: dict[str, list[str]] = {}
        for link_type, source, target in self.links:
            if link_type == "SupportedBy":
                children.setdefault(source, []).append(target)
        out = []
        for name, kind in self.nodes.items():
            if kind != "Goal":
                continue
            kids = children.get(name, [])
            if kids and all(self.nodes[kid] == "Solution" for kid in kids):
                out.append(name)
        return tuple(sorted(out))
