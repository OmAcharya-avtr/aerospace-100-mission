"""Render an assurance case as a Mermaid flowchart.

Mermaid is used because GitHub renders it inline from a fenced ``mermaid``
block, so a case stays reviewable in a pull request with no image hosting and no
binary asset that can drift from the YAML.

**Where the rendering is not GSN.** GSN Standard v3 draws a Goal as a
rectangle, a Strategy as a parallelogram, a Solution as a circle and a Context
as a rounded rectangle, and Mermaid has all four. It draws an Assumption and a
Justification as ellipses annotated 'A' and 'J', and the *Undeveloped* decorator
as a small diamond below the element; Mermaid has none of those three. The
substitutions this module makes are therefore:

================  ===========================  =============================
GSN element       GSN Standard v3 shape        Mermaid shape used here
================  ===========================  =============================
Goal              rectangle                    rectangle ``["..."]``
Strategy          parallelogram                parallelogram ``[/"..."/]``
Solution          circle                       circle ``(("..."))``
Context           rounded rectangle            stadium ``(["..."])``
Assumption        ellipse marked 'A'           hexagon ``{{"A: ..."}}``
Justification     ellipse marked 'J'           asymmetric ``>"J: ..."]``
Undeveloped       diamond below the element    ``(undeveloped)`` in the label
SupportedBy       solid line, filled arrow     ``-->``
InContextOf       solid line, hollow arrow      ``--o``
================  ===========================  =============================

A reader who knows GSN will recognise the first four without being told and must
be told about the last five. That is why this table is in the module docstring
and in the README rather than only in a commit message.

Units: ``max_label_chars`` is a character count.
"""

from __future__ import annotations

from .checks import CheckReport, Severity
from .model import AssuranceCase, EdgeKind, Node, NodeKind

__all__ = ["MERMAID_SHAPES", "render_mermaid"]

#: Mermaid shape delimiters per GSN element type, as ``(open, close)``.
MERMAID_SHAPES: dict[NodeKind, tuple[str, str]] = {
    NodeKind.GOAL: ('["', '"]'),
    NodeKind.STRATEGY: ('[/"', '"/]'),
    NodeKind.SOLUTION: ('(("', '"))'),
    NodeKind.CONTEXT: ('(["', '"])'),
    NodeKind.ASSUMPTION: ('{{"', '"}}'),
    NodeKind.JUSTIFICATION: ('>"', '"]'),
}

_EDGE_ARROWS: dict[EdgeKind, str] = {
    EdgeKind.SUPPORTED_BY: "-->",
    EdgeKind.IN_CONTEXT_OF: "--o",
}

_KIND_MARKER: dict[NodeKind, str] = {
    NodeKind.ASSUMPTION: "A",
    NodeKind.JUSTIFICATION: "J",
}

_CLASS_DEFS = (
    "    classDef gsnGoal fill:#eef4ff,stroke:#2b4c7e,stroke-width:1px;",
    "    classDef gsnStrategy fill:#f3eeff,stroke:#5a3e8e,stroke-width:1px;",
    "    classDef gsnSolution fill:#eefaf0,stroke:#2f6b45,stroke-width:1px;",
    "    classDef gsnContext fill:#f7f7f7,stroke:#555555,stroke-width:1px;",
    "    classDef gsnAssumption fill:#fffbe6,stroke:#8a6d1f,stroke-width:1px;",
    "    classDef gsnJustification fill:#fffbe6,stroke:#8a6d1f,stroke-width:1px;",
    "    classDef gsnFinding fill:#ffe9e9,stroke:#b3261e,stroke-width:3px;",
)

_KIND_CLASS: dict[NodeKind, str] = {
    NodeKind.GOAL: "gsnGoal",
    NodeKind.STRATEGY: "gsnStrategy",
    NodeKind.SOLUTION: "gsnSolution",
    NodeKind.CONTEXT: "gsnContext",
    NodeKind.ASSUMPTION: "gsnAssumption",
    NodeKind.JUSTIFICATION: "gsnJustification",
}


def _escape(text: str) -> str:
    """Make ``text`` safe inside a quoted Mermaid label.

    Mermaid's quoted-label form cannot contain a raw double quote; the HTML
    entity ``#quot;`` is the documented substitute. Newlines become ``<br/>``
    so a multi-line GSN statement keeps its line breaks.
    """
    out = text.replace('"', "#quot;")
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    return out.replace("\n", "<br/>")


def _label(node: Node, max_label_chars: int) -> str:
    """Build the quoted label text for ``node``.

    Runs of spaces and tabs inside a line are collapsed so a YAML folded block
    does not produce ragged labels, but an explicit newline is kept and becomes
    ``<br/>`` in :func:`_escape`. An earlier draft collapsed newlines along with
    the rest of the whitespace, which silently disabled the ``<br/>`` handling
    this module documents; ``tests/test_mermaid.py`` now pins the behaviour.
    """
    lines = [" ".join(line.split()) for line in node.statement.splitlines()]
    statement = "\n".join(line for line in lines if line) or node.statement.strip()
    if max_label_chars > 0 and len(statement) > max_label_chars:
        statement = statement[: max_label_chars - 1].rstrip() + "…"
    marker = _KIND_MARKER.get(node.kind)
    prefix = f"{node.node_id} [{marker}]" if marker else node.node_id
    suffix = " (undeveloped)" if node.undeveloped else ""
    return _escape(f"{prefix}: {statement}{suffix}")


def _mermaid_id(node_id: str) -> str:
    """Map a case node id to a Mermaid node identifier.

    Mermaid identifiers may not contain whitespace or its own punctuation, and
    GSN project identifiers sometimes do. Every character outside
    ``[A-Za-z0-9_]`` is replaced with ``_`` and the original id stays visible in
    the label, so the diagram is still readable against the YAML.
    """
    safe = "".join(c if (c.isalnum() and c.isascii()) or c == "_" else "_" for c in node_id)
    return f"n_{safe}"


def render_mermaid(
    case: AssuranceCase,
    *,
    report: CheckReport | None = None,
    direction: str = "TD",
    max_label_chars: int = 70,
    fence: bool = False,
) -> str:
    """Render ``case`` as a Mermaid ``flowchart``.

    Args:
        case: The parsed case.
        report: When given, every node named by an ``ERROR`` finding is styled
            with the ``gsnFinding`` class, so the diagram shows where the case
            is incomplete. ``WARNING`` and ``INFO`` findings are not styled,
            because highlighting a declared-undeveloped goal in red would
            misrepresent it.
        direction: Mermaid flowchart direction, one of ``TD``, ``TB``, ``BT``,
            ``LR``, ``RL``.
        max_label_chars: Truncate a statement longer than this, with an ellipsis.
            0 or negative disables truncation. Truncation affects the diagram
            only; no check reads the label.
        fence: Wrap the output in a ```` ```mermaid ```` fence, ready to paste
            into a README.

    Returns:
        The diagram source, newline-terminated.

    Raises:
        ValueError: ``direction`` is not a Mermaid flowchart direction.
    """
    valid_directions = ("TD", "TB", "BT", "LR", "RL")
    if direction not in valid_directions:
        raise ValueError(
            f"direction must be one of {list(valid_directions)}, got {direction!r}"
        )

    flagged: set[str] = set()
    if report is not None:
        for finding in report.findings:
            if finding.severity is Severity.ERROR:
                flagged.update(finding.node_ids)

    lines: list[str] = [f"flowchart {direction}"]
    for node_id, node in case.nodes.items():
        open_delim, close_delim = MERMAID_SHAPES[node.kind]
        lines.append(
            f"    {_mermaid_id(node_id)}{open_delim}{_label(node, max_label_chars)}{close_delim}"
        )
    for edge in case.edges:
        arrow = _EDGE_ARROWS[edge.kind]
        lines.append(f"    {_mermaid_id(edge.source)} {arrow} {_mermaid_id(edge.target)}")

    lines.extend(_CLASS_DEFS)
    for kind in NodeKind:
        members = [
            _mermaid_id(n.node_id)
            for n in case.nodes_of_kind(kind)
            if n.node_id not in flagged
        ]
        if members:
            lines.append(f"    class {','.join(members)} {_KIND_CLASS[kind]};")
    if flagged:
        ordered = [_mermaid_id(i) for i in case.nodes if i in flagged]
        lines.append(f"    class {','.join(ordered)} gsnFinding;")

    body = "\n".join(lines) + "\n"
    if fence:
        return f"```mermaid\n{body}```\n"
    return body
