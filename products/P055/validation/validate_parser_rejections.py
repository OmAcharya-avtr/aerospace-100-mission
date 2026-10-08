"""Exercise every rejection path in the parser and print the message each gives.

Purpose: show that a malformed case is rejected with a message that names the
location and says what to do, rather than with a traceback or a silent pass.
Each row below is one malformed document; the script asserts that the expected
exception type is raised and prints the message verbatim.

Run: ``python validation/validate_parser_rejections.py``
Output: ``validation/validate_parser_rejections_output.txt``
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any

import _bootstrap

from assuregraph import CaseFormatError, CaseIOError, load_case, parse_case


def _node(node_id: str, kind: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"id": node_id, "type": kind, "statement": f"statement {node_id}"}
    out.update(extra)
    return out


def _doc(nodes: list[dict[str, Any]], edges: list[Any] | None = None, **extra: Any) -> dict:
    out: dict[str, Any] = {"name": "rejection probe", "nodes": nodes}
    if edges is not None:
        out["edges"] = edges
    out.update(extra)
    return out


CASES: list[tuple[str, Any]] = [
    ("document is a list, not a mapping", [1, 2]),
    ("document is null", None),
    ("unknown top-level key", {**_doc([_node("G1", "goal")]), "nodez": []}),
    ("near-miss top-level key", {**_doc([_node("G1", "goal")]), "edgs": []}),
    ("missing name", {"nodes": [_node("G1", "goal")]}),
    ("name is a number", {"name": 7, "nodes": [_node("G1", "goal")]}),
    ("name is blank", {"name": "  ", "nodes": [_node("G1", "goal")]}),
    ("missing nodes", {"name": "x"}),
    ("nodes is a mapping", {"name": "x", "nodes": {"G1": "goal"}}),
    ("nodes is empty", {"name": "x", "nodes": []}),
    ("node is a string", {"name": "x", "nodes": ["G1"]}),
    ("node has no id", {"name": "x", "nodes": [{"type": "goal", "statement": "s"}]}),
    (
        "node id is an integer",
        {"name": "x", "nodes": [{"id": 1, "type": "goal", "statement": "s"}]},
    ),
    ("duplicate node id", _doc([_node("G1", "goal"), _node("G1", "goal")])),
    ("node has no type", {"name": "x", "nodes": [{"id": "G1", "statement": "s"}]}),
    ("unknown node type", _doc([_node("G1", "claim")])),
    ("near-miss node type", _doc([_node("G1", "goals")])),
    ("node has no statement", {"name": "x", "nodes": [{"id": "G1", "type": "goal"}]}),
    ("unknown node key", _doc([_node("G1", "goal", colour="red")])),
    ("undeveloped is a string", _doc([_node("G1", "goal", undeveloped="yes")])),
    (
        "undeveloped on a solution",
        _doc([_node("Sn1", "solution", evidence={"path": "a.txt"}, undeveloped=True)]),
    ),
    ("discharged on a goal", _doc([_node("G1", "goal", discharged=True)])),
    (
        "discharged_by without discharged",
        _doc([_node("G1", "goal"), _node("A1", "assumption", discharged_by="G1")]),
    ),
    (
        "dangling discharged_by",
        _doc([_node("A1", "assumption", discharged=True, discharged_by="Sn9")]),
    ),
    (
        "discharged_by is an integer",
        _doc([_node("A1", "assumption", discharged=True, discharged_by=3)]),
    ),
    ("solution without evidence", _doc([_node("Sn1", "solution")])),
    ("evidence on a goal", _doc([_node("G1", "goal", evidence={"path": "a.txt"})])),
    ("evidence is a string", _doc([_node("Sn1", "solution", evidence="a.txt")])),
    ("evidence without path", _doc([_node("Sn1", "solution", evidence={"sha256": "a" * 64})])),
    (
        "unknown evidence key",
        _doc([_node("Sn1", "solution", evidence={"path": "a.txt", "md5": "x"})]),
    ),
    ("digest too short", _doc([_node("Sn1", "solution", evidence={"path": "a", "sha256": "abc"})])),
    (
        "digest not hexadecimal",
        _doc([_node("Sn1", "solution", evidence={"path": "a", "sha256": "z" * 64})]),
    ),
    (
        "digest is an integer",
        _doc([_node("Sn1", "solution", evidence={"path": "a", "sha256": 1234})]),
    ),
    (
        "recorded is an integer",
        _doc([_node("Sn1", "solution", evidence={"path": "a", "recorded": 2026})]),
    ),
    ("edges is a mapping", {**_doc([_node("G1", "goal")]), "edges": {"from": "G1"}}),
    ("edge is a string", _doc([_node("G1", "goal")], ["G1->G2"])),
    (
        "edge without from",
        _doc([_node("G1", "goal")], [{"to": "G1", "type": "supported_by"}]),
    ),
    (
        "unknown relationship type",
        _doc(
            [_node("G1", "goal"), _node("G2", "goal")],
            [{"from": "G1", "to": "G2", "type": "because"}],
        ),
    ),
    (
        "edge to an unknown node",
        _doc([_node("G1", "goal")], [{"from": "G1", "to": "G2", "type": "supported_by"}]),
    ),
    (
        "edge from an unknown node",
        _doc([_node("G1", "goal")], [{"from": "G0", "to": "G1", "type": "supported_by"}]),
    ),
    (
        "self loop",
        _doc([_node("G1", "goal")], [{"from": "G1", "to": "G1", "type": "supported_by"}]),
    ),
    (
        "duplicate edge",
        _doc(
            [_node("G1", "goal"), _node("G2", "goal")],
            [
                {"from": "G1", "to": "G2", "type": "supported_by"},
                {"from": "G1", "to": "G2", "type": "supported_by"},
            ],
        ),
    ),
    (
        "solution as an edge source",
        _doc(
            [_node("Sn1", "solution", evidence={"path": "a"}), _node("G1", "goal")],
            [{"from": "Sn1", "to": "G1", "type": "supported_by"}],
        ),
    ),
    (
        "context as an edge source",
        _doc(
            [_node("C1", "context"), _node("G1", "goal")],
            [{"from": "C1", "to": "G1", "type": "supported_by"}],
        ),
    ),
    (
        "supported_by terminating on a context",
        _doc(
            [_node("G1", "goal"), _node("C1", "context")],
            [{"from": "G1", "to": "C1", "type": "supported_by"}],
        ),
    ),
    (
        "in_context_of terminating on a goal",
        _doc(
            [_node("G1", "goal"), _node("G2", "goal")],
            [{"from": "G1", "to": "G2", "type": "in_context_of"}],
        ),
    ),
    ("top_goals is a string", {**_doc([_node("G1", "goal")]), "top_goals": "G1"}),
    ("top goal is an integer", {**_doc([_node("G1", "goal")]), "top_goals": [1]}),
    ("unknown top goal", {**_doc([_node("G1", "goal")]), "top_goals": ["G9"]}),
    ("top goal is a strategy", {**_doc([_node("S1", "strategy")]), "top_goals": ["S1"]}),
    ("duplicate top goal", {**_doc([_node("G1", "goal")]), "top_goals": ["G1", "G1"]}),
]


def main() -> int:
    start = time.perf_counter()
    lines = [
        "Parser rejection paths -- assuregraph 0.1.0",
        "",
        "Each row is one malformed document. The parser must raise CaseFormatError",
        "(or CaseIOError for an I/O problem) with a message that names the location.",
        "A row that raised nothing, or raised the wrong type, is printed as FAIL.",
        "",
    ]
    rejected = 0
    failed = 0
    located = 0
    for index, (label, document) in enumerate(CASES, start=1):
        try:
            parse_case(document)
        except CaseFormatError as exc:
            rejected += 1
            if exc.location is not None:
                located += 1
            lines.append(f"{index:>3}. {label}")
            lines.append(f"     CaseFormatError: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            lines.append(f"{index:>3}. {label}")
            lines.append(f"     FAIL wrong exception type {type(exc).__name__}: {exc}")
        else:
            failed += 1
            lines.append(f"{index:>3}. {label}")
            lines.append("     FAIL accepted a malformed document")

    io_cases: list[tuple[str, str]] = [
        ("missing file", "/nonexistent-directory-for-validation/case.yaml"),
        ("a directory", _bootstrap.EXAMPLES),
    ]
    lines.append("")
    lines.append("I/O rejection paths (CaseIOError expected):")
    io_rejected = 0
    for label, path in io_cases:
        try:
            load_case(path)
        except CaseIOError as exc:
            io_rejected += 1
            relative = str(exc).replace(_bootstrap.REPO_ROOT, "<repo>")
            lines.append(f"  - {label}: CaseIOError: {relative}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            lines.append(f"  - {label}: FAIL {type(exc).__name__}: {exc}")
        else:
            failed += 1
            lines.append(f"  - {label}: FAIL accepted")

    with tempfile.TemporaryDirectory() as scratch:
        bad_yaml = os.path.join(scratch, "bad.yaml")
        with open(bad_yaml, "w", encoding="utf-8") as handle:
            handle.write("name: [unclosed\n")
        try:
            load_case(bad_yaml)
        except CaseIOError as exc:
            io_rejected += 1
            redacted = str(exc).replace(scratch, "<scratch>")
            lines.append(f"  - unparsable YAML: CaseIOError: {redacted}")
        else:
            failed += 1
            lines.append("  - unparsable YAML: FAIL accepted")

    elapsed = time.perf_counter() - start
    lines.extend(
        [
            "",
            "Summary",
            f"  schema rejection paths exercised : {len(CASES)}",
            f"  rejected with CaseFormatError   : {rejected}",
            f"  of those, message carried a YAML location : {located}",
            f"  I/O rejection paths exercised   : {len(io_cases) + 1}",
            f"  rejected with CaseIOError       : {io_rejected}",
            f"  FAILURES                        : {failed}",
            f"  elapsed                         : {elapsed:.3f} s",
        ]
    )
    text = "\n".join(lines) + "\n"
    target = __file__.replace(".py", "_output.txt")
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(text)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
