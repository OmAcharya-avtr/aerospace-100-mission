"""Check the Mermaid renderer: shape mapping, escaping, determinism, completeness.

What is checked:

* every one of the six Core GSN element types renders with the shape documented
  in :mod:`assuregraph.mermaid`, and the three substitutions the module declares
  (Assumption, Justification, Undeveloped) are the ones it declares;
* the output declares exactly one node line per node and one edge line per edge
  for each shipped example;
* double quotes become ``#quot;`` and newlines become ``<br/>``, so a statement
  cannot break the diagram;
* rendering is deterministic -- two renders of the same case are byte-identical,
  which is what lets a diagram be committed and diffed.

Not checked here: that Mermaid renders the result as intended. That requires a
Mermaid implementation, which is not available in this container. A seven-node
excerpt carrying one node of each element type was checked once against an
external Mermaid renderer during the build; that is recorded in VALIDATION.md as
an external check, not as something this script re-runs, and the full diagrams
were never passed to a renderer.

Run: ``python validation/validate_mermaid.py``
Output: ``validation/validate_mermaid_output.txt``
"""

from __future__ import annotations

import os

import _bootstrap

from assuregraph import (
    MERMAID_SHAPES,
    NodeKind,
    load_case,
    parse_case,
    render_mermaid,
    run_checks,
)

EXPECTED_SHAPES = {
    "goal": ('["', '"]'),
    "strategy": ('[/"', '"/]'),
    "solution": ('(("', '"))'),
    "context": ('(["', '"])'),
    "assumption": ('{{"', '"}}'),
    "justification": ('>"', '"]'),
}

ALL_KINDS_DOCUMENT = {
    "name": "one of each element type",
    "top_goals": ["G1"],
    "nodes": [
        {"id": "G1", "type": "goal", "statement": "the top claim"},
        {"id": "S1", "type": "strategy", "statement": "the argument step"},
        {
            "id": "Sn1",
            "type": "solution",
            "statement": "the artifact",
            "evidence": {"path": "evidence/nothing.txt"},
        },
        {"id": "C1", "type": "context", "statement": "the scope"},
        {"id": "A1", "type": "assumption", "statement": "the assumption"},
        {"id": "J1", "type": "justification", "statement": "the justification"},
    ],
    "edges": [
        {"from": "G1", "to": "S1", "type": "supported_by"},
        {"from": "S1", "to": "Sn1", "type": "supported_by"},
        {"from": "G1", "to": "C1", "type": "in_context_of"},
        {"from": "G1", "to": "A1", "type": "in_context_of"},
        {"from": "G1", "to": "J1", "type": "in_context_of"},
    ],
}


def main() -> int:
    lines = [
        "Mermaid rendering -- assuregraph 0.1.0",
        "",
        "GSN Standard v3 (SCSC-141C, May 2021) draws Assumption and Justification as",
        "annotated ellipses and Undeveloped as a diamond. Mermaid has none of those,",
        "so this package substitutes a hexagon, an asymmetric shape and a label marker,",
        "and says so. The substitutions are checked below, not hidden.",
        "",
        "Shape mapping, declared against rendered:",
    ]
    failures = 0

    case = parse_case(ALL_KINDS_DOCUMENT, base_dir="/nonexistent-base")
    diagram = render_mermaid(case)
    for kind in NodeKind:
        declared = MERMAID_SHAPES[kind]
        expected = EXPECTED_SHAPES[kind.value]
        node_id = {"goal": "G1", "strategy": "S1", "solution": "Sn1", "context": "C1",
                   "assumption": "A1", "justification": "J1"}[kind.value]
        fragment = f"n_{node_id}{declared[0]}"
        present = fragment in diagram
        matches = declared == expected
        lines.append(
            f"  {kind.value:<14} declared {declared[0]!r:>8} {declared[1]!r:<8} "
            f"rendered {'yes' if present else 'NO'}  "
            f"matches module table: {'yes' if matches else 'NO -- FAIL'}"
        )
        failures += not (present and matches)

    lines.extend(
        [
            "",
            "Relationship arrows:",
            f"  supported_by  -> {'-->' if 'n_G1 --> n_S1' in diagram else 'MISSING -- FAIL'}",
            f"  in_context_of -> {'--o' if 'n_G1 --o n_C1' in diagram else 'MISSING -- FAIL'}",
        ]
    )
    failures += "n_G1 --> n_S1" not in diagram
    failures += "n_G1 --o n_C1" not in diagram

    markers = [("A1 [A]:", "Assumption"), ("J1 [J]:", "Justification")]
    lines.append("")
    lines.append("Letter markers that stand in for the GSN ellipse annotation:")
    for fragment, label in markers:
        present = fragment in diagram
        verdict = "yes" if present else "NO -- FAIL"
        lines.append(f"  {label:<14} {fragment!r:<12} present: {verdict}")
        failures += not present

    undeveloped = render_mermaid(
        parse_case(
            {
                "name": "undeveloped",
                "nodes": [
                    {"id": "G1", "type": "goal", "statement": "not elaborated",
                     "undeveloped": True}
                ],
            }
        )
    )
    present = "(undeveloped)" in undeveloped
    lines.append(
        f"  Undeveloped    '(undeveloped)' in label present: {'yes' if present else 'NO -- FAIL'}"
    )
    failures += not present

    quoted = render_mermaid(
        parse_case({"name": "q", "nodes": [{"id": "G1", "type": "goal",
                                           "statement": 'the "primary" claim'}]})
    )
    escaped = "#quot;primary#quot;" in quoted and '"primary"' not in quoted
    broken = render_mermaid(
        parse_case({"name": "n", "nodes": [{"id": "G1", "type": "goal",
                                            "statement": "first\nsecond"}]})
    )
    wrapped = "first<br/>second" in broken
    lines.extend(
        [
            "",
            "Escaping:",
            f"  double quote becomes '#quot;'     : {'yes' if escaped else 'NO -- FAIL'}",
            f"  newline becomes '<br/>'          : {'yes' if wrapped else 'NO -- FAIL'}",
        ]
    )
    failures += not escaped
    failures += not wrapped

    lines.append("")
    lines.append("Completeness and determinism on the shipped cases:")
    lines.append(
        f"  {'case':<22} {'nodes':>6} {'node lines':>11} {'edges':>6} {'edge lines':>11} "
        f"{'identical reruns':>17}"
    )
    for filename in ("complete_case.yaml", "incomplete_case.yaml", "cyclic_case.yaml"):
        path = os.path.join(_bootstrap.EXAMPLES, filename)
        shipped = load_case(path)
        first = render_mermaid(shipped, report=run_checks(shipped), max_label_chars=48)
        second = render_mermaid(shipped, report=run_checks(shipped), max_label_chars=48)
        declarations = [ln.strip() for ln in first.splitlines() if ln.strip().startswith("n_")]
        node_lines = [ln for ln in declarations if " --" not in ln]
        edge_lines = [ln for ln in declarations if " --" in ln]
        same = first == second
        ok = len(node_lines) == len(shipped.nodes) and len(edge_lines) == len(shipped.edges)
        failures += not (ok and same)
        lines.append(
            f"  {filename:<22} {len(shipped.nodes):>6} {len(node_lines):>11} "
            f"{len(shipped.edges):>6} {len(edge_lines):>11} {'yes' if same else 'NO -- FAIL':>17}"
        )

    lines.extend(
        [
            "",
            "External syntax check (performed once during the build, not by this script):",
            "  A seven-node excerpt of the complete_case.yaml diagram, carrying one node",
            "  of each of the six element types plus both relationship arrows, was passed",
            "  to a Mermaid renderer on 2026-10-08. It reported valid=True,",
            "  diagramType=flowchart, rendered all 7 node labels, and produced SVG",
            "  containing rect, polygon and circle elements, which is consistent with the",
            "  four GSN shapes plus the two substitutions. The full 18-node diagram was",
            "  NOT passed to a renderer. See VALIDATION.md section 6.",
            "",
            f"FAILURES: {failures}",
        ]
    )
    target = __file__.replace(".py", "_output.txt")
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
