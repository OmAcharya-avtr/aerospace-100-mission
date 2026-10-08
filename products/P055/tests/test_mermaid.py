"""Tests for the Mermaid renderer, including the documented shape substitutions."""

from __future__ import annotations

import pytest
from conftest import COMPLETE_CASE, CYCLIC_CASE, document, edge, node

from assuregraph import NodeKind, load_case, parse_case, render_mermaid, run_checks


def _all_kinds_case():
    return parse_case(
        document(
            [
                node("G1", "goal"),
                node("S1", "strategy"),
                node("Sn1", "solution", evidence={"path": "a.txt"}),
                node("C1", "context"),
                node("A1", "assumption"),
                node("J1", "justification"),
            ],
            [
                edge("G1", "S1"),
                edge("S1", "Sn1"),
                edge("G1", "C1", "in_context_of"),
                edge("G1", "A1", "in_context_of"),
                edge("G1", "J1", "in_context_of"),
            ],
            top_goals=["G1"],
        )
    )


def test_header_names_the_direction() -> None:
    assert render_mermaid(_all_kinds_case(), direction="LR").startswith("flowchart LR")


def test_rejects_an_invalid_direction() -> None:
    with pytest.raises(ValueError, match="direction must be one of"):
        render_mermaid(_all_kinds_case(), direction="sideways")


@pytest.mark.parametrize(
    ("node_id", "fragment"),
    [
        ("G1", 'n_G1["'),
        ("S1", 'n_S1[/"'),
        ("Sn1", 'n_Sn1(("'),
        ("C1", 'n_C1(["'),
        ("A1", 'n_A1{{"'),
        ("J1", 'n_J1>"'),
    ],
)
def test_each_element_type_gets_its_documented_shape(node_id: str, fragment: str) -> None:
    assert fragment in render_mermaid(_all_kinds_case())


def test_supported_by_is_a_filled_arrow_and_in_context_of_is_not() -> None:
    text = render_mermaid(_all_kinds_case())
    assert "n_G1 --> n_S1" in text
    assert "n_G1 --o n_C1" in text


def test_assumption_and_justification_carry_their_letter_markers() -> None:
    text = render_mermaid(_all_kinds_case())
    assert "A1 [A]:" in text
    assert "J1 [J]:" in text


def test_undeveloped_is_marked_in_the_label() -> None:
    case = parse_case(document([node("G1", "goal", undeveloped=True)]))
    assert "(undeveloped)" in render_mermaid(case)


def test_double_quotes_are_escaped() -> None:
    case = parse_case(document([node("G1", "goal", 'the "primary" claim')]))
    text = render_mermaid(case)
    assert "#quot;primary#quot;" in text
    assert '"primary"' not in text


def test_newlines_become_line_breaks() -> None:
    case = parse_case(document([node("G1", "goal", "one\ntwo")]))
    assert "one<br/>two" in render_mermaid(case)


def test_labels_are_truncated_with_an_ellipsis() -> None:
    case = parse_case(document([node("G1", "goal", "x" * 200)]))
    text = render_mermaid(case, max_label_chars=30)
    assert "…" in text
    assert "x" * 200 not in text


def test_zero_disables_truncation() -> None:
    case = parse_case(document([node("G1", "goal", "x" * 200)]))
    assert "x" * 200 in render_mermaid(case, max_label_chars=0)


def test_node_ids_with_punctuation_are_sanitised() -> None:
    case = parse_case(document([node("REQ-1.2 a", "goal")]))
    text = render_mermaid(case)
    assert "n_REQ_1_2_a[" in text
    # The original id stays visible in the label so the diagram matches the YAML.
    assert "REQ-1.2 a:" in text


def test_highlight_classes_the_error_nodes_only() -> None:
    case = load_case(CYCLIC_CASE)
    report = run_checks(case)
    text = render_mermaid(case, report=report)
    assert "gsnFinding" in text
    finding_line = [ln for ln in text.splitlines() if ln.strip().endswith("gsnFinding;")][0]
    # The cyclic case's error names G1, S1, G2, S2 and not Sn1.
    assert "n_Sn1" not in finding_line


def test_without_a_report_nothing_is_flagged() -> None:
    text = render_mermaid(load_case(CYCLIC_CASE))
    assert "gsnFinding;" not in text.split("classDef gsnFinding")[-1]


def test_fence_wraps_the_output() -> None:
    text = render_mermaid(_all_kinds_case(), fence=True)
    assert text.startswith("```mermaid\n")
    assert text.rstrip().endswith("```")


def test_every_node_and_edge_appears_once() -> None:
    case = load_case(COMPLETE_CASE)
    text = render_mermaid(case)
    declarations = [ln for ln in text.splitlines() if ln.strip().startswith("n_")]
    node_lines = [ln for ln in declarations if " --" not in ln]
    edge_lines = [ln for ln in declarations if " --" in ln]
    assert len(node_lines) == len(case.nodes)
    assert len(edge_lines) == len(case.edges)


def test_render_is_deterministic() -> None:
    case = load_case(COMPLETE_CASE)
    assert render_mermaid(case) == render_mermaid(case)


def test_all_six_kinds_have_a_shape_entry() -> None:
    from assuregraph import MERMAID_SHAPES

    assert set(MERMAID_SHAPES) == set(NodeKind)
