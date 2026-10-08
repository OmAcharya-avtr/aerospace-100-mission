"""Tests for accepted documents and for what parsing preserves."""

from __future__ import annotations

import os

import yaml
from conftest import COMPLETE_CASE, document, edge, node

from assuregraph import EdgeKind, NodeKind, load_case, parse_case


def test_minimal_case_is_one_goal() -> None:
    case = parse_case(document([node("G1", "goal", "the only claim")]))
    assert case.name == "test case"
    assert list(case.nodes) == ["G1"]
    assert case.edges == ()


def test_edges_key_may_be_absent() -> None:
    case = parse_case({"name": "x", "nodes": [node("G1", "goal")]})
    assert case.edges == ()


def test_document_order_of_nodes_is_preserved() -> None:
    ids = ["G1", "S1", "G2", "Sn1"]
    doc = document(
        [
            node("G1", "goal"),
            node("S1", "strategy"),
            node("G2", "goal"),
            node("Sn1", "solution", evidence={"path": "a.txt"}),
        ],
        [edge("G1", "S1"), edge("S1", "G2"), edge("G2", "Sn1")],
    )
    assert list(parse_case(doc).nodes) == ids


def test_document_order_of_edges_is_preserved() -> None:
    doc = document(
        [node("G1", "goal"), node("G2", "goal"), node("C1", "context")],
        [edge("G1", "C1", "in_context_of"), edge("G1", "G2")],
    )
    case = parse_case(doc)
    assert [e.kind for e in case.edges] == [EdgeKind.IN_CONTEXT_OF, EdgeKind.SUPPORTED_BY]


def test_evidence_fields_round_trip() -> None:
    digest = "0" * 64
    doc = document(
        [
            node(
                "Sn1",
                "solution",
                evidence={"path": "sub/a.txt", "sha256": digest, "recorded": "2026-01-01"},
            )
        ]
    )
    ev = parse_case(doc).nodes["Sn1"].evidence
    assert ev is not None
    assert (ev.path, ev.sha256, ev.recorded) == ("sub/a.txt", digest, "2026-01-01")


def test_all_six_element_types_parse() -> None:
    doc = document(
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
    )
    case = parse_case(doc)
    assert {n.kind for n in case.nodes.values()} == set(NodeKind)


def test_base_dir_is_the_case_files_directory() -> None:
    case = load_case(COMPLETE_CASE)
    assert case.base_dir == os.path.dirname(os.path.abspath(COMPLETE_CASE))


def test_shipped_complete_case_parses() -> None:
    case = load_case(COMPLETE_CASE)
    assert len(case.nodes) == 18
    assert len(case.edges) == 17
    assert case.top_goals == ("G1",)


def test_load_case_accepts_a_path_like(tmp_path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump(document([node("G1", "goal")]), sort_keys=False), encoding="utf-8"
    )
    assert parse_case(yaml.safe_load(path.read_text())).name == load_case(path).name


def test_statement_may_be_multiline() -> None:
    doc = document([node("G1", "goal", "line one\nline two")])
    assert parse_case(doc).nodes["G1"].statement == "line one\nline two"
