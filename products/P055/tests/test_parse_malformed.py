"""One test per rejection path in :mod:`assuregraph.parse`.

Each test asserts the exception type and that the message names the offending
location, because an error that does not say where it is costs the user the
same time as no error at all.
"""

from __future__ import annotations

import pytest
from conftest import document, edge, node

from assuregraph import CaseFormatError, CaseIOError, load_case, parse_case


def test_rejects_non_mapping_document() -> None:
    with pytest.raises(CaseFormatError, match="expected a mapping, found list"):
        parse_case([1, 2, 3])


def test_rejects_none_document() -> None:
    with pytest.raises(CaseFormatError, match="expected a mapping, found NoneType"):
        parse_case(None)


def test_rejects_unknown_top_level_key() -> None:
    doc = document([node("G1", "goal")])
    doc["nodez"] = []
    with pytest.raises(CaseFormatError, match="unknown key 'nodez'"):
        parse_case(doc)


def test_unknown_top_level_key_suggests_the_right_one() -> None:
    doc = document([node("G1", "goal")])
    doc["edgs"] = []
    with pytest.raises(CaseFormatError, match="did you mean 'edges'"):
        parse_case(doc)


def test_rejects_missing_name() -> None:
    with pytest.raises(CaseFormatError, match="missing required key 'name'"):
        parse_case({"nodes": [node("G1", "goal")]})


def test_rejects_non_string_name() -> None:
    with pytest.raises(CaseFormatError, match="key 'name' must be a non-empty string"):
        parse_case({"name": 7, "nodes": [node("G1", "goal")]})


def test_rejects_blank_name() -> None:
    with pytest.raises(CaseFormatError, match="key 'name' must be a non-empty string"):
        parse_case({"name": "   ", "nodes": [node("G1", "goal")]})


def test_rejects_missing_nodes() -> None:
    with pytest.raises(CaseFormatError, match="missing required key 'nodes'"):
        parse_case({"name": "x"})


def test_rejects_non_list_nodes() -> None:
    with pytest.raises(CaseFormatError, match="key 'nodes' must be a list, found dict"):
        parse_case({"name": "x", "nodes": {"G1": "goal"}})


def test_rejects_empty_nodes() -> None:
    with pytest.raises(CaseFormatError, match="at least one Goal"):
        parse_case({"name": "x", "nodes": []})


def test_rejects_non_mapping_node() -> None:
    with pytest.raises(CaseFormatError, match=r"nodes\[0\]: expected a mapping, found str"):
        parse_case({"name": "x", "nodes": ["G1"]})


def test_rejects_node_without_id() -> None:
    with pytest.raises(CaseFormatError, match=r"nodes\[0\]: missing required key 'id'"):
        parse_case({"name": "x", "nodes": [{"type": "goal", "statement": "s"}]})


def test_rejects_non_string_node_id() -> None:
    with pytest.raises(CaseFormatError, match="key 'id' must be a non-empty string"):
        parse_case({"name": "x", "nodes": [{"id": 1, "type": "goal", "statement": "s"}]})


def test_rejects_duplicate_node_id() -> None:
    doc = document([node("G1", "goal"), node("G1", "goal")])
    with pytest.raises(CaseFormatError, match="duplicate node id 'G1'"):
        parse_case(doc)


def test_rejects_node_without_type() -> None:
    with pytest.raises(CaseFormatError, match="missing required key 'type'"):
        parse_case({"name": "x", "nodes": [{"id": "G1", "statement": "s"}]})


def test_rejects_unknown_node_type() -> None:
    doc = document([node("G1", "claim")])
    with pytest.raises(CaseFormatError, match="unknown node type 'claim'"):
        parse_case(doc)


def test_unknown_node_type_lists_the_six_valid_ones() -> None:
    doc = document([node("G1", "claim")])
    with pytest.raises(CaseFormatError, match="six Core GSN element types"):
        parse_case(doc)


def test_unknown_node_type_suggests_a_near_miss() -> None:
    doc = document([node("G1", "goals")])
    with pytest.raises(CaseFormatError, match="did you mean 'goal'"):
        parse_case(doc)


def test_rejects_node_without_statement() -> None:
    with pytest.raises(CaseFormatError, match="missing required key 'statement'"):
        parse_case({"name": "x", "nodes": [{"id": "G1", "type": "goal"}]})


def test_rejects_unknown_node_key() -> None:
    doc = document([node("G1", "goal", colour="red")])
    with pytest.raises(CaseFormatError, match="unknown key 'colour'"):
        parse_case(doc)


def test_rejects_non_boolean_undeveloped() -> None:
    doc = document([node("G1", "goal", undeveloped="yes")])
    with pytest.raises(CaseFormatError, match="must be true or false"):
        parse_case(doc)


def test_rejects_undeveloped_on_a_solution() -> None:
    doc = document(
        [node("Sn1", "solution", evidence={"path": "a.txt"}, undeveloped=True)]
    )
    with pytest.raises(CaseFormatError, match="applies only to a Goal or a Strategy"):
        parse_case(doc)


def test_rejects_discharged_on_a_goal() -> None:
    doc = document([node("G1", "goal", discharged=True)])
    with pytest.raises(CaseFormatError, match="apply only to an Assumption"):
        parse_case(doc)


def test_rejects_discharged_by_without_discharged() -> None:
    doc = document([node("G1", "goal"), node("A1", "assumption", discharged_by="G1")])
    with pytest.raises(CaseFormatError, match="apply only to an Assumption|'discharged' is not"):
        parse_case(doc)


def test_rejects_assumption_discharged_by_without_flag() -> None:
    doc = document([node("G1", "goal"), node("A1", "assumption", discharged_by="G1")])
    with pytest.raises(CaseFormatError, match="set discharged: true or remove discharged_by"):
        parse_case(doc)


def test_rejects_dangling_discharged_by() -> None:
    doc = document([node("A1", "assumption", discharged=True, discharged_by="Sn9")])
    with pytest.raises(CaseFormatError, match="references unknown node id 'Sn9'"):
        parse_case(doc)


def test_rejects_non_string_discharged_by() -> None:
    doc = document([node("A1", "assumption", discharged=True, discharged_by=3)])
    with pytest.raises(CaseFormatError, match="'discharged_by' must be a node id string"):
        parse_case(doc)


def test_rejects_solution_without_evidence() -> None:
    doc = document([node("Sn1", "solution")])
    with pytest.raises(CaseFormatError, match="a Solution must cite an evidence artifact"):
        parse_case(doc)


def test_rejects_evidence_on_a_goal() -> None:
    doc = document([node("G1", "goal", evidence={"path": "a.txt"})])
    with pytest.raises(CaseFormatError, match="only a Solution may carry 'evidence'"):
        parse_case(doc)


def test_rejects_non_mapping_evidence() -> None:
    doc = document([node("Sn1", "solution", evidence="a.txt")])
    with pytest.raises(CaseFormatError, match="expected a mapping, found str"):
        parse_case(doc)


def test_rejects_evidence_without_path() -> None:
    doc = document([node("Sn1", "solution", evidence={"sha256": "a" * 64})])
    with pytest.raises(CaseFormatError, match="missing required key 'path'"):
        parse_case(doc)


def test_rejects_unknown_evidence_key() -> None:
    doc = document([node("Sn1", "solution", evidence={"path": "a.txt", "md5": "x"})])
    with pytest.raises(CaseFormatError, match="unknown key 'md5'"):
        parse_case(doc)


def test_rejects_short_digest() -> None:
    doc = document([node("Sn1", "solution", evidence={"path": "a.txt", "sha256": "abc"})])
    with pytest.raises(CaseFormatError, match="exactly 64 lowercase hexadecimal characters"):
        parse_case(doc)


def test_rejects_non_hex_digest() -> None:
    doc = document([node("Sn1", "solution", evidence={"path": "a.txt", "sha256": "z" * 64})])
    with pytest.raises(CaseFormatError, match="exactly 64 lowercase hexadecimal characters"):
        parse_case(doc)


def test_rejects_non_string_digest() -> None:
    doc = document([node("Sn1", "solution", evidence={"path": "a.txt", "sha256": 1234})])
    with pytest.raises(CaseFormatError, match="key 'sha256' must be a string"):
        parse_case(doc)


def test_accepts_uppercase_digest_and_normalises_it() -> None:
    digest = "AB" * 32
    doc = document([node("Sn1", "solution", evidence={"path": "a.txt", "sha256": digest})])
    case = parse_case(doc)
    assert case.nodes["Sn1"].evidence is not None
    assert case.nodes["Sn1"].evidence.sha256 == digest.lower()


def test_rejects_non_string_recorded() -> None:
    doc = document([node("Sn1", "solution", evidence={"path": "a.txt", "recorded": 2026})])
    with pytest.raises(CaseFormatError, match="key 'recorded' must be a string"):
        parse_case(doc)


def test_rejects_non_list_edges() -> None:
    doc = document([node("G1", "goal")])
    doc["edges"] = {"from": "G1"}
    with pytest.raises(CaseFormatError, match="key 'edges' must be a list, found dict"):
        parse_case(doc)


def test_rejects_non_mapping_edge() -> None:
    doc = document([node("G1", "goal")], ["G1->G2"])
    with pytest.raises(CaseFormatError, match=r"edges\[0\]: expected a mapping"):
        parse_case(doc)


def test_rejects_edge_without_from() -> None:
    doc = document([node("G1", "goal")], [{"to": "G1", "type": "supported_by"}])
    with pytest.raises(CaseFormatError, match="missing required key 'from'"):
        parse_case(doc)


def test_rejects_unknown_edge_type() -> None:
    doc = document([node("G1", "goal"), node("G2", "goal")], [edge("G1", "G2", "because")])
    with pytest.raises(CaseFormatError, match="unknown relationship type 'because'"):
        parse_case(doc)


def test_rejects_edge_to_unknown_node() -> None:
    doc = document([node("G1", "goal")], [edge("G1", "G2")])
    with pytest.raises(CaseFormatError, match="to references unknown node id 'G2'"):
        parse_case(doc)


def test_rejects_edge_from_unknown_node() -> None:
    doc = document([node("G1", "goal")], [edge("G0", "G1")])
    with pytest.raises(CaseFormatError, match="from references unknown node id 'G0'"):
        parse_case(doc)


def test_rejects_self_loop() -> None:
    doc = document([node("G1", "goal")], [edge("G1", "G1")])
    with pytest.raises(CaseFormatError, match="may not be related to itself"):
        parse_case(doc)


def test_rejects_duplicate_edge() -> None:
    doc = document([node("G1", "goal"), node("G2", "goal")], [edge("G1", "G2"), edge("G1", "G2")])
    with pytest.raises(CaseFormatError, match="duplicate supported_by relationship"):
        parse_case(doc)


def test_rejects_solution_as_edge_source() -> None:
    doc = document(
        [node("Sn1", "solution", evidence={"path": "a.txt"}), node("G1", "goal")],
        [edge("Sn1", "G1")],
    )
    with pytest.raises(CaseFormatError, match="a solution may not be the source"):
        parse_case(doc)


def test_rejects_context_as_edge_source() -> None:
    doc = document([node("C1", "context"), node("G1", "goal")], [edge("C1", "G1")])
    with pytest.raises(CaseFormatError, match="a context may not be the source"):
        parse_case(doc)


def test_rejects_supported_by_a_context() -> None:
    # Core GSN: SupportedBy may not terminate on a Context.
    doc = document([node("G1", "goal"), node("C1", "context")], [edge("G1", "C1")])
    with pytest.raises(CaseFormatError, match="does not permit goal --supported_by--> context"):
        parse_case(doc)


def test_rejects_in_context_of_a_goal() -> None:
    # Core GSN: InContextOf may not terminate on a Goal.
    doc = document([node("G1", "goal"), node("G2", "goal")], [edge("G1", "G2", "in_context_of")])
    with pytest.raises(CaseFormatError, match="does not permit goal --in_context_of--> goal"):
        parse_case(doc)


def test_rejects_non_list_top_goals() -> None:
    doc = document([node("G1", "goal")])
    doc["top_goals"] = "G1"
    with pytest.raises(CaseFormatError, match="key 'top_goals' must be a list, found str"):
        parse_case(doc)


def test_rejects_non_string_top_goal() -> None:
    doc = document([node("G1", "goal")], top_goals=[1])
    with pytest.raises(CaseFormatError, match="must be a node id string"):
        parse_case(doc)


def test_rejects_unknown_top_goal() -> None:
    doc = document([node("G1", "goal")], top_goals=["G9"])
    with pytest.raises(CaseFormatError, match="unknown node id 'G9'"):
        parse_case(doc)


def test_rejects_top_goal_that_is_not_a_goal() -> None:
    doc = document([node("S1", "strategy")], top_goals=["S1"])
    with pytest.raises(CaseFormatError, match="a top goal must be a Goal"):
        parse_case(doc)


def test_rejects_duplicate_top_goal() -> None:
    doc = document([node("G1", "goal")], top_goals=["G1", "G1"])
    with pytest.raises(CaseFormatError, match="duplicate top goal 'G1'"):
        parse_case(doc)


def test_load_case_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(CaseIOError, match="cannot read case file"):
        load_case(str(tmp_path / "nope.yaml"))


def test_load_case_rejects_invalid_yaml(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(CaseIOError, match="is not valid YAML"):
        load_case(str(path))


def test_load_case_rejects_a_directory(tmp_path) -> None:
    with pytest.raises(CaseIOError, match="cannot read case file"):
        load_case(str(tmp_path))


def test_error_message_carries_the_location_prefix() -> None:
    doc = document([node("G1", "goal"), node("G1", "goal")])
    with pytest.raises(CaseFormatError) as info:
        parse_case(doc)
    assert info.value.location == "nodes[1]"
    assert str(info.value).startswith("nodes[1]: ")
