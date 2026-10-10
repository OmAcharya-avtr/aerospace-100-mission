"""Configuration loading, validation and round-tripping."""

from __future__ import annotations

import json

import pytest

from traceaudit import TraceConfig


@pytest.mark.verifies("REQ-021")
def test_default_config_uses_the_req_nnn_convention():
    cfg = TraceConfig.default()
    assert cfg.id_pattern == r"REQ-\d{3}"
    assert cfg.skip_code_fences is True
    assert cfg.claim_from_node_id is True
    assert cfg.heuristic_assertions is True
    assert cfg.ignored_codes == ()


@pytest.mark.verifies("REQ-021")
def test_declaration_regex_is_built_from_the_id_pattern():
    cfg = TraceConfig(id_pattern=r"SR-\d{4}")
    assert r"(?P<id>SR-\d{4})" in cfg.declaration_regex
    assert "id" in __import__("re").compile(cfg.declaration_regex).groupindex


@pytest.mark.verifies("REQ-021")
def test_to_dict_round_trips_through_from_dict():
    cfg = TraceConfig(id_pattern=r"SR-\d{4}", ignored_codes=("TA006",),
                      claim_value_separators="|")
    assert TraceConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.verifies("REQ-021")
def test_from_json_file_reads_a_configuration(tmp_path):
    path = tmp_path / "traceaudit.json"
    path.write_text(json.dumps({"id_pattern": "SR-[0-9]{4}", "ignored_codes": ["TA006"]}),
                    encoding="utf-8")
    cfg = TraceConfig.from_json_file(path)
    assert cfg.id_pattern == "SR-[0-9]{4}"
    assert cfg.ignored_codes == ("TA006",)
    assert cfg.skip_code_fences is True


@pytest.mark.verifies("REQ-021")
def test_unknown_configuration_key_is_rejected_by_name():
    with pytest.raises(ValueError, match="unknown configuration key\\(s\\): idpattern"):
        TraceConfig.from_dict({"idpattern": "REQ"})


@pytest.mark.verifies("REQ-021")
def test_a_list_valued_key_given_as_a_string_is_rejected():
    with pytest.raises(ValueError, match="must be a list of strings"):
        TraceConfig.from_dict({"ignored_codes": "TA006"})


@pytest.mark.verifies("REQ-021")
def test_an_invalid_id_pattern_is_rejected():
    with pytest.raises(ValueError, match="id_pattern is not a valid regular expression"):
        TraceConfig(id_pattern="REQ-[0-9")


@pytest.mark.verifies("REQ-021")
def test_an_empty_id_pattern_is_rejected():
    with pytest.raises(ValueError, match="id_pattern must be a non-empty"):
        TraceConfig(id_pattern="")


@pytest.mark.verifies("REQ-021")
def test_an_id_pattern_defining_the_id_group_is_rejected():
    with pytest.raises(ValueError, match="must not define a group named 'id'"):
        TraceConfig(id_pattern=r"(?P<id>REQ-\d{3})")


@pytest.mark.verifies("REQ-021")
def test_a_declaration_pattern_without_the_id_group_is_rejected():
    with pytest.raises(ValueError, match="must define a named group 'id'"):
        TraceConfig(declaration_pattern=r"^(REQ-\d{3})$")


@pytest.mark.verifies("REQ-021")
def test_an_invalid_declaration_pattern_is_rejected():
    with pytest.raises(ValueError, match="declaration_pattern is not a valid"):
        TraceConfig(declaration_pattern=r"^(?P<id>REQ-[0-9$")


@pytest.mark.verifies("REQ-021")
def test_an_invalid_node_id_claim_pattern_is_rejected():
    with pytest.raises(ValueError, match="node_id_claim_pattern is not a valid"):
        TraceConfig(node_id_claim_pattern=r"REQ-(\d{3}")


@pytest.mark.verifies("REQ-021")
def test_an_unformattable_claim_template_is_rejected():
    with pytest.raises(ValueError, match="not formattable with positional groups"):
        TraceConfig(node_id_claim_template="REQ-{id}")


@pytest.mark.verifies("REQ-021")
def test_empty_claim_property_names_are_rejected():
    with pytest.raises(ValueError, match="must name at least one junit property"):
        TraceConfig(claim_property_names=())


@pytest.mark.verifies("REQ-021")
def test_empty_claim_value_separators_are_rejected():
    with pytest.raises(ValueError, match="must contain at least one character"):
        TraceConfig(claim_value_separators="")


@pytest.mark.verifies("REQ-022")
def test_a_missing_configuration_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="configuration file not found"):
        TraceConfig.from_json_file(tmp_path / "absent.json")


@pytest.mark.verifies("REQ-022")
def test_invalid_json_raises_value_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="is not valid JSON"):
        TraceConfig.from_json_file(path)


@pytest.mark.verifies("REQ-022")
def test_a_json_array_is_rejected_with_a_type_error(tmp_path):
    path = tmp_path / "array.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(TypeError, match="must be a JSON object"):
        TraceConfig.from_json_file(path)


@pytest.mark.verifies("REQ-021")
def test_config_is_frozen():
    import dataclasses

    cfg = TraceConfig.default()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.id_pattern = "other"
