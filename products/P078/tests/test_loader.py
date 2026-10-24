"""The loader is strict on purpose: a permissive loader checks nothing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from conftest import LOADABLE_CASES

from cmdguard.loader import case_from_mapping, case_to_mapping, dump_case, load_case
from cmdguard.model import Verity


def _minimal() -> dict:
    return {
        "name": "m",
        "dictionary": {"commands": [{"id": "A"}]},
        "mode_machine": {"modes": ["NOMINAL"], "transitions": {"NOMINAL": []}},
        "initial_state": {"mode": "NOMINAL"},
        "sequence": [{"t_s": 0.0, "command": "A"}],
    }


@pytest.mark.parametrize("name", LOADABLE_CASES)
def test_every_committed_case_loads(case_dir: Path, name: str) -> None:
    case = load_case(case_dir / f"{name}.yaml")
    assert case.name == name
    assert len(case.sequence) > 0


@pytest.mark.parametrize("name", LOADABLE_CASES)
def test_round_trip_is_byte_stable(case_dir: Path, name: str) -> None:
    """Dump, reload, dump again: the second dump equals the first.

    This is what makes ``validation/regenerate_cases.py`` able to assert that
    the committed files are unchanged by regeneration.
    """
    case = load_case(case_dir / f"{name}.yaml")
    first = dump_case(case)
    second = dump_case(case_from_mapping(yaml.safe_load(first)))
    assert first == second


@pytest.mark.parametrize("name", LOADABLE_CASES)
def test_round_trip_preserves_semantics(case_dir: Path, name: str) -> None:
    from cmdguard.trace import forward_execute

    case = load_case(case_dir / f"{name}.yaml")
    again = case_from_mapping(yaml.safe_load(dump_case(case)))
    assert forward_execute(case).fingerprint() == forward_execute(again).fingerprint()


def test_missing_declarations_is_rejected(case_dir: Path) -> None:
    with pytest.raises(ValueError, match="unknown key"):
        load_case(case_dir / "missing_declarations.yaml")


def test_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="case file not found"):
        load_case(tmp_path / "absent.yaml")


def test_empty_file_raises_valueerror(tmp_path: Path) -> None:
    target = tmp_path / "empty.yaml"
    target.write_text("")
    with pytest.raises(ValueError, match="case file is empty"):
        load_case(target)


def test_unknown_top_level_key_rejected() -> None:
    raw = _minimal()
    raw["extra"] = 1
    with pytest.raises(ValueError, match="unknown top-level key"):
        case_from_mapping(raw)


@pytest.mark.parametrize("key", ["dictionary", "mode_machine", "initial_state", "sequence"])
def test_missing_required_key_rejected(key: str) -> None:
    raw = _minimal()
    del raw[key]
    with pytest.raises(ValueError, match=f"required key '{key}' is missing"):
        case_from_mapping(raw)


def test_empty_command_list_rejected() -> None:
    raw = _minimal()
    raw["dictionary"]["commands"] = []
    with pytest.raises(ValueError, match="non-empty list of commands"):
        case_from_mapping(raw)


def test_duplicate_command_rejected() -> None:
    raw = _minimal()
    raw["dictionary"]["commands"] = [{"id": "A"}, {"id": "A"}]
    with pytest.raises(ValueError, match="declared twice"):
        case_from_mapping(raw)


def test_non_string_id_rejected() -> None:
    raw = _minimal()
    raw["dictionary"]["commands"] = [{"id": 7}]
    with pytest.raises(ValueError, match="'id' must be a string"):
        case_from_mapping(raw)


def test_required_modes_must_be_a_list() -> None:
    raw = _minimal()
    raw["dictionary"]["commands"][0]["required_modes"] = "NOMINAL"
    with pytest.raises(ValueError, match="got the string"):
        case_from_mapping(raw)


def test_irreversible_must_be_boolean() -> None:
    raw = _minimal()
    raw["dictionary"]["commands"][0]["irreversible"] = "yes"
    with pytest.raises(ValueError, match="must be true or false"):
        case_from_mapping(raw)


def test_resource_draw_must_be_numeric() -> None:
    raw = _minimal()
    raw["dictionary"]["commands"][0]["resource_draw"] = {"power_w": "lots"}
    with pytest.raises(ValueError, match="expected a number"):
        case_from_mapping(raw)


def test_unknown_step_key_rejected() -> None:
    raw = _minimal()
    raw["sequence"][0]["when"] = 1.0
    with pytest.raises(ValueError, match=r"case.sequence\[0\]: unknown key"):
        case_from_mapping(raw)


def test_precondition_of_unknown_rejected_through_the_loader() -> None:
    raw = _minimal()
    raw["dictionary"]["commands"][0]["preconditions"] = {"p": "unknown"}
    with pytest.raises(ValueError, match="not a condition that can be met"):
        case_from_mapping(raw)


def test_mode_effect_must_be_a_string() -> None:
    raw = _minimal()
    raw["dictionary"]["commands"][0]["mode_effect"] = 3
    with pytest.raises(ValueError, match="'mode_effect' must be a string"):
        case_from_mapping(raw)


def test_exclusion_group_must_be_a_string() -> None:
    raw = _minimal()
    raw["dictionary"]["commands"][0]["exclusion_group"] = 3
    with pytest.raises(ValueError, match="'exclusion_group' must be a string"):
        case_from_mapping(raw)


def test_name_defaults_to_the_file_stem(tmp_path: Path) -> None:
    raw = _minimal()
    del raw["name"]
    target = tmp_path / "stem_name.yaml"
    target.write_text(yaml.safe_dump(raw))
    assert load_case(target).name == "stem_name"


def test_boolean_predicates_parse_from_yaml_booleans() -> None:
    raw = _minimal()
    raw["initial_state"]["predicates"] = {"p": True, "q": False, "r": "unknown"}
    case = case_from_mapping(raw)
    assert case.initial.verity("p") is Verity.TRUE
    assert case.initial.verity("q") is Verity.FALSE
    assert case.initial.verity("r") is Verity.UNKNOWN


def test_case_to_mapping_omits_defaults() -> None:
    mapping = case_to_mapping(case_from_mapping(_minimal()))
    entry = mapping["dictionary"]["commands"][0]
    assert set(entry) == {"id", "subsystem"}
