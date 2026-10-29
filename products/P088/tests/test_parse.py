"""Reading and writing dictionary documents: aliases, extras, errors, round trips."""

from __future__ import annotations

import json

import pytest

from tmtcdiff.model import Dictionary, Parameter
from tmtcdiff.parse import (
    DictionaryFormatError,
    dictionary_from_mapping,
    dictionary_to_mapping,
    dumps,
    load_dictionary,
    loads,
    parameter_from_mapping,
    round_trip,
)

MINIMAL = {"version": "1.0", "parameters": [{"name": "A"}]}


class TestParameterFromMapping:
    def test_minimal_entry(self) -> None:
        assert parameter_from_mapping({"name": "A"}).name == "A"

    @pytest.mark.parametrize(
        ("alias", "canonical", "value"),
        [
            ("mnemonic", "name", "A"),
            ("type", "data_type", "int"),
            ("bits", "bit_width", 16),
            ("width", "bit_width", 16),
            ("factor", "scale", 0.5),
            ("gain", "scale", 0.5),
            ("bias", "offset", 2.0),
            ("units", "unit", "V"),
            ("apid", "packet", "PWR_HK"),
            ("desc", "description", "text"),
            ("min", "low_limit", 1.0),
            ("max", "high_limit", 9.0),
            ("endianness", "byte_order", "little"),
        ],
    )
    def test_alias_maps_to_the_canonical_field(
        self, alias: str, canonical: str, value: object
    ) -> None:
        raw = {"name": "A"} if alias != "mnemonic" else {}
        raw[alias] = value
        parameter = parameter_from_mapping(raw)
        assert getattr(parameter, canonical) == value

    def test_unmodelled_key_is_carried_as_an_extra(self) -> None:
        parameter = parameter_from_mapping({"name": "A", "calibration_id": 77})
        assert parameter.extra_map == {"calibration_id": "77"}

    def test_strict_mode_rejects_an_unmodelled_key(self) -> None:
        with pytest.raises(DictionaryFormatError, match="unmodelled keys"):
            parameter_from_mapping({"name": "A", "calibration_id": 77}, strict=True)

    def test_missing_name_names_the_accepted_keys(self) -> None:
        with pytest.raises(DictionaryFormatError, match="no 'name' or 'mnemonic' key"):
            parameter_from_mapping({"unit": "V"})

    def test_enum_from_string_keys(self) -> None:
        parameter = parameter_from_mapping(
            {"name": "A", "data_type": "enum", "enum": {"0": "OFF", "1": "ON"}}
        )
        assert parameter.enum == ((0, "OFF"), (1, "ON"))

    def test_enum_from_pairs(self) -> None:
        parameter = parameter_from_mapping(
            {"name": "A", "data_type": "enum", "enum": [[1, "ON"], [0, "OFF"]]}
        )
        assert parameter.enum == ((0, "OFF"), (1, "ON"))

    def test_enum_from_records(self) -> None:
        parameter = parameter_from_mapping(
            {
                "name": "A",
                "data_type": "enum",
                "enum": [{"code": 1, "label": "ON"}, {"code": 0, "label": "OFF"}],
            }
        )
        assert parameter.enum_map == {0: "OFF", 1: "ON"}

    def test_enum_record_missing_a_key_is_named(self) -> None:
        with pytest.raises(DictionaryFormatError, match="needs both 'code' and 'label'"):
            parameter_from_mapping({"name": "A", "enum": [{"code": 1}]})

    def test_enum_key_that_is_not_an_integer_is_named(self) -> None:
        with pytest.raises(DictionaryFormatError, match="not an integer code"):
            parameter_from_mapping({"name": "A", "enum": {"nought": "OFF"}})

    def test_model_validation_error_is_wrapped_with_the_parameter_name(self) -> None:
        with pytest.raises(DictionaryFormatError, match="parameter 'A'"):
            parameter_from_mapping({"name": "A", "scale": 0.0})

    def test_non_mapping_entry_rejected(self) -> None:
        with pytest.raises(DictionaryFormatError, match="must be a mapping"):
            parameter_from_mapping(["name", "A"])  # type: ignore[arg-type]

    def test_a_key_and_its_alias_together_are_rejected(self) -> None:
        with pytest.raises(DictionaryFormatError, match="alias"):
            parameter_from_mapping({"name": "A", "mnemonic": "B"})


class TestDictionaryFromMapping:
    def test_minimal_document(self) -> None:
        assert dictionary_from_mapping(MINIMAL).version == "1.0"

    def test_missing_version_rejected(self) -> None:
        with pytest.raises(DictionaryFormatError, match="no 'version' key"):
            dictionary_from_mapping({"parameters": []})

    def test_parameters_must_be_a_list(self) -> None:
        with pytest.raises(DictionaryFormatError, match="must be a list"):
            dictionary_from_mapping({"version": "1.0", "parameters": {"name": "A"}})

    def test_unknown_top_level_key_is_carried(self) -> None:
        dictionary = dictionary_from_mapping({**MINIMAL, "issued": "2026-10-29"})
        assert dict(dictionary.extra) == {"issued": "2026-10-29"}

    def test_strict_mode_rejects_an_unknown_top_level_key(self) -> None:
        with pytest.raises(DictionaryFormatError, match="unmodelled top-level keys"):
            dictionary_from_mapping({**MINIMAL, "issued": "2026-10-29"}, strict=True)

    def test_command_arguments_default_to_file_order(self) -> None:
        document = {
            "version": "1.0",
            "commands": [{"name": "C", "arguments": [{"name": "A"}, {"name": "B"}]}],
        }
        command = dictionary_from_mapping(document).commands[0]
        assert [(a.name, a.position) for a in command.arguments] == [("A", 0), ("B", 1)]

    def test_explicit_argument_positions_win_over_file_order(self) -> None:
        document = {
            "version": "1.0",
            "commands": [
                {
                    "name": "C",
                    "arguments": [{"name": "A", "position": 1}, {"name": "B", "position": 0}],
                }
            ],
        }
        command = dictionary_from_mapping(document).commands[0]
        assert [a.name for a in command.arguments] == ["B", "A"]

    def test_arguments_must_be_a_list(self) -> None:
        with pytest.raises(DictionaryFormatError, match="'arguments' must be a list"):
            dictionary_from_mapping(
                {"version": "1.0", "commands": [{"name": "C", "arguments": {"name": "A"}}]}
            )

    def test_non_mapping_document_rejected(self) -> None:
        with pytest.raises(DictionaryFormatError, match="must be a mapping"):
            dictionary_from_mapping([MINIMAL])  # type: ignore[arg-type]


class TestTextIo:
    def test_loads_round_trips(self, dictionary: Dictionary) -> None:
        assert round_trip(dictionary).parameters == dictionary.parameters

    def test_round_trip_preserves_commands(self, dictionary: Dictionary) -> None:
        assert round_trip(dictionary).commands == dictionary.commands

    def test_round_trip_preserves_extras(self) -> None:
        parameter = Parameter(name="A", extra=(("calib", "77"),))
        dictionary = Dictionary(version="1.0", parameters=(parameter,))
        assert round_trip(dictionary).parameters[0].extra_map == {"calib": "77"}

    def test_dumps_is_valid_json(self, dictionary: Dictionary) -> None:
        assert json.loads(dumps(dictionary))["version"] == "1.0"

    def test_dumps_ends_with_a_newline(self, dictionary: Dictionary) -> None:
        assert dumps(dictionary).endswith("\n")

    def test_mapping_omits_an_absent_enumeration(self) -> None:
        one = Dictionary(version="1.0", parameters=(Parameter(name="A"),))
        assert "enum" not in dictionary_to_mapping(one)["parameters"][0]

    def test_mapping_omits_absent_limits(self) -> None:
        one = Dictionary(version="1.0", parameters=(Parameter(name="A"),))
        assert "low_limit" not in dictionary_to_mapping(one)["parameters"][0]

    def test_bad_json_is_reported_as_a_format_error(self) -> None:
        with pytest.raises(DictionaryFormatError, match="not valid JSON"):
            loads("{not json")

    def test_loads_rejects_bytes(self) -> None:
        with pytest.raises(TypeError, match="expects str"):
            loads(b"{}")  # type: ignore[arg-type]

    def test_load_dictionary_reads_a_file(self, tmp_path, dictionary: Dictionary) -> None:
        target = tmp_path / "d.json"
        target.write_text(dumps(dictionary), encoding="utf-8")
        assert load_dictionary(target).version == "1.0"

    def test_load_dictionary_records_the_file_name_as_the_source(
        self, tmp_path, dictionary: Dictionary
    ) -> None:
        target = tmp_path / "d.json"
        target.write_text(dumps(dictionary), encoding="utf-8")
        assert load_dictionary(target).source == "d.json"

    def test_missing_file_is_reported_as_a_format_error(self, tmp_path) -> None:
        with pytest.raises(DictionaryFormatError, match="cannot read"):
            load_dictionary(tmp_path / "absent.json")
