"""Data model: construction, validation, normalisation and known-answer arithmetic."""

from __future__ import annotations

import math

import pytest

from tmtcdiff.model import (
    DATA_TYPES,
    MAX_BIT_WIDTH,
    Command,
    CommandArgument,
    Dictionary,
    Parameter,
)


class TestParameterConstruction:
    def test_defaults_are_a_byte_wide_unsigned_field(self) -> None:
        parameter = Parameter(name="X")
        assert parameter.data_type == "uint"
        assert parameter.bit_offset == 0
        assert parameter.bit_width == 8
        assert parameter.byte_order == "big"
        assert parameter.scale == 1.0
        assert parameter.offset == 0.0
        assert parameter.enum == ()

    def test_name_is_stripped(self) -> None:
        assert Parameter(name="  BATT_T_1 ").name == "BATT_T_1"

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_name_rejected(self, blank: str) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Parameter(name=blank)

    def test_non_string_name_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be a string"):
            Parameter(name=7)  # type: ignore[arg-type]

    @pytest.mark.parametrize("data_type", DATA_TYPES)
    def test_every_declared_data_type_is_accepted(self, data_type: str) -> None:
        enum = ((0, "A"),) if data_type in ("enum", "uint", "int", "bool") else ()
        assert Parameter(name="X", data_type=data_type, enum=enum).data_type == data_type

    def test_data_type_is_lower_cased(self) -> None:
        assert Parameter(name="X", data_type="UINT").data_type == "uint"

    def test_unknown_data_type_names_the_alternatives(self) -> None:
        with pytest.raises(ValueError, match="is not one of"):
            Parameter(name="X", data_type="fixed_point")

    @pytest.mark.parametrize(
        ("written", "expected"),
        [("big", "big"), ("little", "little"), ("BE", "big"), ("le", "little"), ("MSB", "big")],
    )
    def test_byte_order_aliases_normalise(self, written: str, expected: str) -> None:
        assert Parameter(name="X", byte_order=written).byte_order == expected

    def test_unknown_byte_order_rejected(self) -> None:
        with pytest.raises(ValueError, match="byte_order"):
            Parameter(name="X", byte_order="middle")

    def test_negative_bit_offset_rejected(self) -> None:
        with pytest.raises(ValueError, match="bit_offset must be >= 0"):
            Parameter(name="X", bit_offset=-1)

    @pytest.mark.parametrize("width", [0, -8])
    def test_non_positive_bit_width_rejected(self, width: int) -> None:
        with pytest.raises(ValueError, match="bit_width must be > 0"):
            Parameter(name="X", bit_width=width)

    def test_bit_width_above_the_maximum_rejected(self) -> None:
        with pytest.raises(ValueError, match="exceeds the supported maximum"):
            Parameter(name="X", bit_width=MAX_BIT_WIDTH + 1)

    def test_bool_is_not_an_int_for_bit_width(self) -> None:
        with pytest.raises(TypeError, match="must be an int"):
            Parameter(name="X", bit_width=True)  # type: ignore[arg-type]

    def test_zero_scale_rejected_with_the_reason(self) -> None:
        with pytest.raises(ValueError, match="cannot be inverted"):
            Parameter(name="X", scale=0.0)

    @pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
    def test_non_finite_scale_rejected(self, value: float) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            Parameter(name="X", scale=value)

    def test_limits_the_wrong_way_round_rejected(self) -> None:
        with pytest.raises(ValueError, match="is above high_limit"):
            Parameter(name="X", low_limit=10.0, high_limit=1.0)

    def test_equal_limits_accepted(self) -> None:
        assert Parameter(name="X", low_limit=5.0, high_limit=5.0).low_limit == 5.0

    def test_only_one_limit_is_allowed(self) -> None:
        assert Parameter(name="X", low_limit=1.0).high_limit is None


class TestEnumerations:
    def test_enum_is_sorted_by_code(self) -> None:
        parameter = Parameter(name="X", data_type="enum", enum=((2, "C"), (0, "A"), (1, "B")))
        assert parameter.enum == ((0, "A"), (1, "B"), (2, "C"))

    def test_enum_accepts_a_mapping(self) -> None:
        parameter = Parameter(name="X", data_type="enum", enum={1: "B", 0: "A"})
        assert parameter.enum_map == {0: "A", 1: "B"}

    def test_label_map_inverts_the_enumeration(self) -> None:
        parameter = Parameter(name="X", data_type="enum", enum=((0, "A"), (1, "B")))
        assert parameter.label_map == {"A": 0, "B": 1}

    def test_duplicate_codes_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate enum codes"):
            Parameter(name="X", data_type="enum", enum=((0, "A"), (0, "B")))

    def test_duplicate_labels_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate enum labels"):
            Parameter(name="X", data_type="enum", enum=((0, "A"), (1, "A")))

    def test_empty_label_rejected(self) -> None:
        with pytest.raises(ValueError, match="is empty"):
            Parameter(name="X", data_type="enum", enum=((0, ""),))

    def test_code_too_wide_for_the_field_rejected(self) -> None:
        # 4 bits hold 0..15; code 16 does not fit.
        with pytest.raises(ValueError, match="does not fit in bit_width"):
            Parameter(name="X", data_type="enum", bit_width=4, enum=((16, "A"),))

    def test_negative_code_on_an_unsigned_field_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsigned"):
            Parameter(name="X", data_type="uint", enum=((-1, "A"),))

    def test_negative_code_on_a_signed_field_accepted(self) -> None:
        assert Parameter(name="X", data_type="int", enum=((-1, "A"),)).enum == ((-1, "A"),)

    def test_float_cannot_carry_an_enumeration(self) -> None:
        with pytest.raises(ValueError, match="cannot carry an enumeration"):
            Parameter(name="X", data_type="float", enum=((0, "A"),))

    def test_enum_given_as_a_string_rejected(self) -> None:
        with pytest.raises(TypeError, match="not a string"):
            Parameter(name="X", enum="0:A")  # type: ignore[arg-type]

    def test_enum_entry_of_the_wrong_length_rejected(self) -> None:
        with pytest.raises(TypeError, match="code, label"):
            Parameter(name="X", enum=[(0, "A", "extra")])  # type: ignore[list-item]


class TestEngineeringValue:
    def test_known_answer_battery(self, battery: Parameter) -> None:
        # Hand calculation: 0.0625 K/count * 4800 counts = 300.0 K, offset 0.
        assert battery.engineering(4800) == pytest.approx(300.0, rel=1e-12, abs=0.0)

    def test_known_answer_with_offset(self) -> None:
        # 0.0625 * 4800 = 300.0; 300.0 + (-273.15) = 26.85 exactly in decimal.
        parameter = Parameter(name="X", data_type="int", bit_width=12, scale=0.0625, offset=-273.15)
        assert parameter.engineering(4800) == pytest.approx(26.85, rel=1e-12, abs=0.0)

    def test_zero_raw_returns_the_offset(self) -> None:
        parameter = Parameter(name="X", scale=3.0, offset=-1.5)
        assert parameter.engineering(0) == pytest.approx(-1.5, rel=1e-12, abs=0.0)

    def test_linearity(self) -> None:
        parameter = Parameter(name="X", scale=0.25, offset=7.0)
        left = parameter.engineering(4) - parameter.engineering(0)
        right = parameter.engineering(8) - parameter.engineering(4)
        assert left == pytest.approx(right, rel=1e-12, abs=0.0)


class TestStructuralSignature:
    def test_signature_ignores_the_name(self, battery: Parameter) -> None:
        assert battery.structural_signature() == battery.with_fields(
            name="SOMETHING_ELSE"
        ).structural_signature()

    def test_signature_ignores_the_path_and_description(self, battery: Parameter) -> None:
        other = battery.with_fields(path="", description="")
        assert battery.structural_signature() == other.structural_signature()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("data_type", "uint"),
            ("bit_offset", 4),
            ("bit_width", 16),
            ("byte_order", "little"),
            ("scale", 0.125),
            ("offset", 1.0),
            ("unit", "degC"),
            ("low_limit", 200.0),
            ("high_limit", 400.0),
            ("packet", "OTHER_HK"),
        ],
    )
    def test_signature_sees_every_structural_field(
        self, battery: Parameter, field: str, value: object
    ) -> None:
        other = battery.with_fields(**{field: value})
        assert battery.structural_signature() != other.structural_signature()


class TestCommandArgument:
    def test_defaults(self) -> None:
        argument = CommandArgument(name="DUTY", position=0)
        assert argument.data_type == "uint"
        assert argument.bit_width == 8
        assert argument.scale == 1.0

    def test_negative_position_rejected(self) -> None:
        with pytest.raises(ValueError, match="position must be >= 0"):
            CommandArgument(name="DUTY", position=-1)

    def test_zero_scale_rejected(self) -> None:
        with pytest.raises(ValueError, match="scale must be non-zero"):
            CommandArgument(name="DUTY", position=0, scale=0.0)

    def test_signature_excludes_the_name(self) -> None:
        left = CommandArgument(name="A", position=0)
        right = CommandArgument(name="B", position=0)
        assert left.structural_signature() == right.structural_signature()


class TestCommand:
    def test_arguments_sorted_by_position(self) -> None:
        command = Command(
            name="C",
            arguments=(
                CommandArgument(name="B", position=1),
                CommandArgument(name="A", position=0),
            ),
        )
        assert [argument.name for argument in command.arguments] == ["A", "B"]

    def test_argument_map(self, heater: Command) -> None:
        assert sorted(heater.argument_map) == ["DUTY", "HEATER_ID"]

    def test_duplicate_argument_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate argument names"):
            Command(
                name="C",
                arguments=(
                    CommandArgument(name="A", position=0),
                    CommandArgument(name="A", position=1),
                ),
            )

    def test_duplicate_positions_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate argument positions"):
            Command(
                name="C",
                arguments=(
                    CommandArgument(name="A", position=0),
                    CommandArgument(name="B", position=0),
                ),
            )

    def test_non_argument_rejected(self) -> None:
        with pytest.raises(TypeError, match="CommandArgument"):
            Command(name="C", arguments=("A",))  # type: ignore[arg-type]

    def test_signature_includes_argument_names(self, heater: Command) -> None:
        renamed = Command(
            name=heater.name,
            arguments=(
                CommandArgument(name="OTHER", position=0, bit_width=8),
                CommandArgument(name="DUTY", position=1, bit_width=8, unit="percent"),
            ),
        )
        assert heater.structural_signature() != renamed.structural_signature()


class TestDictionary:
    def test_length_counts_parameters_and_commands(self, dictionary: Dictionary) -> None:
        assert len(dictionary) == 3

    def test_items_are_sorted_by_name(self, dictionary: Dictionary) -> None:
        assert [p.name for p in dictionary.parameters] == ["AOCS_MODE", "BATT_T_1"]

    def test_item_names_merges_both_kinds(self, dictionary: Dictionary) -> None:
        assert dictionary.item_names == ("AOCS_MODE", "BATT_T_1", "SET_HEATER")

    def test_resolve_by_name(self, dictionary: Dictionary) -> None:
        assert dictionary.resolve("BATT_T_1") == "BATT_T_1"

    def test_resolve_by_path(self, dictionary: Dictionary) -> None:
        assert dictionary.resolve("TM/PWR/BATT_T_1") == "BATT_T_1"

    def test_resolve_unknown_returns_none(self, dictionary: Dictionary) -> None:
        assert dictionary.resolve("NOT_THERE") is None

    def test_duplicate_parameter_names_rejected(self, battery: Parameter) -> None:
        with pytest.raises(ValueError, match="duplicate parameter names"):
            Dictionary(version="1.0", parameters=(battery, battery))

    def test_duplicate_command_names_rejected(self, heater) -> None:
        with pytest.raises(ValueError, match="duplicate command names"):
            Dictionary(version="1.0", commands=(heater, heater))

    def test_a_name_used_for_both_kinds_is_rejected(self, battery: Parameter) -> None:
        with pytest.raises(ValueError, match="both a parameter and a command"):
            Dictionary(
                version="1.0",
                parameters=(battery,),
                commands=(Command(name=battery.name),),
            )

    def test_blank_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="version"):
            Dictionary(version="  ")

    def test_enum_labels_map_to_owner_and_code(self, dictionary: Dictionary) -> None:
        assert dictionary.enum_labels["SAFE_MODE"] == ("AOCS_MODE", 1)

    def test_path_map_skips_items_with_no_path(self) -> None:
        parameters = (Parameter(name="A"), Parameter(name="B", path="TM/B"))
        assert Dictionary(version="1.0", parameters=parameters).path_map == {"TM/B": "B"}

    def test_wrong_member_type_rejected(self) -> None:
        with pytest.raises(TypeError, match="Parameter instances"):
            Dictionary(version="1.0", parameters=("not a parameter",))  # type: ignore[arg-type]
