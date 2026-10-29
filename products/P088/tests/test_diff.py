"""The semantic diff: per-field classification, renames, limits and ranking."""

from __future__ import annotations

import pytest

from tmtcdiff.changes import ChangeClass, Severity
from tmtcdiff.diff import (
    DEFAULT_NUMERIC_ABS_TOL,
    DEFAULT_NUMERIC_REL_TOL,
    DEFAULT_RENAME_THRESHOLD,
    DiffConfig,
    DiffLimits,
    IncomparableField,
    compare_commands,
    compare_parameters,
    diff_dictionaries,
)
from tmtcdiff.model import Command, CommandArgument, Dictionary, Parameter


def _classes(old: Parameter, new: Parameter, config: DiffConfig | None = None):
    changes, _ = compare_parameters(old, new, config or DiffConfig())
    return [change.change_class for change in changes]


class TestCompareParameters:
    def test_identical_parameters_produce_nothing(self, battery: Parameter) -> None:
        assert _classes(battery, battery) == []

    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("data_type", "uint", ChangeClass.RETYPED),
            ("bit_offset", 8, ChangeClass.BIT_SHIFTED),
            ("bit_width", 16, ChangeClass.BIT_SHIFTED),
            ("byte_order", "little", ChangeClass.ENDIANNESS_CHANGED),
            ("scale", 0.125, ChangeClass.RESCALED),
            ("offset", -273.15, ChangeClass.RESCALED),
            ("unit", "degC", ChangeClass.UNIT_CHANGED),
            ("packet", "OTHER_HK", ChangeClass.PACKET_MOVED),
            ("low_limit", 200.0, ChangeClass.LIMITS_CHANGED),
            ("high_limit", 400.0, ChangeClass.LIMITS_CHANGED),
            ("path", "TM/OTHER/BATT_T_1", ChangeClass.PATH_MOVED),
            ("description", "other text", ChangeClass.DESCRIPTION_CHANGED),
        ],
    )
    def test_each_field_maps_to_its_class(
        self, battery: Parameter, field: str, value: object, expected: ChangeClass
    ) -> None:
        assert _classes(battery, battery.with_fields(**{field: value})) == [expected]

    def test_a_rescale_reports_the_ratio(self, battery: Parameter) -> None:
        changes, _ = compare_parameters(
            battery, battery.with_fields(scale=0.125), DiffConfig()
        )
        assert "scales by 2" in changes[0].detail

    def test_an_offset_change_reports_the_shift(self, battery: Parameter) -> None:
        changes, _ = compare_parameters(
            battery, battery.with_fields(offset=-273.15), DiffConfig()
        )
        assert "-273.15" in changes[0].detail

    def test_two_bit_fields_produce_two_changes(self, battery: Parameter) -> None:
        classes = _classes(battery, battery.with_fields(bit_offset=8, bit_width=16))
        assert classes == [ChangeClass.BIT_SHIFTED, ChangeClass.BIT_SHIFTED]

    def test_a_scale_change_below_the_tolerance_is_not_reported(
        self, battery: Parameter
    ) -> None:
        tiny = battery.with_fields(scale=battery.scale * (1.0 + 1e-15))
        assert _classes(battery, tiny) == []

    def test_a_scale_change_above_the_tolerance_is_reported(self, battery: Parameter) -> None:
        bigger = battery.with_fields(scale=battery.scale * (1.0 + 1e-9))
        assert _classes(battery, bigger) == [ChangeClass.RESCALED]

    def test_a_zero_tolerance_reports_the_smallest_representable_change(
        self, battery: Parameter
    ) -> None:
        import math

        nudged = battery.with_fields(scale=math.nextafter(battery.scale, math.inf))
        assert _classes(battery, nudged, DiffConfig(numeric_rel_tol=0.0)) == [
            ChangeClass.RESCALED
        ]

    def test_a_renamed_pair_files_changes_under_the_old_name(self, battery: Parameter) -> None:
        changes, _ = compare_parameters(
            battery, battery.with_fields(name="NEW", unit="degC"), DiffConfig()
        )
        assert changes[0].name == "BATT_T_1"
        assert changes[0].new_name == "NEW"


class TestEnumerationComparison:
    def test_a_relabelled_code_is_reported(self, mode: Parameter) -> None:
        changed = mode.with_fields(enum=((0, "IDLE"), (1, "NOMINAL_MODE"), (2, "SAFE_MODE")))
        changes, _ = compare_parameters(mode, changed, DiffConfig())
        assert changes[0].change_class is ChangeClass.REENUMERATED
        assert "codes relabelled" in changes[0].detail

    def test_a_recoded_label_is_reported(self, mode: Parameter) -> None:
        changed = mode.with_fields(enum=((0, "IDLE"), (1, "NOMINAL_MODE"), (2, "SAFE_MODE")))
        changes, _ = compare_parameters(mode, changed, DiffConfig())
        assert "labels recoded" in changes[0].detail

    def test_a_removed_label_says_the_test_never_fires(self, mode: Parameter) -> None:
        changed = mode.with_fields(enum=mode.enum[:-1])
        changes, _ = compare_parameters(mode, changed, DiffConfig())
        assert "never fires" in changes[0].detail

    def test_an_added_label_only_is_said_to_change_nothing_existing(
        self, mode: Parameter
    ) -> None:
        changed = mode.with_fields(enum=(*mode.enum, (3, "SCIENCE_MODE")))
        changes, _ = compare_parameters(mode, changed, DiffConfig())
        assert "labels added only" in changes[0].detail

    def test_reordering_an_enumeration_is_not_a_change(self, mode: Parameter) -> None:
        reordered = mode.with_fields(enum=tuple(reversed(mode.enum)))
        assert _classes(mode, reordered) == []


class TestCompareCommands:
    def test_identical_commands_produce_nothing(self, heater: Command) -> None:
        changes, _ = compare_commands(heater, heater, DiffConfig())
        assert changes == ()

    def test_swapped_positions_are_reported_as_reordered(self, heater: Command) -> None:
        swapped = Command(
            name=heater.name,
            path=heater.path,
            description=heater.description,
            arguments=(
                CommandArgument(name="DUTY", position=0, bit_width=8, unit="percent"),
                CommandArgument(name="HEATER_ID", position=1, bit_width=8),
            ),
        )
        changes, _ = compare_commands(heater, swapped, DiffConfig())
        assert {change.change_class for change in changes} == {ChangeClass.ARGUMENT_REORDERED}
        assert len(changes) == 2

    def test_a_removed_argument_is_loud(self, heater: Command) -> None:
        trimmed = Command(name=heater.name, arguments=heater.arguments[:1], path=heater.path,
                          description=heater.description)
        changes, _ = compare_commands(heater, trimmed, DiffConfig())
        assert changes[0].change_class is ChangeClass.ARGUMENT_REMOVED
        assert changes[0].severity is Severity.LOUD

    def test_an_added_argument_is_loud(self, heater: Command) -> None:
        extended = Command(
            name=heater.name,
            path=heater.path,
            description=heater.description,
            arguments=(*heater.arguments, CommandArgument(name="EXTRA", position=2)),
        )
        changes, _ = compare_commands(heater, extended, DiffConfig())
        assert changes[0].change_class is ChangeClass.ARGUMENT_ADDED

    def test_a_widened_argument_is_retyped(self, heater: Command) -> None:
        widened = Command(
            name=heater.name,
            path=heater.path,
            description=heater.description,
            arguments=(
                CommandArgument(name="HEATER_ID", position=0, bit_width=16),
                heater.arguments[1],
            ),
        )
        changes, _ = compare_commands(heater, widened, DiffConfig())
        assert changes[0].change_class is ChangeClass.ARGUMENT_RETYPED

    def test_a_rescaled_argument_is_silent(self, heater: Command) -> None:
        rescaled = Command(
            name=heater.name,
            path=heater.path,
            description=heater.description,
            arguments=(
                CommandArgument(name="HEATER_ID", position=0, bit_width=8),
                CommandArgument(name="DUTY", position=1, bit_width=8, unit="percent", scale=2.0),
            ),
        )
        changes, _ = compare_commands(heater, rescaled, DiffConfig())
        assert changes[0].change_class is ChangeClass.ARGUMENT_RESCALED
        assert changes[0].is_silent

    def test_an_argument_unit_change_is_reported_as_a_unit_change(self, heater: Command) -> None:
        changed = Command(
            name=heater.name,
            path=heater.path,
            description=heater.description,
            arguments=(
                CommandArgument(name="HEATER_ID", position=0, bit_width=8),
                CommandArgument(name="DUTY", position=1, bit_width=8, unit="fraction"),
            ),
        )
        changes, _ = compare_commands(heater, changed, DiffConfig())
        assert changes[0].change_class is ChangeClass.UNIT_CHANGED

    def test_reordering_the_declaration_without_moving_positions_is_not_a_change(
        self, heater: Command
    ) -> None:
        reordered = Command(
            name=heater.name,
            path=heater.path,
            description=heater.description,
            arguments=tuple(reversed(heater.arguments)),
        )
        changes, _ = compare_commands(heater, reordered, DiffConfig())
        assert changes == ()


class TestIncomparableFields:
    def test_a_field_only_in_the_old_version(self, battery: Parameter) -> None:
        old = battery.with_fields(extra=(("calib", "77"),))
        _, incomparable = compare_parameters(old, battery, DiffConfig())
        assert incomparable[0].presence == "old_only"

    def test_a_field_only_in_the_new_version(self, battery: Parameter) -> None:
        new = battery.with_fields(extra=(("calib", "77"),))
        _, incomparable = compare_parameters(battery, new, DiffConfig())
        assert incomparable[0].presence == "new_only"

    def test_a_field_that_differs_is_named_as_unmodelled(self, battery: Parameter) -> None:
        old = battery.with_fields(extra=(("calib", "77"),))
        new = battery.with_fields(extra=(("calib", "78"),))
        _, incomparable = compare_parameters(old, new, DiffConfig())
        assert incomparable[0].presence == "unmodelled_differs"
        assert "cannot say" in incomparable[0].detail

    def test_an_identical_unmodelled_field_is_not_reported(self, battery: Parameter) -> None:
        same = battery.with_fields(extra=(("calib", "77"),))
        _, incomparable = compare_parameters(same, same, DiffConfig())
        assert incomparable == ()

    def test_presence_is_validated(self) -> None:
        with pytest.raises(ValueError, match="presence must be"):
            IncomparableField("parameter", "A", "f", "maybe")


class TestDiffDictionaries:
    def test_type_is_checked(self, dictionary: Dictionary) -> None:
        with pytest.raises(TypeError, match="two Dictionary instances"):
            diff_dictionaries(dictionary, "not a dictionary")  # type: ignore[arg-type]

    def test_a_dictionary_against_itself_is_empty(self, dictionary: Dictionary) -> None:
        assert diff_dictionaries(dictionary, dictionary).changes == ()

    def test_versions_are_carried(self, dictionary: Dictionary) -> None:
        result = diff_dictionaries(dictionary, dictionary)
        assert (result.old_version, result.new_version) == ("1.0", "1.0")

    def test_item_counts_are_carried(self, dictionary: Dictionary) -> None:
        result = diff_dictionaries(dictionary, dictionary)
        assert result.old_item_count == result.new_item_count == 3

    def test_a_removal_is_loud(self, dictionary: Dictionary, battery: Parameter) -> None:
        trimmed = Dictionary(
            version="1.1",
            parameters=tuple(p for p in dictionary.parameters if p.name != battery.name),
            commands=dictionary.commands,
        )
        result = diff_dictionaries(dictionary, trimmed)
        assert [c.change_class for c in result.loud] == [ChangeClass.REMOVED]

    def test_an_addition_is_advisory(self, dictionary: Dictionary) -> None:
        extended = Dictionary(
            version="1.1",
            parameters=(*dictionary.parameters, Parameter(name="NEW_ONE")),
            commands=dictionary.commands,
        )
        result = diff_dictionaries(dictionary, extended)
        assert [c.change_class for c in result.advisory] == [ChangeClass.ADDED]

    def test_a_rename_is_detected_and_mapped(
        self, dictionary: Dictionary, battery: Parameter
    ) -> None:
        renamed = battery.with_fields(name="BATTERY_TEMP_1", path="TM/PWR/BATTERY_TEMP_1")
        new = Dictionary(
            version="1.1",
            parameters=tuple(
                renamed if p.name == battery.name else p for p in dictionary.parameters
            ),
            commands=dictionary.commands,
        )
        result = diff_dictionaries(dictionary, new)
        assert result.rename_map == {"BATT_T_1": "BATTERY_TEMP_1"}

    def test_rename_detection_can_be_disabled(
        self, dictionary: Dictionary, battery: Parameter
    ) -> None:
        renamed = battery.with_fields(name="BATTERY_TEMP_1")
        new = Dictionary(
            version="1.1",
            parameters=tuple(
                renamed if p.name == battery.name else p for p in dictionary.parameters
            ),
            commands=dictionary.commands,
        )
        result = diff_dictionaries(dictionary, new, DiffConfig(detect_renames=False))
        assert result.rename_map == {}
        assert ChangeClass.REMOVED in result.class_counts

    def test_a_rename_carrying_a_silent_change_reports_both(
        self, dictionary: Dictionary, battery: Parameter
    ) -> None:
        renamed = battery.with_fields(name="BATTERY_TEMP_1", unit="degC")
        new = Dictionary(
            version="1.1",
            parameters=tuple(
                renamed if p.name == battery.name else p for p in dictionary.parameters
            ),
            commands=dictionary.commands,
        )
        result = diff_dictionaries(dictionary, new)
        classes = {change.change_class for change in result.changes_for("BATT_T_1")}
        assert ChangeClass.RENAMED in classes
        assert ChangeClass.UNIT_CHANGED in classes

    def test_changes_are_ranked_silent_first(
        self, dictionary: Dictionary, battery: Parameter
    ) -> None:
        new = Dictionary(
            version="1.1",
            parameters=(battery.with_fields(unit="degC"),),
            commands=dictionary.commands,
        )
        result = diff_dictionaries(dictionary, new)
        ranks = [change.severity.rank for change in result.changes]
        assert ranks == sorted(ranks)

    def test_semantic_excludes_advisory(self, dictionary: Dictionary) -> None:
        extended = Dictionary(
            version="1.1",
            parameters=(*dictionary.parameters, Parameter(name="NEW_ONE")),
            commands=dictionary.commands,
        )
        result = diff_dictionaries(dictionary, extended)
        assert result.semantic == ()

    def test_changed_item_names_lists_only_semantic_changes(
        self, dictionary: Dictionary, battery: Parameter
    ) -> None:
        new = Dictionary(
            version="1.1",
            parameters=tuple(
                p.with_fields(description="revised") if p.name == battery.name else p
                for p in dictionary.parameters
            ),
            commands=dictionary.commands,
        )
        result = diff_dictionaries(dictionary, new)
        assert result.changed_item_names == ()

    def test_class_counts_sum_to_the_change_count(self, dictionary: Dictionary) -> None:
        new = Dictionary(
            version="1.1",
            parameters=(dictionary.parameters[0],),
            commands=(),
        )
        result = diff_dictionaries(dictionary, new)
        assert sum(result.class_counts.values()) == len(result.changes)


class TestDiffConfig:
    def test_defaults(self) -> None:
        config = DiffConfig()
        assert config.rename_threshold == DEFAULT_RENAME_THRESHOLD
        assert config.numeric_rel_tol == DEFAULT_NUMERIC_REL_TOL
        assert config.numeric_abs_tol == DEFAULT_NUMERIC_ABS_TOL
        assert config.strategy == "optimal"

    def test_the_absolute_tolerance_default_is_zero(self) -> None:
        # Deliberate: math.isclose passes when EITHER bound is met, so a
        # non-zero absolute default would silently replace the relative bound
        # for small scale factors.
        assert DEFAULT_NUMERIC_ABS_TOL == 0.0

    @pytest.mark.parametrize("threshold", [-0.5, 1.5])
    def test_threshold_is_validated(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="rename_threshold"):
            DiffConfig(rename_threshold=threshold)

    def test_negative_tolerance_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            DiffConfig(numeric_rel_tol=-1e-9)

    def test_non_finite_tolerance_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            DiffConfig(numeric_rel_tol=float("inf"))

    def test_weight_map_is_none_by_default(self) -> None:
        assert DiffConfig().weight_map is None

    def test_weight_map_round_trips(self) -> None:
        weights = (("structural", 0.5), ("name", 0.3), ("description", 0.2))
        assert DiffConfig(weights=weights).weight_map == dict(weights)


class TestDiffLimits:
    def test_no_reasons_means_determinable(self) -> None:
        limits = DiffLimits(0, 0, 0, 0, 0.0, 0.8)
        assert limits.determinable

    def test_any_reason_means_not_determinable(self) -> None:
        limits = DiffLimits(0, 0, 0, 0, 0.0, 0.8, reasons=("something",))
        assert not limits.determinable

    def test_a_non_zero_tolerance_is_always_a_reason(self, dictionary: Dictionary) -> None:
        result = diff_dictionaries(dictionary, dictionary)
        assert any("relative" in reason for reason in result.limits.reasons)

    def test_a_zero_tolerance_leaves_no_reason_on_an_unchanged_pair(
        self, dictionary: Dictionary
    ) -> None:
        result = diff_dictionaries(dictionary, dictionary, DiffConfig(numeric_rel_tol=0.0))
        assert result.limits.determinable

    def test_disabled_rename_detection_is_a_reason(self, dictionary: Dictionary) -> None:
        result = diff_dictionaries(
            dictionary, dictionary, DiffConfig(numeric_rel_tol=0.0, detect_renames=False)
        )
        assert any("rename detection was disabled" in r for r in result.limits.reasons)

    def test_incomparable_fields_are_counted(
        self, dictionary: Dictionary, battery: Parameter
    ) -> None:
        new = Dictionary(
            version="1.1",
            parameters=tuple(
                p.with_fields(extra=(("calib", "77"),)) if p.name == battery.name else p
                for p in dictionary.parameters
            ),
            commands=dictionary.commands,
        )
        result = diff_dictionaries(dictionary, new, DiffConfig(numeric_rel_tol=0.0))
        assert result.limits.incomparable_fields == 1
        assert not result.limits.determinable
