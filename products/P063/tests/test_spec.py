"""The declarative input layer: parsing, validation and lookups."""

from __future__ import annotations

import pytest

from safemodetree import (
    SEVERITY_CATEGORIES,
    Condition,
    Literal,
    Precondition,
    load_spec_text,
    packaged_spec_names,
    packaged_spec_path,
)


def test_packaged_specs_are_listed_and_loadable():
    names = packaged_spec_names()
    assert "reaction_wheel_fmea" in names
    assert "planted_unreachable" in names
    for name in names:
        assert packaged_spec_path(name).is_file()


def test_packaged_spec_path_rejects_unknown_name():
    with pytest.raises(KeyError, match="no packaged declaration"):
        packaged_spec_path("not_a_declaration")


def test_minimal_yaml_round_trip(minimal_yaml):
    spec = load_spec_text(minimal_yaml)
    assert spec.name == "minimal"
    assert spec.mode_ids == ("ops", "hold")
    assert spec.safe_mode_ids == ("hold",)
    assert spec.fault_ids == ("f_one",)
    assert spec.response("r_one").cost_min == pytest.approx(7.5)
    assert spec.observability.signature("f_one") == (1,)


def test_fmea_shape(fmea):
    assert len(fmea.modes) == 4
    assert len(fmea.faults) == 8
    assert len(fmea.responses) == 13
    assert len(fmea.observability.channels) == 7
    assert fmea.safe_mode_ids == ("sun_safe", "survival")
    assert fmea.initial_mode == "fine_pointing"


def test_fmea_declares_two_ambiguity_classes(fmea):
    groups = fmea.observability.indistinguishable_groups()
    assert groups == (
        ("pcdu_shunt_stuck", "sa_string_open"),
        ("st_blinding", "st_latchup"),
    )


def test_severity_is_ordinal_and_ranked(fmea):
    assert SEVERITY_CATEGORIES == ("I", "II", "III", "IV")
    assert fmea.fault("bat_cell_short").severity_rank == 0
    assert fmea.fault("rw_tach_stale").severity_rank == 2


def test_lookups_raise_keyerror_on_unknown_ids(fmea):
    for getter, bad in (
        (fmea.mode, "no_such_mode"),
        (fmea.fault, "no_such_fault"),
        (fmea.response, "no_such_response"),
        (fmea.condition, "no_such_condition"),
    ):
        with pytest.raises(KeyError):
            getter(bad)


def test_fault_occurs_in_respects_declaration_and_absorbing(fmea):
    assert fmea.fault_occurs_in("st_latchup", "fine_pointing")
    assert not fmea.fault_occurs_in("st_latchup", "sun_safe")
    assert not fmea.fault_occurs_in("st_latchup", "survival")


def test_literal_parsing_both_negation_forms():
    assert Literal.parse("a") == Literal("a", True)
    assert Literal.parse("!a") == Literal("a", False)
    assert Literal.parse("not a") == Literal("a", False)
    assert str(Literal("a", False)) == "!a"


def test_precondition_parse_accepts_string_list_and_none():
    assert Precondition.parse(None).is_trivial
    assert Precondition.parse("a").literals == (Literal("a", True),)
    assert len(Precondition.parse(["a", "!b"]).literals) == 2
    assert str(Precondition.parse(["a", "!b"])) == "a and !b"
    assert str(Precondition.parse(None)) == "true"


def test_precondition_conflict_detects_contradiction(fmea):
    contradiction = Precondition(
        (Literal("wheels_available", True), Literal("wheels_available", False))
    )
    message = fmea.precondition_conflict(contradiction, "fine_pointing")
    assert message is not None
    assert "wheels_available" in message


def test_precondition_conflict_detects_always_false(fmea):
    message = fmea.precondition_conflict(
        Precondition((Literal("redundant_wheel_available"),)), "fine_pointing"
    )
    assert message is not None
    assert "always_false" in message


def test_precondition_conflict_detects_mode_unavailability(fmea):
    assert fmea.precondition_can_hold(
        Precondition((Literal("tracker_powered"),)), "fine_pointing"
    )
    message = fmea.precondition_conflict(
        Precondition((Literal("tracker_powered"),)), "sun_safe"
    )
    assert message is not None
    assert "sun_safe" in message


def test_negative_literal_never_conflicts_on_availability(fmea):
    assert fmea.precondition_can_hold(
        Precondition((Literal("tracker_powered", positive=False),)), "sun_safe"
    )


def test_condition_can_hold_in_honours_empty_mode_list():
    condition = Condition(id="k", available_in_modes=())
    assert condition.can_hold_in("anything")
    assert not Condition(id="k", always_false=True).can_hold_in("anything")


def test_observability_expected_vector_and_nominal(fmea):
    nominal = fmea.observability.nominal_vector()
    assert nominal[fmea.observability.channel_ids.index("bus_voltage")] == pytest.approx(28.0)
    expected = fmea.observability.expected_vector(fmea.observability.signature("bat_cell_short"))
    assert expected[fmea.observability.channel_ids.index("bus_voltage")] == pytest.approx(24.2)


def test_observability_expected_vector_rejects_wrong_width(fmea):
    with pytest.raises(ValueError, match="expected 7"):
        fmea.observability.expected_vector((1, 0))


def test_channel_margin_sign_and_scale(fmea):
    channel = next(c for c in fmea.observability.channels if c.id == "bus_voltage")
    # Hand arithmetic: direction 'below', threshold 26.0 V, noise_sd 0.40 V.
    # A reading of 25.6 V is 0.4 V beyond the threshold, i.e. +1.0 sigma.
    assert channel.margin(25.6) == pytest.approx(1.0, abs=1e-12)
    assert channel.margin(26.4) == pytest.approx(-1.0, abs=1e-12)
    assert channel.asserted(25.6)
    assert not channel.asserted(26.4)


def test_channel_exactly_on_threshold_does_not_assert(fmea):
    channel = next(c for c in fmea.observability.channels if c.id == "wheel_current")
    assert not channel.asserted(channel.threshold)


def test_summary_mentions_counts(fmea):
    text = fmea.summary()
    assert "8 faults" in text
    assert "2 declared ambiguity class(es)" in text
