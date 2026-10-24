"""Declaration-level validation. Every error message names the offending key."""

from __future__ import annotations

import pytest

from cmdguard.model import (
    Case,
    CommandDictionary,
    CommandSequence,
    CommandSpec,
    InitialState,
    ModeMachine,
    ResourceLimits,
    SequenceStep,
    Verity,
    parse_verity,
    steps_from_pairs,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, Verity.TRUE),
        (False, Verity.FALSE),
        (None, Verity.UNKNOWN),
        ("true", Verity.TRUE),
        ("FALSE", Verity.FALSE),
        ("  Unknown ", Verity.UNKNOWN),
        (Verity.TRUE, Verity.TRUE),
    ],
)
def test_parse_verity_accepts(value: object, expected: Verity) -> None:
    assert parse_verity(value) is expected


@pytest.mark.parametrize("value", ["maybe", 1, 0.0, [], {}])
def test_parse_verity_rejects(value: object) -> None:
    with pytest.raises(ValueError, match="not one of true, false, unknown"):
        parse_verity(value)


def test_precondition_cannot_require_unknown() -> None:
    with pytest.raises(ValueError, match="not a condition that can be met"):
        CommandSpec(identifier="A", preconditions={"p": Verity.UNKNOWN})


def test_effect_may_be_unknown() -> None:
    spec = CommandSpec(identifier="A", effects={"p": Verity.UNKNOWN})
    assert spec.effects["p"] is Verity.UNKNOWN


def test_negative_resource_draw_rejected() -> None:
    with pytest.raises(ValueError, match="which is negative"):
        CommandSpec(identifier="A", resource_draw={"power_w": -1.0})


def test_boolean_resource_draw_rejected() -> None:
    with pytest.raises(ValueError, match="must be a number"):
        CommandSpec(identifier="A", resource_draw={"power_w": True})


@pytest.mark.parametrize("field", ["hold_s", "min_interval_s", "min_gap_after_s"])
def test_negative_duration_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} is -1.0"):
        CommandSpec(identifier="A", **{field: -1.0})


def test_empty_identifier_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        CommandSpec(identifier="   ")


def test_draw_keys_sorted() -> None:
    spec = CommandSpec(identifier="A", resource_draw={"z": 1.0, "a": 2.0})
    assert spec.draw_keys == ("a", "z")


def test_resource_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        ResourceLimits({"power_w": 0.0})


def test_limit_for_unknown_key_is_none() -> None:
    assert ResourceLimits({"power_w": 10.0}).limit_for("data") is None


def test_empty_dictionary_rejected() -> None:
    with pytest.raises(ValueError, match="dictionary is empty"):
        CommandDictionary(commands={})


def test_dictionary_key_must_match_identifier() -> None:
    with pytest.raises(ValueError, match="does not match command identifier"):
        CommandDictionary(commands={"B": CommandSpec(identifier="A")})


def test_dictionary_membership_and_identifiers() -> None:
    dictionary = CommandDictionary(
        commands={
            "B": CommandSpec(identifier="B", exclusion_group="g"),
            "A": CommandSpec(identifier="A", exclusion_group="g"),
        }
    )
    assert "A" in dictionary and "Z" not in dictionary
    assert dictionary.identifiers == ("A", "B")
    assert dictionary.exclusion_members("g") == ("A", "B")
    assert dictionary.get("Z") is None


def test_mode_machine_rejects_duplicates_and_unknown_modes() -> None:
    with pytest.raises(ValueError, match="duplicate mode"):
        ModeMachine(modes=("A", "A"), transitions={})
    with pytest.raises(ValueError, match="undeclared mode 'Z'"):
        ModeMachine(modes=("A",), transitions={"A": ("Z",)})
    with pytest.raises(ValueError, match="from undeclared mode"):
        ModeMachine(modes=("A",), transitions={"Z": ("A",)})


def test_mode_machine_self_loop_is_implicit() -> None:
    machine = ModeMachine(modes=("A", "B"), transitions={"A": ("B",)})
    assert machine.allows("A", "A")
    assert machine.allows("A", "B")
    assert not machine.allows("B", "A")
    assert machine.reachable_from("A") == ("B",)
    assert machine.reachable_from("B") == ()


def test_initial_state_undeclared_predicate_is_unknown() -> None:
    state = InitialState(mode="A", predicates={"p": Verity.TRUE})
    assert state.verity("p") is Verity.TRUE
    assert state.verity("q") is Verity.UNKNOWN


def test_sequence_must_not_go_backwards() -> None:
    with pytest.raises(ValueError, match="non-decreasing in time"):
        CommandSequence(
            (SequenceStep(t_s=10.0, command="A"), SequenceStep(t_s=5.0, command="B"))
        )


def test_sequence_equal_times_allowed() -> None:
    sequence = CommandSequence(
        (SequenceStep(t_s=5.0, command="A"), SequenceStep(t_s=5.0, command="B"))
    )
    assert len(sequence) == 2
    assert sequence.duration_s == 0.0


def test_empty_sequence_duration_is_zero() -> None:
    assert CommandSequence(()).duration_s == 0.0


def test_steps_from_pairs() -> None:
    sequence = steps_from_pairs([(0.0, "A"), (1.0, "B")])
    assert [s.command for s in sequence] == ["A", "B"]


def _machine() -> ModeMachine:
    return ModeMachine(modes=("SAFE", "NOMINAL"), transitions={"NOMINAL": ("SAFE",)})


def test_case_rejects_undeclared_initial_mode() -> None:
    with pytest.raises(ValueError, match="is not a declared mode"):
        Case(
            name="c",
            dictionary=CommandDictionary(commands={"A": CommandSpec(identifier="A")}),
            machine=_machine(),
            initial=InitialState(mode="SCIENCE"),
            sequence=CommandSequence(()),
        )


def test_case_rejects_gate_on_undeclared_mode() -> None:
    with pytest.raises(ValueError, match="gates on undeclared mode"):
        Case(
            name="c",
            dictionary=CommandDictionary(
                commands={"A": CommandSpec(identifier="A", required_modes=("SCIENCE",))}
            ),
            machine=_machine(),
            initial=InitialState(mode="NOMINAL"),
            sequence=CommandSequence(()),
        )


def test_case_rejects_transition_to_undeclared_mode() -> None:
    with pytest.raises(ValueError, match="transitions to undeclared mode"):
        Case(
            name="c",
            dictionary=CommandDictionary(
                commands={"A": CommandSpec(identifier="A", mode_effect="SCIENCE")}
            ),
            machine=_machine(),
            initial=InitialState(mode="NOMINAL"),
            sequence=CommandSequence(()),
        )


def test_case_reports_undeclared_commands_used() -> None:
    case = Case(
        name="c",
        dictionary=CommandDictionary(commands={"A": CommandSpec(identifier="A")}),
        machine=_machine(),
        initial=InitialState(mode="NOMINAL"),
        sequence=steps_from_pairs([(0.0, "A"), (1.0, "Z"), (2.0, "Z")]),
    )
    assert case.undeclared_commands == ("Z",)
