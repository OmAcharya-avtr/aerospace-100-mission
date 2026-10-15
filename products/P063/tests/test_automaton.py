"""The response automaton: state space, enabling rules and the assumption."""

from __future__ import annotations

import math

import pytest

from safemodetree import AutomatonState, DetectionAssumption, build_automaton


def test_state_space_is_modes_times_faults_plus_one(fmea):
    automaton = build_automaton(fmea)
    assert len(automaton) == len(fmea.modes) * (len(fmea.faults) + 1)
    assert len(automaton) == 36


def test_states_are_sorted_and_totally_ordered(fmea):
    automaton = build_automaton(fmea)
    assert list(automaton.states) == sorted(automaton.states)
    assert AutomatonState("a", None) < AutomatonState("a", "f")
    assert AutomatonState("a", "f") <= AutomatonState("b", None)
    assert AutomatonState("b", None) > AutomatonState("a", "z")
    assert AutomatonState("a", None) >= AutomatonState("a", None)


def test_state_comparison_with_other_types_is_not_implemented():
    state = AutomatonState("a", None)
    assert state.__lt__(1) is NotImplemented
    assert state.__le__(1) is NotImplemented
    assert state.__gt__(1) is NotImplemented
    assert state.__ge__(1) is NotImplemented


def test_absorbing_mode_has_no_outgoing_transition(fmea):
    automaton = build_automaton(fmea)
    for state in automaton.states:
        if fmea.mode(state.mode).absorbing:
            assert automaton.outgoing(state) == ()


def test_fault_states_exclude_modes_the_fault_cannot_occur_in(fmea):
    automaton = build_automaton(fmea)
    modes = {state.mode for state in automaton.fault_states}
    assert modes == {"fine_pointing", "coarse_pointing"}
    assert len(automaton.fault_states) == 16


def test_safe_states_are_exactly_the_safe_modes(fmea):
    automaton = build_automaton(fmea)
    assert {state.mode for state in automaton.safe_states} == {"sun_safe", "survival"}


def test_vacuous_response_contributes_no_transition(fmea):
    automaton = build_automaton(fmea)
    labels = {transition.response.id for transition in automaton.transitions}
    assert "rw_swap_to_spare" not in labels
    assert "st_power_cycle_in_safe" not in labels


def test_self_loop_is_dropped(minimal):
    automaton = build_automaton(minimal)
    for transition in automaton.transitions:
        assert transition.source != transition.target


def test_latency_budget_disables_slow_fault_responses(fmea):
    tight = build_automaton(fmea, DetectionAssumption(latency_budget_s=60.0))
    labels = {t.response.id for t in tight.transitions}
    assert "array_reconfigure" not in labels
    assert "pcdu_reset" not in labels
    assert "st_power_cycle" in labels
    assert "safe_mode_entry" in labels


def test_assume_perfect_detection_restores_everything(fmea):
    optimistic = build_automaton(
        fmea, DetectionAssumption(latency_budget_s=0.0, require_detection=False)
    )
    labels = {t.response.id for t in optimistic.transitions}
    assert "array_reconfigure" in labels


def test_undetectable_fault_admits_only_fault_agnostic_responses(minimal):
    blind = minimal.__class__(
        name=minimal.name,
        modes=minimal.modes,
        faults=(minimal.faults[0].__class__(
            id="f_one", effect="x", severity="II", detectable=False, detection_latency_s=0.0
        ),),
        responses=minimal.responses,
        conditions=minimal.conditions,
        observability=minimal.observability,
        initial_mode=minimal.initial_mode,
    )
    automaton = build_automaton(blind)
    assert automaton.outgoing(AutomatonState("ops", "f_one")) == ()


def test_is_actionable_uses_declared_latency(fmea):
    assumption = DetectionAssumption(latency_budget_s=30.0)
    assert assumption.is_actionable(fmea, "rw_bearing_drag")
    assert not assumption.is_actionable(fmea, "st_latchup")


def test_assumption_describe_text():
    assert "unbounded" in DetectionAssumption().describe()
    assert "15 s" in DetectionAssumption(latency_budget_s=15.0).describe()
    assert "perfect" in DetectionAssumption(require_detection=False).describe()


def test_incoming_and_outgoing_agree(fmea):
    automaton = build_automaton(fmea)
    total_out = sum(len(automaton.outgoing(s)) for s in automaton.states)
    total_in = sum(len(automaton.incoming(s)) for s in automaton.states)
    assert total_out == total_in == len(automaton.transitions)


def test_unknown_state_raises_keyerror(fmea):
    automaton = build_automaton(fmea)
    with pytest.raises(KeyError):
        automaton.outgoing(AutomatonState("nowhere", None))


def test_summary_and_str_are_informative(fmea):
    automaton = build_automaton(fmea)
    assert "36 states" in automaton.summary()
    assert str(AutomatonState("m", None)) == "(m, no_fault)"
    transition = automaton.transitions[0]
    assert "->" in str(transition)
    assert math.isfinite(transition.cost_min)


def test_iteration_and_len(fmea):
    automaton = build_automaton(fmea)
    assert len(list(iter(automaton))) == len(automaton)


def test_assumption_type_errors():
    with pytest.raises(TypeError, match="real number"):
        DetectionAssumption(latency_budget_s="soon")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a bool"):
        DetectionAssumption(require_detection=1)  # type: ignore[arg-type]
