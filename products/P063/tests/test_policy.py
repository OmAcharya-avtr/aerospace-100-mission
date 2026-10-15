"""The policy, and the confidence it consumes."""

from __future__ import annotations

import math

import pytest

from safemodetree import NOMINAL_LABEL, DetectionAssumption, FaultResponsePolicy


@pytest.fixture
def policy(fmea) -> FaultResponsePolicy:
    return FaultResponsePolicy(fmea, confidence_threshold=0.7)


def test_high_confidence_commands_the_fault_specific_response(policy):
    decision = policy.decide("fine_pointing", "st_latchup", 0.95)
    assert decision.response_id == "st_power_cycle"
    assert not decision.routed_conservative
    assert decision.ambiguity_class == ("st_blinding", "st_latchup")
    assert "0.950" in decision.reason


def test_low_confidence_routes_to_the_conservative_response(policy):
    decision = policy.decide("fine_pointing", "st_latchup", 0.45)
    assert decision.response_id == "coarse_downmode"
    assert decision.routed_conservative
    assert "conservative route" in decision.reason


def test_the_threshold_itself_does_not_route(policy):
    assert not policy.decide("fine_pointing", "st_latchup", 0.7).routed_conservative
    assert policy.decide("fine_pointing", "st_latchup", 0.69).routed_conservative


def test_nominal_identification_commands_nothing(policy):
    decision = policy.decide("fine_pointing", NOMINAL_LABEL, 0.99)
    assert decision.response_id is None
    assert decision.reason == "no fault identified"
    assert decision.ambiguity_class == ()
    assert policy.cost_of(decision, "st_latchup") == pytest.approx(0.0)


def test_cost_known_answers(policy):
    # Hand arithmetic from the declaration, as in test_reach.py:
    #   correct st_power_cycle on st_latchup        -> 25.0 min
    #   st_power_cycle on st_blinding               -> 25.0 + 45.0 = 70.0 min
    #   conservative coarse_downmode on st_latchup  -> 8.0 + 45.0 = 53.0 min
    correct = policy.decide("fine_pointing", "st_latchup", 0.95)
    assert policy.cost_of(correct, "st_latchup") == pytest.approx(25.0, abs=1e-12)
    assert policy.cost_of(correct, "st_blinding") == pytest.approx(70.0, abs=1e-12)
    routed = policy.decide("fine_pointing", "st_latchup", 0.3)
    assert policy.cost_of(routed, "st_latchup") == pytest.approx(53.0, abs=1e-12)
    assert policy.cost_of(routed, "st_blinding") == pytest.approx(53.0, abs=1e-12)


def test_conservative_cost_is_the_same_for_every_class_member(policy):
    routed = policy.decide("fine_pointing", "st_blinding", 0.1)
    for member in ("st_blinding", "st_latchup"):
        assert policy.cost_of(routed, member) == pytest.approx(53.0, abs=1e-12)


def test_oracle_cost_takes_the_cheaper_route(policy):
    assert policy.oracle_cost("fine_pointing", "st_blinding") == pytest.approx(6.0, abs=1e-12)
    assert policy.oracle_cost("fine_pointing", "st_latchup") == pytest.approx(25.0, abs=1e-12)
    assert policy.oracle_cost("fine_pointing", NOMINAL_LABEL) == pytest.approx(0.0)


def test_oracle_falls_back_to_the_conservative_route(policy):
    # st_blinding has no response declared out of coarse_pointing, so the
    # oracle must take the conservative route, 45.0 min.
    assert policy.oracle_cost("coarse_pointing", "st_blinding") == pytest.approx(
        45.0, abs=1e-12
    )


def test_misroute_detection(policy):
    specific = policy.decide("fine_pointing", "st_latchup", 0.95)
    assert policy.is_misrouted(specific, "st_blinding")
    assert not policy.is_misrouted(specific, "st_latchup")
    conservative = policy.decide("fine_pointing", "st_latchup", 0.2)
    assert not policy.is_misrouted(conservative, "st_blinding")
    assert not policy.is_misrouted(conservative, NOMINAL_LABEL)


def test_false_alarm_costs_the_response_only(policy):
    decision = policy.decide("fine_pointing", "st_latchup", 0.95)
    assert policy.cost_of(decision, NOMINAL_LABEL) == pytest.approx(25.0, abs=1e-12)


def test_threshold_zero_never_routes_and_two_always_does(fmea):
    never = FaultResponsePolicy(fmea, confidence_threshold=0.0)
    always = FaultResponsePolicy(fmea, confidence_threshold=2.0)
    assert not never.decide("fine_pointing", "st_latchup", 0.0).routed_conservative
    assert always.decide("fine_pointing", "st_latchup", 1.0).routed_conservative


def test_ambiguity_class_is_a_singleton_for_an_unambiguous_fault(policy):
    assert policy.ambiguity_class_of("bat_cell_short", "fine_pointing") == (
        "bat_cell_short",
    )


def test_ambiguity_class_drops_members_that_cannot_occur(fmea):
    policy = FaultResponsePolicy(fmea)
    assert policy.ambiguity_class_of("st_latchup", "sun_safe") == ("st_latchup",)


def test_specific_route_is_absent_where_no_response_names_the_fault(policy):
    response_id, cost = policy.specific_route("st_blinding", "coarse_pointing")
    assert response_id is None
    assert math.isinf(cost)


def test_no_conservative_route_falls_back_to_the_specific_one(minimal):
    policy = FaultResponsePolicy(minimal, confidence_threshold=0.9)
    decision = policy.decide("ops", "f_one", 0.1)
    assert decision.response_id == "r_one"
    assert not decision.routed_conservative
    assert "no conservative route" in decision.reason


def test_tight_latency_budget_forces_the_conservative_route(fmea):
    policy = FaultResponsePolicy(
        fmea, DetectionAssumption(latency_budget_s=60.0), confidence_threshold=0.5
    )
    decision = policy.decide("fine_pointing", "sa_string_open", 0.99)
    assert decision.response_id == "coarse_downmode"
    assert decision.routed_conservative
    assert "no response names this fault here" in decision.reason


def test_reaches_safe_is_true_everywhere_in_the_shipped_declaration(policy, fmea):
    for fault_id in fmea.fault_ids:
        decision = policy.decide("fine_pointing", fault_id, 0.95)
        assert policy.reaches_safe(decision, fault_id)


def test_reaches_safe_is_false_in_the_planted_trap(planted):
    policy = FaultResponsePolicy(planted, confidence_threshold=0.0)
    decision = policy.decide("ops", "sensor_fail", 1.0)
    assert decision.response_id == "sensor_downmode"
    assert not policy.reaches_safe(decision, "sensor_fail")
    assert math.isinf(policy.cost_of(decision, "sensor_fail"))


def test_outcome_mode_and_decision_str(policy):
    decision = policy.decide("fine_pointing", "st_latchup", 0.95)
    assert policy.outcome_mode(decision, "st_latchup") == "coarse_pointing"
    assert "st_power_cycle" in str(decision)
    quiet = policy.decide("fine_pointing", NOMINAL_LABEL, 1.0)
    assert policy.outcome_mode(quiet, "st_latchup") is None
    assert "no action" in str(quiet)


def test_properties_expose_the_declaration_and_automaton(policy, fmea):
    assert policy.spec is fmea
    assert len(policy.automaton) == 36
    assert policy.confidence_threshold == pytest.approx(0.7)
