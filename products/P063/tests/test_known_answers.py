"""Known-answer tests. Every expected value is derived in the comment above it.

These are the hand calculations the medium validation class requires. They use
the shipped declaration's literal declared numbers, so a change to the
declaration breaks them deliberately.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from safemodetree import (
    AutomatonState,
    DetectionAssumption,
    FaultResponsePolicy,
    ThresholdLogicClassifier,
    analyse,
    brier_score,
    build_automaton,
    min_cost_to_safe,
    reachability_report,
    reliability_curve,
    resolution_cost,
)


def test_state_and_transition_counts(fmea):
    # 4 modes x (8 faults + 1 no-fault) = 36 states, by hand.
    automaton = build_automaton(fmea)
    assert len(automaton) == 36
    # survival is absorbing, so its 9 states have no outgoing transition.
    absorbing_states = [s for s in automaton.states if s.mode == "survival"]
    assert len(absorbing_states) == 9
    assert all(automaton.outgoing(s) == () for s in absorbing_states)


def test_cheapest_route_costs(fmea):
    # safe_mode_entry declares from_mode: coarse_pointing, cost 45.0 min.
    # coarse_downmode declares from_mode: fine_pointing, to coarse_pointing,
    # cost 8.0 min. survival_entry is from any mode at 480.0 min.
    #   from coarse_pointing: min(45.0, 480.0)             = 45.0, 1 hop
    #   from fine_pointing:   min(8.0 + 45.0, 480.0) = 53.0        , 2 hops
    report = reachability_report(fmea)
    by_state = {row.state: row for row in report.rows}
    fine = by_state[AutomatonState("fine_pointing", "rw_tach_stale")]
    coarse = by_state[AutomatonState("coarse_pointing", "rw_tach_stale")]
    assert fine.cost_min == pytest.approx(53.0, abs=1e-12)
    assert fine.hops == 2
    assert coarse.cost_min == pytest.approx(45.0, abs=1e-12)
    assert coarse.hops == 1


def test_bat_cell_bypass_is_cheaper_than_the_ladder_for_its_own_fault(fmea):
    # bat_cell_bypass goes straight to sun_safe at 60.0 min and clears the
    # fault, so its resolution cost is 60.0 min; the conservative ladder out of
    # fine_pointing is 53.0 min. For this fault the declared specific response
    # is the DEARER option, which the policy's oracle must notice.
    policy = FaultResponsePolicy(fmea, confidence_threshold=0.0)
    specific_id, specific_cost = policy.specific_route("bat_cell_short", "fine_pointing")
    assert specific_id == "bat_cell_bypass"
    assert specific_cost == pytest.approx(60.0, abs=1e-12)
    conservative_id, conservative_cost = policy.conservative_route(
        "bat_cell_short", "fine_pointing"
    )
    assert conservative_id == "coarse_downmode"
    assert conservative_cost == pytest.approx(53.0, abs=1e-12)
    assert policy.oracle_cost("fine_pointing", "bat_cell_short") == pytest.approx(
        53.0, abs=1e-12
    )


def test_mis_route_cost_arithmetic(fmea):
    # st_wait_and_reacquire costs 6.0 min and clears st_blinding only.
    # Commanded while st_latchup is active it changes nothing about the fault,
    # leaves the system in fine_pointing, and the declared ladder out of
    # fine_pointing then costs 53.0 min: 6.0 + 53.0 = 59.0 min.
    automaton = build_automaton(fmea)
    costs = min_cost_to_safe(automaton)
    wait = fmea.response("st_wait_and_reacquire")
    assert resolution_cost(
        automaton, costs, "fine_pointing", "st_latchup", wait
    ) == pytest.approx(59.0, abs=1e-12)


def test_conservatism_cost_arithmetic(fmea):
    # Class [st_blinding, st_latchup] in fine_pointing:
    #   specific costs 6.0 and 25.0, mean 15.5
    #   conservative route 8.0 + 45.0 = 53.0
    #   conservatism cost 53.0 - 15.5 = 37.5 min
    report = analyse(fmea)
    item = next(
        c
        for c in report.ambiguity
        if c.faults == ("st_blinding", "st_latchup") and c.mode == "fine_pointing"
    )
    assert item.conservatism_cost_min == pytest.approx(37.5, abs=1e-12)


def test_threshold_confidence_arithmetic(fmea):
    # Rule 4 of the declared threshold logic: an ambiguity class of size m
    # returns 1/m. Both declared classes have m = 2, so 0.5.
    classifier = ThresholdLogicClassifier(fmea)
    for fault_id in ("st_blinding", "st_latchup", "sa_string_open", "pcdu_shunt_stuck"):
        _label, confidence = classifier.classify_word(
            fmea.observability.signature(fault_id)
        )
        assert confidence == pytest.approx(0.5, abs=1e-15)


def test_channel_margin_arithmetic(fmea):
    # wheel_torque_error: threshold 3.00 mN*m, noise_sd 0.60 mN*m, 'above'.
    # A reading of 4.20 mN*m is 1.20 mN*m past the threshold = +2.0 sigma.
    channel = next(c for c in fmea.observability.channels if c.id == "wheel_torque_error")
    assert channel.margin(4.20) == pytest.approx(2.0, abs=1e-12)
    # array_current: threshold 1.20 A, noise_sd 0.18 A, 'below'.
    # A reading of 0.84 A is 0.36 A below = +2.0 sigma of asserted margin.
    channel = next(c for c in fmea.observability.channels if c.id == "array_current")
    assert channel.margin(0.84) == pytest.approx(2.0, abs=1e-12)


def test_ece_known_answer_two_bins():
    # Four predictions, two bins over [0, 1]:
    #   [0.0, 0.5): 0.10 wrong, 0.40 wrong -> acc 0.0, conf 0.25, gap 0.25,
    #     weight 0.5 -> 0.125
    #   [0.5, 1.0]: 0.60 right, 0.90 right -> acc 1.0, conf 0.75, gap 0.25,
    #     weight 0.5 -> 0.125
    #   ECE = 0.250
    result = reliability_curve(
        np.array([0.10, 0.40, 0.60, 0.90]), np.array([False, False, True, True]), n_bins=2
    )
    assert result.ece == pytest.approx(0.25, abs=1e-12)


def test_brier_known_answer_three_classes():
    # One row, three classes, true class 'b':
    #   (0.2-0)^2 + (0.5-1)^2 + (0.3-0)^2 = 0.04 + 0.25 + 0.09 = 0.38
    assert brier_score(
        np.array([[0.2, 0.5, 0.3]]), np.array(["b"]), ("a", "b", "c")
    ) == pytest.approx(0.38, abs=1e-12)


def test_planted_case_known_answers(planted):
    # sensor_downmode is the only response for sensor_fail and it targets
    # `degraded`, declared absorbing and not safe. Nothing leaves an absorbing
    # mode, so (ops, sensor_fail) has no path to safe_hold or orphan_safe.
    # actuator_safe_hold reaches safe_hold at 11.0 min in one hop.
    report = reachability_report(planted)
    by_state = {row.state: row for row in report.rows}
    sensor = by_state[AutomatonState("ops", "sensor_fail")]
    actuator = by_state[AutomatonState("ops", "actuator_fail")]
    assert not sensor.reachable
    assert math.isinf(sensor.cost_min)
    assert actuator.reachable
    assert actuator.cost_min == pytest.approx(11.0, abs=1e-12)
    assert actuator.path == ("actuator_safe_hold",)
    assert report.unreachable_safe_modes == ("orphan_safe",)


def test_latency_budget_boundary_is_inclusive(fmea):
    # rw_bearing_drag declares detection_latency_s: 30.0. A budget of exactly
    # 30.0 s must admit it; 29.9 s must not.
    assert DetectionAssumption(latency_budget_s=30.0).is_actionable(fmea, "rw_bearing_drag")
    assert not DetectionAssumption(latency_budget_s=29.9).is_actionable(
        fmea, "rw_bearing_drag"
    )
