"""Reachability, cheapest paths and the brute-force cross-check."""

from __future__ import annotations

import math

import pytest

from safemodetree import (
    AutomatonState,
    DetectionAssumption,
    build_automaton,
    exhaustive_min_cost_to_safe,
    exhaustive_safe_reachable,
    min_cost_to_safe,
    outcome_state,
    preferred_specific_response,
    reachability_report,
    resolution_cost,
    safe_reachable_set,
)


def test_fixpoint_matches_exhaustive_on_the_shipped_declaration(fmea):
    automaton = build_automaton(fmea)
    assert safe_reachable_set(automaton) == exhaustive_safe_reachable(automaton)


def test_dijkstra_matches_exhaustive_costs_on_the_shipped_declaration(fmea):
    automaton = build_automaton(fmea)
    fixpoint = min_cost_to_safe(automaton)
    brute = exhaustive_min_cost_to_safe(automaton)
    assert set(fixpoint) == set(brute)
    for state, (cost, _path, _hops) in fixpoint.items():
        assert cost == pytest.approx(brute[state], rel=1e-12, abs=1e-12)


def test_fixpoint_matches_exhaustive_on_the_planted_declaration(planted):
    automaton = build_automaton(planted)
    assert safe_reachable_set(automaton) == exhaustive_safe_reachable(automaton)


def test_safe_states_are_their_own_witnesses(fmea):
    automaton = build_automaton(fmea)
    costs = min_cost_to_safe(automaton)
    for state in automaton.safe_states:
        assert costs[state] == (0.0, (), 0)


def test_cheapest_route_from_fine_pointing_is_the_mode_ladder(fmea):
    # Hand arithmetic from the declaration: safe_mode_entry has
    # from_mode: coarse_pointing, so the only route out of fine_pointing is
    # coarse_downmode (8.0 min) followed by safe_mode_entry (45.0 min),
    # total 53.0 min in 2 hops. The alternative, survival_entry, is 480.0 min.
    report = reachability_report(fmea)
    row = next(r for r in report.rows if r.state == AutomatonState("fine_pointing", "st_latchup"))
    assert row.reachable
    assert row.hops == 2
    assert row.cost_min == pytest.approx(53.0, abs=1e-12)
    assert row.path == ("coarse_downmode", "safe_mode_entry")
    assert row.safe_mode == "sun_safe"


def test_cheapest_route_from_coarse_pointing_is_one_hop(fmea):
    report = reachability_report(fmea)
    row = next(
        r for r in report.rows if r.state == AutomatonState("coarse_pointing", "st_latchup")
    )
    assert row.hops == 1
    assert row.cost_min == pytest.approx(45.0, abs=1e-12)


def test_all_fault_states_reach_a_safe_mode_in_the_shipped_declaration(fmea):
    report = reachability_report(fmea)
    assert report.all_reachable
    assert report.unreachable_rows == ()
    assert report.unreachable_safe_modes == ()
    assert report.worst_cost_min == pytest.approx(53.0, abs=1e-12)


def test_planted_case_is_detected(planted):
    report = reachability_report(planted)
    assert not report.all_reachable
    assert [str(r.state) for r in report.unreachable_rows] == ["(ops, sensor_fail)"]
    assert report.unreachable_safe_modes == ("orphan_safe",)
    assert math.isinf(report.worst_cost_min)


def test_reachability_table_and_row_strings(fmea, planted):
    assert "cost_min" in reachability_report(fmea).table()
    report = reachability_report(planted)
    text = "\n".join(str(row) for row in report.rows)
    assert "NO SAFE MODE REACHABLE" in text
    assert "safe_hold" in text


def test_latency_budget_changes_the_witness_but_not_reachability(fmea):
    tight = reachability_report(fmea, DetectionAssumption(latency_budget_s=60.0))
    assert tight.all_reachable
    assert tight.worst_cost_min == pytest.approx(53.0, abs=1e-12)


def test_outcome_state_only_clears_the_matching_fault(fmea):
    wait = fmea.response("st_wait_and_reacquire")
    assert outcome_state(wait, "fine_pointing", "st_blinding") == AutomatonState(
        "fine_pointing", None
    )
    assert outcome_state(wait, "fine_pointing", "st_latchup") == AutomatonState(
        "fine_pointing", "st_latchup"
    )


def test_fault_agnostic_response_with_clears_fault_clears_anything(fmea):
    agnostic = fmea.response("coarse_downmode")
    assert outcome_state(agnostic, "fine_pointing", "st_latchup") == AutomatonState(
        "coarse_pointing", "st_latchup"
    )


def test_resolution_cost_known_answers(fmea):
    # Hand arithmetic, all from the declaration:
    #   correct wait:        6.0, clears st_blinding            -> 6.0 min
    #   correct power cycle: 25.0, clears st_latchup            -> 25.0 min
    #   wait on a latch-up:  6.0 + cost_to_safe(fine, latchup)  -> 6.0 + 53.0 = 59.0 min
    #   power cycle on blinding: 25.0 + cost_to_safe(coarse, blinding) -> 25.0 + 45.0 = 70.0
    automaton = build_automaton(fmea)
    costs = min_cost_to_safe(automaton)
    wait = fmea.response("st_wait_and_reacquire")
    cycle = fmea.response("st_power_cycle")
    assert resolution_cost(automaton, costs, "fine_pointing", "st_blinding", wait) == pytest.approx(
        6.0, abs=1e-12
    )
    assert resolution_cost(
        automaton, costs, "fine_pointing", "st_latchup", cycle
    ) == pytest.approx(25.0, abs=1e-12)
    assert resolution_cost(automaton, costs, "fine_pointing", "st_latchup", wait) == pytest.approx(
        59.0, abs=1e-12
    )
    assert resolution_cost(
        automaton, costs, "fine_pointing", "st_blinding", cycle
    ) == pytest.approx(70.0, abs=1e-12)


def test_preferred_specific_response_known_answers(fmea):
    automaton = build_automaton(fmea)
    costs = min_cost_to_safe(automaton)
    assert preferred_specific_response(automaton, costs, "fine_pointing", "st_blinding") == (
        "st_wait_and_reacquire",
        6.0,
    )
    response_id, cost = preferred_specific_response(
        automaton, costs, "coarse_pointing", "st_blinding"
    )
    assert response_id is None
    assert math.isinf(cost)


def test_exhaustive_depth_zero_finds_only_safe_states(fmea):
    automaton = build_automaton(fmea)
    assert exhaustive_safe_reachable(automaton, max_depth=0) == frozenset(automaton.safe_states)


def test_exhaustive_rejects_negative_depth(fmea):
    automaton = build_automaton(fmea)
    with pytest.raises(ValueError, match="max_depth must be >= 0"):
        exhaustive_safe_reachable(automaton, max_depth=-1)
    with pytest.raises(ValueError, match="max_depth must be >= 0"):
        exhaustive_min_cost_to_safe(automaton, max_depth=-1)


def test_witness_path_cost_is_the_sum_of_declared_costs(fmea):
    report = reachability_report(fmea)
    for row in report.rows:
        if not row.reachable:
            continue
        declared = sum(fmea.response(rid).cost_min for rid in row.path)
        assert row.cost_min == pytest.approx(declared, rel=1e-12, abs=1e-12)
