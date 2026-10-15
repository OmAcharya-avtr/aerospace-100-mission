"""The findings: vacuous responses, missing responses and ambiguity classes."""

from __future__ import annotations

import math

import pytest

from safemodetree import (
    DetectionAssumption,
    ambiguity_classes,
    analyse,
    build_automaton,
    conservative_response,
    min_cost_to_safe,
    vacuous_responses,
)


def test_shipped_declaration_finding_counts(fmea):
    report = analyse(fmea)
    assert report.finding_counts == {
        "fault_states_without_safe_mode": 0,
        "unreachable_safe_modes": 0,
        "fault_states_without_response": 0,
        "faults_without_response": 0,
        "vacuous_responses": 2,
        "conflicting_ambiguity_classes": 4,
        "classes_without_conservative_response": 0,
    }
    assert report.total_findings == 6
    assert not report.ok


def test_vacuous_responses_are_named_with_reasons(fmea):
    vacuous = {item.response_id: item for item in vacuous_responses(fmea)}
    assert set(vacuous) == {"rw_swap_to_spare", "st_power_cycle_in_safe"}
    assert "always_false" in str(vacuous["rw_swap_to_spare"])
    assert "sun_safe" in str(vacuous["st_power_cycle_in_safe"])


def test_trivial_precondition_is_never_vacuous(minimal):
    assert vacuous_responses(minimal) == ()


def test_response_from_an_absorbing_mode_is_vacuous(fmea):
    # survival is declared absorbing, so a response declared out of it is dead.
    extended = fmea.__class__(
        name=fmea.name,
        modes=fmea.modes,
        faults=fmea.faults,
        responses=(
            *fmea.responses,
            fmea.responses[0].__class__(
                id="impossible", from_mode="survival", to_mode="sun_safe", cost_min=1.0
            ),
        ),
        conditions=fmea.conditions,
        observability=fmea.observability,
        initial_mode=fmea.initial_mode,
    )
    ids = {item.response_id for item in vacuous_responses(extended)}
    assert "impossible" in ids


def test_planted_declaration_findings(planted):
    report = analyse(planted)
    assert report.finding_counts["fault_states_without_safe_mode"] == 1
    assert report.finding_counts["unreachable_safe_modes"] == 1
    assert report.total_findings == 2
    text = report.report()
    assert "(ops, sensor_fail)" in text
    assert "orphan_safe" in text


def test_ambiguity_classes_known_answers(fmea):
    classes = {(c.faults, c.mode): c for c in ambiguity_classes(fmea)}
    assert len(classes) == 4
    star = classes[(("st_blinding", "st_latchup"), "fine_pointing")]
    # Hand arithmetic: st_wait_and_reacquire costs 6.0 min and clears
    # st_blinding; st_power_cycle costs 25.0 min and clears st_latchup. The
    # cheapest fault-agnostic route out of fine_pointing is coarse_downmode
    # (8.0) then safe_mode_entry (45.0) = 53.0. The mean fault-specific cost is
    # (6.0 + 25.0) / 2 = 15.5, so the conservatism cost is 53.0 - 15.5 = 37.5.
    assert star.conflicting
    assert star.preferred == {
        "st_blinding": "st_wait_and_reacquire",
        "st_latchup": "st_power_cycle",
    }
    assert star.best_specific_cost_min["st_blinding"] == pytest.approx(6.0, abs=1e-12)
    assert star.best_specific_cost_min["st_latchup"] == pytest.approx(25.0, abs=1e-12)
    assert star.conservative_response == "coarse_downmode"
    assert star.conservative_cost_min == pytest.approx(53.0, abs=1e-12)
    assert star.conservatism_cost_min == pytest.approx(37.5, abs=1e-12)
    assert star.signature == fmea.observability.signature("st_blinding")


def test_power_class_conservatism_cost(fmea):
    classes = {(c.faults, c.mode): c for c in ambiguity_classes(fmea)}
    power = classes[(("pcdu_shunt_stuck", "sa_string_open"), "fine_pointing")]
    # array_reconfigure 15.0 and pcdu_reset 35.0, mean 25.0; conservative 53.0.
    assert power.conservatism_cost_min == pytest.approx(28.0, abs=1e-12)


def test_coarse_pointing_class_has_one_unavailable_specific_response(fmea):
    classes = {(c.faults, c.mode): c for c in ambiguity_classes(fmea)}
    star = classes[(("st_blinding", "st_latchup"), "coarse_pointing")]
    assert star.preferred["st_blinding"] is None
    assert math.isinf(star.best_specific_cost_min["st_blinding"])
    assert star.conflicting
    assert star.conservative_response == "safe_mode_entry"


def test_tight_latency_budget_removes_the_power_class_conflict(fmea):
    report = analyse(fmea, DetectionAssumption(latency_budget_s=60.0))
    conflicting = {(c.faults, c.mode) for c in report.conflicting_ambiguity}
    assert (("pcdu_shunt_stuck", "sa_string_open"), "fine_pointing") not in conflicting
    assert (("st_blinding", "st_latchup"), "fine_pointing") in conflicting
    assert report.faults_over_latency_budget == ("sa_string_open", "pcdu_shunt_stuck")


def test_undetectable_faults_are_listed(fmea):
    assert analyse(fmea).undetectable_faults == ()


def test_conservative_response_returns_none_when_nothing_is_agnostic(minimal):
    automaton = build_automaton(minimal)
    costs = min_cost_to_safe(automaton)
    response_id, cost = conservative_response(
        minimal, automaton, costs, ("f_one",), "ops"
    )
    assert response_id is None
    assert math.isinf(cost)


def test_ambiguity_class_str_and_report_strings(fmea):
    report = analyse(fmea)
    text = report.report()
    assert "ambiguity classes" in text
    assert "CONFLICTING" in text
    assert "total findings: 6" in text
    assert "conservative=" in str(report.ambiguity[0])


def test_declaration_with_no_ambiguity_reports_none(minimal):
    assert ambiguity_classes(minimal) == ()
    report = analyse(minimal)
    assert report.conflicting_ambiguity == ()
    assert report.classes_without_conservative_response == ()


def test_fault_state_without_response_is_reported(minimal):
    stripped = minimal.__class__(
        name=minimal.name,
        modes=minimal.modes,
        faults=minimal.faults,
        responses=(),
        conditions=minimal.conditions,
        observability=minimal.observability,
        initial_mode=minimal.initial_mode,
    )
    report = analyse(stripped)
    assert report.finding_counts["fault_states_without_response"] == 1
    assert report.finding_counts["faults_without_response"] == 1
    assert "no enabled response" in report.report()
