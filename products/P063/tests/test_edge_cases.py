"""Degenerate declarations and boundary inputs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from safemodetree import (
    NOMINAL_LABEL,
    AutomatonState,
    DetectionAssumption,
    Fault,
    FaultResponsePolicy,
    Mode,
    ObservabilityModel,
    ObservationChannel,
    Precondition,
    Response,
    SafeModeSpec,
    analyse,
    build_automaton,
    exhaustive_safe_reachable,
    load_spec_text,
    make_observations,
    min_cost_to_safe,
    reachability_report,
    reliability_curve,
    safe_reachable_set,
    split_observations,
)


def test_a_declaration_whose_only_mode_is_safe_and_absorbing(minimal_yaml):
    text = minimal_yaml.replace(
        "modes:\n  - id: ops\n  - id: hold\n    safe: true",
        "modes:\n  - id: ops\n    safe: true\n    absorbing: true\n  - id: hold\n    safe: true",
    )
    spec = load_spec_text(text)
    report = reachability_report(spec)
    # Every fault state is already in a safe mode, so each is reachable at zero
    # cost with an empty witness path.
    assert report.all_reachable
    assert all(row.cost_min == pytest.approx(0.0) for row in report.rows)
    assert all(row.hops == 0 for row in report.rows)


def test_a_declaration_with_no_responses_at_all(minimal_yaml):
    text = minimal_yaml.split("responses:")[0] + minimal_yaml[minimal_yaml.index("channels:"):]
    spec = load_spec_text(text)
    automaton = build_automaton(spec)
    assert automaton.transitions == ()
    report = analyse(spec)
    assert report.finding_counts["fault_states_without_safe_mode"] == 1
    assert report.finding_counts["faults_without_response"] == 1


def test_every_mode_safe_means_nothing_is_ever_a_finding(minimal_yaml):
    text = minimal_yaml.replace("  - id: ops\n", "  - id: ops\n    safe: true\n")
    spec = load_spec_text(text)
    report = reachability_report(spec)
    assert report.all_reachable
    assert report.unreachable_safe_modes == ()


def test_single_mode_declaration(minimal_yaml):
    text = minimal_yaml.replace("  - id: ops\n  - id: hold\n    safe: true",
                                "  - id: ops\n    safe: true")
    text = text.replace("to_mode: hold", "to_mode: ops")
    spec = load_spec_text(text)
    automaton = build_automaton(spec)
    # The only response would be a self-loop in mode, but it clears nothing, so
    # it changes no state and is dropped.
    assert automaton.transitions == ()
    assert len(automaton) == 2


def test_zero_cost_responses_give_a_zero_cost_route(minimal_yaml):
    spec = load_spec_text(minimal_yaml.replace("cost_min: 7.5", "cost_min: 0.0"))
    report = reachability_report(spec)
    row = next(r for r in report.rows if r.state == AutomatonState("ops", "f_one"))
    assert row.cost_min == pytest.approx(0.0, abs=1e-12)
    assert row.hops == 1


def test_a_fault_with_zero_detection_latency_survives_a_zero_budget(minimal_yaml):
    spec = load_spec_text(minimal_yaml.replace("detection_latency_s: 5.0",
                                               "detection_latency_s: 0.0"))
    automaton = build_automaton(spec, DetectionAssumption(latency_budget_s=0.0))
    assert len(automaton.transitions) == 1


def test_an_undetectable_fault_with_no_agnostic_response_is_stranded(minimal_yaml):
    spec = load_spec_text(minimal_yaml.replace("severity: \"II\"",
                                               "severity: \"II\"\n    detectable: false"))
    report = analyse(spec)
    assert report.undetectable_faults == ("f_one",)
    assert report.finding_counts["fault_states_without_safe_mode"] == 1
    assert report.finding_counts["fault_states_without_response"] == 1


def test_an_enormous_latency_budget_is_accepted(fmea):
    assumption = DetectionAssumption(latency_budget_s=1e12)
    assert all(assumption.is_actionable(fmea, fault_id) for fault_id in fmea.fault_ids)


def test_an_infinite_latency_budget_describes_as_unbounded():
    assert "unbounded" in DetectionAssumption(latency_budget_s=math.inf).describe()


def test_one_sample_per_fault_still_splits(fmea):
    observations = make_observations(fmea, n_per_fault=8, seed=3)
    split = split_observations(observations, train_fraction=0.5, calibration_fraction=0.25,
                               seed=4)
    assert sum(split.sizes()) == observations.n_samples


def test_reliability_curve_with_one_bin_and_one_sample():
    result = reliability_curve(np.array([0.5]), np.array([True]), n_bins=1)
    assert result.ece == pytest.approx(0.5, abs=1e-12)
    assert int(result.bin_counts[0]) == 1


def test_policy_with_a_threshold_of_exactly_one(fmea):
    policy = FaultResponsePolicy(fmea, confidence_threshold=1.0)
    # A confidence of exactly 1 is NOT strictly below 1, so it is not routed.
    assert not policy.decide("fine_pointing", "st_latchup", 1.0).routed_conservative
    assert policy.decide("fine_pointing", "st_latchup", 0.999999).routed_conservative


def test_cost_of_a_decision_that_commands_nothing_is_zero(fmea):
    policy = FaultResponsePolicy(fmea)
    decision = policy.decide("fine_pointing", NOMINAL_LABEL, 1.0)
    assert policy.cost_of(decision, "st_latchup") == pytest.approx(0.0)
    assert policy.reaches_safe(decision, "st_latchup")


def test_reaches_safe_for_a_nominal_truth_with_nothing_commanded(fmea):
    policy = FaultResponsePolicy(fmea)
    decision = policy.decide("fine_pointing", NOMINAL_LABEL, 1.0)
    assert policy.reaches_safe(decision, NOMINAL_LABEL)


def test_an_all_absorbing_declaration_has_no_transitions():
    spec = SafeModeSpec(
        name="frozen",
        modes=(Mode(id="a", safe=True, absorbing=True), Mode(id="b", absorbing=True)),
        faults=(Fault(id="f", occurs_in_modes=()),),
        responses=(Response(id="r", to_mode="a", cost_min=1.0, precondition=Precondition(())),),
        observability=ObservabilityModel(
            channels=(
                ObservationChannel(
                    id="c", unit="V", nominal=0.0, threshold=1.0, faulted=2.0,
                    direction="above", noise_sd=0.5,
                ),
            ),
            signatures={"f": (1,)},
        ),
        initial_mode="a",
    )
    automaton = build_automaton(spec)
    assert automaton.transitions == ()
    assert automaton.fault_states == ()
    report = reachability_report(spec)
    assert report.rows == ()
    assert report.worst_cost_min == pytest.approx(0.0)
    assert set(report.unreachable_safe_modes) == {"a"}
    assert safe_reachable_set(automaton) == exhaustive_safe_reachable(automaton)
    assert set(min_cost_to_safe(automaton)) == set(automaton.safe_states)
