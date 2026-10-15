"""Known answers, each derived by hand in the printed text beside it.

Every number here is computed from the literal declared values in
``specs/reaction_wheel_fmea.yaml`` and compared against arithmetic written out
in full. Nothing is fitted and nothing is random.

Run:
    PYTHONPATH=src python validation/validate_known_answers.py
"""

from __future__ import annotations

import math

import numpy as np

from safemodetree import (
    AutomatonState,
    DetectionAssumption,
    FaultResponsePolicy,
    ThresholdLogicClassifier,
    analyse,
    brier_score,
    build_automaton,
    load_spec,
    min_cost_to_safe,
    packaged_spec_path,
    reachability_report,
    reliability_curve,
    resolution_cost,
)

TOL = 1e-12
CHECKS: list[tuple[str, bool]] = []


def record(label: str, ok: bool) -> None:
    """Print and record one check."""
    CHECKS.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def close(measured: float, expected: float, tol: float = TOL) -> bool:
    """Compare with a tolerance on both sides."""
    return abs(measured - expected) <= tol + tol * abs(expected)


def section_structure(spec) -> None:
    """Automaton size, by hand."""
    print("1. automaton size")
    print("-" * 76)
    automaton = build_automaton(spec)
    print("  4 declared modes x (8 declared faults + 1 no-fault) = 36 states")
    print(f"  measured: {len(automaton)} states, {len(automaton.transitions)} transitions")
    record("state count is 36", len(automaton) == 36)
    print("  survival is declared absorbing: its 9 states have no outgoing transition")
    absorbing = [s for s in automaton.states if s.mode == "survival"]
    record(
        "the 9 survival states have no outgoing transition",
        len(absorbing) == 9 and all(automaton.outgoing(s) == () for s in absorbing),
    )
    print(
        "  faults are declared to occur only in fine_pointing and coarse_pointing, "
        "so 8 x 2 = 16 fault states"
    )
    record("fault-state count is 16", len(automaton.fault_states) == 16)
    print()


def section_reachability(spec) -> None:
    """Cheapest-path arithmetic, by hand."""
    print("2. cheapest route to a declared safe mode")
    print("-" * 76)
    print("  declared: coarse_downmode fine_pointing -> coarse_pointing, 8.0 min")
    print("            safe_mode_entry coarse_pointing -> sun_safe,      45.0 min")
    print("            survival_entry  any -> survival,                 480.0 min")
    print("  from coarse_pointing: min(45.0, 480.0) = 45.0 min in 1 hop")
    print("  from fine_pointing:   min(8.0 + 45.0, 480.0) = 53.0 min in 2 hops")
    report = reachability_report(spec)
    rows = {row.state: row for row in report.rows}
    fine = rows[AutomatonState("fine_pointing", "rw_tach_stale")]
    coarse = rows[AutomatonState("coarse_pointing", "rw_tach_stale")]
    print(f"  measured fine_pointing:   {fine.cost_min:.12f} min, {fine.hops} hops")
    print(f"  measured coarse_pointing: {coarse.cost_min:.12f} min, {coarse.hops} hops")
    record("fine_pointing route is 53.0 min in 2 hops",
           close(fine.cost_min, 53.0) and fine.hops == 2)
    record("coarse_pointing route is 45.0 min in 1 hop",
           close(coarse.cost_min, 45.0) and coarse.hops == 1)
    record("every fault state reaches a declared safe mode", report.all_reachable)
    record("no declared safe mode is unreachable", report.unreachable_safe_modes == ())
    print()


def section_resolution_costs(spec) -> None:
    """Mis-route arithmetic, by hand."""
    print("3. resolution cost of a right and a wrong response")
    print("-" * 76)
    automaton = build_automaton(spec)
    costs = min_cost_to_safe(automaton)
    wait = spec.response("st_wait_and_reacquire")
    cycle = spec.response("st_power_cycle")
    cases = [
        ("st_wait_and_reacquire on st_blinding  (correct, clears)", wait, "st_blinding", 6.0,
         "6.0 declared, fault cleared, nothing onward"),
        ("st_power_cycle on st_latchup          (correct, clears)", cycle, "st_latchup", 25.0,
         "25.0 declared, fault cleared, nothing onward"),
        ("st_wait_and_reacquire on st_latchup   (WRONG)", wait, "st_latchup", 59.0,
         "6.0 + ladder from fine_pointing 53.0 = 59.0"),
        ("st_power_cycle on st_blinding         (WRONG)", cycle, "st_blinding", 70.0,
         "25.0 + ladder from coarse_pointing 45.0 = 70.0"),
    ]
    for label, response, true_fault, expected, derivation in cases:
        measured = resolution_cost(automaton, costs, "fine_pointing", true_fault, response)
        print(f"  {label}")
        print(f"      {derivation}")
        print(f"      measured {measured:.12f} min, expected {expected:.1f} min")
        record(f"{label.strip()} costs {expected:.1f} min", close(measured, expected))
    print()


def section_ambiguity(spec) -> None:
    """Conservatism arithmetic, by hand."""
    print("4. ambiguity classes and the cost of conservatism")
    print("-" * 76)
    report = analyse(spec)
    print("  declared signatures make two pairs indistinguishable:")
    for group in spec.observability.indistinguishable_groups():
        print(f"      {', '.join(group)} share signature "
              f"{spec.observability.signature(group[0])}")
    star = next(
        c for c in report.ambiguity
        if c.faults == ("st_blinding", "st_latchup") and c.mode == "fine_pointing"
    )
    power = next(
        c for c in report.ambiguity
        if c.faults == ("pcdu_shunt_stuck", "sa_string_open") and c.mode == "fine_pointing"
    )
    print("  star-tracker class in fine_pointing:")
    print("      specific 6.0 (wait) and 25.0 (power cycle), mean 15.5")
    print("      conservative 8.0 (coarse_downmode) + 45.0 (safe_mode_entry) = 53.0")
    print("      cost of conservatism 53.0 - 15.5 = 37.5 min")
    print(f"      measured {star.conservatism_cost_min:.12f} min")
    record("star-tracker conservatism cost is 37.5 min",
           close(star.conservatism_cost_min, 37.5))
    print("  power class in fine_pointing:")
    print("      specific 15.0 (array_reconfigure) and 35.0 (pcdu_reset), mean 25.0")
    print("      conservative 53.0; cost of conservatism 53.0 - 25.0 = 28.0 min")
    print(f"      measured {power.conservatism_cost_min:.12f} min")
    record("power conservatism cost is 28.0 min", close(power.conservatism_cost_min, 28.0))
    record("all four ambiguity classes are reported conflicting",
           len(report.conflicting_ambiguity) == 4)
    record("two responses are reported vacuous", len(report.vacuous) == 2)
    print()


def section_oracle(spec) -> None:
    """The case where the declared specific response is the dearer one."""
    print("5. the declared specific response is not always the cheaper one")
    print("-" * 76)
    policy = FaultResponsePolicy(spec, confidence_threshold=0.0)
    specific_id, specific = policy.specific_route("bat_cell_short", "fine_pointing")
    conservative_id, conservative = policy.conservative_route("bat_cell_short", "fine_pointing")
    oracle = policy.oracle_cost("fine_pointing", "bat_cell_short")
    print(f"  bat_cell_short: specific {specific_id} = {specific:.1f} min")
    print(f"                  conservative {conservative_id} = {conservative:.1f} min")
    print(f"  the oracle therefore takes the conservative route: {oracle:.1f} min")
    record("bat_cell_bypass costs the declared 60.0 min", close(specific, 60.0))
    record("the conservative ladder costs 53.0 min", close(conservative, 53.0))
    record("the oracle takes the cheaper of the two, 53.0 min", close(oracle, 53.0))
    print()


def section_threshold_logic(spec) -> None:
    """Declared threshold-logic confidence arithmetic."""
    print("6. declared threshold-logic confidence")
    print("-" * 76)
    classifier = ThresholdLogicClassifier(spec)
    nominal = np.array(spec.observability.nominal_vector())
    label, confidence = classifier.predict_with_confidence(nominal.reshape(1, -1))
    print(f"  the declared nominal vector gives label {label[0]!r} at confidence "
          f"{confidence[0]:.12f}")
    record("the nominal vector is labelled nominal at confidence 1",
           label[0] == "nominal" and close(float(confidence[0]), 1.0))
    print("  rule 3: a unique declared signature gives confidence 1")
    unique_label, unique_conf = classifier.classify_word(
        spec.observability.signature("bat_cell_short")
    )
    print(f"      bat_cell_short -> {unique_label!r} at {unique_conf:.12f}")
    record("a unique signature gives confidence 1", close(unique_conf, 1.0))
    print("  rule 4: an ambiguity class of size m = 2 gives confidence 1/2")
    class_label, class_conf = classifier.classify_word(
        spec.observability.signature("st_blinding")
    )
    print(f"      shared star-tracker signature -> {class_label!r} at {class_conf:.12f}")
    record("an ambiguity class of size 2 gives confidence 0.5", close(class_conf, 0.5))
    print("  rule 5: attitude_residual alone is at Hamming distance 1 from three")
    print("          declared signatures (rw_motor_open, st_blinding, st_latchup),")
    print("          so the confidence is 1 / (3 * (1 + 1)) = 1/6 = 0.166666666667")
    index = spec.observability.channel_ids.index("attitude_residual")
    word = tuple(1 if position == index else 0 for position in range(7))
    hamming_label, hamming_conf = classifier.classify_word(word)
    print(f"      measured -> {hamming_label!r} at {hamming_conf:.12f}")
    record("the Hamming-1 tie of three gives 1/6", close(hamming_conf, 1.0 / 6.0))
    print()


def section_channels(spec) -> None:
    """Channel margin arithmetic."""
    print("7. channel margin arithmetic")
    print("-" * 76)
    cases = [
        ("wheel_torque_error", 4.20, 2.0, "threshold 3.00 mN*m, sd 0.60, (4.20-3.00)/0.60"),
        ("array_current", 0.84, 2.0, "threshold 1.20 A, sd 0.18, (1.20-0.84)/0.18"),
        ("bus_voltage", 25.60, 1.0, "threshold 26.00 V, sd 0.40, (26.00-25.60)/0.40"),
        ("attitude_residual", 0.80, 3.0, "threshold 0.50 deg, sd 0.10, (0.80-0.50)/0.10"),
    ]
    for channel_id, value, expected, derivation in cases:
        channel = next(c for c in spec.observability.channels if c.id == channel_id)
        measured = channel.margin(value)
        print(f"  {channel_id}: {derivation} = {expected:.1f} sigma")
        print(f"      measured {measured:.12f}")
        record(f"{channel_id} margin at {value} is {expected:.1f} sigma",
               close(measured, expected))
    print("  a reading exactly on the threshold does not assert (strict comparison)")
    channel = next(c for c in spec.observability.channels if c.id == "wheel_current")
    record("a reading on the threshold does not assert", not channel.asserted(channel.threshold))
    print()


def section_metrics() -> None:
    """Calibration-metric arithmetic."""
    print("8. calibration metric arithmetic")
    print("-" * 76)
    print("  ECE over 5 bins on confidences 0.9, 0.9, 0.6, 0.2 with")
    print("  correctness True, False, True, False:")
    print("      [0.8,1.0] acc 0.5 conf 0.9 gap 0.4 weight 0.50 -> 0.200")
    print("      [0.6,0.8) acc 1.0 conf 0.6 gap 0.4 weight 0.25 -> 0.100")
    print("      [0.2,0.4) acc 0.0 conf 0.2 gap 0.2 weight 0.25 -> 0.050")
    print("      ECE = 0.350, MCE = 0.400")
    result = reliability_curve(
        np.array([0.9, 0.9, 0.6, 0.2]), np.array([True, False, True, False]), n_bins=5
    )
    print(f"      measured ECE {result.ece:.12f}, MCE {result.mce:.12f}")
    record("ECE is 0.35", close(result.ece, 0.35))
    record("MCE is 0.40", close(result.mce, 0.40))
    print("  multi-class Brier on one row, three classes, true class 'b':")
    print("      (0.2-0)^2 + (0.5-1)^2 + (0.3-0)^2 = 0.04 + 0.25 + 0.09 = 0.38")
    measured = brier_score(np.array([[0.2, 0.5, 0.3]]), np.array(["b"]), ("a", "b", "c"))
    print(f"      measured {measured:.12f}")
    record("Brier is 0.38", close(measured, 0.38))
    print()


def section_latency(spec) -> None:
    """Latency-budget boundary arithmetic."""
    print("9. detection latency budget boundary")
    print("-" * 76)
    print("  rw_bearing_drag declares detection_latency_s: 30.0")
    print("  the comparison is latency <= budget, so 30.0 s admits it and 29.9 s does not")
    record("a 30.0 s budget admits a 30.0 s latency",
           DetectionAssumption(latency_budget_s=30.0).is_actionable(spec, "rw_bearing_drag"))
    record("a 29.9 s budget does not",
           not DetectionAssumption(latency_budget_s=29.9).is_actionable(spec, "rw_bearing_drag"))
    tight = analyse(spec, DetectionAssumption(latency_budget_s=60.0))
    print("  with a 60 s budget the two 120 s power faults lose their specific responses")
    print(f"      over budget: {', '.join(tight.faults_over_latency_budget)}")
    print(f"      conflicting ambiguity classes fall from 4 to "
          f"{len(tight.conflicting_ambiguity)}")
    record("the two power faults are reported over budget",
           set(tight.faults_over_latency_budget) == {"sa_string_open", "pcdu_shunt_stuck"})
    record("conflicting classes fall to 2", len(tight.conflicting_ambiguity) == 2)
    record("reachability survives the tighter budget", tight.reachability.all_reachable)
    print()


def main() -> None:
    """Run every known-answer section."""
    spec = load_spec(packaged_spec_path("reaction_wheel_fmea"))
    print("safemodetree known answers")
    print("=" * 76)
    print(f"declaration: {spec.summary()}")
    print(f"tolerance: {TOL:g} absolute and relative on every comparison")
    print()
    section_structure(spec)
    section_reachability(spec)
    section_resolution_costs(spec)
    section_ambiguity(spec)
    section_oracle(spec)
    section_threshold_logic(spec)
    section_channels(spec)
    section_metrics()
    section_latency(spec)
    failed = [label for label, ok in CHECKS if not ok]
    print(f"summary: {len(CHECKS) - len(failed)} of {len(CHECKS)} checks passed")
    if failed:
        print("FAILED CHECKS (recorded, not suppressed):")
        for label in failed:
            print(f"  {label}")
    assert not failed, failed
    assert math.isfinite(TOL)
    print("ALL KNOWN-ANSWER CHECKS PASSED")


if __name__ == "__main__":
    main()
