"""Baseline 1 under hand-calculated known answers, one per violation kind.

Every expected value in this file is derived by hand in the test body or in the
committed case file's ``note`` fields. Where a comparison is on a
floating-point statistic it is made with a tolerance on both sides; no test
here asserts equality of a boolean produced by a strict threshold comparison,
which is the one thing floating-point arithmetic cannot promise.
"""

from __future__ import annotations

import pytest
from conftest import CLEAN_CASES, tiny_case

from cmdguard.model import CommandSpec, EffectPolicy, Verity
from cmdguard.trace import (
    RESOURCE_REL_TOL,
    TIME_TOL_S,
    ViolationKind,
    forward_execute,
    peak_draw,
)

_T, _F, _U = Verity.TRUE, Verity.FALSE, Verity.UNKNOWN


# --------------------------------------------------------------------------
# the committed hand-constructed cases
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", CLEAN_CASES)
def test_negative_control_reports_nothing(cases: dict, name: str) -> None:
    """The negative control: a clean case must report no violation at all."""
    report = forward_execute(cases[name])
    assert report.n_violations == 0, report.kind_counts()
    assert report.clean
    assert report.gate_blocking == ()
    assert report.ordering_defects == ()
    assert report.involved_steps() == ()


def test_nominal_pass_peak_power_is_hand_calculated(cases: dict) -> None:
    """At t=80 s the held power is heater 24 + enable 18 + scan 46 = 88 W."""
    report = forward_execute(cases["nominal_science_pass"])
    peaks = peak_draw(report)
    assert peaks["power_w"] == pytest.approx(88.0, abs=1e-9)
    assert peaks["data_rate_mbps"] == pytest.approx(22.0, abs=1e-9)
    assert report.final_mode == "NOMINAL"


def test_late_payload_enable_is_one_violation_with_the_right_trace(cases: dict) -> None:
    """The specification's own example, down to which step the trace names."""
    report = forward_execute(cases["late_payload_enable"])
    assert report.kind_counts() == {"precondition_false": 1}
    violation = report.violations[0]
    assert violation.step == 6
    assert violation.command == "PAYLOAD_SCAN"
    assert violation.kind is ViolationKind.PRECONDITION_FALSE
    assert "payload_enabled" in violation.detail
    # The trace is the deliverable: it must name step 5, PAYLOAD_SAFE, t=80 s.
    first = violation.trace[0]
    assert first.step == 5
    assert first.command == "PAYLOAD_SAFE"
    assert first.t_s == pytest.approx(80.0, abs=1e-9)
    assert "set payload_enabled=false" in first.note
    assert report.involved_steps() == (5, 6)


def test_irreversible_before_verify_is_gate_blocking(cases: dict) -> None:
    report = forward_execute(cases["irreversible_before_verify"])
    assert report.kind_counts() == {"irreversible_unverified": 1}
    violation = report.violations[0]
    assert violation.step == 1
    assert violation.gate_blocking
    assert violation.trace[0].step is None
    assert "attitude_verified=unknown" in violation.trace[0].note


def test_resource_and_exclusion_matches_hand_derivation(cases: dict) -> None:
    """Six violations, each hand-derived in the case file's note fields."""
    report = forward_execute(cases["resource_and_exclusion"])
    assert report.kind_counts() == {
        "mutual_exclusion": 1,
        "resource_limit": 3,
        "timing_gap": 1,
        "timing_interval": 1,
    }
    by_key = {(v.step, v.kind.value): v for v in report.violations}

    # step 2 HEATER_B at t=60 against HEATER_A's hold [10, 210): 150 s remain.
    assert by_key[(2, "mutual_exclusion")].shortfall == pytest.approx(150.0, abs=1e-9)
    # step 3 XMIT: 40 + 40 + 55 = 135 W against 100 W -> shortfall 35 W.
    assert by_key[(3, "resource_limit")].shortfall == pytest.approx(35.0, abs=1e-9)
    # step 4 SSR_WRITE 10 s after XMIT, which declares a 20 s gap.
    assert by_key[(4, "timing_gap")].shortfall == pytest.approx(10.0, abs=1e-9)
    # step 5 SSR_READ: 30 + 25 = 55 Mbps against 40 -> shortfall 15 Mbps.
    assert by_key[(5, "resource_limit")].shortfall == pytest.approx(15.0, abs=1e-9)
    # step 6 XMIT repeats 130 s after t=70 against a 300 s interval.
    assert by_key[(6, "timing_interval")].shortfall == pytest.approx(170.0, abs=1e-9)
    assert len(report.ordering_defects) == 6


def test_resource_trace_names_contributors_largest_first(cases: dict) -> None:
    report = forward_execute(cases["resource_and_exclusion"])
    violation = next(
        v for v in report.violations if v.step == 3 and v.kind is ViolationKind.RESOURCE_LIMIT
    )
    assert [entry.command for entry in violation.trace] == ["HEATER_A", "HEATER_B"]


# --------------------------------------------------------------------------
# one minimal case per kind
# --------------------------------------------------------------------------


def test_undeclared_command_leaves_state_untouched() -> None:
    case = tiny_case(
        [(0.0, "SET_P"), (1.0, "GHOST"), (2.0, "NEED_P")], predicates={"p": _F}
    )
    report = forward_execute(case)
    assert report.kind_counts() == {"undeclared_command": 1}
    assert report.snapshots[1].predicates["p"] is _T
    assert report.snapshots[2].command == "NEED_P"


def test_mode_gate_trace_names_the_transition() -> None:
    commands = [
        CommandSpec(identifier="GO_SCIENCE", mode_effect="SCIENCE"),
        CommandSpec(identifier="GO_NOMINAL", mode_effect="NOMINAL"),
        CommandSpec(identifier="SCIENCE_ONLY", required_modes=("SCIENCE",)),
    ]
    case = tiny_case(
        [(0.0, "GO_SCIENCE"), (1.0, "GO_NOMINAL"), (2.0, "SCIENCE_ONLY")], commands=commands
    )
    report = forward_execute(case)
    assert report.kind_counts() == {"mode_gate": 1}
    violation = report.violations[0]
    assert violation.trace[0].step == 1
    assert violation.trace[0].command == "GO_NOMINAL"


def test_illegal_transition_is_reported_and_not_applied() -> None:
    commands = [CommandSpec(identifier="JUMP", mode_effect="SCIENCE")]
    case = tiny_case(
        [(0.0, "JUMP")],
        commands=commands,
        mode="SAFE",
        transitions={"SAFE": ("NOMINAL",), "NOMINAL": ("SAFE", "SCIENCE"), "SCIENCE": ()},
    )
    report = forward_execute(case)
    assert report.kind_counts() == {"illegal_transition": 1}
    assert report.final_mode == "SAFE"
    assert "not applied" in report.violations[0].detail


def test_precondition_false_and_unknown_are_distinguished() -> None:
    commands = [
        CommandSpec(identifier="NEED_P", preconditions={"p": _T}),
        CommandSpec(identifier="NEED_Q", preconditions={"q": _T}),
    ]
    case = tiny_case(
        [(0.0, "NEED_P"), (1.0, "NEED_Q")], commands=commands, predicates={"p": _F, "q": _U}
    )
    report = forward_execute(case)
    assert report.kind_counts() == {"precondition_false": 1, "precondition_unknown": 1}


def test_unknown_never_satisfies_a_precondition() -> None:
    """An undeclared predicate is unverified, not false, and not satisfiable."""
    commands = [CommandSpec(identifier="NEED_P", preconditions={"p": _F})]
    case = tiny_case([(0.0, "NEED_P")], commands=commands, predicates={})
    report = forward_execute(case)
    assert report.kind_counts() == {"precondition_unknown": 1}


def test_effect_unknown_invalidates_a_verification() -> None:
    commands = [
        CommandSpec(identifier="SLEW", effects={"p": _U}),
        CommandSpec(identifier="NEED_P", preconditions={"p": _T}),
    ]
    case = tiny_case([(0.0, "SLEW"), (1.0, "NEED_P")], commands=commands, predicates={"p": _T})
    report = forward_execute(case)
    assert report.kind_counts() == {"precondition_unknown": 1}
    assert report.violations[0].trace[0].step == 0


def test_irreversible_unknown_outranks_precondition_unknown() -> None:
    commands = [CommandSpec(identifier="FIRE", preconditions={"p": _T}, irreversible=True)]
    case = tiny_case([(0.0, "FIRE")], commands=commands, predicates={"p": _U})
    report = forward_execute(case)
    assert report.kind_counts() == {"irreversible_unverified": 1}


def test_irreversible_false_precondition_is_not_the_gate_case() -> None:
    """A false precondition on an irreversible command is an ordinary finding.

    The gate exists for the unverified case. A false precondition is caught by
    anything, including a per-command lint, so it is not escalated.
    """
    commands = [CommandSpec(identifier="FIRE", preconditions={"p": _T}, irreversible=True)]
    case = tiny_case([(0.0, "FIRE")], commands=commands, predicates={"p": _F})
    report = forward_execute(case)
    assert report.kind_counts() == {"precondition_false": 1}
    assert report.gate_blocking == ()


def test_hold_interval_is_half_open() -> None:
    """A hold ending exactly at t is expired at t, not active.

    HOLDER holds the group over [0, 10). TAKER at t=10 is legal; at t=9.999 it
    is not. The boundary is asserted away from it on both sides rather than on
    it, because the comparison is a floating-point threshold.
    """
    commands = [
        CommandSpec(identifier="HOLDER", hold_s=10.0, exclusion_group="g"),
        CommandSpec(identifier="TAKER", hold_s=1.0, exclusion_group="g"),
    ]
    at_boundary = forward_execute(tiny_case([(0.0, "HOLDER"), (10.0, "TAKER")], commands=commands))
    inside = forward_execute(tiny_case([(0.0, "HOLDER"), (9.99, "TAKER")], commands=commands))
    assert at_boundary.n_violations == 0
    assert inside.kind_counts() == {"mutual_exclusion": 1}


def test_timing_gap_tolerance_does_not_fire_one_ulp_short() -> None:
    """A gap short by less than TIME_TOL_S is not a defect in any sequence."""
    commands = [
        CommandSpec(identifier="FIRST", min_gap_after_s=10.0),
        CommandSpec(identifier="SECOND"),
    ]
    inside = forward_execute(
        tiny_case([(0.0, "FIRST"), (10.0 - 0.1 * TIME_TOL_S, "SECOND")], commands=commands)
    )
    outside = forward_execute(
        tiny_case([(0.0, "FIRST"), (10.0 - 1000.0 * TIME_TOL_S, "SECOND")], commands=commands)
    )
    assert inside.n_violations == 0
    assert outside.kind_counts() == {"timing_gap": 1}


def test_resource_relative_tolerance_is_applied() -> None:
    commands = [CommandSpec(identifier="DRAW", resource_draw={"power_w": 100.0})]
    limits = {"power_w": 100.0 * (1.0 - 0.1 * RESOURCE_REL_TOL)}
    inside = forward_execute(tiny_case([(0.0, "DRAW")], commands=commands, limits=limits))
    assert inside.n_violations == 0


def test_undeclared_resource_key_is_unchecked() -> None:
    commands = [CommandSpec(identifier="DRAW", resource_draw={"torque_nm": 1e9})]
    report = forward_execute(tiny_case([(0.0, "DRAW")], commands=commands, limits={"power_w": 1.0}))
    assert report.n_violations == 0


def test_min_interval_applies_only_to_the_same_command() -> None:
    commands = [
        CommandSpec(identifier="A", min_interval_s=100.0),
        CommandSpec(identifier="B", min_interval_s=100.0),
    ]
    report = forward_execute(tiny_case([(0.0, "A"), (1.0, "B"), (2.0, "A")], commands=commands))
    assert report.kind_counts() == {"timing_interval": 1}
    assert report.violations[0].step == 2


# --------------------------------------------------------------------------
# effect policy, determinism, snapshots
# --------------------------------------------------------------------------


def test_effect_policy_block_suppresses_a_violating_command_effects() -> None:
    commands = [
        CommandSpec(identifier="BAD_SET", preconditions={"q": _T}, effects={"p": _T}),
        CommandSpec(identifier="NEED_P", preconditions={"p": _T}),
    ]
    steps = [(0.0, "BAD_SET"), (1.0, "NEED_P")]
    applied = forward_execute(
        tiny_case(steps, commands=commands, predicates={"p": _F, "q": _F}), EffectPolicy.APPLY
    )
    blocked = forward_execute(
        tiny_case(steps, commands=commands, predicates={"p": _F, "q": _F}), EffectPolicy.BLOCK
    )
    assert applied.kind_counts() == {"precondition_false": 1}
    assert blocked.kind_counts() == {"precondition_false": 2}
    assert applied.effect_policy is EffectPolicy.APPLY


def test_effect_policy_must_be_an_enum_member() -> None:
    with pytest.raises(TypeError, match="must be an EffectPolicy"):
        forward_execute(tiny_case([(0.0, "SET_P")]), "apply")  # type: ignore[arg-type]


def test_determinism_under_rerun(cases: dict) -> None:
    """Asserted by test, as the build rules require: ten runs, one fingerprint."""
    for case in cases.values():
        prints = {forward_execute(case).fingerprint() for _ in range(10)}
        assert len(prints) == 1


def test_fingerprint_changes_when_a_finding_changes() -> None:
    clean = forward_execute(tiny_case([(0.0, "SET_P"), (1.0, "NEED_P")], predicates={"p": _F}))
    dirty = forward_execute(tiny_case([(0.0, "NEED_P"), (1.0, "SET_P")], predicates={"p": _F}))
    assert clean.fingerprint() != dirty.fingerprint()


def test_empty_sequence_is_clean_and_has_no_snapshots() -> None:
    report = forward_execute(tiny_case([]))
    assert report.clean
    assert report.snapshots == ()
    assert peak_draw(report) == {}


def test_snapshot_count_equals_step_count(cases: dict) -> None:
    for case in cases.values():
        assert len(forward_execute(case).snapshots) == len(case.sequence)


def test_of_kind_filters(cases: dict) -> None:
    report = forward_execute(cases["resource_and_exclusion"])
    assert len(report.of_kind(ViolationKind.RESOURCE_LIMIT)) == 3
    assert report.of_kind(ViolationKind.MODE_GATE) == ()
