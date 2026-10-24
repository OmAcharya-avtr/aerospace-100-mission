"""The gate refuses. Two values, neither of which is the word "safe"."""

from __future__ import annotations

import pytest
from conftest import tiny_case

from cmdguard.gate import Verdict, gate_case, irreversible_gate
from cmdguard.model import CommandSpec, EffectPolicy, Verity
from cmdguard.trace import forward_execute

_T, _U = Verity.TRUE, Verity.UNKNOWN


def test_gate_refuses_the_hand_constructed_case(cases: dict) -> None:
    result = gate_case(cases["irreversible_before_verify"])
    assert result.verdict is Verdict.REFUSE
    assert result.refused
    assert result.exit_code == 2
    assert len(result.blocking) == 1
    assert result.blocking[0].command == "ANT_DEPLOY"
    assert result.n_irreversible_commands == 1
    assert result.n_irreversible_executions == 1


def test_gate_does_not_refuse_the_corrected_case(cases: dict) -> None:
    result = gate_case(cases["irreversible_after_verify"])
    assert result.verdict is Verdict.NO_REFUSAL_FOUND
    assert not result.refused
    assert result.exit_code == 0
    assert result.blocking == ()


def test_the_clean_verdict_is_never_rendered_as_safe(cases: dict) -> None:
    text = gate_case(cases["irreversible_after_verify"]).render()
    assert "NO_REFUSAL_FOUND" in text
    assert "not a statement that the sequence is safe to uplink" in text
    assert "safe to uplink." in text
    lowered = text.lower()
    for forbidden in ("flight-safe", "certified", "mission-ready", "production-ready"):
        assert forbidden not in lowered


def test_refusal_render_includes_the_trace(cases: dict) -> None:
    text = gate_case(cases["irreversible_before_verify"]).render()
    assert "REFUSED" in text
    assert "because declared initial state: attitude_verified=unknown" in text


def test_gate_refuses_under_both_effect_policies(cases: dict) -> None:
    for policy in EffectPolicy:
        assert gate_case(cases["irreversible_before_verify"], policy).refused


def test_gate_rejects_a_report_from_another_case(cases: dict) -> None:
    report = forward_execute(cases["nominal_science_pass"])
    with pytest.raises(ValueError, match="will not reason about a report from a different case"):
        irreversible_gate(report, cases["late_payload_enable"])


def test_gate_counts_executions_not_declarations() -> None:
    commands = [
        CommandSpec(identifier="FIRE", preconditions={"p": _T}, irreversible=True),
        CommandSpec(identifier="SET_P", effects={"p": _T}),
        CommandSpec(identifier="OTHER", irreversible=True),
    ]
    case = tiny_case(
        [(0.0, "SET_P"), (1.0, "FIRE"), (2.0, "FIRE")], commands=commands, predicates={"p": _U}
    )
    result = gate_case(case)
    assert result.n_irreversible_commands == 2
    assert result.n_irreversible_executions == 2
    assert not result.refused


def test_gate_refuses_when_the_verification_comes_after() -> None:
    """The ordering the gate exists for, in four lines."""
    commands = [
        CommandSpec(identifier="FIRE", preconditions={"p": _T}, irreversible=True),
        CommandSpec(identifier="VERIFY_P", effects={"p": _T}),
    ]
    before = tiny_case(
        [(0.0, "FIRE"), (1.0, "VERIFY_P")], commands=commands, predicates={"p": _U}
    )
    after = tiny_case(
        [(0.0, "VERIFY_P"), (1.0, "FIRE")], commands=commands, predicates={"p": _U}
    )
    assert gate_case(before).refused
    assert not gate_case(after).refused


def test_gate_with_no_irreversible_command_does_not_refuse() -> None:
    assert not gate_case(tiny_case([(0.0, "SET_P")])).refused
