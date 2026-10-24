"""Baseline 2, including the measurement of what it structurally cannot see."""

from __future__ import annotations

import pytest
from conftest import tiny_case

from cmdguard.lint import LintStrength, static_lint
from cmdguard.model import CommandSpec, Verity
from cmdguard.trace import ORDERING_KINDS, ViolationKind, forward_execute

_T, _F, _U = Verity.TRUE, Verity.FALSE, Verity.UNKNOWN


def test_lint_finds_the_undeclared_command() -> None:
    case = tiny_case([(0.0, "GHOST")])
    report = static_lint(case)
    assert report.kind_counts() == {"undeclared_command": 1}


def test_lint_cannot_see_an_ordering_defect() -> None:
    """The defining weakness, asserted rather than asserted about.

    SET_P at step 0 makes NEED_P at step 1 legal. The lint evaluates NEED_P
    against the declared initial state, where p is false, so it reports a
    violation the forward walk does not confirm -- and it is blind in the other
    direction too, which the next test shows.
    """
    case = tiny_case([(0.0, "SET_P"), (1.0, "NEED_P")], predicates={"p": _F})
    forward = forward_execute(case)
    lint = static_lint(case)
    assert forward.n_violations == 0
    assert lint.n_findings == 1
    assert lint.findings[0].step == 1


def test_lint_misses_a_real_precondition_defect() -> None:
    """p is true initially and a command falsifies it. The lint sees nothing."""
    commands = [
        CommandSpec(identifier="CLEAR_P", effects={"p": _F}),
        CommandSpec(identifier="NEED_P", preconditions={"p": _T}),
    ]
    case = tiny_case([(0.0, "CLEAR_P"), (1.0, "NEED_P")], commands=commands, predicates={"p": _T})
    assert forward_execute(case).kind_counts() == {"precondition_false": 1}
    assert static_lint(case).clean


@pytest.mark.parametrize("strength", list(LintStrength))
def test_lint_cannot_see_mode_exclusion_or_accumulated_resource(
    cases: dict, strength: LintStrength
) -> None:
    """Neither strength finds a mutual exclusion or an accumulated overdraw."""
    lint = static_lint(cases["resource_and_exclusion"], strength)
    kinds = set(lint.kind_counts())
    assert "mutual_exclusion" not in kinds
    assert "resource_limit" not in kinds


def test_with_timing_finds_the_gap_and_the_interval(cases: dict) -> None:
    case = cases["resource_and_exclusion"]
    weak = static_lint(case, LintStrength.PER_COMMAND)
    strong = static_lint(case, LintStrength.WITH_TIMING)
    assert "timing_gap" not in weak.kind_counts()
    assert strong.kind_counts() == {"timing_gap": 1, "timing_interval": 1}


def test_with_timing_reproduces_the_forward_timing_findings_exactly(cases: dict) -> None:
    """The timing checks are a function of the time column, so they must agree."""
    case = cases["resource_and_exclusion"]
    forward = forward_execute(case)
    lint = static_lint(case, LintStrength.WITH_TIMING)
    timing_kinds = {ViolationKind.TIMING_GAP.value, ViolationKind.TIMING_INTERVAL.value}
    forward_timing = {
        (v.step, v.kind.value) for v in forward.violations if v.kind.value in timing_kinds
    }
    lint_timing = {ident for ident in lint.identities() if ident[1] in timing_kinds}
    assert forward_timing == lint_timing


def test_lint_single_command_resource_check_fires() -> None:
    commands = [CommandSpec(identifier="HOG", resource_draw={"power_w": 200.0})]
    case = tiny_case([(0.0, "HOG")], commands=commands, limits={"power_w": 100.0})
    assert static_lint(case).kind_counts() == {"resource_limit": 1}


def test_lint_mode_gate_uses_only_the_initial_mode() -> None:
    commands = [
        CommandSpec(identifier="GO_SCIENCE", mode_effect="SCIENCE"),
        CommandSpec(identifier="SCIENCE_ONLY", required_modes=("SCIENCE",)),
    ]
    case = tiny_case([(0.0, "GO_SCIENCE"), (1.0, "SCIENCE_ONLY")], commands=commands)
    assert forward_execute(case).clean
    assert static_lint(case).kind_counts() == {"mode_gate": 1}


def test_lint_distinguishes_irreversible_unverified() -> None:
    commands = [CommandSpec(identifier="FIRE", preconditions={"p": _T}, irreversible=True)]
    case = tiny_case([(0.0, "FIRE")], commands=commands, predicates={"p": _U})
    assert static_lint(case).kind_counts() == {"irreversible_unverified": 1}


def test_lint_identity_matches_the_trace_identity(cases: dict) -> None:
    case = cases["late_payload_enable"]
    forward = {(v.step, v.kind.value) for v in forward_execute(case).violations}
    lint = static_lint(case).identities()
    assert forward & lint == {(6, "precondition_false")}


def test_lint_rejects_a_non_enum_strength() -> None:
    with pytest.raises(TypeError, match="must be a LintStrength"):
        static_lint(tiny_case([]), "per_command")  # type: ignore[arg-type]


def test_ordering_kinds_are_exactly_the_sequence_aware_ones() -> None:
    """A guard on the taxonomy: the miss-rate split is computed from this tuple."""
    assert set(ORDERING_KINDS) == {
        ViolationKind.MODE_GATE,
        ViolationKind.ILLEGAL_TRANSITION,
        ViolationKind.TIMING_INTERVAL,
        ViolationKind.TIMING_GAP,
        ViolationKind.MUTUAL_EXCLUSION,
        ViolationKind.RESOURCE_LIMIT,
    }


def test_lint_is_order_invariant_in_its_per_command_strength() -> None:
    """Reversing the sequence changes nothing. That is the baseline's nature."""
    forward_order = tiny_case([(0.0, "SET_P"), (1.0, "NEED_P")], predicates={"p": _F})
    reverse_order = tiny_case([(0.0, "NEED_P"), (1.0, "SET_P")], predicates={"p": _F})
    left = {(f.command, f.kind) for f in static_lint(forward_order).findings}
    right = {(f.command, f.kind) for f in static_lint(reverse_order).findings}
    assert left == right
