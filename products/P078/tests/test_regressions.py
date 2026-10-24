"""Counterexamples found by Hypothesis, pinned with literal inputs.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is
lost on a fresh clone. Every counterexample this build found is reproduced here
from literals, with a note saying what the original claim was and why it was
wrong. A test in this file failing means a defect has come back, not that a
search got lucky.
"""

from __future__ import annotations

import pytest

from cmdguard.model import (
    Case,
    CommandDictionary,
    CommandSequence,
    CommandSpec,
    EffectPolicy,
    InitialState,
    ModeMachine,
    ResourceLimits,
    SequenceStep,
    Verity,
)
from cmdguard.trace import ViolationKind, forward_execute


def _mode_gated_illegal_transition_case() -> Case:
    """The literal counterexample Hypothesis minimised to, rebuilt by hand.

    One command, ``C0``: permitted only in mode ``M1``, declaring a transition
    to ``M1``, executed from mode ``M0`` in a machine whose transition relation
    from ``M0`` is empty.
    """
    spec = CommandSpec(
        identifier="C0",
        subsystem="a",
        required_modes=("M1",),
        mode_effect="M1",
    )
    return Case(
        name="regression_mode_gated_illegal_transition",
        dictionary=CommandDictionary(
            commands={"C0": spec}, resource_limits=ResourceLimits({})
        ),
        machine=ModeMachine(modes=("M0", "M1"), transitions={"M0": (), "M1": ()}),
        initial=InitialState(mode="M0", predicates={}),
        sequence=CommandSequence((SequenceStep(t_s=0.0, command="C0"),)),
    )


def test_block_policy_can_report_fewer_violations_than_apply() -> None:
    """Hypothesis counterexample, found 2026-10-24 on this build.

    The falsified claim was that ``EffectPolicy.BLOCK`` never reports fewer
    violations than ``EffectPolicy.APPLY``, on the reasoning that blocking a
    state change can only make later preconditions harder to satisfy. It is
    wrong twice over. Here it is wrong because ``illegal_transition`` is raised
    while applying effects: under APPLY the command raises the mode gate *and*
    the forbidden transition, under BLOCK it raises only the mode gate, because
    a command that changes no state cannot make an illegal transition. The
    second way it is wrong is that suppressing an effect which sets a predicate
    false can make a later precondition *pass*.

    The behaviour is correct and is pinned rather than changed. What was fixed
    was the property, which now asserts the identity that holds: the two
    policies agree on whether the sequence is clean, and on the first violating
    step.
    """
    case = _mode_gated_illegal_transition_case()
    applied = forward_execute(case, EffectPolicy.APPLY)
    blocked = forward_execute(case, EffectPolicy.BLOCK)
    assert applied.kind_counts() == {"illegal_transition": 1, "mode_gate": 1}
    assert blocked.kind_counts() == {"mode_gate": 1}
    assert blocked.n_violations < applied.n_violations
    # The identity that does hold.
    assert applied.clean == blocked.clean
    assert min(v.step for v in applied.violations) == min(
        v.step for v in blocked.violations
    )


def test_illegal_transition_does_not_advance_the_mode() -> None:
    """The conservative reading, pinned: a forbidden transition is not credited."""
    case = _mode_gated_illegal_transition_case()
    report = forward_execute(case)
    assert report.final_mode == "M0"
    violation = next(
        v for v in report.violations if v.kind is ViolationKind.ILLEGAL_TRANSITION
    )
    assert "the transition is not applied" in violation.detail


def test_self_transition_is_never_an_illegal_transition() -> None:
    """A command declaring a transition to the mode it is already in is legal.

    Found while minimising the case above: with an empty transition relation,
    every ``mode_effect`` equal to the current mode would otherwise be reported
    as illegal, which would make a no-op a defect.
    """
    spec = CommandSpec(identifier="STAY", mode_effect="M0")
    case = Case(
        name="regression_self_transition",
        dictionary=CommandDictionary(commands={"STAY": spec}),
        machine=ModeMachine(modes=("M0", "M1"), transitions={"M0": (), "M1": ()}),
        initial=InitialState(mode="M0"),
        sequence=CommandSequence((SequenceStep(t_s=0.0, command="STAY"),)),
    )
    assert forward_execute(case).clean


def test_zero_length_hold_still_counts_at_its_own_instant() -> None:
    """A command with ``hold_s = 0`` draws at its execution instant.

    Pinned because the half-open hold interval ``[t, t)`` is empty, and an
    implementation that evaluated the ceiling only against *other* live holds
    would miss a single command that exceeds a ceiling on its own.
    """
    spec = CommandSpec(identifier="HOG", resource_draw={"power_w": 150.0}, hold_s=0.0)
    case = Case(
        name="regression_zero_hold",
        dictionary=CommandDictionary(
            commands={"HOG": spec}, resource_limits=ResourceLimits({"power_w": 100.0})
        ),
        machine=ModeMachine(modes=("M0",), transitions={"M0": ()}),
        initial=InitialState(mode="M0"),
        sequence=CommandSequence((SequenceStep(t_s=0.0, command="HOG"),)),
    )
    report = forward_execute(case)
    assert report.kind_counts() == {"resource_limit": 1}
    assert report.violations[0].shortfall == pytest.approx(50.0, abs=1e-9)


def test_undeclared_predicate_in_initial_state_is_unverified_not_false() -> None:
    """Pinned because the convenient default is the dangerous one.

    A precondition requiring ``p = false`` against an initial state that does
    not mention ``p`` must report ``precondition_unknown``, not pass. An
    implementation defaulting absent predicates to ``FALSE`` would pass it, and
    an irreversible command behind such a precondition would pass the gate.
    """
    spec = CommandSpec(
        identifier="FIRE", preconditions={"p": Verity.FALSE}, irreversible=True
    )
    case = Case(
        name="regression_absent_predicate",
        dictionary=CommandDictionary(commands={"FIRE": spec}),
        machine=ModeMachine(modes=("M0",), transitions={"M0": ()}),
        initial=InitialState(mode="M0", predicates={}),
        sequence=CommandSequence((SequenceStep(t_s=0.0, command="FIRE"),)),
    )
    report = forward_execute(case)
    assert report.kind_counts() == {"irreversible_unverified": 1}
    assert report.gate_blocking


def test_compress_gap_family_no_longer_uses_min_interval_as_a_gap() -> None:
    """Measured defect, pinned: the no-op count exposed the wrong arithmetic.

    The first version of ``cmdguard.defects.compress_gap`` took
    ``max(previous.min_gap_after_s, own.min_interval_s)`` as the spacing to
    break, and moved the step to a quarter of that after the *previous* step.
    ``min_interval_s`` is measured from the command's own previous execution,
    not from the previous step, so on a command that had not run before, the
    edit broke nothing: 18 of 40 injections were no-ops. The family now targets
    ``min_gap_after_s`` only, and this test asserts an application rate of 1 on
    the first twelve generator seeds.
    """
    from cmdguard.dataset import generate_clean_case
    from cmdguard.defects import inject

    for seed in range(12):
        _, record = inject(generate_clean_case(seed), "compress_gap", seed)
        assert record.applied, f"compress_gap was a no-op on seed {seed}"
