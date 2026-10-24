"""Hypothesis property tests over randomly declared cases.

Hand-written cases check the arithmetic the author thought of. These check
identities that must hold for every declaration, and on P071 of this mission
property tests found six equation defects that the hand-written suite did not.
Any counterexample Hypothesis finds here is pinned as a named regression test
with literal inputs in ``test_regressions.py``, because ``.hypothesis/`` is
gitignored and a falsifying example found by one run is lost on a fresh clone.

Floating point
--------------
Two of the properties are exact-arithmetic identities that floating point only
respects under conditions this module arranges deliberately:

*Resource scaling.* Multiplying every declared draw and every ceiling by the
same power of two is exact in binary floating point, and the rounding of a sum
is scale-invariant under a power of two, so the comparison
``total > limit * (1 + rel_tol)`` cannot change. The strategies therefore draw
power-of-two scale factors, never arbitrary ones.

*Time scaling.* The timing comparisons carry an absolute tolerance
(``TIME_TOL_S``) which does **not** scale with the times, so the identity holds
only when every margin is either exactly zero or far larger than the tolerance.
The strategies therefore draw times and durations from an integer grid, where
the smallest non-zero margin is 1 s against a tolerance of 1e-9 s.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cmdguard.lint import LintStrength, static_lint
from cmdguard.loader import case_from_mapping, dump_case
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
from cmdguard.ranker import involved_mask, precision_at_k, recall_at_k
from cmdguard.trace import ViolationKind, forward_execute

PREDICATES = ("p", "q", "r")
MODES = ("M0", "M1")
RESOURCES = ("power_w", "data_rate_mbps")
TIMING_KINDS = frozenset(
    {
        ViolationKind.TIMING_GAP.value,
        ViolationKind.TIMING_INTERVAL.value,
        ViolationKind.MUTUAL_EXCLUSION.value,
    }
)

SETTINGS = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

_verity = st.sampled_from([Verity.TRUE, Verity.FALSE])
_grid_duration = st.sampled_from([0, 1, 2, 5, 10, 30, 60, 120])
_grid_draw = st.sampled_from([0.0, 0.5, 4.0, 10.0, 25.0, 40.0, 60.0])
_grid_limit = st.sampled_from([0.5, 10.0, 40.0, 100.0])
SCALES = st.sampled_from([0.25, 0.5, 2.0, 4.0])


@st.composite
def command_specs(draw: st.DrawFn, count: int) -> list[CommandSpec]:
    """A list of ``count`` distinct commands over the small declaration space."""
    specs: list[CommandSpec] = []
    for index in range(count):
        preconditions = {
            name: draw(_verity)
            for name in PREDICATES
            if draw(st.booleans())
        }
        effects = {
            name: draw(st.sampled_from(list(Verity)))
            for name in PREDICATES
            if draw(st.booleans())
        }
        resource_draw = {
            key: draw(_grid_draw) for key in RESOURCES if draw(st.booleans())
        }
        specs.append(
            CommandSpec(
                identifier=f"C{index}",
                subsystem=draw(st.sampled_from(["a", "b"])),
                required_modes=tuple(
                    mode for mode in MODES if draw(st.booleans())
                ),
                preconditions=preconditions,
                effects=effects,
                mode_effect=draw(st.sampled_from([None, *MODES])),
                resource_draw={k: v for k, v in resource_draw.items() if v > 0.0},
                hold_s=float(draw(_grid_duration)),
                min_interval_s=float(draw(_grid_duration)),
                min_gap_after_s=float(draw(_grid_duration)),
                irreversible=draw(st.booleans()),
                exclusion_group=draw(st.sampled_from([None, "g"])),
            )
        )
    return specs


@st.composite
def random_cases(draw: st.DrawFn) -> Case:
    """A randomly declared, structurally valid case on the integer time grid."""
    specs = draw(command_specs(draw(st.integers(min_value=1, max_value=4))))
    identifiers = [spec.identifier for spec in specs]
    limits = {key: draw(_grid_limit) for key in RESOURCES if draw(st.booleans())}
    increments = draw(
        st.lists(st.integers(min_value=0, max_value=200), min_size=0, max_size=8)
    )
    times: list[float] = []
    cursor = 0
    for increment in increments:
        cursor += increment
        times.append(float(cursor))
    commands = draw(
        st.lists(
            st.sampled_from(identifiers), min_size=len(times), max_size=len(times)
        )
    ) if times else []
    predicates = {
        name: draw(st.sampled_from(list(Verity)))
        for name in PREDICATES
        if draw(st.booleans())
    }
    return Case(
        name="prop",
        dictionary=CommandDictionary(
            commands={spec.identifier: spec for spec in specs},
            resource_limits=ResourceLimits(limits),
        ),
        machine=ModeMachine(
            modes=MODES,
            transitions={
                "M0": tuple(m for m in MODES if draw(st.booleans())),
                "M1": tuple(m for m in MODES if draw(st.booleans())),
            },
        ),
        initial=InitialState(mode=draw(st.sampled_from(MODES)), predicates=predicates),
        sequence=CommandSequence(
            tuple(
                SequenceStep(t_s=t, command=c) for t, c in zip(times, commands, strict=True)
            )
        ),
    )


def _identities(case: Case, policy: EffectPolicy = EffectPolicy.APPLY) -> set[tuple[int, str]]:
    return {(v.step, v.kind.value) for v in forward_execute(case, policy).violations}


@SETTINGS
@given(random_cases())
def test_forward_execution_is_deterministic(case: Case) -> None:
    """Required by the build rules and asserted over arbitrary declarations."""
    prints = {forward_execute(case).fingerprint() for _ in range(3)}
    assert len(prints) == 1


@SETTINGS
@given(random_cases())
def test_violations_are_sorted_and_in_range(case: Case) -> None:
    report = forward_execute(case)
    keys = [v.key() for v in report.violations]
    assert keys == sorted(keys)
    assert all(0 <= v.step < len(case.sequence) for v in report.violations)
    assert all(v.t_s == case.sequence.steps[v.step].t_s for v in report.violations)


@SETTINGS
@given(random_cases())
def test_one_snapshot_per_step(case: Case) -> None:
    report = forward_execute(case)
    assert len(report.snapshots) == len(case.sequence)
    assert [s.step for s in report.snapshots] == list(range(len(case.sequence)))


@SETTINGS
@given(random_cases())
def test_prefix_violations_are_exactly_the_early_violations(case: Case) -> None:
    """A later step cannot change what an earlier step did.

    The forward walk is causal, so truncating the sequence must leave the
    violations of the retained steps exactly as they were. This is the property
    that would break first if any check reached forward in the sequence.
    """
    from dataclasses import replace

    full = _identities(case)
    for cut in range(len(case.sequence) + 1):
        prefix = replace(case, sequence=CommandSequence(case.sequence.steps[:cut]))
        assert _identities(prefix) == {key for key in full if key[0] < cut}


@SETTINGS
@given(random_cases(), SCALES)
def test_resource_violations_are_invariant_under_power_of_two_scaling(
    case: Case, scale: float
) -> None:
    """Scaling every draw and every ceiling by 2^k changes no resource finding.

    Exact in binary floating point: see the module docstring.
    """
    from dataclasses import replace

    scaled_commands = {
        identifier: replace(
            spec,
            resource_draw={k: v * scale for k, v in spec.resource_draw.items()},
        )
        for identifier, spec in case.dictionary.commands.items()
    }
    scaled = replace(
        case,
        dictionary=CommandDictionary(
            commands=scaled_commands,
            resource_limits=ResourceLimits(
                {k: v * scale for k, v in case.dictionary.resource_limits.limits.items()}
            ),
        ),
    )
    kind = ViolationKind.RESOURCE_LIMIT.value
    before = {key for key in _identities(case) if key[1] == kind}
    after = {key for key in _identities(scaled) if key[1] == kind}
    assert before == after


@SETTINGS
@given(random_cases(), st.sampled_from([2.0, 4.0]))
def test_timing_violations_are_invariant_under_power_of_two_time_scaling(
    case: Case, scale: float
) -> None:
    """Scaling every time and every declared duration by 2^k changes no timing
    finding.

    Exact on the integer time grid the strategy draws from: the smallest
    non-zero margin is 1 s against an absolute tolerance of 1e-9 s, so no
    comparison sits inside the tolerance band.
    """
    from dataclasses import replace

    scaled_commands = {
        identifier: replace(
            spec,
            hold_s=spec.hold_s * scale,
            min_interval_s=spec.min_interval_s * scale,
            min_gap_after_s=spec.min_gap_after_s * scale,
        )
        for identifier, spec in case.dictionary.commands.items()
    }
    scaled = replace(
        case,
        dictionary=CommandDictionary(
            commands=scaled_commands, resource_limits=case.dictionary.resource_limits
        ),
        sequence=CommandSequence(
            tuple(replace(s, t_s=s.t_s * scale) for s in case.sequence.steps)
        ),
    )
    before = {key for key in _identities(case) if key[1] in TIMING_KINDS}
    after = {key for key in _identities(scaled) if key[1] in TIMING_KINDS}
    assert before == after


@SETTINGS
@given(random_cases())
def test_both_effect_policies_agree_on_whether_the_sequence_is_clean(case: Case) -> None:
    """The two policies always agree on the clean/not-clean verdict.

    Up to the first violating step the two executions are identical, and if
    there is no violating step then every effect is applied under both
    policies, so the sequences cannot disagree about emptiness. This matters
    because it makes the negative control policy-independent: a clean case is
    clean under both readings of what a rejected command does.

    Note what is **not** asserted. An earlier version of this test claimed that
    BLOCK never reports fewer violations than APPLY. Hypothesis falsified it in
    seven examples, and the counterexample is pinned in ``test_regressions.py``
    as ``test_block_policy_can_report_fewer_violations_than_apply``: a command
    gated to a mode it is not in, which also declares a transition the machine
    forbids, raises two violations under APPLY and one under BLOCK, because a
    command that changes no state cannot make an illegal transition. The claim
    was wrong in both directions -- suppressing a falsifying effect can also
    make a later precondition pass -- and the identity below is what actually
    holds.
    """
    assert forward_execute(case, EffectPolicy.APPLY).clean == forward_execute(
        case, EffectPolicy.BLOCK
    ).clean


@SETTINGS
@given(random_cases())
def test_both_effect_policies_agree_on_the_first_violating_step(case: Case) -> None:
    """Identical up to and including the checks of the first violating step.

    The only finding the two policies may differ on at that step is
    ``illegal_transition``, which is raised while applying effects and is
    therefore not raised at all when the effects are blocked.
    """
    apply_report = forward_execute(case, EffectPolicy.APPLY)
    block_report = forward_execute(case, EffectPolicy.BLOCK)
    if not apply_report.violations:
        assert not block_report.violations
        return
    first = min(v.step for v in apply_report.violations)
    assert first == min(v.step for v in block_report.violations)
    transition = ViolationKind.ILLEGAL_TRANSITION.value
    left = {
        (v.step, v.kind.value)
        for v in apply_report.violations
        if v.step == first and v.kind.value != transition
    }
    right = {
        (v.step, v.kind.value)
        for v in block_report.violations
        if v.step == first and v.kind.value != transition
    }
    assert left == right


@SETTINGS
@given(random_cases())
def test_unconstrained_declarations_never_violate(case: Case) -> None:
    """Strip every constraint and the walk must find nothing but the gate-free
    truth: no preconditions, no gating, no timing, no draws, no exclusions."""
    from dataclasses import replace

    bare = {
        identifier: CommandSpec(identifier=identifier, effects=dict(spec.effects))
        for identifier, spec in case.dictionary.commands.items()
    }
    stripped = replace(
        case,
        dictionary=CommandDictionary(commands=bare, resource_limits=ResourceLimits({})),
    )
    assert forward_execute(stripped).clean


@SETTINGS
@given(random_cases())
def test_per_command_lint_is_invariant_under_reordering(case: Case) -> None:
    """The baseline's defining weakness, as a property over all declarations.

    Reversing the command column leaves the per-command lint's multiset of
    (command, kind) findings unchanged, because the lint has no sequence state.
    """
    from collections import Counter
    from dataclasses import replace

    reversed_case = replace(
        case,
        sequence=CommandSequence(
            tuple(
                SequenceStep(t_s=step.t_s, command=command)
                for step, command in zip(
                    case.sequence.steps,
                    [s.command for s in reversed(case.sequence.steps)],
                    strict=True,
                )
            )
        ),
    )
    left = Counter(
        (f.command, f.kind) for f in static_lint(case, LintStrength.PER_COMMAND).findings
    )
    right = Counter(
        (f.command, f.kind)
        for f in static_lint(reversed_case, LintStrength.PER_COMMAND).findings
    )
    assert left == right


@SETTINGS
@given(random_cases())
def test_yaml_round_trip_preserves_the_finding_set(case: Case) -> None:
    again = case_from_mapping(yaml.safe_load(dump_case(case)))
    assert forward_execute(case).fingerprint() == forward_execute(again).fingerprint()


@SETTINGS
@given(random_cases())
def test_involved_steps_are_in_range_and_sorted(case: Case) -> None:
    report = forward_execute(case)
    involved = report.involved_steps()
    assert list(involved) == sorted(set(involved))
    assert all(0 <= step < len(case.sequence) for step in involved)
    mask = involved_mask(report, len(case.sequence))
    assert mask.sum() == len(involved)


@SETTINGS
@given(random_cases(), st.integers(min_value=1, max_value=8))
def test_precision_and_recall_at_k_are_fractions(case: Case, k: int) -> None:
    report = forward_execute(case)
    labels = involved_mask(report, len(case.sequence))
    order = list(range(len(case.sequence)))
    for value in (precision_at_k(order, labels, k), recall_at_k(order, labels, k)):
        assert np.isnan(value) or 0.0 <= value <= 1.0


@SETTINGS
@given(random_cases())
def test_gate_refuses_exactly_when_a_gate_blocking_kind_is_present(case: Case) -> None:
    from cmdguard.gate import gate_case

    report = forward_execute(case)
    assert gate_case(case).refused == bool(report.gate_blocking)


def test_hypothesis_is_actually_installed() -> None:
    """A guard: this file is worthless if the plugin silently did not load."""
    import hypothesis

    assert hypothesis.__version__
    assert pytest is not None
