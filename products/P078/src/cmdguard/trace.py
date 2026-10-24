"""Baseline 1: forward symbolic execution of a command sequence. Ground truth.

A linear command sequence has no branches. Symbolic execution of a program with
one path enumerates that path exactly (King 1976, "Symbolic execution and
program testing", *Communications of the ACM* 19(7), 385-394), so this walk is
not an approximation of the sequence's behaviour against the declared model: it
*is* the sequence's behaviour against the declared model. Everything else in
this package is measured against it.

What makes this module worth more than a loop over preconditions is the
provenance. Each violation carries the chain of earlier steps that produced the
state it tripped over. "Step 47 PAYLOAD_SCAN is illegal" tells a reviewer to go
and read the sequence. "Step 47 PAYLOAD_SCAN requires payload_enabled=true but
it is false; it was set false at step 12 t=430.0 s by PAYLOAD_SAFE and no later
step set it true" tells them what to change. The second is a diff; the first is
a task.

Cost
----
One pass over the sequence. Per step the work is proportional to the number of
preconditions, the number of active holds and the number of declared resource
keys, all small and none growing with sequence length, so the walk is linear in
the number of steps with a small constant. Measured timings are in
``validation/validate_determinism_output.txt`` and in the README table that is
explicitly declared unstable.

Floating point
--------------
Timing comparisons are threshold comparisons on sums of floating-point times,
so they are made with an absolute tolerance (:data:`TIME_TOL_S`) rather than
with a bare ``<``. A gap that is short by less than a nanosecond is not a
defect in anybody's command sequence, and a strict comparison on
``t - t_prev < required`` flips on the last bit of a subtraction. The same
applies to resource totals (:data:`RESOURCE_REL_TOL`): a draw that exceeds its
ceiling by one part in 10^9 is reported as within the ceiling, and the
docstrings say so rather than claiming exactness the arithmetic cannot give.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

from .model import (
    Case,
    CommandSpec,
    EffectPolicy,
    ModeMachine,
    Verity,
)

__all__ = [
    "ViolationKind",
    "Provenance",
    "Violation",
    "StateSnapshot",
    "CheckReport",
    "forward_execute",
    "TIME_TOL_S",
    "RESOURCE_REL_TOL",
    "GATE_BLOCKING_KINDS",
]

#: Absolute tolerance, in seconds, on every timing threshold comparison.
TIME_TOL_S = 1e-9

#: Relative tolerance on every resource ceiling comparison.
RESOURCE_REL_TOL = 1e-9


class ViolationKind(enum.Enum):
    """The violation taxonomy. Stable strings; the report and CLI print these."""

    UNDECLARED_COMMAND = "undeclared_command"
    MODE_GATE = "mode_gate"
    ILLEGAL_TRANSITION = "illegal_transition"
    PRECONDITION_FALSE = "precondition_false"
    PRECONDITION_UNKNOWN = "precondition_unknown"
    IRREVERSIBLE_UNVERIFIED = "irreversible_unverified"
    TIMING_INTERVAL = "timing_interval"
    TIMING_GAP = "timing_gap"
    MUTUAL_EXCLUSION = "mutual_exclusion"
    RESOURCE_LIMIT = "resource_limit"


#: Kinds the irreversible-command gate refuses on. See :mod:`cmdguard.gate`.
GATE_BLOCKING_KINDS = (ViolationKind.IRREVERSIBLE_UNVERIFIED,)

#: Kinds that only a forward walk through the sequence can find. A per-command
#: static lint has no sequence state and therefore cannot find any of these
#: except by accident; the measurement of how often it does is in
#: :mod:`cmdguard.benchmark`.
ORDERING_KINDS = (
    ViolationKind.MODE_GATE,
    ViolationKind.ILLEGAL_TRANSITION,
    ViolationKind.TIMING_INTERVAL,
    ViolationKind.TIMING_GAP,
    ViolationKind.MUTUAL_EXCLUSION,
    ViolationKind.RESOURCE_LIMIT,
)


@dataclass(frozen=True)
class Provenance:
    """Where a piece of state came from.

    ``step`` is ``None`` for the declared initial state, which is the honest
    answer when nothing in the sequence touched the predicate: the reviewer has
    to go and look at the declaration, not at the sequence.
    """

    step: int | None
    command: str | None
    t_s: float | None
    note: str

    def render(self) -> str:
        """One line naming the step, command, time and reason."""
        if self.step is None:
            return f"declared initial state: {self.note}"
        return f"step {self.step} t={self.t_s:.1f} s {self.command}: {self.note}"


@dataclass(frozen=True)
class Violation:
    """One violation, with the state trace that reaches it.

    ``trace`` is ordered oldest first and ends with the step that tripped. It is
    the deliverable: a violation without it is not actionable.
    """

    step: int
    t_s: float
    command: str
    kind: ViolationKind
    detail: str
    trace: tuple[Provenance, ...] = ()
    shortfall: float | None = None

    @property
    def gate_blocking(self) -> bool:
        """True when the irreversible-command gate must refuse on this."""
        return self.kind in GATE_BLOCKING_KINDS

    @property
    def ordering_defect(self) -> bool:
        """True when only a sequence-aware check can find this violation."""
        return self.kind in ORDERING_KINDS

    def render(self, indent: str = "  ") -> str:
        """Multi-line human-readable rendering, trace included."""
        head = f"step {self.step:>3d} t={self.t_s:>9.1f} s  {self.command:<22s} {self.kind.value}"
        lines = [head, f"{indent}{self.detail}"]
        if self.shortfall is not None:
            lines.append(f"{indent}shortfall: {self.shortfall:.6g}")
        for entry in self.trace:
            lines.append(f"{indent}because {entry.render()}")
        return "\n".join(lines)

    def key(self) -> tuple[int, str, str]:
        """Deterministic sort key: step, then kind, then detail."""
        return (self.step, self.kind.value, self.detail)


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable record of the state reached after a step. One per step.

    Used by the report, the timeline figure and the trace tests. Holding a
    snapshot per step is what lets a reviewer answer "what was the mode at step
    47" without re-running anything, and the memory cost is linear and small.
    """

    step: int
    t_s: float
    command: str
    mode: str
    predicates: Mapping[str, Verity]
    active_draw: Mapping[str, float]
    violations: tuple[Violation, ...]


@dataclass
class _Hold:
    """A command's resource draw and exclusion claim, live until ``t_end``."""

    step: int
    command: str
    t_start: float
    t_end: float
    draw: Mapping[str, float]
    exclusion_group: str | None


@dataclass
class _State:
    """Mutable executor state. Not public; the public record is StateSnapshot."""

    mode: str
    predicates: dict[str, Verity]
    predicate_source: dict[str, Provenance]
    mode_source: Provenance
    last_exec_t: dict[str, float] = field(default_factory=dict)
    last_exec_step: dict[str, int] = field(default_factory=dict)
    holds: list[_Hold] = field(default_factory=list)
    prev: tuple[int, str, float, float] | None = None  # step, command, t, min_gap_after_s


@dataclass(frozen=True)
class CheckReport:
    """Result of a forward execution: every violation, every snapshot.

    ``clean`` says that no violation was found against the declared model. It
    does **not** say the sequence is safe to uplink, and the renderer in
    :mod:`cmdguard.report` prints that distinction every time, because the
    distinction is the entire epistemic content of this package.
    """

    case_name: str
    violations: tuple[Violation, ...]
    snapshots: tuple[StateSnapshot, ...]
    effect_policy: EffectPolicy
    final_mode: str
    final_predicates: Mapping[str, Verity]

    @property
    def clean(self) -> bool:
        """True when no violation was found against the declared model."""
        return not self.violations

    @property
    def n_violations(self) -> int:
        """Number of violations found."""
        return len(self.violations)

    def of_kind(self, kind: ViolationKind) -> tuple[Violation, ...]:
        """Violations of one kind, in step order."""
        return tuple(v for v in self.violations if v.kind is kind)

    def kind_counts(self) -> dict[str, int]:
        """Violation count per kind, keyed by the kind's stable string."""
        counts: dict[str, int] = {}
        for violation in self.violations:
            counts[violation.kind.value] = counts.get(violation.kind.value, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def gate_blocking(self) -> tuple[Violation, ...]:
        """Violations the irreversible-command gate must refuse on."""
        return tuple(v for v in self.violations if v.gate_blocking)

    @property
    def ordering_defects(self) -> tuple[Violation, ...]:
        """Violations only a sequence-aware check can find."""
        return tuple(v for v in self.violations if v.ordering_defect)

    def involved_steps(self) -> tuple[int, ...]:
        """Steps that are either a violating step or named in a violation trace.

        This is the label the risk ranker in :mod:`cmdguard.ranker` is trained
        and scored against: the set of steps a reviewer would have to read in
        order to understand every violation in the sequence.
        """
        involved: set[int] = set()
        for violation in self.violations:
            involved.add(violation.step)
            for entry in violation.trace:
                if entry.step is not None:
                    involved.add(entry.step)
        return tuple(sorted(involved))

    def fingerprint(self) -> str:
        """Order-insensitive digest of the finding set, for determinism tests.

        Built from the violation tuple rather than from object identity, so two
        runs agreeing on this string agree on every step, kind, detail and
        trace entry that was reported.
        """
        import hashlib

        parts = [self.case_name, self.effect_policy.value, self.final_mode]
        for violation in self.violations:
            parts.append(
                "|".join(
                    [
                        str(violation.step),
                        f"{violation.t_s:.9f}",
                        violation.command,
                        violation.kind.value,
                        violation.detail,
                        ";".join(e.render() for e in violation.trace),
                        "none" if violation.shortfall is None else f"{violation.shortfall:.9e}",
                    ]
                )
            )
        for name in sorted(self.final_predicates):
            parts.append(f"{name}={self.final_predicates[name].value}")
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:32]


def _expire_holds(holds: list[_Hold], t_s: float) -> list[_Hold]:
    """Drop holds that ended at or before ``t_s``.

    A hold ending exactly at ``t_s`` is expired, not active: the interval is
    half-open ``[t_start, t_end)``. The convention matters for the
    hand-calculated exclusion cases in ``tests/test_trace.py`` and is stated
    there.
    """
    return [h for h in holds if h.t_end > t_s + TIME_TOL_S]


def _active_draw(holds: Iterable[_Hold]) -> dict[str, float]:
    """Sum of held draws per resource key."""
    total: dict[str, float] = {}
    for hold in holds:
        for key in sorted(hold.draw):
            total[key] = total.get(key, 0.0) + float(hold.draw[key])
    return dict(sorted(total.items()))


def _predicate_reason(name: str, required: Verity, actual: Verity) -> str:
    """The detail line for a failed precondition."""
    return (
        f"requires {name}={required.value} but it is {actual.value} at this step"
        if actual is not Verity.UNKNOWN
        else f"requires {name}={required.value} but it has not been verified"
    )


def _no_later_setter_note(name: str, required: Verity) -> str:
    """The sentence that turns a violation into a diff."""
    return f"no step between that point and this one set {name} to {required.value}"


def forward_execute(
    case: Case,
    effect_policy: EffectPolicy = EffectPolicy.APPLY,
) -> CheckReport:
    """Walk ``case.sequence`` through the mode machine and report every violation.

    Exact for a single linear sequence against the declared model: there is one
    path and it is enumerated. Not exact about anything else, and in particular
    not a statement that the sequence is safe to uplink.

    Parameters
    ----------
    case:
        Dictionary, mode machine, initial state and candidate sequence.
    effect_policy:
        :class:`~cmdguard.model.EffectPolicy`. ``APPLY`` (default) applies a
        violating command's declared effects anyway, so independent later
        defects are still found. ``BLOCK`` applies nothing from a command that
        violated, modelling an onboard rejection.

    Returns
    -------
    CheckReport
        Violations in step order with their provenance traces, plus one
        :class:`StateSnapshot` per step.

    Notes
    -----
    Order of checks within a step is fixed and is part of the contract, because
    the first violation a step raises is the one a reviewer reads first:
    undeclared command, timing, mode gate, preconditions (irreversible first),
    mutual exclusion, resource ceiling. Effects are applied last.
    """
    if not isinstance(effect_policy, EffectPolicy):
        raise TypeError(f"effect_policy must be an EffectPolicy, got {type(effect_policy)!r}")

    machine: ModeMachine = case.machine
    state = _State(
        mode=case.initial.mode,
        predicates={k: v for k, v in sorted(case.initial.predicates.items())},
        predicate_source={
            name: Provenance(None, None, None, f"{name}={value.value}")
            for name, value in sorted(case.initial.predicates.items())
        },
        mode_source=Provenance(None, None, None, f"mode={case.initial.mode}"),
    )

    violations: list[Violation] = []
    snapshots: list[StateSnapshot] = []

    for index, step in enumerate(case.sequence.steps):
        t_s = float(step.t_s)
        step_violations: list[Violation] = []
        spec = case.dictionary.get(step.command)

        state.holds = _expire_holds(state.holds, t_s)

        if spec is None:
            step_violations.append(
                Violation(
                    step=index,
                    t_s=t_s,
                    command=step.command,
                    kind=ViolationKind.UNDECLARED_COMMAND,
                    detail=(
                        f"command {step.command!r} is not in the declared dictionary, so "
                        "nothing about its preconditions, draw or effects is known and the "
                        "state after this step is the state before it"
                    ),
                    trace=(Provenance(None, None, None, "dictionary declares no such command"),),
                )
            )
            violations.extend(step_violations)
            snapshots.append(
                StateSnapshot(
                    step=index,
                    t_s=t_s,
                    command=step.command,
                    mode=state.mode,
                    predicates=dict(sorted(state.predicates.items())),
                    active_draw=_active_draw(state.holds),
                    violations=tuple(step_violations),
                )
            )
            state.prev = (index, step.command, t_s, 0.0)
            continue

        step_violations.extend(_check_timing(index, t_s, spec, state))
        step_violations.extend(_check_mode_gate(index, t_s, spec, state))
        step_violations.extend(_check_preconditions(index, t_s, spec, state))
        step_violations.extend(_check_exclusion(index, t_s, spec, state, case))
        step_violations.extend(_check_resources(index, t_s, spec, state, case))

        if effect_policy is EffectPolicy.APPLY or not step_violations:
            step_violations.extend(_apply_effects(index, t_s, spec, state, machine))

        if spec.hold_s > 0.0 or spec.resource_draw or spec.exclusion_group is not None:
            state.holds.append(
                _Hold(
                    step=index,
                    command=spec.identifier,
                    t_start=t_s,
                    t_end=t_s + float(spec.hold_s),
                    draw=dict(spec.resource_draw),
                    exclusion_group=spec.exclusion_group,
                )
            )

        state.last_exec_t[spec.identifier] = t_s
        state.last_exec_step[spec.identifier] = index
        state.prev = (index, spec.identifier, t_s, float(spec.min_gap_after_s))

        ordered = tuple(sorted(step_violations, key=lambda v: v.key()))
        violations.extend(ordered)
        snapshots.append(
            StateSnapshot(
                step=index,
                t_s=t_s,
                command=step.command,
                mode=state.mode,
                predicates=dict(sorted(state.predicates.items())),
                active_draw=_active_draw(state.holds),
                violations=ordered,
            )
        )

    return CheckReport(
        case_name=case.name,
        violations=tuple(sorted(violations, key=lambda v: v.key())),
        snapshots=tuple(snapshots),
        effect_policy=effect_policy,
        final_mode=state.mode,
        final_predicates=dict(sorted(state.predicates.items())),
    )


def _check_timing(index: int, t_s: float, spec: CommandSpec, state: _State) -> list[Violation]:
    """Minimum inter-command gap and minimum same-command interval.

    Both are ``elapsed + tol < required`` comparisons, never ``elapsed <
    required``: see :data:`TIME_TOL_S`. The shortfall reported is
    ``required - elapsed`` in seconds, which is the number an operator edits the
    sequence by.
    """
    found: list[Violation] = []

    if state.prev is not None:
        prev_step, prev_command, prev_t, required_gap = state.prev
        elapsed = t_s - prev_t
        if required_gap > 0.0 and elapsed + TIME_TOL_S < required_gap:
            found.append(
                Violation(
                    step=index,
                    t_s=t_s,
                    command=spec.identifier,
                    kind=ViolationKind.TIMING_GAP,
                    detail=(
                        f"executes {elapsed:.3f} s after the previous command, which declares "
                        f"min_gap_after_s={required_gap:.3f}"
                    ),
                    trace=(
                        Provenance(
                            prev_step,
                            prev_command,
                            prev_t,
                            f"declares min_gap_after_s={required_gap:.3f} and ran at "
                            f"t={prev_t:.1f} s",
                        ),
                    ),
                    shortfall=required_gap - elapsed,
                )
            )

    last_t = state.last_exec_t.get(spec.identifier)
    if last_t is not None and spec.min_interval_s > 0.0:
        elapsed = t_s - last_t
        if elapsed + TIME_TOL_S < spec.min_interval_s:
            found.append(
                Violation(
                    step=index,
                    t_s=t_s,
                    command=spec.identifier,
                    kind=ViolationKind.TIMING_INTERVAL,
                    detail=(
                        f"repeats {elapsed:.3f} s after its own previous execution, which is "
                        f"shorter than min_interval_s={spec.min_interval_s:.3f}"
                    ),
                    trace=(
                        Provenance(
                            state.last_exec_step[spec.identifier],
                            spec.identifier,
                            last_t,
                            "previous execution of the same command",
                        ),
                    ),
                    shortfall=spec.min_interval_s - elapsed,
                )
            )
    return found


def _check_mode_gate(index: int, t_s: float, spec: CommandSpec, state: _State) -> list[Violation]:
    """Mode gating. The trace names the step that put the vehicle in this mode."""
    if not spec.required_modes or state.mode in spec.required_modes:
        return []
    permitted = ", ".join(spec.required_modes)
    return [
        Violation(
            step=index,
            t_s=t_s,
            command=spec.identifier,
            kind=ViolationKind.MODE_GATE,
            detail=(
                f"is permitted only in mode(s) [{permitted}] but the vehicle is in "
                f"{state.mode}"
            ),
            trace=(
                state.mode_source,
                Provenance(
                    index,
                    spec.identifier,
                    t_s,
                    f"no step between that point and this one left {state.mode} for one of "
                    f"[{permitted}]",
                ),
            ),
        )
    ]


def _check_preconditions(
    index: int, t_s: float, spec: CommandSpec, state: _State
) -> list[Violation]:
    """Declared preconditions, three-valued.

    An irreversible command with an unverified precondition raises
    :attr:`ViolationKind.IRREVERSIBLE_UNVERIFIED` rather than
    ``PRECONDITION_UNKNOWN``. That is the distinction the gate in
    :mod:`cmdguard.gate` refuses on, and collapsing the two would make the gate
    unimplementable.
    """
    found: list[Violation] = []
    for name in sorted(spec.preconditions):
        required = spec.preconditions[name]
        actual = state.predicates.get(name, Verity.UNKNOWN)
        if actual is required:
            continue
        source = state.predicate_source.get(
            name, Provenance(None, None, None, f"{name} is not declared in the initial state")
        )
        trace = (
            source,
            Provenance(index, spec.identifier, t_s, _no_later_setter_note(name, required)),
        )
        if actual is Verity.UNKNOWN and spec.irreversible:
            kind = ViolationKind.IRREVERSIBLE_UNVERIFIED
            detail = (
                f"is declared irreversible and requires {name}={required.value}, which has "
                "not been verified at this step; an irreversible command whose precondition "
                "is unverified is the gate-blocking case"
            )
        elif actual is Verity.UNKNOWN:
            kind = ViolationKind.PRECONDITION_UNKNOWN
            detail = _predicate_reason(name, required, actual)
        else:
            kind = ViolationKind.PRECONDITION_FALSE
            detail = _predicate_reason(name, required, actual)
        found.append(
            Violation(
                step=index,
                t_s=t_s,
                command=spec.identifier,
                kind=kind,
                detail=detail,
                trace=trace,
            )
        )
    return found


def _check_exclusion(
    index: int, t_s: float, spec: CommandSpec, state: _State, case: Case
) -> list[Violation]:
    """Mutual exclusion over live holds in the same declared group."""
    if spec.exclusion_group is None:
        return []
    found: list[Violation] = []
    for hold in state.holds:
        if hold.exclusion_group != spec.exclusion_group:
            continue
        members = ", ".join(case.dictionary.exclusion_members(spec.exclusion_group))
        found.append(
            Violation(
                step=index,
                t_s=t_s,
                command=spec.identifier,
                kind=ViolationKind.MUTUAL_EXCLUSION,
                detail=(
                    f"is in exclusion group {spec.exclusion_group!r} (members: {members}) and "
                    f"{hold.command} still holds that group until t={hold.t_end:.1f} s"
                ),
                trace=(
                    Provenance(
                        hold.step,
                        hold.command,
                        hold.t_start,
                        f"took exclusion group {spec.exclusion_group!r} for "
                        f"{hold.t_end - hold.t_start:.1f} s",
                    ),
                ),
                shortfall=hold.t_end - t_s,
            )
        )
    return found


def _check_resources(
    index: int, t_s: float, spec: CommandSpec, state: _State, case: Case
) -> list[Violation]:
    """Held resource draw against the declared ceiling.

    The total is the sum over live holds plus this command's own draw. The
    comparison is ``total > limit * (1 + rel_tol)``; see
    :data:`RESOURCE_REL_TOL`. The trace names every live contributor, sorted by
    descending draw, because the one to move is usually the biggest.
    """
    if not spec.resource_draw:
        return []
    limits = case.dictionary.resource_limits
    found: list[Violation] = []
    held = _active_draw(state.holds)
    for key in spec.draw_keys:
        limit = limits.limit_for(key)
        if limit is None:
            continue
        total = held.get(key, 0.0) + float(spec.resource_draw[key])
        if total <= limit * (1.0 + RESOURCE_REL_TOL):
            continue
        contributors = sorted(
            (h for h in state.holds if key in h.draw and h.draw[key] > 0.0),
            key=lambda h: (-float(h.draw[key]), h.step),
        )
        trace = tuple(
            Provenance(
                h.step,
                h.command,
                h.t_start,
                f"holds {float(h.draw[key]):.6g} of {key} until t={h.t_end:.1f} s",
            )
            for h in contributors
        ) or (Provenance(None, None, None, f"no other command holds {key} at this step"),)
        found.append(
            Violation(
                step=index,
                t_s=t_s,
                command=spec.identifier,
                kind=ViolationKind.RESOURCE_LIMIT,
                detail=(
                    f"brings held {key} to {total:.6g} against a declared ceiling of "
                    f"{limit:.6g} (its own draw is {float(spec.resource_draw[key]):.6g})"
                ),
                trace=trace,
                shortfall=total - limit,
            )
        )
    return found


def _apply_effects(
    index: int, t_s: float, spec: CommandSpec, state: _State, machine: ModeMachine
) -> list[Violation]:
    """Apply declared effects; report an undeclared mode transition.

    A transition the machine forbids is reported and **not applied**: the
    executor does not credit a command with a state change the declared machine
    says it cannot make. The consequence is that later mode-gated commands are
    then evaluated in the pre-transition mode, which is the conservative
    reading and is recorded in the README under Limitations.
    """
    found: list[Violation] = []
    for name in sorted(spec.effects):
        value = spec.effects[name]
        state.predicates[name] = value
        state.predicate_source[name] = Provenance(
            index, spec.identifier, t_s, f"set {name}={value.value}"
        )
    if spec.mode_effect is not None and spec.mode_effect != state.mode:
        if machine.allows(state.mode, spec.mode_effect):
            state.mode = spec.mode_effect
            state.mode_source = Provenance(
                index, spec.identifier, t_s, f"transitioned to {spec.mode_effect}"
            )
        else:
            reachable = ", ".join(machine.reachable_from(state.mode)) or "nothing"
            found.append(
                Violation(
                    step=index,
                    t_s=t_s,
                    command=spec.identifier,
                    kind=ViolationKind.ILLEGAL_TRANSITION,
                    detail=(
                        f"transitions to {spec.mode_effect} but the declared machine allows "
                        f"only [{reachable}] from {state.mode}; the transition is not applied"
                    ),
                    trace=(
                        state.mode_source,
                        Provenance(
                            index,
                            spec.identifier,
                            t_s,
                            f"declares mode_effect={spec.mode_effect}",
                        ),
                    ),
                )
            )
    return found


def peak_draw(report: CheckReport) -> dict[str, float]:
    """Peak held draw per resource key over the whole sequence.

    Taken from the per-step snapshots, so it is the peak at command execution
    instants and not a continuous maximum. A draw profile between two commands
    is flat by construction here, so the two coincide except that a hold which
    ends between two commands is not sampled at its own end; that is harmless
    for a maximum.
    """
    peak: dict[str, float] = {}
    for snapshot in report.snapshots:
        for key, value in snapshot.active_draw.items():
            if not math.isfinite(value):  # pragma: no cover - defensive
                continue
            peak[key] = max(peak.get(key, 0.0), float(value))
    return dict(sorted(peak.items()))


def with_policy(case: Case, policy: EffectPolicy) -> CheckReport:
    """Convenience: ``forward_execute(case, policy)``. Kept for the CLI."""
    return forward_execute(case, policy)


def restate(report: CheckReport, case_name: str) -> CheckReport:
    """Return ``report`` with a different case name. Used by the benchmarks."""
    return replace(report, case_name=case_name)
