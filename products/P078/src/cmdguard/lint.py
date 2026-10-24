"""Baseline 2: per-command static precondition lint. The thing people do.

This is not a straw man and it is not useless. A per-command lint catches the
defect class that dominates in practice -- somebody typed a command that is not
in the dictionary, or asked for a payload scan when the declared initial state
has the payload disabled -- it runs in a text editor, it needs no mode machine,
and it never reports a cascade. It is what a mission's own ``check_seq.py``
usually is.

What it cannot do is see the sequence. Every check here is evaluated against
**the declared initial state**, which is the only state a per-command check
has. A command that was legal at t=0 and illegal at t=430 s because step 12
safed the payload is, to this lint, legal. That is the defect class the forward
walk in :mod:`cmdguard.trace` exists for, and measuring how much of it the lint
misses is a deliverable of this product rather than a rhetorical point: see
:func:`cmdguard.benchmark.lint_miss_rate` and
``validation/validate_lint_missrate_output.txt``.

Two strengths are implemented, so that the measured miss rate is not a
measurement against the weakest thing that could be called a lint:

``LintStrength.PER_COMMAND``
    Strictly one command at a time. Dictionary membership, preconditions and
    mode gate against the declared initial state, and the command's own
    resource draw against the ceiling on its own.

``LintStrength.WITH_TIMING``
    The above, plus a timing pass over the sequence's time column:
    ``min_gap_after_s`` between adjacent steps and ``min_interval_s`` between
    successive occurrences of the same command. No state model, no mode, no
    accumulation -- arithmetic on the time column, which a shell script can do.

Neither strength can find a mode-gate defect that a mode transition created, an
exclusion overlap, a resource ceiling crossed by accumulation, or a
precondition falsified by an earlier command. Those are
:data:`cmdguard.trace.ORDERING_KINDS` and they are structurally out of reach,
not merely unimplemented.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .model import Case, Verity
from .trace import TIME_TOL_S, ViolationKind

__all__ = ["LintStrength", "LintFinding", "LintReport", "static_lint"]


class LintStrength(enum.Enum):
    """How much of the sequence the lint is allowed to look at."""

    PER_COMMAND = "per_command"
    WITH_TIMING = "with_timing"


@dataclass(frozen=True)
class LintFinding:
    """One lint finding. Same ``(step, kind)`` identity as a trace violation.

    Sharing the identity is what makes the miss rate computable: a finding and
    a violation are the same finding when they agree on the step and the kind.
    """

    step: int
    t_s: float
    command: str
    kind: ViolationKind
    detail: str

    def identity(self) -> tuple[int, str]:
        """``(step, kind)`` -- the key the miss rate is computed over."""
        return (self.step, self.kind.value)

    def render(self) -> str:
        """One-line rendering. The lint has no trace to print."""
        return (
            f"step {self.step:>3d} t={self.t_s:>9.1f} s  {self.command:<22s} "
            f"{self.kind.value}: {self.detail}"
        )


@dataclass(frozen=True)
class LintReport:
    """Findings from one lint pass."""

    case_name: str
    strength: LintStrength
    findings: tuple[LintFinding, ...]

    @property
    def clean(self) -> bool:
        """True when the lint found nothing. Says nothing about the sequence."""
        return not self.findings

    @property
    def n_findings(self) -> int:
        """Number of findings."""
        return len(self.findings)

    def identities(self) -> frozenset[tuple[int, str]]:
        """The ``(step, kind)`` set, for comparison against a trace report."""
        return frozenset(f.identity() for f in self.findings)

    def kind_counts(self) -> dict[str, int]:
        """Finding count per kind."""
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.kind.value] = counts.get(finding.kind.value, 0) + 1
        return dict(sorted(counts.items()))


def static_lint(
    case: Case,
    strength: LintStrength = LintStrength.PER_COMMAND,
) -> LintReport:
    """Lint ``case.sequence`` one command at a time against the initial state.

    Parameters
    ----------
    case:
        The same input the forward executor takes. The mode machine is read
        only for the initial mode; the transition relation is not used, because
        a per-command lint has no notion of having transitioned.
    strength:
        :class:`LintStrength`. ``WITH_TIMING`` adds a timing pass over the time
        column.

    Returns
    -------
    LintReport
        Findings in step order, with the same ``(step, kind)`` identity the
        forward executor uses.
    """
    if not isinstance(strength, LintStrength):
        raise TypeError(f"strength must be a LintStrength, got {type(strength)!r}")

    initial_mode = case.initial.mode
    findings: list[LintFinding] = []

    for index, step in enumerate(case.sequence.steps):
        spec = case.dictionary.get(step.command)
        t_s = float(step.t_s)
        if spec is None:
            findings.append(
                LintFinding(
                    step=index,
                    t_s=t_s,
                    command=step.command,
                    kind=ViolationKind.UNDECLARED_COMMAND,
                    detail=f"command {step.command!r} is not in the declared dictionary",
                )
            )
            continue

        if spec.required_modes and initial_mode not in spec.required_modes:
            findings.append(
                LintFinding(
                    step=index,
                    t_s=t_s,
                    command=spec.identifier,
                    kind=ViolationKind.MODE_GATE,
                    detail=(
                        f"is permitted only in [{', '.join(spec.required_modes)}] and the "
                        f"declared initial mode is {initial_mode}"
                    ),
                )
            )

        for name in sorted(spec.preconditions):
            required = spec.preconditions[name]
            declared = case.initial.verity(name)
            if declared is required:
                continue
            if declared is Verity.UNKNOWN and spec.irreversible:
                kind = ViolationKind.IRREVERSIBLE_UNVERIFIED
            elif declared is Verity.UNKNOWN:
                kind = ViolationKind.PRECONDITION_UNKNOWN
            else:
                kind = ViolationKind.PRECONDITION_FALSE
            findings.append(
                LintFinding(
                    step=index,
                    t_s=t_s,
                    command=spec.identifier,
                    kind=kind,
                    detail=(
                        f"requires {name}={required.value}; the declared initial state has "
                        f"{name}={declared.value}"
                    ),
                )
            )

        for key in spec.draw_keys:
            limit = case.dictionary.resource_limits.limit_for(key)
            own = float(spec.resource_draw[key])
            if limit is not None and own > limit:
                findings.append(
                    LintFinding(
                        step=index,
                        t_s=t_s,
                        command=spec.identifier,
                        kind=ViolationKind.RESOURCE_LIMIT,
                        detail=(
                            f"draws {own:.6g} of {key} on its own against a ceiling of "
                            f"{limit:.6g}"
                        ),
                    )
                )

    if strength is LintStrength.WITH_TIMING:
        findings.extend(_timing_pass(case))

    return LintReport(
        case_name=case.name,
        strength=strength,
        findings=tuple(sorted(findings, key=lambda f: (f.step, f.kind.value, f.detail))),
    )


def _timing_pass(case: Case) -> list[LintFinding]:
    """Arithmetic on the time column. No state, no mode, no accumulation.

    This is deliberately as strong as a timing check can be made without a
    state model, so that the measured miss rate in
    ``validation/validate_lint_missrate_output.txt`` is a measurement against a
    competent baseline. It reproduces the forward executor's timing findings
    exactly, including the tolerance convention, because the forward executor's
    timing checks are themselves a function of the time column alone.
    """
    findings: list[LintFinding] = []
    last_seen: dict[str, float] = {}
    previous: tuple[str, float, float] | None = None

    for index, step in enumerate(case.sequence.steps):
        spec = case.dictionary.get(step.command)
        t_s = float(step.t_s)
        if spec is None:
            previous = (step.command, t_s, 0.0)
            continue

        if previous is not None:
            _, prev_t, required_gap = previous
            elapsed = t_s - prev_t
            if required_gap > 0.0 and elapsed + TIME_TOL_S < required_gap:
                findings.append(
                    LintFinding(
                        step=index,
                        t_s=t_s,
                        command=spec.identifier,
                        kind=ViolationKind.TIMING_GAP,
                        detail=(
                            f"executes {elapsed:.3f} s after the previous command, which "
                            f"declares min_gap_after_s={required_gap:.3f}"
                        ),
                    )
                )

        last_t = last_seen.get(spec.identifier)
        if last_t is not None and spec.min_interval_s > 0.0:
            elapsed = t_s - last_t
            if elapsed + TIME_TOL_S < spec.min_interval_s:
                findings.append(
                    LintFinding(
                        step=index,
                        t_s=t_s,
                        command=spec.identifier,
                        kind=ViolationKind.TIMING_INTERVAL,
                        detail=(
                            f"repeats {elapsed:.3f} s after its own previous execution against "
                            f"min_interval_s={spec.min_interval_s:.3f}"
                        ),
                    )
                )

        last_seen[spec.identifier] = t_s
        previous = (spec.identifier, t_s, float(spec.min_gap_after_s))

    return findings
