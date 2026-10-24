"""Rendering. The state trace is the product, so the renderer is not cosmetic.

Three renderers:

:func:`render_check`
    The forward-execution report, violations with their provenance traces.
:func:`render_comparison`
    Forward execution against the static lint on the same case, with what each
    found, what only the forward walk found, and what the lint reported that
    the forward walk did not.
:func:`render_state_at`
    The state reached after a given step, for the reviewer who asks "what mode
    was the vehicle in at step 47" and wants the answer without re-running.

Every clean verdict is printed with the sentence that bounds it. A report that
says "no violations" and stops invites exactly the inference this package
cannot support, and the sentence costs one line.
"""

from __future__ import annotations

from .lint import LintReport
from .model import Case
from .trace import CheckReport, ViolationKind, peak_draw

__all__ = [
    "CLEAN_DISCLAIMER",
    "render_check",
    "render_comparison",
    "render_state_at",
    "render_dictionary_summary",
]

#: Printed under every clean verdict, in every renderer, without exception.
CLEAN_DISCLAIMER = (
    "No violation was found against the declared model. That is a statement about the "
    "declarations in this case and about nothing else. It is not a statement that the "
    "sequence is safe to uplink."
)


def _header(title: str, width: int = 78) -> list[str]:
    return ["=" * width, title, "=" * width]


def render_check(report: CheckReport, case: Case | None = None) -> str:
    """Render a forward-execution report, traces included.

    Parameters
    ----------
    report:
        The report to render.
    case:
        Optional; when given, the summary line records the sequence length and
        span, which the report alone does not carry for an empty sequence.
    """
    lines = _header(f"cmdguard forward execution: {report.case_name}")
    n_steps = len(report.snapshots) if case is None else len(case.sequence)
    span = 0.0 if case is None else case.sequence.duration_s
    lines.append(f"steps                 : {n_steps}")
    lines.append(f"sequence span         : {span:.1f} s")
    lines.append(f"effect policy         : {report.effect_policy.value}")
    lines.append(f"violations            : {report.n_violations}")
    lines.append(f"ordering-only defects : {len(report.ordering_defects)}")
    lines.append(f"gate-blocking         : {len(report.gate_blocking)}")
    lines.append(f"final mode            : {report.final_mode}")
    peaks = peak_draw(report)
    if peaks:
        rendered = ", ".join(f"{k}={v:.6g}" for k, v in peaks.items())
        lines.append(f"peak held draw        : {rendered}")
    if report.clean:
        lines.append("")
        lines.append(CLEAN_DISCLAIMER)
        return "\n".join(lines)

    counts = report.kind_counts()
    lines.append("")
    lines.append("by kind:")
    for kind, count in counts.items():
        lines.append(f"  {kind:<26s} {count}")
    lines.append("")
    lines.append("violations, with the state trace that reaches each one:")
    for violation in report.violations:
        lines.append("")
        lines.append(violation.render(indent="    "))
    return "\n".join(lines)


def render_comparison(report: CheckReport, lint: LintReport) -> str:
    """Render forward execution against the static lint on the same case."""
    if report.case_name != lint.case_name:
        raise ValueError(
            f"cannot compare a forward report for {report.case_name!r} against a lint report "
            f"for {lint.case_name!r}"
        )
    forward_ids = frozenset((v.step, v.kind.value) for v in report.violations)
    lint_ids = lint.identities()
    missed = sorted(forward_ids - lint_ids)
    extra = sorted(lint_ids - forward_ids)

    lines = _header(f"cmdguard baseline comparison: {report.case_name}")
    lines.append(f"lint strength                     : {lint.strength.value}")
    lines.append(f"forward execution (ground truth)  : {len(forward_ids)} violation(s)")
    lines.append(f"static lint                       : {len(lint_ids)} finding(s)")
    lines.append(f"found by forward, missed by lint  : {len(missed)}")
    lines.append(f"reported by lint, not by forward  : {len(extra)}")
    denominator = len(forward_ids)
    if denominator:
        lines.append(f"lint miss rate on this case       : {len(missed) / denominator:.6f}")
    else:
        lines.append("lint miss rate on this case       : undefined (no violation to miss)")
    if missed:
        lines.append("")
        lines.append("missed by the lint:")
        for step, kind in missed:
            match = next(
                v for v in report.violations if v.step == step and v.kind.value == kind
            )
            lines.append("")
            lines.append(match.render(indent="    "))
    if extra:
        lines.append("")
        lines.append("reported by the lint and not by the forward walk:")
        for step, kind in extra:
            finding = next(
                f for f in lint.findings if f.step == step and f.kind.value == kind
            )
            lines.append(f"  {finding.render()}")
        lines.append(
            "  A lint finding the forward walk does not confirm is a false positive against "
            "the declared model: the condition the lint read from the initial state was "
            "changed by an earlier command before the step was reached."
        )
    if not missed and not extra:
        lines.append("")
        lines.append("The lint agreed with the forward walk on this case, finding by finding.")
    return "\n".join(lines)


def render_state_at(report: CheckReport, step: int) -> str:
    """Render the state reached after ``step``.

    Raises
    ------
    IndexError
        When ``step`` is not a step of the sequence this report came from.
    """
    if step < 0 or step >= len(report.snapshots):
        raise IndexError(
            f"step {step} is outside the sequence, which has {len(report.snapshots)} step(s)"
        )
    snapshot = report.snapshots[step]
    lines = _header(f"state after step {step} of {report.case_name}")
    lines.append(f"t                : {snapshot.t_s:.1f} s")
    lines.append(f"command          : {snapshot.command}")
    lines.append(f"mode             : {snapshot.mode}")
    lines.append(f"violations here  : {len(snapshot.violations)}")
    if snapshot.active_draw:
        rendered = ", ".join(f"{k}={v:.6g}" for k, v in snapshot.active_draw.items())
        lines.append(f"held draw        : {rendered}")
    lines.append("predicates:")
    for name in sorted(snapshot.predicates):
        lines.append(f"  {name:<28s} {snapshot.predicates[name].value}")
    for violation in snapshot.violations:
        lines.append("")
        lines.append(violation.render(indent="    "))
    return "\n".join(lines)


def render_dictionary_summary(case: Case) -> str:
    """Render what was declared, so a reviewer can check the input, not the output."""
    lines = _header(f"declarations: {case.name}")
    if case.description:
        lines.append(case.description)
        lines.append("")
    lines.append(f"commands declared        : {len(case.dictionary.commands)}")
    irreversible = sorted(
        s.identifier for s in case.dictionary.commands.values() if s.irreversible
    )
    lines.append(f"irreversible commands    : {len(irreversible)}")
    if irreversible:
        lines.append(f"  {', '.join(irreversible)}")
    groups = sorted(
        {s.exclusion_group for s in case.dictionary.commands.values() if s.exclusion_group}
    )
    lines.append(f"mutual-exclusion groups  : {len(groups)}")
    for group in groups:
        lines.append(f"  {group}: {', '.join(case.dictionary.exclusion_members(group))}")
    limits = case.dictionary.resource_limits.limits
    lines.append(f"resource ceilings        : {len(limits)}")
    for key in sorted(limits):
        lines.append(f"  {key:<24s} {float(limits[key]):.6g}")
    lines.append(f"modes                    : {', '.join(case.machine.modes)}")
    lines.append(f"initial mode             : {case.initial.mode}")
    unknown = sorted(
        name
        for name, value in case.initial.predicates.items()
        if value.value == "unknown"
    )
    lines.append(f"predicates declared      : {len(case.initial.predicates)}")
    lines.append(f"  unverified at t=0      : {len(unknown)}")
    if unknown:
        lines.append(f"    {', '.join(unknown)}")
    lines.append(f"sequence steps           : {len(case.sequence)}")
    lines.append(f"sequence span            : {case.sequence.duration_s:.1f} s")
    undeclared = case.undeclared_commands
    lines.append(f"undeclared commands used : {len(undeclared)}")
    if undeclared:
        lines.append(f"  {', '.join(undeclared)}")
    kinds = ", ".join(k.value for k in ViolationKind)
    lines.append("")
    lines.append(f"violation kinds this package reports: {kinds}")
    return "\n".join(lines)
