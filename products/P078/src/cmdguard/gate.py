"""The irreversible-command gate. It refuses; it does not advise.

Every other output of this package is a report that a human reads. This one is
a verdict with two values, and the reason it exists separately is that the
defect it covers is the one that survives review. A command whose precondition
is *false* is caught by anything, including a per-command lint. A command whose
precondition is *unverified* passes every per-command check, because each one
was evaluated against an assumption, and if that command is irreversible there
is no recovery procedure to fall back on.

So: a sequence in which an irreversible command executes while any of its
declared preconditions is unverified is **refused**. The two readings of
"an irreversible command precedes an unverified precondition" -- the command
runs before the verification step, and the command runs while the condition is
unverified -- are the same condition under forward execution, because the
verification step is exactly what would have set the predicate before the
command's turn came. Both are covered by the one check and this paragraph is
the record of that reasoning.

What the gate does not do
-------------------------
It does not pass anything. :attr:`Verdict.NO_REFUSAL_FOUND` is named for what
was measured and not for what a reader would like it to mean: no gate-blocking
violation was found against the declared model. It is not a statement that the
sequence is safe to uplink, and :meth:`GateResult.render` says so on every
call, including the clean one.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .model import Case, EffectPolicy
from .trace import CheckReport, Violation, forward_execute

__all__ = ["Verdict", "GateResult", "irreversible_gate", "gate_case"]


class Verdict(enum.Enum):
    """Gate outcome. Two values, neither of which is the word "safe"."""

    REFUSE = "refuse"
    NO_REFUSAL_FOUND = "no_refusal_found"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class GateResult:
    """Verdict plus the violations it was reached from."""

    case_name: str
    verdict: Verdict
    blocking: tuple[Violation, ...]
    n_irreversible_commands: int
    n_irreversible_executions: int

    @property
    def refused(self) -> bool:
        """True when the gate refuses the sequence."""
        return self.verdict is Verdict.REFUSE

    @property
    def exit_code(self) -> int:
        """Process exit code for the CLI: 2 on refusal, 0 otherwise."""
        return 2 if self.refused else 0

    def render(self) -> str:
        """Human-readable verdict. Prints the disclaimer in both branches."""
        lines = [
            f"irreversible-command gate: {self.case_name}",
            f"  irreversible commands declared : {self.n_irreversible_commands}",
            f"  irreversible executions in seq : {self.n_irreversible_executions}",
            f"  verdict                        : {self.verdict.value.upper()}",
        ]
        if self.refused:
            lines.append(
                f"  REFUSED on {len(self.blocking)} gate-blocking violation(s). An "
                "irreversible command executes while a declared precondition of it is "
                "unverified."
            )
            for violation in self.blocking:
                lines.append("")
                lines.append(violation.render(indent="    "))
        else:
            lines.append(
                "  No gate-blocking violation was found against the declared model. This is "
                "not a statement that the sequence is safe to uplink."
            )
        return "\n".join(lines)


def irreversible_gate(report: CheckReport, case: Case) -> GateResult:
    """Refuse ``report`` when an irreversible command met an unverified precondition.

    Parameters
    ----------
    report:
        A :class:`~cmdguard.trace.CheckReport` from
        :func:`~cmdguard.trace.forward_execute` on ``case``.
    case:
        The case the report came from, read for the irreversibility flags.

    Returns
    -------
    GateResult
        ``Verdict.REFUSE`` when ``report.gate_blocking`` is non-empty.
    """
    if report.case_name != case.name:
        raise ValueError(
            f"report is for case {report.case_name!r} but case is {case.name!r}; the gate "
            "will not reason about a report from a different case"
        )
    irreversible = {
        spec.identifier for spec in case.dictionary.commands.values() if spec.irreversible
    }
    executions = sum(1 for step in case.sequence.steps if step.command in irreversible)
    blocking = report.gate_blocking
    return GateResult(
        case_name=case.name,
        verdict=Verdict.REFUSE if blocking else Verdict.NO_REFUSAL_FOUND,
        blocking=blocking,
        n_irreversible_commands=len(irreversible),
        n_irreversible_executions=executions,
    )


def gate_case(case: Case, effect_policy: EffectPolicy = EffectPolicy.APPLY) -> GateResult:
    """Forward-execute ``case`` and gate the result in one call."""
    return irreversible_gate(forward_execute(case, effect_policy), case)
