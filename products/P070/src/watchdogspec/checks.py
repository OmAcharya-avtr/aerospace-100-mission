"""The twelve consistency checks, and the report they produce.

Every check is arithmetic over declared numbers. None of them looks at a
running system, and a clean report says only that a specification does not
contradict itself.

Catalogue
---------
=======  =======  =========================================================
code     severity meaning
=======  =======  =========================================================
WS001    error    worst-case detection latency exceeds the liveness deadline
WS002    error    tolerated silence ``k W`` is at least the liveness
                  deadline: the missed-beat tolerance exceeds the deadline
                  it is declared to protect
WS003    error    the escalation ladder cannot reach its final rung within
                  the liveness deadline
WS004    warning  a deadline comparison for this task lands within the
                  comparison tolerance of zero, so the arithmetic does not
                  decide the verdict
WS005    warning  the false-trip rate exceeds the declared budget
WS006    error    a single beat window ``P + J`` already exceeds the
                  liveness deadline
WS007    warning  the supervisor poll period is coarser than the beat window
WS008    error    the jitter bound is not smaller than the heartbeat period
WS009    error    the ladder's final rung is an observing action, so the
                  ladder never attempts a recovery
WS010    warning  rung actions are not in non-decreasing severity order
WS011    warning  declared recovery costs are not non-decreasing across rungs
WS012    error    the task names an escalation ladder that is not declared
=======  =======  =========================================================

Precedence
----------
Three checks can fire on the same arithmetic, because ``W > D`` and
``k W >= D`` both force the detection latency past the deadline. Only the most
specific diagnosis is reported, so that a report carries one line per cause
rather than three per symptom:

* ``WS006`` (one beat window already exceeds the deadline) suppresses
  ``WS002`` and ``WS001`` for that task.
* ``WS002`` (the tolerance exceeds the deadline) suppresses ``WS001``.
* ``WS001`` suppresses ``WS003``: a ladder cannot be blamed for missing a
  deadline the detection had already missed before the first rung started.

``WS002`` is the specification defect the batch specification calls out: a
task whose missed-beat tolerance lets it stay silent for longer than the
deadline that tolerance is supposed to protect. It costs nothing to find,
because ``k W`` and ``D`` are both already declared, and it cannot be fixed by
a faster escalation ladder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .falsetrip import FalseTripRate, false_trip_rate
from .latency import DEFAULT_TOL_MS, Margin, TaskTiming, task_timing
from .model import Spec


class Severity(Enum):
    """How a finding is treated by the exit code."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class CheckSpec:
    """Static description of one check."""

    code: str
    severity: Severity
    summary: str


CHECKS: dict[str, CheckSpec] = {
    spec.code: spec
    for spec in (
        CheckSpec("WS001", Severity.ERROR, "detection latency exceeds the liveness deadline"),
        CheckSpec(
            "WS002",
            Severity.ERROR,
            "tolerated silence exceeds the liveness deadline it is declared to protect",
        ),
        CheckSpec("WS003", Severity.ERROR, "escalation ladder cannot reach its final rung in time"),
        CheckSpec(
            "WS004",
            Severity.WARNING,
            "a deadline comparison lands within the comparison tolerance",
        ),
        CheckSpec("WS005", Severity.WARNING, "false-trip rate exceeds the declared budget"),
        CheckSpec("WS006", Severity.ERROR, "one beat window already exceeds the liveness deadline"),
        CheckSpec(
            "WS007",
            Severity.WARNING,
            "supervisor poll period is coarser than the beat window",
        ),
        CheckSpec("WS008", Severity.ERROR, "jitter bound is not smaller than the heartbeat period"),
        CheckSpec("WS009", Severity.ERROR, "escalation ladder never attempts a recovery"),
        CheckSpec(
            "WS010",
            Severity.WARNING,
            "rung actions are not in non-decreasing severity order",
        ),
        CheckSpec("WS011", Severity.WARNING, "declared recovery costs are not non-decreasing"),
        CheckSpec("WS012", Severity.ERROR, "task names an escalation ladder that is not declared"),
    )
}


def check_spec(code: str) -> CheckSpec:
    """Return the static description of one check.

    Raises
    ------
    KeyError
        If ``code`` is not one of the twelve.
    """
    return CHECKS[code]


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing wrong with a specification."""

    code: str
    severity: Severity
    subject: str
    message: str


@dataclass(frozen=True, slots=True)
class Report:
    """Everything computed about one specification."""

    spec_source: str
    system: str
    findings: tuple[Finding, ...]
    timings: dict[str, TaskTiming]
    false_trips: dict[str, FalseTripRate]
    tol_ms: float

    @property
    def exit_code(self) -> int:
        """``0`` when there is no error finding, ``1`` otherwise."""
        return 1 if any(f.severity is Severity.ERROR for f in self.findings) else 0

    @property
    def is_consistent(self) -> bool:
        """``True`` when no error finding was raised.

        Internal consistency of a declaration. Not a statement about a system.
        """
        return self.exit_code == 0

    def count_by_code(self) -> dict[str, int]:
        """Number of findings per check code, for codes that fired."""
        out: dict[str, int] = {}
        for finding in self.findings:
            out[finding.code] = out.get(finding.code, 0) + 1
        return out

    def count_by_severity(self) -> dict[str, int]:
        """Number of findings per severity, for severities that fired."""
        out: dict[str, int] = {}
        for finding in self.findings:
            out[finding.severity.value] = out.get(finding.severity.value, 0) + 1
        return out


def _ms(value: float) -> str:
    return f"{value:.6g} ms"


def run_checks(spec: Spec, tol_ms: float = DEFAULT_TOL_MS) -> Report:
    """Run every check over a parsed specification.

    Parameters
    ----------
    spec:
        A parsed :class:`watchdogspec.model.Spec`.
    tol_ms:
        Absolute tolerance in ms for every deadline comparison. A comparison
        whose slack is within this band produces ``WS004`` and does not
        produce ``WS001`` or ``WS003``; see
        :func:`watchdogspec.latency.verdict` for why the band exists.

    Returns
    -------
    Report
        Findings in ``(task order, code order)``, plus the timing and
        false-trip arithmetic for every task whose ladder is declared.
    """
    findings: list[Finding] = []
    timings: dict[str, TaskTiming] = {}
    rates: dict[str, FalseTripRate] = {}
    supervisor = spec.supervisor

    for ladder in spec.ladders:
        if not ladder.rungs[-1].action.recovers:
            findings.append(
                Finding(
                    "WS009",
                    Severity.ERROR,
                    ladder.id,
                    f"ladder {ladder.id} ends with '{ladder.rungs[-1].action.value}', an "
                    "observing action: the ladder records the fault and never attempts a "
                    "recovery",
                )
            )
        severities = [rung.action.severity for rung in ladder.rungs]
        for index in range(1, len(severities)):
            if severities[index] < severities[index - 1]:
                findings.append(
                    Finding(
                        "WS010",
                        Severity.WARNING,
                        ladder.id,
                        f"ladder {ladder.id} rung {index + 1} "
                        f"('{ladder.rungs[index].action.value}') is less severe than rung "
                        f"{index} ('{ladder.rungs[index - 1].action.value}')",
                    )
                )
                break
        costs = [rung.recovery_cost for rung in ladder.rungs]
        for index in range(1, len(costs)):
            if costs[index] < costs[index - 1]:
                findings.append(
                    Finding(
                        "WS011",
                        Severity.WARNING,
                        ladder.id,
                        f"ladder {ladder.id} declares rung {index + 1} cost "
                        f"{costs[index]:.6g} below rung {index} cost {costs[index - 1]:.6g}",
                    )
                )
                break

    for task in spec.tasks:
        ladder = spec.ladder(task.escalation)
        if ladder is None:
            findings.append(
                Finding(
                    "WS012",
                    Severity.ERROR,
                    task.id,
                    f"task {task.id} escalates to ladder '{task.escalation}', which is not "
                    "declared in this specification",
                )
            )
            continue

        timing = task_timing(task, supervisor, ladder, tol_ms=tol_ms)
        timings[task.id] = timing
        rate = false_trip_rate(task)
        rates[task.id] = rate

        if task.heartbeat_jitter_ms >= task.heartbeat_period_ms:
            findings.append(
                Finding(
                    "WS008",
                    Severity.ERROR,
                    task.id,
                    f"task {task.id} declares a jitter bound of "
                    f"{_ms(task.heartbeat_jitter_ms)} against a heartbeat period of "
                    f"{_ms(task.heartbeat_period_ms)}: two heartbeats may then arrive out of "
                    "order and the beat-window model does not describe the result",
                )
            )

        window_too_wide = timing.beat_window_ms > task.liveness_deadline_ms + tol_ms
        tolerance_too_wide = (
            task.missed_beats_tolerated > 0 and timing.tolerance_margin is Margin.MISSED
        )

        if window_too_wide:
            findings.append(
                Finding(
                    "WS006",
                    Severity.ERROR,
                    task.id,
                    f"task {task.id} has a beat window of {_ms(timing.beat_window_ms)} against a "
                    f"liveness deadline of {_ms(task.liveness_deadline_ms)}: one heartbeat "
                    "interval already consumes the whole deadline",
                )
            )
        elif tolerance_too_wide:
            findings.append(
                Finding(
                    "WS002",
                    Severity.ERROR,
                    task.id,
                    f"task {task.id} tolerates {task.missed_beats_tolerated} missed beat(s), a "
                    f"silence of {_ms(timing.tolerated_silence_ms)}, against a liveness deadline "
                    f"of {_ms(task.liveness_deadline_ms)}: the tolerance exceeds the deadline it "
                    f"is declared to protect (slack {_ms(timing.tolerance_slack_ms)})",
                )
            )
        elif timing.detection_margin is Margin.MISSED:
            findings.append(
                Finding(
                    "WS001",
                    Severity.ERROR,
                    task.id,
                    f"task {task.id} worst-case detection latency "
                    f"{_ms(timing.detection_latency_ms)} exceeds its liveness deadline "
                    f"{_ms(task.liveness_deadline_ms)} by "
                    f"{_ms(-timing.detection_slack_ms)}, before any escalation rung runs",
                )
            )

        if timing.detection_margin is not Margin.MISSED and timing.ladder_margin is Margin.MISSED:
            findings.append(
                Finding(
                    "WS003",
                    Severity.ERROR,
                    task.id,
                    f"task {task.id} cannot reach rung {len(ladder.rungs)} "
                    f"('{ladder.rungs[-1].action.value}') of ladder {ladder.id} within its "
                    f"liveness deadline: worst case "
                    f"{_ms(timing.full_ladder_latency_ms)} against "
                    f"{_ms(task.liveness_deadline_ms)}; deepest rung reachable is "
                    f"{timing.deepest_rung_reachable}",
                )
            )

        marginal: list[str] = []
        if timing.detection_margin is Margin.MARGINAL:
            marginal.append(f"detection slack {timing.detection_slack_ms:.3e} ms")
        if timing.ladder_margin is Margin.MARGINAL:
            marginal.append(f"final-rung slack {timing.ladder_slack_ms:.3e} ms")
        if task.missed_beats_tolerated > 0 and timing.tolerance_margin is Margin.MARGINAL:
            marginal.append(f"tolerated-silence slack {timing.tolerance_slack_ms:.3e} ms")
        if marginal:
            findings.append(
                Finding(
                    "WS004",
                    Severity.WARNING,
                    task.id,
                    f"task {task.id} has {', '.join(marginal)}, within the comparison "
                    f"tolerance of {tol_ms:.3e} ms: whether those deadlines are met is not "
                    "decided by this arithmetic",
                )
            )

        if supervisor.poll_period_ms > timing.beat_window_ms:
            findings.append(
                Finding(
                    "WS007",
                    Severity.WARNING,
                    task.id,
                    f"supervisor poll period {_ms(supervisor.poll_period_ms)} is coarser than "
                    f"task {task.id} beat window {_ms(timing.beat_window_ms)}: the poll "
                    "granularity, not the heartbeat, dominates the detection latency",
                )
            )

        budget = task.false_trip_budget_per_hour
        if budget is not None and rate.per_hour > budget:
            findings.append(
                Finding(
                    "WS005",
                    Severity.WARNING,
                    task.id,
                    f"task {task.id} false-trip rate {rate.per_hour:.6g} per hour exceeds its "
                    f"declared budget of {budget:.6g} per hour (jitter sigma "
                    f"{_ms(task.jitter_sigma_ms)}, trip threshold {_ms(rate.threshold_ms)}, "
                    f"{rate.z:.4g} sigma of the gap)",
                )
            )

    order = {code: index for index, code in enumerate(CHECKS)}
    findings.sort(key=lambda f: (f.subject, order[f.code], f.message))
    return Report(
        spec_source=spec.source,
        system=spec.system,
        findings=tuple(findings),
        timings=timings,
        false_trips=rates,
        tol_ms=tol_ms,
    )
