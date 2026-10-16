"""Immutable data model for a declared watchdog and liveness specification.

Units
-----
Every time in this package is in **milliseconds (ms)**. Costs are in
dimensionless **declared cost units**: the specification author chooses what
they mean (lost frames, joules, mission-seconds) and this package only ever
compares them to one another.

Symbols used throughout the package, fixed here once
----------------------------------------------------
=========  =====================================================  ======
symbol     meaning                                                units
=========  =====================================================  ======
``P``      nominal heartbeat period of a supervised task          ms
``J``      bound on the deviation of the receipt-to-receipt gap
           from ``P`` (absorbs emission jitter and transport
           variation; see :class:`Task`)                          ms
``sigma``  standard deviation of the emission deviation in the
           stochastic jitter model                                ms
``k``      number of consecutive deadline expiries the
           supervisor tolerates before declaring a fault          --
``D``      liveness deadline the escalation must complete within  ms
``W``      beat window, ``P + J``                                 ms
``Q``      supervisor poll period                                 ms
``Jq``     bound on supervisor poll release jitter                ms
``O``      supervisor declaration overhead                        ms
=========  =====================================================  ======

The bounded-jitter release model (a nominal period plus a bounded deviation)
is the one used in fixed-priority response-time analysis: Audsley, N. C.,
Burns, A., Richardson, M. F., Tindell, K. and Wellings, A. J., "Applying new
scheduling theory to static priority pre-emptive scheduling", *Software
Engineering Journal*, vol. 8, no. 5, 1993, pp. 284-292,
doi:10.1049/sej.1993.0034; and Tindell, K. and Clark, J., "Holistic
schedulability analysis for distributed hard real-time systems",
*Microprocessors and Microsystems*, vol. 18, no. 4, 1994, pp. 40-50,
doi:10.1016/0141-9331(94)90080-9.

Nothing in this module computes anything. It holds what was declared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Action(Enum):
    """One escalation action, ordered by how much of the system it disturbs.

    The ordering is a declared convention of this package, not a standard.
    ``LOG`` and ``NOTIFY`` are **observing** actions: they record the fault and
    change nothing. ``RESTART_TASK``, ``SAFE_MODE`` and ``RESET`` are
    **recovering** actions: they change the state of the system in an attempt
    to remove the fault. A ladder whose last rung only observes has no
    recovery at all, which is check ``WS009``.

    The ladder vocabulary follows the common supervision pattern described in
    Burns, A. and Wellings, A., *Real-Time Systems and Programming Languages*,
    4th edition, Addison-Wesley, 2009, ISBN 978-0-321-41745-9, and in Kopetz,
    H., *Real-Time Systems: Design Principles for Distributed Embedded
    Applications*, 2nd edition, Springer, 2011, ISBN 978-1-4419-8236-0.
    """

    LOG = "log"
    NOTIFY = "notify"
    RESTART_TASK = "restart_task"
    SAFE_MODE = "safe_mode"
    RESET = "reset"

    @property
    def severity(self) -> int:
        """Position in the declared severity order, 0 (``LOG``) to 4 (``RESET``)."""
        return _SEVERITY[self]

    @property
    def recovers(self) -> bool:
        """``True`` when this action attempts to change the state of the system."""
        return _SEVERITY[self] >= 2


_SEVERITY: dict[Action, int] = {
    Action.LOG: 0,
    Action.NOTIFY: 1,
    Action.RESTART_TASK: 2,
    Action.SAFE_MODE: 3,
    Action.RESET: 4,
}


@dataclass(frozen=True, slots=True)
class Rung:
    """One rung of an escalation ladder.

    Attributes
    ----------
    action:
        What the supervisor does at this rung.
    latency_ms:
        Worst-case time, in ms, from the start of this rung to the instant at
        which the supervisor has either completed the action or given up on it
        and is ready to start the next rung. It therefore includes both the
        action itself and whatever confirmation timeout follows it. Declared,
        never measured by this package.
    recovery_cost:
        Declared cost of executing this rung, in dimensionless cost units.
    note:
        Free text. Not interpreted.
    """

    action: Action
    latency_ms: float
    recovery_cost: float = 0.0
    note: str = ""


@dataclass(frozen=True, slots=True)
class Ladder:
    """An ordered escalation ladder, rung 1 first.

    ``rungs`` is non-empty by construction of the parser. The ladder is
    executed in order: rung ``i`` starts when rung ``i - 1`` has finished or
    timed out.
    """

    id: str
    rungs: tuple[Rung, ...]
    description: str = ""

    @property
    def total_latency_ms(self) -> float:
        """Sum of every rung latency, in ms. Time to run the whole ladder."""
        return float(sum(rung.latency_ms for rung in self.rungs))

    @property
    def total_cost(self) -> float:
        """Sum of every declared rung cost, in dimensionless cost units."""
        return float(sum(rung.recovery_cost for rung in self.rungs))

    def cumulative_latency_ms(self) -> tuple[float, ...]:
        """Cumulative latency after each rung, in ms. Length equals ``len(rungs)``."""
        out: list[float] = []
        running = 0.0
        for rung in self.rungs:
            running += rung.latency_ms
            out.append(running)
        return tuple(out)


@dataclass(frozen=True, slots=True)
class Supervisor:
    """The supervising entity: how often it looks, and how late it may be.

    Attributes
    ----------
    poll_period_ms:
        ``Q``. Nominal period at which the supervisor evaluates its deadline
        table. An internal deadline expiry is not acted on until the next
        poll.
    poll_jitter_ms:
        ``Jq``. Bound on the release jitter of that poll, in the sense of
        Audsley et al. 1993.
    declaration_overhead_ms:
        ``O``. Bound on the time from the poll observing an expiry to the fault
        being declared and the first escalation rung starting.
    name:
        Free text.
    """

    poll_period_ms: float
    poll_jitter_ms: float = 0.0
    declaration_overhead_ms: float = 0.0
    name: str = "supervisor"

    @property
    def observation_delay_ms(self) -> float:
        """``Q + Jq + O``: worst-case delay from internal expiry to declaration, ms."""
        return self.poll_period_ms + self.poll_jitter_ms + self.declaration_overhead_ms


@dataclass(frozen=True, slots=True)
class Task:
    """A supervised task, as declared.

    The jitter bound ``J`` is defined on the **receipt-to-receipt gap at the
    supervisor**: the specification asserts that, while the task is healthy,
    two consecutive heartbeats are never more than ``P + J`` apart as the
    supervisor sees them. ``J`` therefore absorbs emission jitter, transport
    variation and supervisor input queueing in one declared number. This is a
    modelling choice; it is stated here because the arithmetic in
    :mod:`watchdogspec.latency` is only as good as it.

    Attributes
    ----------
    heartbeat_period_ms:
        ``P``, strictly positive.
    heartbeat_jitter_ms:
        ``J``, non-negative. Check ``WS008`` fires when ``J >= P``.
    jitter_sigma_ms:
        ``sigma``, non-negative, standard deviation of the emission deviation
        in the **stochastic** model used only for the false-trip rate. Zero
        means the author declares the jitter bound to be hard, and the
        false-trip rate is then reported as exactly zero.
    missed_beats_tolerated:
        ``k``, a non-negative integer. The supervisor declares a fault on the
        ``k + 1``-th consecutive deadline expiry.
    liveness_deadline_ms:
        ``D``, strictly positive. Measured from the receipt of the last
        heartbeat the supervisor accepted -- the last instant at which it had
        positive evidence of liveness. Every latency in this package uses the
        same origin, which is the only way the comparison means anything.
    escalation:
        Identifier of the :class:`Ladder` run when this task is declared faulty.
    false_trip_budget_per_hour:
        Optional declared budget. ``None`` disables check ``WS005``.
    """

    id: str
    heartbeat_period_ms: float
    missed_beats_tolerated: int
    liveness_deadline_ms: float
    escalation: str
    heartbeat_jitter_ms: float = 0.0
    jitter_sigma_ms: float = 0.0
    false_trip_budget_per_hour: float | None = None
    name: str = ""
    note: str = ""

    @property
    def beat_window_ms(self) -> float:
        """``W = P + J``: the longest healthy gap between two receipts, in ms."""
        return self.heartbeat_period_ms + self.heartbeat_jitter_ms

    @property
    def tolerated_silence_ms(self) -> float:
        """``k * W``: longest silence the supervisor accepts as healthy, in ms.

        This is the quantity compared against the liveness deadline by check
        ``WS002``. A tolerance that lets the task be silent for longer than the
        deadline it is declared to protect cannot protect it, whatever the
        escalation ladder does afterwards.
        """
        return self.missed_beats_tolerated * self.beat_window_ms


@dataclass(frozen=True, slots=True)
class Spec:
    """A whole declared specification: one supervisor, some ladders, some tasks."""

    system: str
    supervisor: Supervisor
    ladders: tuple[Ladder, ...]
    tasks: tuple[Task, ...]
    notes: str = ""
    source: str = "<string>"
    spec_version: int = 1
    _ladder_index: dict[str, Ladder] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_ladder_index", {ladder.id: ladder for ladder in self.ladders})

    def ladder(self, ladder_id: str) -> Ladder | None:
        """Return the ladder with this identifier, or ``None`` if undeclared."""
        return self._ladder_index.get(ladder_id)

    def task(self, task_id: str) -> Task | None:
        """Return the task with this identifier, or ``None`` if undeclared."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
