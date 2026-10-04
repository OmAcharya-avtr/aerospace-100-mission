"""Periodic task model and period/deadline arithmetic, with units checked.

Model (Liu & Layland 1973, Sec. 2, assumptions A1-A4): a task set of ``n``
independent periodic tasks, each task ``i`` released every ``T_i`` seconds,
requiring at most ``C_i`` seconds of processor time per release, to be
completed by a relative deadline ``D_i`` seconds after release, on a single
processor with preemption and zero context-switch cost.

Reference
    C. L. Liu and J. W. Layland, "Scheduling Algorithms for Multiprogramming
    in a Hard-Real-Time Environment", *Journal of the ACM* **20**(1), 46-61
    (1973). Assumptions: single processor, fully preemptive, tasks
    independent (no shared resources, no precedence), deadlines equal to
    periods (``D_i = T_i``, the *implicit-deadline* case), release jitter
    zero, overheads zero.

This module permits ``D_i <= T_i`` (*constrained* deadlines), which is outside
Liu & Layland's assumption A4 and is therefore handled only by response-time
analysis (:mod:`rtclock.schedulability`), never by a utilization bound. The
utilization-bound functions raise if given a constrained-deadline set, rather
than silently applying a bound outside its validity range.

Units: every duration is seconds (s). Utilization is dimensionless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["PeriodicTask", "TaskSet", "hyperperiod_s"]


@dataclass(frozen=True)
class PeriodicTask:
    """One periodic task.

    Attributes:
        name: identifier, non-empty.
        period_s: release period ``T``, units s, must be > 0.
        wcet_s: worst-case execution time ``C``, units s, must be > 0.
        deadline_s: relative deadline ``D``, units s. Defaults to
            ``period_s`` (implicit deadline). Must satisfy
            ``wcet_s <= deadline_s <= period_s``.
        priority: integer priority; **larger means higher priority**. If left
            at ``None``, :meth:`TaskSet.rate_monotonic` assigns it.
    """

    name: str
    period_s: float
    wcet_s: float
    deadline_s: float | None = None
    priority: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        for label, value in (("period_s", self.period_s), ("wcet_s", self.wcet_s)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{label} must be a real number, got {type(value).__name__}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite, got {value}")
            if float(value) <= 0.0:
                raise ValueError(f"{label} must be > 0 s, got {value}")
        object.__setattr__(self, "period_s", float(self.period_s))
        object.__setattr__(self, "wcet_s", float(self.wcet_s))
        if self.deadline_s is None:
            object.__setattr__(self, "deadline_s", self.period_s)
        else:
            d = float(self.deadline_s)
            if not math.isfinite(d) or d <= 0.0:
                raise ValueError(f"deadline_s must be a finite value > 0 s, got {self.deadline_s}")
            if d > self.period_s:
                raise ValueError(
                    f"deadline_s ({d} s) > period_s ({self.period_s} s): arbitrary-deadline "
                    "task sets are outside the scope of this package"
                )
            if d < self.wcet_s:
                raise ValueError(
                    f"deadline_s ({d} s) < wcet_s ({self.wcet_s} s): task can never complete"
                )
            object.__setattr__(self, "deadline_s", d)

    @property
    def utilization(self) -> float:
        """Processor utilization ``C / T``, dimensionless."""
        return self.wcet_s / self.period_s

    @property
    def density(self) -> float:
        """Density ``C / min(D, T)``, dimensionless. Equals utilization when ``D = T``."""
        return self.wcet_s / min(self.deadline_s, self.period_s)

    @property
    def has_implicit_deadline(self) -> bool:
        """True when ``D == T`` exactly."""
        return self.deadline_s == self.period_s

    def releases_in(self, window_s: float) -> int:
        """Number of releases in a window of ``window_s`` seconds starting at a release.

        ``ceil(window / T)`` -- the count used by the response-time recurrence.

        Raises:
            ValueError: if ``window_s`` is negative.
        """
        if window_s < 0.0:
            raise ValueError(f"window_s must be >= 0 s, got {window_s}")
        return math.ceil(window_s / self.period_s)


@dataclass
class TaskSet:
    """An ordered collection of :class:`PeriodicTask`, names unique."""

    tasks: list[PeriodicTask] = field(default_factory=list)

    def __post_init__(self) -> None:
        for t in self.tasks:
            if not isinstance(t, PeriodicTask):
                raise TypeError(f"tasks must be PeriodicTask, got {type(t).__name__}")
        names = [t.name for t in self.tasks]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"task names must be unique; duplicated: {dupes}")

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def __getitem__(self, index: int) -> PeriodicTask:
        return self.tasks[index]

    @property
    def total_utilization(self) -> float:
        """``sum(C_i / T_i)``, dimensionless."""
        return math.fsum(t.utilization for t in self.tasks)

    @property
    def total_density(self) -> float:
        """``sum(C_i / min(D_i, T_i))``, dimensionless."""
        return math.fsum(t.density for t in self.tasks)

    @property
    def all_implicit_deadlines(self) -> bool:
        """True when every task has ``D == T``."""
        return all(t.has_implicit_deadline for t in self.tasks)

    def rate_monotonic(self) -> TaskSet:
        """Return a copy with rate-monotonic priorities assigned.

        Shortest period gets the highest priority (Liu & Layland 1973,
        Theorem 2: RM is optimal among fixed-priority assignments for
        implicit-deadline sets). Ties are broken by the task's position in the
        original list, which is arbitrary but deterministic. Priorities are
        ``n-1`` down to ``0`` with larger meaning higher.
        """
        order = sorted(range(len(self.tasks)), key=lambda i: (self.tasks[i].period_s, i))
        n = len(self.tasks)
        new: list[PeriodicTask | None] = [None] * n
        for rank, idx in enumerate(order):
            t = self.tasks[idx]
            new[idx] = PeriodicTask(t.name, t.period_s, t.wcet_s, t.deadline_s, n - 1 - rank)
        return TaskSet([t for t in new if t is not None])

    def deadline_monotonic(self) -> TaskSet:
        """Return a copy with deadline-monotonic priorities assigned.

        Shortest relative deadline gets the highest priority. Optimal among
        fixed-priority assignments for constrained-deadline sets
        (Leung & Whitehead 1982; see also Buttazzo, *Hard Real-Time Computing
        Systems*, 3rd ed., Springer 2011, Sec. 4.5).
        """
        order = sorted(range(len(self.tasks)), key=lambda i: (self.tasks[i].deadline_s, i))
        n = len(self.tasks)
        new: list[PeriodicTask | None] = [None] * n
        for rank, idx in enumerate(order):
            t = self.tasks[idx]
            new[idx] = PeriodicTask(t.name, t.period_s, t.wcet_s, t.deadline_s, n - 1 - rank)
        return TaskSet([t for t in new if t is not None])

    def by_priority(self) -> list[PeriodicTask]:
        """Tasks sorted highest priority first.

        Raises:
            ValueError: if any task has no priority assigned.
        """
        missing = [t.name for t in self.tasks if t.priority is None]
        if missing:
            raise ValueError(
                f"tasks {missing} have no priority; call rate_monotonic() or "
                "deadline_monotonic() first, or set priority explicitly"
            )
        return sorted(self.tasks, key=lambda t: -int(t.priority or 0))

    def higher_priority_than(self, task: PeriodicTask) -> list[PeriodicTask]:
        """Tasks with strictly higher priority than ``task``."""
        if task.priority is None:
            raise ValueError(f"task {task.name!r} has no priority assigned")
        return [t for t in self.tasks if t.priority is not None and t.priority > task.priority]

    def lower_priority_than(self, task: PeriodicTask) -> list[PeriodicTask]:
        """Tasks with strictly lower priority than ``task``."""
        if task.priority is None:
            raise ValueError(f"task {task.name!r} has no priority assigned")
        return [t for t in self.tasks if t.priority is not None and t.priority < task.priority]


def hyperperiod_s(periods_s: list[float], quantum_s: float = 1.0e-9) -> float:
    """Least common multiple of a list of periods, computed on an integer grid.

    Periods are rounded to the nearest multiple of ``quantum_s`` (default 1 ns)
    and the integer LCM is taken, because the LCM of floats is not well
    defined. The returned value is exact for periods that are exact multiples
    of the quantum, and is otherwise the LCM of the rounded values -- which is
    the honest answer, not an approximation of an undefined quantity.

    Args:
        periods_s: strictly positive periods, units s.
        quantum_s: grid step, units s, must be > 0.

    Returns:
        Hyperperiod, units s.

    Raises:
        ValueError: on an empty list, a non-positive period, or a period that
            rounds to zero on the given grid.
    """
    if not periods_s:
        raise ValueError("periods_s must be non-empty")
    if quantum_s <= 0.0:
        raise ValueError(f"quantum_s must be > 0 s, got {quantum_s}")
    ticks = []
    for p in periods_s:
        if p <= 0.0:
            raise ValueError(f"every period must be > 0 s, got {p}")
        n = round(p / quantum_s)
        if n <= 0:
            raise ValueError(f"period {p} s rounds to zero on a {quantum_s} s grid")
        ticks.append(int(n))
    return math.lcm(*ticks) * quantum_s
