"""Utilization bounds, response-time analysis and priority-ceiling blocking.

Every test in this module is a *sufficient* condition, a *necessary* one, or
*exact*, and each function's docstring says which. Confusing the three is the
single most common error in applied schedulability analysis, so the
:class:`SchedulabilityResult` returned by every test carries the word.

References
    [LL73] C. L. Liu and J. W. Layland, "Scheduling Algorithms for
        Multiprogramming in a Hard-Real-Time Environment", *Journal of the
        ACM* **20**(1), 46-61 (1973). Theorem 5 gives the rate-monotonic
        least upper bound ``n (2^(1/n) - 1)``; Theorem 7 gives the
        earliest-deadline-first condition ``U <= 1``.
    [JP86] M. Joseph and P. Pandya, "Finding Response Times in a Real-Time
        System", *The Computer Journal* **29**(5) (1986). Worst-case
        response time of a fixed-priority preemptive task set.
    [ABRTW93] N. C. Audsley, A. Burns, M. Richardson, K. Tindell and
        A. J. Wellings, "Applying new scheduling theory to static priority
        pre-emptive scheduling", *Software Engineering Journal* **8**(5)
        (1993). The fixed-point recurrence solved iteratively here.
    [SRL90] L. Sha, R. Rajkumar and J. P. Lehoczky, "Priority Inheritance
        Protocols: An Approach to Real-Time Synchronization", *IEEE
        Transactions on Computers* **39**(9) (1990). Under the priority
        ceiling protocol a task is blocked at most once per release.
    [But11] G. C. Buttazzo, *Hard Real-Time Computing Systems: Predictable
        Scheduling Algorithms and Applications*, 3rd ed., Springer (2011).
        Textbook anchor for all of the above, plus the hyperbolic bound of
        Bini, Buttazzo and Buttazzo (Sec. 4.3).

Shared assumptions for everything in this module, from [LL73] Sec. 2 unless
noted: one processor; fully preemptive scheduling with zero preemption and
context-switch cost; tasks independent except for the shared resources
modelled explicitly in :func:`priority_ceiling_blocking`; no release jitter;
execution times are worst case and are an upper bound, not a distribution;
periods and deadlines are exact. Units: seconds throughout; utilizations
dimensionless.

Validity range: ``1 <= n <= 10**4`` tasks, all durations strictly positive
and finite. The utilization bounds apply only to implicit-deadline sets
(``D_i = T_i``); the functions raise if given anything else rather than
returning a number outside its own validity range.

Known limitation of the recurrence
----------------------------------
The fixed point of ``R_i = C_i + B_i + sum ceil(R_i/T_j) C_j`` is the exact
worst-case response time only while ``R_i <= T_i``. Once ``R_i`` exceeds the
task's own period, more than one release of task ``i`` can be pending at
once and the exact analysis requires examining every release inside the
level-``i`` busy period (Lehoczky 1990; see [But11] Sec. 4.6). This package
reports the recurrence value, flags the deadline miss, and does not claim the
number is the exact worst case in that regime -- a task with ``R_i > T_i`` is
unschedulable either way, so the flag is sound even where the number is not
exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .taskset import PeriodicTask, TaskSet

__all__ = [
    "RESPONSE_TIME_MAX_ITERATIONS",
    "ResponseTime",
    "SchedulabilityResult",
    "edf_test",
    "hyperbolic_bound_test",
    "priority_ceiling_blocking",
    "response_time",
    "response_time_analysis",
    "rm_utilization_bound",
    "rm_utilization_test",
]

RESPONSE_TIME_MAX_ITERATIONS: int = 10_000
"""Iteration cap for the response-time fixed point. The recurrence is
monotone increasing and bounded above by the deadline test, so it either
converges or exceeds the deadline; the cap exists so a pathological input
cannot hang a caller."""


@dataclass(frozen=True)
class SchedulabilityResult:
    """Outcome of a schedulability test.

    Attributes:
        schedulable: the test's verdict. Read it together with ``strength``:
            ``False`` from a sufficient test means *unknown*, not *infeasible*.
        strength: ``"sufficient"``, ``"necessary"`` or ``"exact"``.
        test: name of the test.
        utilization: total utilization of the set, dimensionless.
        bound: the threshold compared against, dimensionless, or ``None`` when
            the test is not of threshold form.
        margin: ``bound - utilization`` when a bound exists, else ``None``.
        detail: one-line human-readable explanation.
    """

    schedulable: bool
    strength: str
    test: str
    utilization: float
    bound: float | None
    margin: float | None
    detail: str


@dataclass(frozen=True)
class ResponseTime:
    """Worst-case response time of one task.

    Attributes:
        name: task name.
        response_s: worst-case response time ``R``, units s.
        deadline_s: relative deadline ``D``, units s.
        blocking_s: blocking term ``B`` used, units s.
        iterations: fixed-point iterations taken (0 if the first estimate was
            already a fixed point).
        converged: True if a fixed point was reached within
            :data:`RESPONSE_TIME_MAX_ITERATIONS`.
        stopped_past_deadline: True if iteration stopped early because the
            iterate had already passed the deadline. The recurrence is
            non-decreasing, so once ``R > D`` no further iteration can bring
            it back and the task is unschedulable; stopping there is what
            keeps an overloaded set (``sum U > 1`` over the higher-priority
            tasks) from iterating to overflow.
        iterates: the full sequence of iterates ``R^(0), R^(1), ...``, units s,
            retained so a hand computation can be compared step by step.
    """

    name: str
    response_s: float
    deadline_s: float
    blocking_s: float
    iterations: int
    converged: bool
    iterates: tuple[float, ...]
    stopped_past_deadline: bool = False

    @property
    def meets_deadline(self) -> bool:
        """True when ``R <= D`` and the fixed point converged."""
        return self.converged and self.response_s <= self.deadline_s

    @property
    def slack_s(self) -> float:
        """``D - R``, units s. Negative means the deadline is missed."""
        return self.deadline_s - self.response_s


def rm_utilization_bound(n: int) -> float:
    """Rate-monotonic least upper bound ``U_lub(n) = n (2^(1/n) - 1)``.

    Reference [LL73] Theorem 5. Dimensionless. Decreasing in ``n``, from
    ``1.0`` at ``n = 1`` to ``ln 2 = 0.693147...`` as ``n -> inf``.

    Validity: ``n >= 1``. Applies to implicit-deadline (``D = T``) sets under
    rate-monotonic priority assignment on one processor with the [LL73]
    assumptions. The bound is *sufficient*: a set with ``U`` above it may
    still be schedulable (a harmonic set is schedulable up to ``U = 1``).

    Args:
        n: number of tasks, integer >= 1.

    Returns:
        Utilization bound, dimensionless.

    Raises:
        TypeError: if ``n`` is not an integer.
        ValueError: if ``n < 1``.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return n * (2.0 ** (1.0 / n) - 1.0)


def rm_utilization_test(task_set: TaskSet) -> SchedulabilityResult:
    """Liu & Layland sufficient test for rate-monotonic scheduling.

    Compares ``U = sum(C_i / T_i)`` against :func:`rm_utilization_bound`.
    Reference [LL73] Theorem 5. **Sufficient, not necessary.**

    Raises:
        ValueError: on an empty set, or if any task has ``D != T`` (the bound
            is not valid for constrained deadlines -- use
            :func:`response_time_analysis`).
    """
    if len(task_set) == 0:
        raise ValueError("task_set must contain at least one task")
    if not task_set.all_implicit_deadlines:
        offenders = [t.name for t in task_set if not t.has_implicit_deadline]
        raise ValueError(
            f"the Liu & Layland utilization bound assumes D == T; tasks {offenders} "
            "have D < T. Use response_time_analysis() instead."
        )
    n = len(task_set)
    u = task_set.total_utilization
    bound = rm_utilization_bound(n)
    ok = u <= bound
    return SchedulabilityResult(
        schedulable=ok,
        strength="sufficient",
        test="Liu & Layland RM utilization bound",
        utilization=u,
        bound=bound,
        margin=bound - u,
        detail=(
            f"U = {u:.12f} {'<=' if ok else '>'} n(2^(1/n)-1) = {bound:.12f} for n = {n}; "
            + (
                "schedulable under RM"
                if ok
                else "test inconclusive -- above a sufficient bound, run RTA"
            )
        ),
    )


def hyperbolic_bound_test(task_set: TaskSet) -> SchedulabilityResult:
    """Hyperbolic sufficient test: ``prod(U_i + 1) <= 2``.

    Bini, Buttazzo and Buttazzo, "Rate Monotonic Analysis: the Hyperbolic
    Bound"; see [But11] Sec. 4.3. **Sufficient, not necessary**, and strictly
    tighter than :func:`rm_utilization_test` -- every set the Liu & Layland
    bound accepts is accepted here, and some it rejects are accepted here.

    The reported ``utilization`` field is the total utilization for
    comparability; the quantity actually tested is the product, which appears
    in ``detail``.

    Raises:
        ValueError: on an empty set or a constrained-deadline set.
    """
    if len(task_set) == 0:
        raise ValueError("task_set must contain at least one task")
    if not task_set.all_implicit_deadlines:
        offenders = [t.name for t in task_set if not t.has_implicit_deadline]
        raise ValueError(
            f"the hyperbolic bound assumes D == T; tasks {offenders} have D < T. "
            "Use response_time_analysis() instead."
        )
    product = 1.0
    for t in task_set:
        product *= t.utilization + 1.0
    ok = product <= 2.0
    return SchedulabilityResult(
        schedulable=ok,
        strength="sufficient",
        test="hyperbolic bound",
        utilization=task_set.total_utilization,
        bound=2.0,
        margin=2.0 - product,
        detail=(
            f"prod(U_i + 1) = {product:.12f} {'<=' if ok else '>'} 2; "
            + ("schedulable under RM" if ok else "test inconclusive -- run RTA")
        ),
    )


def edf_test(task_set: TaskSet) -> SchedulabilityResult:
    """Earliest-deadline-first feasibility.

    For implicit deadlines (``D_i = T_i``) the condition ``U <= 1`` is
    **necessary and sufficient** -- exact -- on one processor with the [LL73]
    assumptions ([LL73] Theorem 7).

    For constrained deadlines (``D_i < T_i`` for some ``i``) the density test
    ``sum(C_i / D_i) <= 1`` is only **sufficient**; the exact test is the
    processor-demand criterion, which this package does not implement. The
    returned ``strength`` says which case applies, and ``bound`` is 1.0 in
    both.

    Raises:
        ValueError: on an empty set.
    """
    if len(task_set) == 0:
        raise ValueError("task_set must contain at least one task")
    if task_set.all_implicit_deadlines:
        u = task_set.total_utilization
        ok = u <= 1.0
        return SchedulabilityResult(
            schedulable=ok,
            strength="exact",
            test="EDF utilization test (implicit deadlines)",
            utilization=u,
            bound=1.0,
            margin=1.0 - u,
            detail=f"U = {u:.12f} {'<=' if ok else '>'} 1; necessary and sufficient under EDF",
        )
    d = task_set.total_density
    ok = d <= 1.0
    return SchedulabilityResult(
        schedulable=ok,
        strength="sufficient",
        test="EDF density test (constrained deadlines)",
        utilization=task_set.total_utilization,
        bound=1.0,
        margin=1.0 - d,
        detail=(
            f"sum(C_i/D_i) = {d:.12f} {'<=' if ok else '>'} 1; sufficient only -- the exact "
            "test for D < T is the processor-demand criterion, not implemented here"
        ),
    )


def response_time(
    task: PeriodicTask,
    higher_priority: list[PeriodicTask],
    blocking_s: float = 0.0,
    max_iterations: int = RESPONSE_TIME_MAX_ITERATIONS,
) -> ResponseTime:
    """Worst-case response time of one task by the fixed-point recurrence.

    ``R_i = C_i + B_i + sum_{j in hp(i)} ceil(R_i / T_j) * C_j``

    solved by the monotone iteration of [ABRTW93]::

        R^(0) = C_i + B_i
        R^(m+1) = C_i + B_i + sum_j ceil(R^(m) / T_j) * C_j

    The result is **exact** for the model: fixed-priority preemptive
    scheduling, one processor, zero overheads, no release jitter, critical
    instant at a simultaneous release ([JP86], [LL73] Lemma 1). The sequence
    is non-decreasing and either reaches a fixed point or passes the deadline;
    iteration stops at the first fixed point, at the first iterate strictly
    greater than the deadline, or at ``max_iterations``. Stopping at the
    deadline is not an approximation: the sequence cannot decrease, so a task
    whose iterate has passed its deadline is unschedulable whatever the fixed
    point would have been. It is also what keeps an overloaded set from
    iterating to floating-point overflow.

    Args:
        task: the task under analysis.
        higher_priority: tasks that can preempt it. Their priorities are not
            re-checked here -- :func:`response_time_analysis` does that.
        blocking_s: blocking term ``B``, units s, >= 0. Use
            :func:`priority_ceiling_blocking` to compute it.
        max_iterations: iteration cap, >= 1.

    Returns:
        A :class:`ResponseTime`.

    Raises:
        ValueError: if ``blocking_s`` is negative or ``max_iterations < 1``.
    """
    if blocking_s < 0.0:
        raise ValueError(f"blocking_s must be >= 0 s, got {blocking_s}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")

    deadline = task.deadline_s if task.deadline_s is not None else task.period_s
    base = task.wcet_s + float(blocking_s)
    r = base
    iterates = [r]
    converged = False
    stopped = False
    iterations = 0
    if r > deadline:
        stopped = True
    else:
        for _ in range(max_iterations):
            interference = math.fsum(
                math.ceil(r / j.period_s) * j.wcet_s for j in higher_priority
            )
            nxt = base + interference
            iterations += 1
            iterates.append(nxt)
            if nxt == r:
                converged = True
                break
            r = nxt
            if r > deadline:
                stopped = True
                break
    return ResponseTime(
        name=task.name,
        response_s=r,
        deadline_s=deadline,
        blocking_s=float(blocking_s),
        iterations=iterations,
        converged=converged,
        iterates=tuple(iterates),
        stopped_past_deadline=stopped,
    )


def response_time_analysis(
    task_set: TaskSet,
    blocking_s: dict[str, float] | None = None,
    max_iterations: int = RESPONSE_TIME_MAX_ITERATIONS,
) -> list[ResponseTime]:
    """Response-time analysis for every task in a prioritised set.

    Priorities must already be assigned (call
    :meth:`~rtclock.taskset.TaskSet.rate_monotonic` or
    :meth:`~rtclock.taskset.TaskSet.deadline_monotonic`). Results come back
    highest priority first.

    **Exact** for the model stated in the module docstring. A task whose
    recurrence does not converge within ``max_iterations`` is returned with
    ``converged=False`` and ``meets_deadline=False``; it is never reported as
    schedulable.

    Args:
        task_set: the set, with priorities assigned.
        blocking_s: per-task blocking terms by name, units s. Missing names
            get 0.0.
        max_iterations: iteration cap.

    Returns:
        One :class:`ResponseTime` per task, highest priority first.

    Raises:
        ValueError: on an empty set, an unassigned priority, or a
            ``blocking_s`` key that names no task.
    """
    if len(task_set) == 0:
        raise ValueError("task_set must contain at least one task")
    ordered = task_set.by_priority()
    blocking = dict(blocking_s or {})
    unknown = sorted(set(blocking) - {t.name for t in ordered})
    if unknown:
        raise ValueError(f"blocking_s names tasks not in the set: {unknown}")
    out: list[ResponseTime] = []
    for index, t in enumerate(ordered):
        out.append(
            response_time(
                t,
                higher_priority=ordered[:index],
                blocking_s=blocking.get(t.name, 0.0),
                max_iterations=max_iterations,
            )
        )
    return out


def priority_ceiling_blocking(
    task_set: TaskSet,
    critical_sections: dict[str, dict[str, float]],
    ceilings: dict[str, int] | None = None,
) -> dict[str, float]:
    """Worst-case blocking per task under the priority ceiling protocol.

    Reference [SRL90]; see also [But11] Sec. 7.5. Under PCP a task is blocked
    **at most once** per release, for at most the longest critical section of
    a lower-priority task guarded by a semaphore whose ceiling is greater than
    or equal to the task's priority::

        B_i = max { csec(j, s) : prio(j) < prio(i), ceiling(s) >= prio(i) }

    with ``B_i = 0`` when the set is empty. The ceiling of a semaphore is the
    highest priority of any task that uses it, which is what this function
    computes when ``ceilings`` is not supplied.

    Assumptions carried from [SRL90]: critical sections are properly nested,
    a task does not suspend inside one, every access is through a semaphore
    declared here, and the given durations are worst case. PCP also bounds
    deadlock, which this arithmetic does not model -- it computes the blocking
    term only.

    Args:
        task_set: the set, with priorities assigned.
        critical_sections: ``{task_name: {semaphore: duration_s}}``, durations
            > 0, units s.
        ceilings: optional explicit ``{semaphore: priority}``. Computed from
            ``critical_sections`` when omitted.

    Returns:
        ``{task_name: B_i}`` for every task in the set, units s.

    Raises:
        ValueError: on an unassigned priority, an unknown task name, a
            non-positive duration, or a ceiling for an unknown semaphore.
    """
    ordered = task_set.by_priority()
    known = {t.name: t for t in ordered}
    for tname, sections in critical_sections.items():
        if tname not in known:
            raise ValueError(f"critical_sections names unknown task {tname!r}")
        for sem, dur in sections.items():
            if dur <= 0.0:
                raise ValueError(
                    f"critical section {tname!r}/{sem!r} must have duration > 0 s, got {dur}"
                )

    all_semaphores = {s for sections in critical_sections.values() for s in sections}
    if ceilings is None:
        computed: dict[str, int] = {}
        for tname, sections in critical_sections.items():
            prio = known[tname].priority
            if prio is None:  # pragma: no cover - by_priority() already raises
                raise ValueError(f"task {tname!r} has no priority assigned")
            for sem in sections:
                computed[sem] = max(computed.get(sem, -(2**31)), int(prio))
        ceilings = computed
    else:
        unknown_sem = sorted(set(ceilings) - all_semaphores)
        if unknown_sem:
            raise ValueError(f"ceilings names semaphores never used: {unknown_sem}")
        missing = sorted(all_semaphores - set(ceilings))
        if missing:
            raise ValueError(f"no ceiling given for semaphores: {missing}")

    out: dict[str, float] = {}
    for t in ordered:
        prio_i = int(t.priority or 0)
        candidates = [
            dur
            for tname, sections in critical_sections.items()
            if int(known[tname].priority or 0) < prio_i
            for sem, dur in sections.items()
            if ceilings[sem] >= prio_i
        ]
        out[t.name] = max(candidates) if candidates else 0.0
    return out
