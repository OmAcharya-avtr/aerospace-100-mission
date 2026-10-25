"""Contention and fairness accounting: who was starved, and by how much.

This module is the contribution of the package. Computing visibility windows is
``sgp4`` plus spherical trigonometry, and solving the assignment is HiGHS; both
are mature and neither is novel here. What is not available off the shelf is the
accounting that answers, after the schedule is fixed:

* which requests fell below the contact count they declared they needed, and by
  how many seconds;
* how unequally the satisfied fraction was distributed, as a max-min share and
  as Jain's index;
* whether the priority scheme delivered what it promised, measured as the number
  of **priority inversions** -- pairs where a request of strictly higher
  priority ended with a strictly lower satisfied fraction than one below it.

Definitions, stated because every one of them is a choice
---------------------------------------------------------
**Requested seconds** of a request are ``min_contacts * duration_s``. Not
``max_contacts``: the minimum is the number the requester declared they needed,
and measuring satisfaction against a ceiling would make every schedule look
worse than it is.

**Satisfaction** is allocated seconds over requested seconds, capped at 1 for
the fairness indices and reported uncapped as well. A request whose
``min_contacts`` is zero has no demand, cannot be starved, and is excluded from
the indices; it is listed separately in the report so the exclusion is visible.

**Jain's index** (Jain, Chiu and Hawe, 1984) is
``(sum x)**2 / (n * sum x**2)`` over the capped satisfactions, which lies in
``[1/n, 1]``, equals 1 when all are equal, and is independent of scale. The
all-zero case is scale-free in a degenerate way: every request got nothing, so
the allocation *is* perfectly equal, and this module returns 1.0 there. That is
a convention, not a measurement, it is returned alongside
``all_starved=True``, and it is the reason the max-min share is reported
next to it -- the max-min share of an all-zero allocation is 0.0 and says the
thing the index cannot.

**Priority inversion.** The declared promise of an integer priority is that a
lower number is served no worse. Inversions are counted over ordered pairs of
requests with strictly different priorities, with a tolerance, because a
satisfaction difference of 1e-16 between two requests is a floating-point
artefact and not an inversion. The rate is inversions over comparable pairs.

Jain's index and the max-min share are *descriptions of one schedule*. They do
not say the schedule is fair, they say how equal this allocation is against this
declared demand. A perfectly equal allocation of nothing scores 1.

References
----------
Jain, R., Chiu, D.-M. and Hawe, W. (1984). *A Quantitative Measure of Fairness
and Discrimination for Resource Allocation in Shared Computer Systems*, DEC
Technical Report TR-301.

Bertsekas, D. and Gallager, R. (1992). *Data Networks*, 2nd edition,
Prentice-Hall, section 6.5.2, for max-min fairness as the lexicographic
maximisation of the smallest share.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .assignment import Assignment
from .problem import SchedulingProblem

INVERSION_TOLERANCE = 1e-9
"""Satisfaction difference below which two requests count as equally served."""


def jain_index(shares: list[float]) -> float:
    """Jain's fairness index of ``shares``, dimensionless in ``[1/n, 1]``.

    Returns 1.0 for an empty list and for an all-zero list; see the module
    docstring for why the all-zero convention is a convention.

    Raises
    ------
    ValueError
        If any share is negative.
    """
    values = [float(s) for s in shares]
    if any(v < 0.0 for v in values):
        raise ValueError(f"shares must be non-negative, got {values}")
    if not values:
        return 1.0
    scale = max(values)
    if scale == 0.0:
        return 1.0
    # Normalise by the largest share BEFORE squaring. The index is scale
    # invariant in exact arithmetic, so this changes nothing mathematically, and
    # it is not cosmetic: on the un-normalised expression a share of 5e-324 --
    # the smallest positive double -- has a non-zero sum and a sum of squares
    # that underflows to exactly zero, and the function raised
    # ZeroDivisionError. Hypothesis found that in
    # tests/test_properties.py::TestFairnessIndices; none of the hand-written
    # known-answer tests did. The literal case is pinned in
    # tests/test_regressions.py.
    normalised = [v / scale for v in values]
    total = sum(normalised)
    squared = sum(v * v for v in normalised)
    return float(total * total / (len(values) * squared))


def max_min_share(shares: list[float]) -> float:
    """Smallest share, dimensionless. 1.0 for an empty list.

    The max-min *share* of an allocation is its smallest component. Maximising
    it over all allocations is max-min fairness; this function reports the value
    achieved by the allocation it is given and claims nothing about optimality.
    """
    values = [float(s) for s in shares]
    return min(values) if values else 1.0


@dataclass(frozen=True)
class RequestOutcome:
    """What one request actually got.

    Attributes
    ----------
    request_id, mission : str
        Identifiers.
    priority : int
        1 highest.
    requested_contacts, granted_contacts : int
        Declared minimum and delivered count.
    requested_seconds, allocated_seconds : float
        Seconds.
    satisfaction : float
        ``allocated_seconds / requested_seconds``, capped at 1.0. For a request
        with no declared minimum this is 1.0 and :attr:`has_demand` is False.
    satisfaction_uncapped : float
        The same ratio without the cap, so over-service is visible.
    has_demand : bool
        False when ``min_contacts`` is zero.
    """

    request_id: str
    mission: str
    priority: int
    requested_contacts: int
    granted_contacts: int
    requested_seconds: float
    allocated_seconds: float
    satisfaction: float
    satisfaction_uncapped: float
    has_demand: bool

    @property
    def contacts_short(self) -> int:
        """Contacts below the declared minimum, never negative."""
        return max(0, self.requested_contacts - self.granted_contacts)

    @property
    def seconds_short(self) -> float:
        """Seconds below the declared minimum, never negative."""
        return max(0.0, self.requested_seconds - self.allocated_seconds)

    @property
    def starved(self) -> bool:
        """True when a request with declared demand is below its minimum."""
        return self.has_demand and self.contacts_short > 0


@dataclass(frozen=True)
class Inversion:
    """One pair where priority order and satisfaction order disagree.

    Attributes
    ----------
    higher_request, lower_request : str
        Request ids. ``higher_request`` has the numerically smaller, that is
        more urgent, priority.
    higher_priority, lower_priority : int
        The two priorities.
    higher_satisfaction, lower_satisfaction : float
        Capped satisfactions, with ``higher_satisfaction`` the smaller of the
        two -- which is what makes this an inversion.
    """

    higher_request: str
    lower_request: str
    higher_priority: int
    lower_priority: int
    higher_satisfaction: float
    lower_satisfaction: float

    @property
    def gap(self) -> float:
        """How far the promise was missed, dimensionless satisfaction units."""
        return float(self.lower_satisfaction - self.higher_satisfaction)


@dataclass(frozen=True)
class FairnessReport:
    """Everything derived from one problem and one assignment together.

    Attributes
    ----------
    method : str
        Copied from the assignment, so a report can be read on its own.
    outcomes : tuple of RequestOutcome
        One per request, in declaration order.
    mission_satisfaction : dict
        ``{mission: capped satisfaction}``, aggregated over the mission's
        requests by summing seconds, not by averaging ratios. Averaging ratios
        would let a mission with one tiny satisfied request outscore a mission
        with one large starved one.
    jain : float
        Jain's index over per-request capped satisfactions with declared demand.
    jain_by_mission : float
        The same over per-mission satisfactions. Reported separately because
        they answer different questions and can point in opposite directions
        when one mission owns several spacecraft.
    max_min : float
        Smallest per-request capped satisfaction among requests with demand.
    inversions : tuple of Inversion
        Every priority inversion found.
    comparable_pairs : int
        Ordered request pairs with strictly different priorities and declared
        demand; the denominator of :attr:`inversion_rate`.
    all_starved : bool
        True when every request with demand received nothing, the case in which
        :attr:`jain` returns its degenerate 1.0.
    delivered_by_priority : dict
        ``{priority: mean capped satisfaction}``, the measured delivery of the
        priority scheme.
    """

    method: str
    outcomes: tuple[RequestOutcome, ...]
    mission_satisfaction: dict[str, float]
    jain: float
    jain_by_mission: float
    max_min: float
    inversions: tuple[Inversion, ...]
    comparable_pairs: int
    all_starved: bool
    delivered_by_priority: dict[int, float] = field(default_factory=dict)

    @property
    def starved(self) -> tuple[RequestOutcome, ...]:
        """Requests below their declared minimum."""
        return tuple(o for o in self.outcomes if o.starved)

    @property
    def total_seconds_short(self) -> float:
        """Sum of per-request shortfalls, seconds."""
        return float(sum(o.seconds_short for o in self.outcomes))

    @property
    def inversion_count(self) -> int:
        """Number of priority inversions."""
        return len(self.inversions)

    @property
    def inversion_rate(self) -> float:
        """Inversions over comparable pairs, dimensionless. 0.0 when none exist."""
        if self.comparable_pairs == 0:
            return 0.0
        return float(len(self.inversions)) / float(self.comparable_pairs)

    @property
    def worst_inversion_gap(self) -> float:
        """Largest satisfaction gap across all inversions. 0.0 when none exist."""
        return max((i.gap for i in self.inversions), default=0.0)

    @property
    def priority_monotone(self) -> bool:
        """True when mean delivered satisfaction does not rise with priority number.

        This is the aggregate form of the promise. It can hold while individual
        inversions exist, which is why both are reported.
        """
        items = sorted(self.delivered_by_priority.items())
        return all(
            items[i][1] + INVERSION_TOLERANCE >= items[i + 1][1] for i in range(len(items) - 1)
        )


def analyse(
    problem: SchedulingProblem,
    assignment: Assignment,
    tolerance: float = INVERSION_TOLERANCE,
) -> FairnessReport:
    """Compute the full contention and fairness report.

    Parameters
    ----------
    problem : SchedulingProblem
        The instance the assignment was built on.
    assignment : Assignment
        Any assignment, from any method, feasible or not. This function does not
        check feasibility; :func:`contactsched.problem.check_feasible` does, and
        keeping them separate is what lets a report be produced for a schedule
        that is known to be broken.
    tolerance : float
        Satisfaction difference below which two requests count as equally
        served, dimensionless.

    Returns
    -------
    FairnessReport
    """
    allocated: dict[str, float] = {r.request_id: 0.0 for r in problem.requests}
    granted: dict[str, int] = {r.request_id: 0 for r in problem.requests}
    for index in assignment.selected:
        candidate = problem.candidates[index]
        allocated[candidate.request_id] += candidate.duration_s
        granted[candidate.request_id] += 1

    outcomes: list[RequestOutcome] = []
    for request in problem.requests:
        requested_seconds = request.requested_seconds
        got = allocated[request.request_id]
        has_demand = requested_seconds > 0.0
        uncapped = got / requested_seconds if has_demand else 1.0
        outcomes.append(
            RequestOutcome(
                request_id=request.request_id,
                mission=request.mission,
                priority=int(request.priority),
                requested_contacts=int(request.min_contacts),
                granted_contacts=granted[request.request_id],
                requested_seconds=float(requested_seconds),
                allocated_seconds=float(got),
                satisfaction=float(min(1.0, uncapped)),
                satisfaction_uncapped=float(uncapped),
                has_demand=has_demand,
            )
        )

    with_demand = [o for o in outcomes if o.has_demand]
    shares = [o.satisfaction for o in with_demand]

    mission_requested: dict[str, float] = {}
    mission_allocated: dict[str, float] = {}
    for outcome in outcomes:
        if not outcome.has_demand:
            continue
        mission_requested[outcome.mission] = (
            mission_requested.get(outcome.mission, 0.0) + outcome.requested_seconds
        )
        mission_allocated[outcome.mission] = (
            mission_allocated.get(outcome.mission, 0.0) + outcome.allocated_seconds
        )
    mission_satisfaction = {
        mission: float(min(1.0, mission_allocated[mission] / requested))
        for mission, requested in mission_requested.items()
        if requested > 0.0
    }

    inversions: list[Inversion] = []
    comparable = 0
    for i, first in enumerate(with_demand):
        for second in with_demand[i + 1 :]:
            if first.priority == second.priority:
                continue
            comparable += 1
            if first.priority < second.priority:
                high, low = first, second
            else:
                high, low = second, first
            if low.satisfaction - high.satisfaction > float(tolerance):
                inversions.append(
                    Inversion(
                        higher_request=high.request_id,
                        lower_request=low.request_id,
                        higher_priority=high.priority,
                        lower_priority=low.priority,
                        higher_satisfaction=high.satisfaction,
                        lower_satisfaction=low.satisfaction,
                    )
                )

    by_priority: dict[int, list[float]] = {}
    for outcome in with_demand:
        by_priority.setdefault(outcome.priority, []).append(outcome.satisfaction)
    delivered = {p: float(sum(v) / len(v)) for p, v in by_priority.items()}

    return FairnessReport(
        method=assignment.method,
        outcomes=tuple(outcomes),
        mission_satisfaction=mission_satisfaction,
        jain=jain_index(shares),
        jain_by_mission=jain_index(list(mission_satisfaction.values())),
        max_min=max_min_share(shares),
        inversions=tuple(inversions),
        comparable_pairs=comparable,
        all_starved=bool(with_demand) and all(s == 0.0 for s in shares),
        delivered_by_priority=delivered,
    )
