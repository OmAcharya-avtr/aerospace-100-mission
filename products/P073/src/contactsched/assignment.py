"""The result of scheduling: which candidates were selected, and what that cost.

An :class:`Assignment` is deliberately thin. It carries the selected candidate
indices, the objective value, a status string and free-text notes, and nothing
derived. Everything derived -- per-mission seconds, satisfaction ratios,
fairness indices, starvation -- lives in :mod:`contactsched.fairness` and is
computed from the problem and the assignment together, so that two assignments
produced by different methods are always measured by the same code.
"""

from __future__ import annotations

from dataclasses import dataclass

from .problem import SchedulingProblem

STATUS_OPTIMAL = "optimal"
"""The method proved this is a best assignment over the candidate grid."""

STATUS_HEURISTIC = "heuristic"
"""Feasible, with no optimality claim. The greedy baseline returns this."""

STATUS_INFEASIBLE = "infeasible"
"""No assignment satisfies the constraints the method was asked to enforce."""

STATUS_LIMIT = "limit"
"""The method stopped on a time or node limit; the result may be suboptimal."""


@dataclass(frozen=True)
class Assignment:
    """A set of selected candidates and how it was obtained.

    Attributes
    ----------
    method : str
        Name of the producing method, for reports and tables.
    status : str
        One of the module's status constants.
    selected : tuple of int
        Candidate indices into :attr:`SchedulingProblem.candidates`, sorted.
    objective : float
        Value of the objective the method optimised. Comparable only between
        assignments built on the same problem with the same objective mode.
    objective_mode : str
        ``"weighted_contacts"`` or ``"weighted_seconds"``; see
        :mod:`contactsched.exact`.
    notes : tuple of str
        Diagnostics. For an infeasible exact solve these name the constraints
        that could not be met, which is the output a request renegotiation
        starts from.
    """

    method: str
    status: str
    selected: tuple[int, ...]
    objective: float
    objective_mode: str = "weighted_contacts"
    notes: tuple[str, ...] = ()

    @property
    def contact_count(self) -> int:
        """Number of contacts scheduled."""
        return len(self.selected)

    def contacts_by_request(self, problem: SchedulingProblem) -> dict[str, int]:
        """Contacts granted per request, including zeros."""
        counts = {r.request_id: 0 for r in problem.requests}
        for index in self.selected:
            counts[problem.candidates[index].request_id] += 1
        return counts

    def seconds_by_mission(self, problem: SchedulingProblem) -> dict[str, float]:
        """Allocated contact seconds per mission, including zeros."""
        totals = {m: 0.0 for m in problem.missions}
        for index in self.selected:
            candidate = problem.candidates[index]
            totals[candidate.mission] = totals.get(candidate.mission, 0.0) + candidate.duration_s
        return totals

    def seconds_by_station(self, problem: SchedulingProblem) -> dict[str, float]:
        """Allocated contact seconds per antenna, including zeros."""
        totals = {s.name: 0.0 for s in problem.stations}
        for index in self.selected:
            candidate = problem.candidates[index]
            totals[candidate.station] = totals.get(candidate.station, 0.0) + candidate.duration_s
        return totals

    def timeline(self, problem: SchedulingProblem) -> list[tuple[float, float, str, str, str]]:
        """Selected contacts as ``(start_s, end_s, station, spacecraft, request_id)``.

        Sorted by start time, which is the order an operator reads a schedule in.
        """
        rows = [
            (
                problem.candidates[i].start_s,
                problem.candidates[i].end_s,
                problem.candidates[i].station,
                problem.candidates[i].spacecraft,
                problem.candidates[i].request_id,
            )
            for i in self.selected
        ]
        rows.sort()
        return rows


def objective_value(
    problem: SchedulingProblem, selected: tuple[int, ...], mode: str = "weighted_contacts"
) -> float:
    """Objective of an arbitrary selection, for scoring any method's output.

    Parameters
    ----------
    problem : SchedulingProblem
        The instance.
    selected : tuple of int
        Candidate indices.
    mode : str
        ``"weighted_contacts"`` sums the priority weights.
        ``"weighted_seconds"`` sums weight times contact duration.

    Raises
    ------
    ValueError
        On an unknown mode.
    """
    if mode == "weighted_contacts":
        return float(sum(problem.candidates[i].weight for i in selected))
    if mode == "weighted_seconds":
        return float(
            sum(problem.candidates[i].weight * problem.candidates[i].duration_s for i in selected)
        )
    raise ValueError(
        f"unknown objective mode {mode!r}; expected 'weighted_contacts' or 'weighted_seconds'"
    )
