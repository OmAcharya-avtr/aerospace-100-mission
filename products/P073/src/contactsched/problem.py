"""Requests, candidate slots, and what makes an assignment feasible.

The scheduling model
--------------------
A :class:`ContactRequest` asks for ``min_contacts`` to ``max_contacts``
contiguous contacts of ``duration_s`` seconds each, for one spacecraft, on any
antenna from a declared allowed set. A :class:`Candidate` is one concrete
placement of one such contact: a request, an antenna and a start time inside a
visibility window. Candidates are generated on a declared grid inside each
window, so the model is a *discretisation* of continuous placement and its
optimum is the optimum over that grid and not over the continuum. The grid step
is a parameter, the discretisation is stated in README.md under Limitations, and
``validation/validate_exact.py`` reports how the optimum moves as the grid is
refined.

An assignment is feasible when

1. no two selected candidates on the same antenna overlap, once the antenna's
   turnaround time is added to each;
2. no two selected candidates for the same spacecraft overlap, because a
   spacecraft has one radio;
3. no request takes more than its ``max_contacts``;
4. every selected candidate lies inside the visibility window it was generated
   from.

Minimum contact counts are deliberately **not** in that list. They are a
*fairness constraint* that the exact method can be asked to enforce, and a
quantity the greedy method can miss. Treating a missed minimum as infeasibility
rather than as a measured shortfall would throw away the number this package
exists to report, so :func:`check_feasible` does not require it and
:func:`shortfalls` measures it.

Units: seconds, degrees. Priority is an integer with 1 highest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .station import GroundStation
from .visibility import ContactWindow


def priority_weight(priority: int, base: float = 2.0) -> float:
    """Objective weight of one contact at ``priority``, dimensionless.

    ``base ** (1 - priority)``, so priority 1 has weight 1, priority 2 has
    weight 1/2 at the default base, and so on.

    This function is the *declared promise* of the priority scheme, and naming
    it matters. A geometric weight says that one priority-1 contact is worth
    more than any number of lower-priority contacts only when ``base`` exceeds
    the number of competitors; at ``base=2`` with five priority-2 requests, the
    solver will quite correctly drop a priority-1 contact to gain three
    priority-2 contacts. That is not a bug in the solver, it is what the weights
    asked for, and it is exactly the gap between promise and delivery that
    :mod:`contactsched.fairness` measures.

    Raises
    ------
    ValueError
        If ``priority`` is below 1 or ``base`` is not above 1.
    """
    if int(priority) < 1:
        raise ValueError(f"priority must be 1 or greater, got {priority}")
    if float(base) <= 1.0:
        raise ValueError(f"weight base must be above 1, got {base}")
    return float(base) ** (1 - int(priority))


@dataclass(frozen=True)
class ContactRequest:
    """One mission's demand for contact time with one spacecraft.

    Attributes
    ----------
    request_id : str
        Unique label.
    mission : str
        Owning mission. Fairness is accounted per mission.
    spacecraft : str
        Spacecraft label, matching a :class:`~contactsched.ephemeris.Spacecraft`.
    priority : int
        1 is highest. Converted to an objective weight by
        :func:`priority_weight`.
    duration_s : float
        Required contiguous contact length, seconds. Must be positive.
    min_contacts : int
        Contacts below which the request is *starved*. Not a feasibility
        constraint unless the exact solver is asked to enforce it.
    max_contacts : int
        Contacts above which further contacts have no value. Must be at least
        ``min_contacts`` and at least 1.
    stations : tuple of str or None
        Allowed antenna names, or ``None`` for any antenna.
    """

    request_id: str
    mission: str
    spacecraft: str
    priority: int
    duration_s: float
    min_contacts: int = 1
    max_contacts: int = 2
    stations: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not str(self.request_id):
            raise ValueError("request_id must be a non-empty string")
        if int(self.priority) < 1:
            raise ValueError(f"priority must be 1 or greater, got {self.priority}")
        if float(self.duration_s) <= 0.0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")
        if int(self.min_contacts) < 0:
            raise ValueError(f"min_contacts must be non-negative, got {self.min_contacts}")
        if int(self.max_contacts) < 1:
            raise ValueError(f"max_contacts must be at least 1, got {self.max_contacts}")
        if int(self.max_contacts) < int(self.min_contacts):
            raise ValueError(
                f"max_contacts {self.max_contacts} is below min_contacts {self.min_contacts}"
            )

    @property
    def requested_seconds(self) -> float:
        """Seconds the request would need to reach :attr:`min_contacts`."""
        return float(self.duration_s) * int(self.min_contacts)

    def allows(self, station: str) -> bool:
        """True when ``station`` is in the allowed set."""
        return self.stations is None or station in self.stations


@dataclass(frozen=True)
class Candidate:
    """One placement of one contact: request, antenna, start time.

    Attributes
    ----------
    index : int
        Position in the problem's candidate list; the decision-variable index.
    request_id, mission, spacecraft, station : str
        Identifiers carried for reporting without a lookup.
    start_s, end_s : float
        Seconds after the scenario epoch.
    weight : float
        Objective weight, from :func:`priority_weight`.
    max_elevation_deg : float
        Peak elevation of the window this candidate sits in, degrees. Carried as
        a feature for the conflict predictor and for reporting; it is not used
        by either scheduler.
    """

    index: int
    request_id: str
    mission: str
    spacecraft: str
    station: str
    start_s: float
    end_s: float
    weight: float
    max_elevation_deg: float

    @property
    def duration_s(self) -> float:
        """Contact length, seconds."""
        return float(self.end_s) - float(self.start_s)


@dataclass
class SchedulingProblem:
    """Windows, requests, candidates and the resource rules that bind them.

    Attributes
    ----------
    stations : tuple of GroundStation
        The antennas. Turnaround times come from here.
    requests : tuple of ContactRequest
        The demand.
    windows : tuple of ContactWindow
        The supply, as computed by :mod:`contactsched.visibility`.
    candidates : tuple of Candidate
        Discretised placements.
    slot_step_s : float
        Grid step used to generate candidates, seconds.
    spacecraft_turnaround_s : float
        Separation required between two contacts of the same spacecraft on
        different antennas, seconds. Zero by default: a spacecraft switching
        ground stations has no mechanical constraint, only the radio exclusion
        already enforced by non-overlap.
    """

    stations: tuple[GroundStation, ...]
    requests: tuple[ContactRequest, ...]
    windows: tuple[ContactWindow, ...]
    candidates: tuple[Candidate, ...]
    slot_step_s: float
    spacecraft_turnaround_s: float = 0.0
    _turnaround: dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._turnaround = {s.name: float(s.turnaround_s) for s in self.stations}
        seen: set[str] = set()
        for request in self.requests:
            if request.request_id in seen:
                raise ValueError(f"duplicate request_id {request.request_id!r}")
            seen.add(request.request_id)

    @property
    def request_by_id(self) -> dict[str, ContactRequest]:
        """Requests keyed by :attr:`ContactRequest.request_id`."""
        return {r.request_id: r for r in self.requests}

    @property
    def missions(self) -> tuple[str, ...]:
        """Mission identifiers present in the requests, sorted."""
        return tuple(sorted({r.mission for r in self.requests}))

    def turnaround_s(self, station: str) -> float:
        """Turnaround time of ``station``, seconds."""
        return self._turnaround.get(station, 0.0)

    def candidates_of(self, request_id: str) -> tuple[Candidate, ...]:
        """Candidates belonging to one request, in generation order."""
        return tuple(c for c in self.candidates if c.request_id == request_id)

    def station_conflict(self, a: Candidate, b: Candidate) -> bool:
        """True when ``a`` and ``b`` cannot both be served by their antenna."""
        if a.station != b.station or a.index == b.index:
            return False
        gap = self.turnaround_s(a.station)
        return a.start_s < b.end_s + gap and b.start_s < a.end_s + gap

    def spacecraft_conflict(self, a: Candidate, b: Candidate) -> bool:
        """True when ``a`` and ``b`` need the same spacecraft radio at once."""
        if a.spacecraft != b.spacecraft or a.index == b.index:
            return False
        gap = float(self.spacecraft_turnaround_s)
        return a.start_s < b.end_s + gap and b.start_s < a.end_s + gap

    def conflicts(self, a: Candidate, b: Candidate) -> bool:
        """True when ``a`` and ``b`` cannot both be selected."""
        return self.station_conflict(a, b) or self.spacecraft_conflict(a, b)


def build_candidates(
    requests: tuple[ContactRequest, ...],
    windows: tuple[ContactWindow, ...],
    slot_step_s: float = 120.0,
    weight_base: float = 2.0,
    align_to_window_end: bool = True,
) -> tuple[Candidate, ...]:
    """Discretise placements of every request into every compatible window.

    Parameters
    ----------
    requests : tuple of ContactRequest
        The demand.
    windows : tuple of ContactWindow
        Supply, from :func:`contactsched.visibility.all_windows`.
    slot_step_s : float
        Grid step inside each window, seconds. Must be positive.
    weight_base : float
        Base of the geometric priority weight; see :func:`priority_weight`.
    align_to_window_end : bool
        When true, a candidate flush with the end of the window is added
        whenever the grid does not already land there. This matters: without it
        a request whose duration is just under the window length can be
        unplaceable purely because the grid missed, which is a discretisation
        artefact rather than a contention result.

    Returns
    -------
    tuple of Candidate
        Indexed from zero in generation order.
    """
    if float(slot_step_s) <= 0.0:
        raise ValueError(f"slot_step_s must be positive, got {slot_step_s}")
    candidates: list[Candidate] = []
    for request in requests:
        weight = priority_weight(request.priority, weight_base)
        for window in windows:
            if window.spacecraft != request.spacecraft:
                continue
            if not request.allows(window.station):
                continue
            latest = window.set_s - request.duration_s
            if latest < window.rise_s:
                continue
            starts: list[float] = []
            offset = 0.0
            while window.rise_s + offset <= latest + 1e-9:
                starts.append(window.rise_s + offset)
                offset += float(slot_step_s)
            if align_to_window_end and (not starts or abs(starts[-1] - latest) > 1e-9):
                starts.append(latest)
            for start in starts:
                candidates.append(
                    Candidate(
                        index=len(candidates),
                        request_id=request.request_id,
                        mission=request.mission,
                        spacecraft=request.spacecraft,
                        station=window.station,
                        start_s=float(start),
                        end_s=float(start + request.duration_s),
                        weight=weight,
                        max_elevation_deg=float(window.max_elevation_deg),
                    )
                )
    return tuple(candidates)


def build_problem(
    stations: tuple[GroundStation, ...],
    requests: tuple[ContactRequest, ...],
    windows: tuple[ContactWindow, ...],
    slot_step_s: float = 120.0,
    weight_base: float = 2.0,
    spacecraft_turnaround_s: float = 0.0,
) -> SchedulingProblem:
    """Assemble a :class:`SchedulingProblem` with its candidate set."""
    candidates = build_candidates(requests, windows, slot_step_s, weight_base)
    return SchedulingProblem(
        stations=tuple(stations),
        requests=tuple(requests),
        windows=tuple(windows),
        candidates=candidates,
        slot_step_s=float(slot_step_s),
        spacecraft_turnaround_s=float(spacecraft_turnaround_s),
    )


def check_feasible(problem: SchedulingProblem, selected: tuple[int, ...]) -> list[str]:
    """Return a list of feasibility violations; empty means feasible.

    Violations are returned rather than raised so that a caller measuring a
    heuristic can count them. The checks are the four in the module docstring.
    Minimum contact counts are not checked here; see :func:`shortfalls`.
    """
    problems: list[str] = []
    if len(set(selected)) != len(selected):
        problems.append("a candidate index is selected more than once")
    chosen = [problem.candidates[i] for i in selected]
    for i, a in enumerate(chosen):
        for b in chosen[i + 1 :]:
            if problem.station_conflict(a, b):
                problems.append(
                    f"antenna {a.station} double-booked: {a.request_id} "
                    f"[{a.start_s:.1f}, {a.end_s:.1f}] against {b.request_id} "
                    f"[{b.start_s:.1f}, {b.end_s:.1f}] with "
                    f"{problem.turnaround_s(a.station):.1f} s turnaround"
                )
            if problem.spacecraft_conflict(a, b):
                problems.append(
                    f"spacecraft {a.spacecraft} double-booked: {a.request_id} "
                    f"[{a.start_s:.1f}, {a.end_s:.1f}] against {b.request_id} "
                    f"[{b.start_s:.1f}, {b.end_s:.1f}]"
                )
    counts: dict[str, int] = {}
    for candidate in chosen:
        counts[candidate.request_id] = counts.get(candidate.request_id, 0) + 1
    for request in problem.requests:
        taken = counts.get(request.request_id, 0)
        if taken > request.max_contacts:
            problems.append(
                f"request {request.request_id} has {taken} contacts, above its "
                f"maximum of {request.max_contacts}"
            )
    window_spans: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for window in problem.windows:
        window_spans.setdefault((window.station, window.spacecraft), []).append(
            (window.rise_s, window.set_s)
        )
    for candidate in chosen:
        spans = window_spans.get((candidate.station, candidate.spacecraft), [])
        inside = any(
            start - 1e-6 <= candidate.start_s and candidate.end_s <= end + 1e-6
            for start, end in spans
        )
        if not inside:
            problems.append(
                f"candidate {candidate.index} for {candidate.request_id} at "
                f"{candidate.station} is not contained in any visibility window"
            )
    return problems


def shortfalls(
    problem: SchedulingProblem, selected: tuple[int, ...]
) -> dict[str, tuple[int, int, float]]:
    """Per-request contact shortfall against ``min_contacts``.

    Returns
    -------
    dict
        ``{request_id: (contacts_granted, contacts_short, seconds_short)}`` for
        every request, including the satisfied ones, so a caller can report the
        whole demand rather than only its failures.
    """
    counts: dict[str, int] = {r.request_id: 0 for r in problem.requests}
    for index in selected:
        counts[problem.candidates[index].request_id] += 1
    out: dict[str, tuple[int, int, float]] = {}
    for request in problem.requests:
        granted = counts[request.request_id]
        short = max(0, int(request.min_contacts) - granted)
        out[request.request_id] = (granted, short, short * float(request.duration_s))
    return out
