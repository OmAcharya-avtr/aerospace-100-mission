"""Invariants that must hold for every schedule, with negative controls.

Three kinds of check live here.

**Invariants.** Properties any correct schedule has: no antenna serves two
contacts at once, no spacecraft talks to two antennas at once, no request
exceeds its ceiling, and allocated time never exceeds supply.

**Independently derived bounds.** Checks of a computed quantity against a bound
derived from somewhere else entirely, rather than against a second
implementation of the same expression. Two implementations of one formula agree
on a shared mistake; a bound does not.

**Negative controls.** Deliberate corruptions that the invariant checks must
catch. A negative control is worthless unless it actually fires, so every one of
them asserts that the violation list is non-empty, and every one is anchored on
the *true* value of the quantity it corrupts rather than on an arbitrary offset.
The boundary case is tested too: a corruption exactly at the limit must NOT fire,
because an offset that merely reaches a bound does not cross it.
"""

from __future__ import annotations

import pytest

from contactsched.exact import FairnessPolicy, solve_exact, standalone_capacity
from contactsched.fairness import analyse
from contactsched.greedy import schedule_greedy
from contactsched.problem import (
    Candidate,
    ContactRequest,
    build_problem,
    check_feasible,
    shortfalls,
)
from contactsched.station import GroundStation, HorizonMask
from contactsched.visibility import ContactWindow, supply_seconds_by_station

METHODS = ("greedy", "greedy-demand", "exact")


def station(name: str = "A1", turnaround_s: float = 0.0) -> GroundStation:
    return GroundStation(name, 0.0, 0.0, 0.0, HorizonMask(floor_deg=5.0), turnaround_s)


def window(spacecraft: str, rise: float, close: float, station_name: str = "A1") -> ContactWindow:
    return ContactWindow(
        station_name, spacecraft, "M", rise, close, 45.0, 0.5 * (rise + close), 5.0
    )


@pytest.fixture(scope="module")
def instances():
    """Twelve generated instances over a twelve-hour horizon."""
    from contactsched.dataset import generate_instance

    return [generate_instance(seed, horizon_s=43200.0) for seed in range(40, 52)]


class TestScheduleInvariants:
    def test_no_antenna_or_spacecraft_is_double_booked(self, instances):
        from contactsched.benchmark import run_method

        for instance in instances:
            for method in METHODS:
                result = run_method(instance.problem, method)
                assert result.violations == (), f"seed {instance.seed}, {method}"

    def test_no_request_exceeds_its_ceiling(self, instances):
        for instance in instances:
            assignment = solve_exact(instance.problem)
            counts = assignment.contacts_by_request(instance.problem)
            for request in instance.problem.requests:
                assert counts[request.request_id] <= request.max_contacts

    def test_allocated_contacts_never_exceed_standalone_capacity(self, instances):
        for instance in instances:
            assignment = solve_exact(instance.problem)
            counts = assignment.contacts_by_request(instance.problem)
            for request in instance.problem.requests:
                assert counts[request.request_id] <= standalone_capacity(
                    instance.problem, request.request_id
                ), f"seed {instance.seed}, {request.request_id}"

    def test_enforced_minimums_are_met_or_the_program_is_infeasible(self, instances):
        for instance in instances:
            fair = solve_exact(
                instance.problem, policy=FairnessPolicy(enforce_min_contacts=True)
            )
            if fair.status == "infeasible":
                continue
            short = shortfalls(instance.problem, fair.selected)
            for request_id, (_granted, contacts_short, _seconds) in short.items():
                assert contacts_short == 0, f"seed {instance.seed}, {request_id}"

    def test_contacts_lie_inside_their_windows(self, instances):
        for instance in instances:
            assignment = solve_exact(instance.problem)
            spans: dict[tuple[str, str], list[tuple[float, float]]] = {}
            for w in instance.problem.windows:
                spans.setdefault((w.station, w.spacecraft), []).append((w.rise_s, w.set_s))
            for index in assignment.selected:
                candidate = instance.problem.candidates[index]
                assert any(
                    start - 1e-9 <= candidate.start_s and candidate.end_s <= end + 1e-9
                    for start, end in spans[(candidate.station, candidate.spacecraft)]
                )


class TestIndependentlyDerivedBounds:
    def test_allocated_time_per_antenna_never_exceeds_its_union_supply(self, instances):
        # The bound comes from the window list, through a union that the
        # schedulers never call. An antenna cannot allocate more seconds than it
        # is visible for, counting overlapping visibility once.
        for instance in instances:
            supply = supply_seconds_by_station(list(instance.problem.windows))
            assignment = solve_exact(instance.problem)
            allocated = assignment.seconds_by_station(instance.problem)
            for name, seconds in allocated.items():
                assert seconds <= supply.get(name, 0.0) + 1e-6, f"seed {instance.seed}, {name}"

    def test_total_allocated_time_never_exceeds_total_visibility(self, instances):
        for instance in instances:
            assignment = solve_exact(instance.problem)
            allocated = sum(assignment.seconds_by_station(instance.problem).values())
            visible = sum(w.duration_s for w in instance.problem.windows)
            assert allocated <= visible + 1e-6

    def test_objective_never_exceeds_the_unconstrained_weight_sum(self, instances):
        # Independent bound: no feasible schedule can score more than the sum of
        # the best max_contacts weights available to each request, which is
        # computed here from the requests alone and knows nothing of conflicts.
        for instance in instances:
            ceiling = 0.0
            for request in instance.problem.requests:
                candidates = instance.problem.candidates_of(request.request_id)
                if not candidates:
                    continue
                ceiling += candidates[0].weight * request.max_contacts
            assert solve_exact(instance.problem).objective <= ceiling + 1e-9

    def test_fairness_indices_lie_in_their_declared_ranges(self, instances):
        for instance in instances:
            report = analyse(instance.problem, schedule_greedy(instance.problem))
            with_demand = [o for o in report.outcomes if o.has_demand]
            if with_demand:
                assert 1.0 / len(with_demand) - 1e-12 <= report.jain <= 1.0 + 1e-12
            assert 0.0 <= report.max_min <= 1.0 + 1e-12
            assert 0.0 <= report.inversion_rate <= 1.0 + 1e-12

    def test_seconds_short_never_exceeds_declared_demand(self, instances):
        for instance in instances:
            report = analyse(instance.problem, schedule_greedy(instance.problem))
            demand = sum(r.requested_seconds for r in instance.problem.requests)
            assert report.total_seconds_short <= demand + 1e-9


class TestNegativeControls:
    @pytest.fixture
    def problem(self):
        # A1 carries a 200 s turnaround and SC-1 is visible for 2000 s, so the
        # candidate grid at 200 s holds ten placements of a 200 s contact and
        # every corruption below stays inside the window. That matters: a
        # corruption that also moves a contact out of its window would be caught
        # by the containment check instead of by the one under test, and the
        # control would prove nothing about turnaround enforcement.
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 200.0, min_contacts=2, max_contacts=2),)
        windows = (window("SC-1", 0.0, 2000.0),)
        return build_problem((station(turnaround_s=200.0),), requests, windows, 200.0)

    @staticmethod
    def at(problem, start_s: float) -> Candidate:
        """The grid candidate starting at ``start_s``."""
        return next(c for c in problem.candidates if c.start_s == pytest.approx(start_s))

    @staticmethod
    def moved(template: Candidate, start_s: float) -> Candidate:
        """``template`` relocated to ``start_s``, keeping its identity."""
        return Candidate(
            index=template.index,
            request_id=template.request_id,
            mission=template.mission,
            spacecraft=template.spacecraft,
            station=template.station,
            start_s=start_s,
            end_s=start_s + template.duration_s,
            weight=template.weight,
            max_elevation_deg=template.max_elevation_deg,
        )

    @staticmethod
    def substitute(problem, candidate: Candidate) -> None:
        """Replace one candidate in place, so the corruption is the only change."""
        corrupted = list(problem.candidates)
        corrupted[candidate.index] = candidate
        problem.candidates = tuple(corrupted)

    def test_the_positive_control_is_clean(self, problem):
        # Without this the negative controls below prove nothing: a checker that
        # always fires is not a checker. Contacts at 0 and 1000 are separated by
        # 800 s against a 300 s turnaround.
        first, second = self.at(problem, 0.0), self.at(problem, 1000.0)
        assert check_feasible(problem, (first.index, second.index)) == []

    def test_the_solver_control_is_clean(self, problem):
        assignment = solve_exact(problem)
        assert assignment.contact_count == 2
        assert check_feasible(problem, assignment.selected) == []

    def test_a_separation_exactly_at_the_turnaround_does_not_fire(self, problem):
        # The boundary, taken from the TRUE values: a contact ending at 200 and
        # one starting at 400 are separated by exactly the 200 s turnaround,
        # which is legal. A negative control built on an offset that merely
        # reaches the bound would prove nothing, so the bound is tested from
        # both sides.
        first, second = self.at(problem, 0.0), self.at(problem, 400.0)
        assert first.end_s + problem.turnaround_s("A1") == pytest.approx(
            second.start_s, rel=0.0, abs=0.0
        )
        assert check_feasible(problem, (first.index, second.index)) == []

    def test_a_separation_one_second_inside_the_turnaround_does_fire(self, problem):
        first = self.at(problem, 0.0)
        template = self.at(problem, 400.0)
        self.substitute(problem, self.moved(template, 399.0))
        violations = check_feasible(problem, (first.index, template.index))
        assert violations, "a separation one second inside the turnaround did not fire"
        assert any("double-booked" in v for v in violations)

    def test_an_injected_overlap_fires(self, problem):
        # Anchored on the TRUE end time of the first contact: the second begins
        # 1 s after it ends, 199 s inside the turnaround.
        first = self.at(problem, 0.0)
        template = self.at(problem, 1000.0)
        self.substitute(problem, self.moved(template, first.end_s + 1.0))
        violations = check_feasible(problem, (first.index, template.index))
        assert violations, "the injected turnaround violation did not fire"
        assert any("double-booked" in v for v in violations)

    def test_an_injected_window_escape_fires(self, problem):
        # Moving a contact past the end of the only window must be caught by the
        # containment check, which is a different check from the exclusion one.
        template = self.at(problem, 1000.0)
        self.substitute(problem, self.moved(template, 5000.0))
        violations = check_feasible(problem, (template.index,))
        assert violations
        assert any("not contained in any visibility window" in v for v in violations)

    def test_an_injected_duplicate_selection_fires(self, problem):
        assignment = solve_exact(problem)
        violations = check_feasible(
            problem, assignment.selected + (assignment.selected[0],)
        )
        assert violations
        assert any("more than once" in v for v in violations)

    def test_an_injected_ceiling_breach_fires(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=1, max_contacts=1),)
        windows = tuple(window("SC-1", 500.0 * k, 500.0 * k + 100.0) for k in range(3))
        problem = build_problem((station(),), requests, windows, 100.0)
        violations = check_feasible(problem, (0, 1))
        assert violations
        assert any("above its maximum" in v for v in violations)

    def test_an_injected_spacecraft_clash_fires(self):
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 1),
            ContactRequest("R2", "M1", "SC-1", 2, 100.0, 1, 1),
        )
        windows = (window("SC-1", 0.0, 100.0, "A1"), window("SC-1", 0.0, 100.0, "A2"))
        problem = build_problem((station("A1"), station("A2")), requests, windows, 100.0)
        # One candidate per request per antenna; pick the two on different
        # antennas and the same spacecraft, which is a radio clash.
        pair = tuple(
            c.index
            for c in problem.candidates
            if (c.request_id, c.station) in {("R1", "A1"), ("R2", "A2")}
        )
        violations = check_feasible(problem, pair)
        assert violations
        assert any("spacecraft SC-1 double-booked" in v for v in violations)

    def test_an_injected_starvation_is_measured_not_hidden(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=2, max_contacts=2),)
        windows = (window("SC-1", 0.0, 100.0),)
        problem = build_problem((station(),), requests, windows, 100.0)
        assignment = solve_exact(problem)
        # Feasible, and short by one contact: the invariant check passes and the
        # shortfall measurement reports the gap. Both behaviours are intended.
        assert check_feasible(problem, assignment.selected) == []
        assert shortfalls(problem, assignment.selected)["R1"] == (
            1,
            1,
            pytest.approx(100.0, rel=0.0, abs=0.0),
        )
