"""Baseline 1: priority-ordered first-fit, and the ways it loses.

Each of the three documented failure modes is exhibited here on a hand-built
instance whose optimum can be worked out by hand, with the expected objective
values written out as literals.
"""

from __future__ import annotations

import pytest

from contactsched.assignment import STATUS_HEURISTIC
from contactsched.benchmark import optimality_gap
from contactsched.exact import FairnessPolicy, solve_exact
from contactsched.greedy import schedule_greedy
from contactsched.problem import ContactRequest, build_problem, check_feasible
from contactsched.station import GroundStation, HorizonMask
from contactsched.visibility import ContactWindow


def station(name: str = "A1", turnaround_s: float = 0.0) -> GroundStation:
    return GroundStation(name, 0.0, 0.0, 0.0, HorizonMask(floor_deg=5.0), turnaround_s)


def window(spacecraft: str, rise: float, close: float, station_name: str = "A1") -> ContactWindow:
    return ContactWindow(
        station_name, spacecraft, "M", rise, close, 45.0, 0.5 * (rise + close), 5.0
    )


class TestBasicBehaviour:
    def test_status_is_always_heuristic(self, toy_problem):
        assert schedule_greedy(toy_problem).status == STATUS_HEURISTIC

    def test_result_is_feasible(self, toy_problem):
        assignment = schedule_greedy(toy_problem)
        assert check_feasible(toy_problem, assignment.selected) == []

    def test_result_is_feasible_on_a_generated_instance(self, instance_seed7):
        assignment = schedule_greedy(instance_seed7.problem)
        assert check_feasible(instance_seed7.problem, assignment.selected) == []

    def test_is_deterministic(self, instance_seed7):
        first = schedule_greedy(instance_seed7.problem)
        second = schedule_greedy(instance_seed7.problem)
        assert first.selected == second.selected
        assert first.objective == pytest.approx(second.objective, rel=0.0, abs=0.0)

    def test_selected_indices_are_sorted_and_unique(self, instance_seed7):
        selected = schedule_greedy(instance_seed7.problem).selected
        assert list(selected) == sorted(set(selected))

    def test_respects_max_contacts(self, instance_seed7):
        assignment = schedule_greedy(instance_seed7.problem)
        counts = assignment.contacts_by_request(instance_seed7.problem)
        for request in instance_seed7.problem.requests:
            assert counts[request.request_id] <= request.max_contacts

    def test_demand_order_is_a_different_heuristic(self, instance_seed7):
        by_priority = schedule_greedy(instance_seed7.problem, order="priority")
        by_demand = schedule_greedy(instance_seed7.problem, order="demand")
        assert by_priority.selected != by_demand.selected

    def test_unknown_order_raises(self, toy_problem):
        with pytest.raises(ValueError, match="unknown order"):
            schedule_greedy(toy_problem, order="alphabetical")

    def test_notes_name_the_order(self, toy_problem):
        assert "priority order" in schedule_greedy(toy_problem).notes[0]


class TestFailureModeEarliestStart:
    """Mode 1: taking the earliest feasible slot blocks a request with one slot."""

    @pytest.fixture
    def problem(self):
        # One antenna, zero turnaround. SC-1 is visible 0..300, SC-2 only 0..100.
        # R1 (priority 1, 100 s) can go at 0, 100 or 200. R2 (priority 2, 100 s)
        # can only go at 0. Greedy takes R1 at 0 and destroys R2.
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=1, max_contacts=1),
            ContactRequest("R2", "M2", "SC-2", 2, 100.0, min_contacts=1, max_contacts=1),
        )
        windows = (window("SC-1", 0.0, 300.0), window("SC-2", 0.0, 100.0))
        return build_problem((station(),), requests, windows, slot_step_s=100.0)

    def test_greedy_scores_one(self, problem):
        assignment = schedule_greedy(problem)
        assert assignment.objective == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert assignment.contact_count == 1

    def test_the_exact_optimum_scores_one_and_a_half(self, problem):
        exact = solve_exact(problem)
        assert exact.objective == pytest.approx(1.5, rel=0.0, abs=1e-12)
        assert exact.contact_count == 2

    def test_the_gap_is_one_third(self, problem):
        # (1.5 - 1.0) / 1.5 = 0.3333...
        gap = optimality_gap(schedule_greedy(problem), solve_exact(problem))
        assert gap == pytest.approx(1.0 / 3.0, rel=1e-12, abs=0.0)

    def test_greedy_took_the_earliest_slot(self, problem):
        assignment = schedule_greedy(problem)
        chosen = problem.candidates[assignment.selected[0]]
        assert chosen.request_id == "R1"
        assert chosen.start_s == pytest.approx(0.0, rel=0.0, abs=0.0)


class TestFailureModeDurationBlind:
    """Mode 2: greedy ignores how much time it destroys."""

    @pytest.fixture
    def problem(self):
        # SC-1 visible 0..400, SC-2 visible 0..300. R1 is priority 1 needing
        # 100 s; R2 is priority 2 needing 300 s and has exactly one placement.
        # Under the seconds objective R2 is worth 0.5 * 300 = 150 against R1's
        # 1.0 * 100 = 100.
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=1, max_contacts=1),
            ContactRequest("R2", "M2", "SC-2", 2, 300.0, min_contacts=1, max_contacts=1),
        )
        windows = (window("SC-1", 0.0, 400.0), window("SC-2", 0.0, 300.0))
        return build_problem((station(),), requests, windows, slot_step_s=100.0)

    def test_greedy_scores_one_hundred_weighted_seconds(self, problem):
        assignment = schedule_greedy(problem, objective_mode="weighted_seconds")
        assert assignment.objective == pytest.approx(100.0, rel=0.0, abs=1e-9)

    def test_the_exact_optimum_scores_two_hundred_and_fifty(self, problem):
        exact = solve_exact(problem, objective_mode="weighted_seconds")
        assert exact.objective == pytest.approx(250.0, rel=0.0, abs=1e-9)

    def test_the_exact_solution_places_both(self, problem):
        exact = solve_exact(problem, objective_mode="weighted_seconds")
        assert exact.contact_count == 2
        chosen = {problem.candidates[i].request_id for i in exact.selected}
        assert chosen == {"R1", "R2"}

    def test_the_gap_is_three_fifths(self, problem):
        # (250 - 100) / 250 = 0.6
        gap = optimality_gap(
            schedule_greedy(problem, objective_mode="weighted_seconds"),
            solve_exact(problem, objective_mode="weighted_seconds"),
        )
        assert gap == pytest.approx(0.6, rel=1e-12, abs=0.0)


class TestFailureModeMinimumContactsInvisible:
    """Mode 3: greedy neither enforces nor notices a declared minimum."""

    @pytest.fixture
    def problem(self):
        # One antenna, zero turnaround. Three 100 s slots at 0, 400 and 800 are
        # visible to SC-1; only the first two are visible to SC-2. R1 is
        # priority 1, needs one contact and is allowed three. R2 is priority 2
        # and DECLARES it needs two. Greedy serves R1 to its maximum first and
        # leaves R2 with nothing.
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=1, max_contacts=3),
            ContactRequest("R2", "M2", "SC-2", 2, 100.0, min_contacts=2, max_contacts=2),
        )
        windows = (
            window("SC-1", 0.0, 100.0),
            window("SC-1", 400.0, 500.0),
            window("SC-1", 800.0, 900.0),
            window("SC-2", 0.0, 100.0),
            window("SC-2", 400.0, 500.0),
        )
        return build_problem((station(),), requests, windows, slot_step_s=100.0)

    def test_greedy_starves_the_declared_minimum(self, problem):
        from contactsched.problem import shortfalls

        assignment = schedule_greedy(problem)
        short = shortfalls(problem, assignment.selected)
        assert short["R2"][1] == 2

    def test_the_fairness_constrained_exact_solve_satisfies_it(self, problem):
        exact = solve_exact(problem, policy=FairnessPolicy(enforce_min_contacts=True))
        assert exact.status == "optimal"
        counts = exact.contacts_by_request(problem)
        assert counts["R2"] == 2

    def test_satisfying_the_minimum_costs_objective(self, problem):
        free = solve_exact(problem)
        fair = solve_exact(problem, policy=FairnessPolicy(enforce_min_contacts=True))
        # Free optimum: R1 takes all three slots, 3 x 1.0 = 3.0, and R2 gets
        # nothing. Constrained: R2 must take the slots at 0 and 400, leaving R1
        # the slot at 800, so 1.0 + 2 x 0.5 = 2.0. The declared fairness
        # constraint costs one third of the objective on this instance.
        assert free.objective == pytest.approx(3.0, rel=0.0, abs=1e-12)
        assert fair.objective == pytest.approx(2.0, rel=0.0, abs=1e-12)
        assert fair.objective < free.objective

    def test_the_exact_optimum_also_starves_the_request(self, problem):
        # The starvation is not a greedy artefact: the proven optimum starves R2
        # too, which is why the label used by the conflict predictor is taken
        # from the exact solve and not from the heuristic.
        from contactsched.problem import shortfalls

        free = solve_exact(problem)
        assert shortfalls(problem, free.selected)["R2"][1] == 2

    def test_greedy_is_still_feasible_despite_starving(self, problem):
        assignment = schedule_greedy(problem)
        assert check_feasible(problem, assignment.selected) == []


class TestGreedyCanWin:
    def test_greedy_matches_the_optimum_when_there_is_no_contention(self):
        # Three requests on three antennas with disjoint windows: first-fit is
        # optimal, and a baseline win is reported, not hidden.
        stations = (station("A1"), station("A2"), station("A3"))
        requests = tuple(
            ContactRequest(f"R{i}", f"M{i}", f"SC-{i}", i, 100.0, 1, 1) for i in (1, 2, 3)
        )
        windows = tuple(
            window(f"SC-{i}", 0.0, 100.0, f"A{i}") for i in (1, 2, 3)
        )
        problem = build_problem(stations, requests, windows, slot_step_s=100.0)
        greedy = schedule_greedy(problem)
        exact = solve_exact(problem)
        assert greedy.objective == pytest.approx(exact.objective, rel=0.0, abs=1e-12)
        assert optimality_gap(greedy, exact) == pytest.approx(0.0, rel=0.0, abs=1e-12)

    def test_greedy_never_beats_the_proven_optimum(self, small_instances):
        for instance in small_instances:
            problem = instance.problem
            greedy = schedule_greedy(problem)
            exact = solve_exact(problem)
            assert greedy.objective <= exact.objective + 1e-9, f"seed {instance.seed}"
