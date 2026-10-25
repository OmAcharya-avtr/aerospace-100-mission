"""Exhaustive enumeration, and the exact method measured against it.

This is the check the specification asks for: the exact method against
brute-force enumeration on small instances. The two share no formulation -- one
builds a constraint matrix and calls HiGHS, the other iterates over every
bitmask -- so an error in the clique argument would show up here.
"""

from __future__ import annotations

import pytest

from contactsched.bruteforce import (
    DEFAULT_MAX_CANDIDATES,
    conflict_masks,
    is_feasible_mask,
    request_masks,
    solve_bruteforce,
)
from contactsched.exact import FairnessPolicy, solve_exact
from contactsched.problem import ContactRequest, build_problem, check_feasible
from contactsched.station import GroundStation, HorizonMask
from contactsched.visibility import ContactWindow


def station(name: str = "A1", turnaround_s: float = 0.0) -> GroundStation:
    return GroundStation(name, 0.0, 0.0, 0.0, HorizonMask(floor_deg=5.0), turnaround_s)


def window(spacecraft: str, rise: float, close: float, station_name: str = "A1") -> ContactWindow:
    return ContactWindow(
        station_name, spacecraft, "M", rise, close, 45.0, 0.5 * (rise + close), 5.0
    )


class TestMasks:
    def test_conflict_masks_are_symmetric(self, toy_problem):
        masks = conflict_masks(toy_problem)
        for i, mask in enumerate(masks):
            for j in range(len(masks)):
                assert bool(mask >> j & 1) == bool(masks[j] >> i & 1)

    def test_no_candidate_conflicts_with_itself(self, toy_problem):
        for i, mask in enumerate(conflict_masks(toy_problem)):
            assert not mask >> i & 1

    def test_masks_agree_with_the_conflict_predicate(self, toy_problem):
        masks = conflict_masks(toy_problem)
        for i, a in enumerate(toy_problem.candidates):
            for j, b in enumerate(toy_problem.candidates):
                assert bool(masks[i] >> j & 1) == (
                    toy_problem.conflicts(a, b) and i != j
                )

    def test_request_masks_cover_every_candidate_exactly_once(self, toy_problem):
        masks = request_masks(toy_problem)
        union = 0
        total_bits = 0
        for mask, _limit in masks.values():
            assert union & mask == 0
            union |= mask
            total_bits += mask.bit_count()
        assert total_bits == len(toy_problem.candidates)

    def test_empty_subset_is_feasible(self, toy_problem):
        masks = conflict_masks(toy_problem)
        limits = request_masks(toy_problem)
        assert is_feasible_mask(0, masks, limits)

    def test_a_conflicting_pair_is_infeasible(self, toy_problem):
        masks = conflict_masks(toy_problem)
        limits = request_masks(toy_problem)
        pair = next(
            (i, j)
            for i in range(len(masks))
            for j in range(i + 1, len(masks))
            if masks[i] >> j & 1
        )
        assert not is_feasible_mask((1 << pair[0]) | (1 << pair[1]), masks, limits)

    def test_feasible_masks_agree_with_check_feasible(self, toy_problem):
        masks = conflict_masks(toy_problem)
        limits = request_masks(toy_problem)
        n = len(toy_problem.candidates)
        for subset in range(1 << min(n, 11)):
            selected = tuple(i for i in range(n) if subset >> i & 1)
            assert is_feasible_mask(subset, masks, limits) == (
                check_feasible(toy_problem, selected) == []
            )


class TestKnownAnswers:
    def test_two_conflicting_slots_give_the_higher_weight(self):
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 1),
            ContactRequest("R2", "M2", "SC-2", 3, 100.0, 1, 1),
        )
        windows = (window("SC-1", 0.0, 100.0), window("SC-2", 0.0, 100.0))
        problem = build_problem((station(),), requests, windows, 100.0)
        assignment = solve_bruteforce(problem)
        # Priority 1 weighs 1.0, priority 3 weighs 0.25; only one fits.
        assert assignment.objective == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert problem.candidates[assignment.selected[0]].request_id == "R1"

    def test_three_disjoint_slots_are_all_taken(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 2, 100.0, 1, 3),)
        windows = tuple(window("SC-1", 300.0 * k, 300.0 * k + 100.0) for k in range(3))
        problem = build_problem((station(),), requests, windows, 100.0)
        assignment = solve_bruteforce(problem)
        # Three contacts at priority 2, weight 0.5 each.
        assert assignment.objective == pytest.approx(1.5, rel=0.0, abs=1e-12)
        assert assignment.contact_count == 3

    def test_an_empty_problem_selects_nothing(self):
        requests = (ContactRequest("R1", "M1", "SC-9", 1, 100.0),)
        problem = build_problem((station(),), requests, (), 100.0)
        assignment = solve_bruteforce(problem)
        assert assignment.selected == ()
        assert assignment.objective == pytest.approx(0.0, rel=0.0, abs=0.0)

    def test_notes_record_the_subset_count(self, toy_problem):
        note = solve_bruteforce(toy_problem).notes[0]
        assert str(1 << len(toy_problem.candidates)) in note


class TestAgainstExact:
    def test_matches_the_milp_optimum_on_the_hand_built_problem(self, toy_problem):
        enumerated = solve_bruteforce(toy_problem)
        milp = solve_exact(toy_problem)
        assert enumerated.objective == pytest.approx(milp.objective, rel=0.0, abs=1e-9)

    def test_matches_the_milp_optimum_on_every_small_instance(self, small_instances):
        for instance in small_instances:
            problem = instance.problem
            if len(problem.candidates) > DEFAULT_MAX_CANDIDATES:
                continue
            enumerated = solve_bruteforce(problem)
            milp = solve_exact(problem)
            assert enumerated.objective == pytest.approx(
                milp.objective, rel=0.0, abs=1e-9
            ), f"seed {instance.seed}"

    def test_matches_under_the_seconds_objective(self, small_instances):
        for instance in small_instances[:6]:
            problem = instance.problem
            enumerated = solve_bruteforce(problem, objective_mode="weighted_seconds")
            milp = solve_exact(problem, objective_mode="weighted_seconds")
            assert enumerated.objective == pytest.approx(
                milp.objective, rel=1e-9, abs=1e-6
            ), f"seed {instance.seed}"

    def test_matches_with_minimum_contacts_enforced(self, small_instances):
        for instance in small_instances[:6]:
            problem = instance.problem
            enumerated = solve_bruteforce(problem, min_contacts_required=True)
            milp = solve_exact(problem, policy=FairnessPolicy(enforce_min_contacts=True))
            if milp.status == "infeasible":
                assert enumerated.status == "infeasible", f"seed {instance.seed}"
            else:
                assert enumerated.objective == pytest.approx(
                    milp.objective, rel=0.0, abs=1e-9
                ), f"seed {instance.seed}"

    def test_enumerated_solutions_are_feasible(self, small_instances):
        for instance in small_instances:
            assignment = solve_bruteforce(instance.problem)
            assert check_feasible(instance.problem, assignment.selected) == []


class TestLimits:
    def test_too_many_candidates_raises(self, instance_seed7):
        with pytest.raises(ValueError, match="refusing to enumerate"):
            solve_bruteforce(instance_seed7.problem)

    def test_the_limit_can_be_raised_explicitly(self, toy_problem):
        assert solve_bruteforce(toy_problem, max_candidates=11).status == "optimal"

    def test_an_unknown_objective_mode_raises(self, toy_problem):
        with pytest.raises(ValueError, match="unknown objective mode"):
            solve_bruteforce(toy_problem, objective_mode="utility")

    def test_infeasible_minimums_report_infeasible(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=3, max_contacts=3),)
        windows = (window("SC-1", 0.0, 100.0),)
        problem = build_problem((station(),), requests, windows, 100.0)
        assignment = solve_bruteforce(problem, min_contacts_required=True)
        assert assignment.status == "infeasible"
        assert assignment.selected == ()
