"""Baseline 2: the exact mixed-integer program on HiGHS.

The headline check -- the exact optimum against independent exhaustive
enumeration -- lives in ``test_bruteforce.py`` so that the module under test and
the module that falsifies it are not imported by the same file for the same
reason. This file covers the formulation, the fairness policy, the infeasibility
diagnostics and the backend probe.
"""

from __future__ import annotations

import numpy as np
import pytest

from contactsched.assignment import STATUS_INFEASIBLE, STATUS_OPTIMAL
from contactsched.exact import (
    DEFAULT_BACKEND,
    FairnessPolicy,
    available_backends,
    solve_exact,
    standalone_capacity,
)
from contactsched.problem import ContactRequest, build_problem, check_feasible
from contactsched.station import GroundStation, HorizonMask
from contactsched.visibility import ContactWindow


def station(name: str = "A1", turnaround_s: float = 0.0) -> GroundStation:
    return GroundStation(name, 0.0, 0.0, 0.0, HorizonMask(floor_deg=5.0), turnaround_s)


def window(spacecraft: str, rise: float, close: float, station_name: str = "A1") -> ContactWindow:
    return ContactWindow(
        station_name, spacecraft, "M", rise, close, 45.0, 0.5 * (rise + close), 5.0
    )


class TestBackendProbe:
    def test_highs_is_reported_available(self):
        assert "available via scipy" in available_backends()["highs"]

    def test_pulp_is_reported_present_but_unusable_or_absent(self):
        # The measured fact that justified the deviation from the specification:
        # pulp 4.1.0 ships no solver, so listSolvers(onlyAvailable=True) is
        # empty. If a future environment installs a solver this assertion still
        # holds, because the probe then says "not used by this package".
        detail = available_backends()["pulp"]
        assert detail.startswith(("absent", "present"))

    def test_ortools_is_reported_without_raising(self):
        assert "ortools-cpsat" in available_backends()

    def test_the_probe_never_raises(self):
        assert isinstance(available_backends(), dict)

    def test_an_unknown_backend_raises_and_names_what_is_available(self, toy_problem):
        with pytest.raises(ValueError, match="is not exercised by this package"):
            solve_exact(toy_problem, backend="cbc")

    def test_the_default_backend_is_highs(self):
        assert DEFAULT_BACKEND == "highs"


class TestFormulation:
    def test_returns_an_optimal_status_on_a_feasible_instance(self, toy_problem):
        assert solve_exact(toy_problem).status == STATUS_OPTIMAL

    def test_result_is_feasible(self, toy_problem):
        assignment = solve_exact(toy_problem)
        assert check_feasible(toy_problem, assignment.selected) == []

    def test_result_is_feasible_on_a_generated_instance(self, instance_seed7):
        assignment = solve_exact(instance_seed7.problem)
        assert check_feasible(instance_seed7.problem, assignment.selected) == []

    def test_reported_objective_matches_a_recomputation(self, instance_seed7):
        from contactsched.assignment import objective_value

        assignment = solve_exact(instance_seed7.problem)
        assert assignment.objective == pytest.approx(
            objective_value(instance_seed7.problem, assignment.selected, "weighted_contacts"),
            rel=1e-12,
            abs=0.0,
        )

    def test_turnaround_is_enforced_on_both_sides(self):
        # A1 has a 150 s turnaround and two 100 s slots 100 s apart. Only one can
        # be taken, whichever order the solver considers them in.
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 2),
        )
        windows = (window("SC-1", 0.0, 100.0), window("SC-1", 200.0, 300.0))
        problem = build_problem((station(turnaround_s=150.0),), requests, windows, 100.0)
        assignment = solve_exact(problem)
        assert assignment.contact_count == 1

    def test_a_shorter_turnaround_admits_both(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 2),)
        windows = (window("SC-1", 0.0, 100.0), window("SC-1", 200.0, 300.0))
        problem = build_problem((station(turnaround_s=50.0),), requests, windows, 100.0)
        assert solve_exact(problem).contact_count == 2

    def test_a_spacecraft_cannot_use_two_antennas_at_once(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 2),)
        windows = (window("SC-1", 0.0, 100.0, "A1"), window("SC-1", 0.0, 100.0, "A2"))
        problem = build_problem((station("A1"), station("A2")), requests, windows, 100.0)
        assert solve_exact(problem).contact_count == 1

    def test_max_contacts_is_binding(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 2),)
        windows = tuple(window("SC-1", 200.0 * k, 200.0 * k + 100.0) for k in range(5))
        problem = build_problem((station(),), requests, windows, 100.0)
        assert solve_exact(problem).contact_count == 2

    def test_weighted_seconds_can_prefer_a_different_schedule(self):
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 1),
            ContactRequest("R2", "M2", "SC-2", 2, 300.0, 1, 1),
        )
        windows = (window("SC-1", 0.0, 100.0), window("SC-2", 0.0, 300.0))
        problem = build_problem((station(),), requests, windows, 100.0)
        by_contacts = solve_exact(problem, objective_mode="weighted_contacts")
        by_seconds = solve_exact(problem, objective_mode="weighted_seconds")
        assert by_contacts.objective == pytest.approx(1.0, rel=0.0, abs=1e-12)
        assert by_seconds.objective == pytest.approx(150.0, rel=0.0, abs=1e-9)
        assert by_contacts.selected != by_seconds.selected

    def test_unknown_objective_mode_raises(self, toy_problem):
        with pytest.raises(ValueError, match="unknown objective mode"):
            solve_exact(toy_problem, objective_mode="utility")

    def test_an_empty_candidate_set_raises(self, toy_stations):
        requests = (ContactRequest("R1", "M1", "SC-9", 1, 100.0),)
        problem = build_problem(toy_stations, requests, (), slot_step_s=100.0)
        with pytest.raises(ValueError, match="no candidates"):
            solve_exact(problem)

    def test_notes_record_the_grid_caveat(self, toy_problem):
        assert any("candidate grid" in note for note in solve_exact(toy_problem).notes)


class TestStandaloneCapacity:
    def test_disjoint_slots_all_count(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 9),)
        windows = tuple(window("SC-1", 200.0 * k, 200.0 * k + 100.0) for k in range(4))
        problem = build_problem((station(),), requests, windows, 100.0)
        assert standalone_capacity(problem, "R1") == 4

    def test_overlapping_slots_count_once(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 9),)
        windows = (window("SC-1", 0.0, 150.0),)
        problem = build_problem((station(),), requests, windows, 50.0)
        assert standalone_capacity(problem, "R1") == 1

    def test_zero_for_a_request_with_no_candidates(self, toy_problem):
        assert standalone_capacity(toy_problem, "NOT-A-REQUEST") == 0

    def test_is_an_upper_bound_on_what_any_schedule_delivers(self, instance_seed7):
        assignment = solve_exact(instance_seed7.problem)
        counts = assignment.contacts_by_request(instance_seed7.problem)
        for request in instance_seed7.problem.requests:
            assert counts[request.request_id] <= standalone_capacity(
                instance_seed7.problem, request.request_id
            )


class TestFairnessPolicy:
    def test_default_policy_enforces_nothing(self, toy_problem):
        policy = FairnessPolicy()
        assert policy.enforce_min_contacts is False
        assert policy.max_total_share is None

    @pytest.mark.parametrize("share", [0.0, -0.1, 1.01, 2.0])
    def test_invalid_share_raises(self, share):
        with pytest.raises(ValueError, match="max_total_share"):
            FairnessPolicy(max_total_share=share)

    def test_floor_override_replaces_the_declared_minimum(self):
        policy = FairnessPolicy(min_contacts_floor_override={"R1": 1})
        assert policy.floor_for("R1", 5) == 1
        assert policy.floor_for("R2", 5) == 5

    def test_floor_without_override_is_the_declared_value(self):
        assert FairnessPolicy().floor_for("R1", 3) == 3

    def test_enforcing_minimums_can_raise_the_max_min_share(self):
        from contactsched.fairness import analyse

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
        problem = build_problem((station(),), requests, windows, 100.0)
        free = analyse(problem, solve_exact(problem))
        fair = analyse(problem, solve_exact(problem, policy=FairnessPolicy(True)))
        assert free.max_min == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert fair.max_min == pytest.approx(1.0, rel=0.0, abs=1e-12)

    def test_a_share_cap_limits_one_mission(self):
        # Two missions each with one spacecraft; three slots visible to SC-1 and
        # one to SC-2. Capping any mission at 60 per cent of allocated seconds
        # forces the solver to leave a slot for the second mission.
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 3),
            ContactRequest("R2", "M2", "SC-2", 2, 100.0, 1, 1),
        )
        windows = (
            window("SC-1", 0.0, 100.0),
            window("SC-1", 400.0, 500.0),
            window("SC-1", 800.0, 900.0),
            window("SC-2", 1200.0, 1300.0),
        )
        problem = build_problem((station(),), requests, windows, 100.0)
        free = solve_exact(problem)
        capped = solve_exact(problem, policy=FairnessPolicy(max_total_share=0.6))
        m1_free = free.seconds_by_mission(problem)["M1"]
        total_free = sum(free.seconds_by_mission(problem).values())
        m1_capped = capped.seconds_by_mission(problem)["M1"]
        total_capped = sum(capped.seconds_by_mission(problem).values())
        assert m1_free / total_free > 0.6
        assert m1_capped / total_capped <= 0.6 + 1e-9


class TestInfeasibility:
    @pytest.fixture
    def over_asked(self):
        # R1 declares three contacts but only one slot exists for its spacecraft.
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=3, max_contacts=3),)
        windows = (window("SC-1", 0.0, 100.0),)
        return build_problem((station(),), requests, windows, 100.0)

    def test_status_is_infeasible(self, over_asked):
        assignment = solve_exact(over_asked, policy=FairnessPolicy(enforce_min_contacts=True))
        assert assignment.status == STATUS_INFEASIBLE
        assert assignment.selected == ()

    def test_the_binding_floor_is_named(self, over_asked):
        assignment = solve_exact(over_asked, policy=FairnessPolicy(enforce_min_contacts=True))
        joined = " ".join(assignment.notes)
        assert "R1" in joined
        assert "requires 3 contacts" in joined
        assert "at most 1" in joined

    def test_lowering_the_named_floor_makes_it_feasible(self, over_asked):
        policy = FairnessPolicy(
            enforce_min_contacts=True, min_contacts_floor_override={"R1": 1}
        )
        assert solve_exact(over_asked, policy=policy).status == STATUS_OPTIMAL

    def test_a_jointly_infeasible_instance_says_so(self):
        # Two requests each individually able to take the single slot, each
        # declaring one contact: individually reachable, jointly impossible.
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=1, max_contacts=1),
            ContactRequest("R2", "M2", "SC-2", 2, 100.0, min_contacts=1, max_contacts=1),
        )
        windows = (window("SC-1", 0.0, 100.0), window("SC-2", 0.0, 100.0))
        problem = build_problem((station(),), requests, windows, 100.0)
        assignment = solve_exact(problem, policy=FairnessPolicy(enforce_min_contacts=True))
        assert assignment.status == STATUS_INFEASIBLE
        assert any("infeasibility is joint" in note for note in assignment.notes)

    def test_objective_is_zero_when_infeasible(self, over_asked):
        assignment = solve_exact(over_asked, policy=FairnessPolicy(enforce_min_contacts=True))
        assert assignment.objective == pytest.approx(0.0, rel=0.0, abs=0.0)


class TestDeterminism:
    def test_repeated_solves_give_the_same_objective(self, instance_seed7):
        values = [solve_exact(instance_seed7.problem).objective for _ in range(3)]
        assert np.allclose(values, values[0], rtol=0.0, atol=1e-12)

    def test_repeated_solves_give_the_same_selection(self, instance_seed7):
        first = solve_exact(instance_seed7.problem).selected
        for _ in range(2):
            assert solve_exact(instance_seed7.problem).selected == first
