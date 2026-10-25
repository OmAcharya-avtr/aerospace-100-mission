"""Fairness accounting: Jain's index, max-min share and priority inversions.

Every index value asserted here is hand-calculated and the arithmetic is in the
test comment.
"""

from __future__ import annotations

import pytest

from contactsched.assignment import STATUS_HEURISTIC, Assignment
from contactsched.exact import FairnessPolicy, solve_exact
from contactsched.fairness import (
    INVERSION_TOLERANCE,
    analyse,
    jain_index,
    max_min_share,
)
from contactsched.greedy import schedule_greedy
from contactsched.problem import ContactRequest, build_problem
from contactsched.station import GroundStation, HorizonMask
from contactsched.visibility import ContactWindow


def station(name: str = "A1", turnaround_s: float = 0.0) -> GroundStation:
    return GroundStation(name, 0.0, 0.0, 0.0, HorizonMask(floor_deg=5.0), turnaround_s)


def window(spacecraft: str, rise: float, close: float, station_name: str = "A1") -> ContactWindow:
    return ContactWindow(
        station_name, spacecraft, "M", rise, close, 45.0, 0.5 * (rise + close), 5.0
    )


class TestJainIndex:
    def test_equal_shares_give_one(self):
        # (4)**2 / (4 * 4) = 1 exactly.
        assert jain_index([1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0, rel=0.0, abs=1e-15)

    def test_equal_fractional_shares_give_one(self):
        # (1.0)**2 / (2 * 0.5) = 1 exactly.
        assert jain_index([0.5, 0.5]) == pytest.approx(1.0, rel=0.0, abs=1e-15)

    def test_one_winner_gives_the_lower_bound(self):
        # (1)**2 / (4 * 1) = 0.25 = 1/n.
        assert jain_index([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.25, rel=0.0, abs=1e-15)

    def test_a_two_to_one_split(self):
        # (1.5)**2 / (2 * (1 + 0.25)) = 2.25 / 2.5 = 0.9 exactly.
        assert jain_index([1.0, 0.5]) == pytest.approx(0.9, rel=0.0, abs=1e-15)

    def test_a_three_way_split(self):
        # (1 + 0.5 + 0.5)**2 / (3 * (1 + 0.25 + 0.25)) = 4 / 4.5 = 0.888888...
        assert jain_index([1.0, 0.5, 0.5]) == pytest.approx(
            4.0 / 4.5, rel=1e-15, abs=0.0
        )

    def test_is_scale_invariant(self):
        assert jain_index([2.0, 1.0]) == pytest.approx(jain_index([1.0, 0.5]), rel=1e-15, abs=0.0)

    def test_lies_within_the_declared_range(self):
        for shares in ([1.0], [1.0, 0.0], [0.3, 0.7, 0.1], [1.0] * 7):
            value = jain_index(shares)
            assert 1.0 / len(shares) - 1e-12 <= value <= 1.0 + 1e-12

    def test_empty_gives_one(self):
        assert jain_index([]) == pytest.approx(1.0, rel=0.0, abs=0.0)

    def test_all_zero_gives_the_degenerate_one(self):
        # Documented convention: an all-zero allocation IS perfectly equal.
        assert jain_index([0.0, 0.0, 0.0]) == pytest.approx(1.0, rel=0.0, abs=0.0)

    def test_negative_shares_raise(self):
        with pytest.raises(ValueError, match="non-negative"):
            jain_index([1.0, -0.1])


class TestMaxMinShare:
    def test_is_the_smallest_share(self):
        assert max_min_share([0.4, 0.9, 0.2]) == pytest.approx(0.2, rel=0.0, abs=0.0)

    def test_empty_gives_one(self):
        assert max_min_share([]) == pytest.approx(1.0, rel=0.0, abs=0.0)

    def test_distinguishes_the_all_zero_case_from_jain(self):
        # The pairing that makes the degenerate Jain convention safe to publish.
        assert jain_index([0.0, 0.0]) == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert max_min_share([0.0, 0.0]) == pytest.approx(0.0, rel=0.0, abs=0.0)


class TestAnalyseOutcomes:
    @pytest.fixture
    def problem(self):
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=2, max_contacts=2),
            ContactRequest("R2", "M2", "SC-2", 2, 100.0, min_contacts=2, max_contacts=2),
        )
        windows = (
            window("SC-1", 0.0, 100.0),
            window("SC-1", 400.0, 500.0),
            window("SC-2", 0.0, 100.0),
            window("SC-2", 400.0, 500.0),
        )
        return build_problem((station(),), requests, windows, 100.0)

    def test_nothing_allocated_gives_zero_satisfaction(self, problem):
        empty = Assignment("none", STATUS_HEURISTIC, (), 0.0)
        report = analyse(problem, empty)
        assert all(o.satisfaction == 0.0 for o in report.outcomes)
        assert report.max_min == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert report.all_starved is True
        assert report.jain == pytest.approx(1.0, rel=0.0, abs=0.0)

    def test_total_seconds_short_sums_the_shortfalls(self, problem):
        empty = Assignment("none", STATUS_HEURISTIC, (), 0.0)
        # Each request declared 2 x 100 s and got nothing: 400 s short in total.
        assert analyse(problem, empty).total_seconds_short == pytest.approx(
            400.0, rel=0.0, abs=0.0
        )

    def test_one_request_fully_served_the_other_starved(self, problem):
        indices = tuple(c.index for c in problem.candidates_of("R1"))
        report = analyse(problem, Assignment("x", STATUS_HEURISTIC, indices, 0.0))
        by_id = {o.request_id: o for o in report.outcomes}
        assert by_id["R1"].satisfaction == pytest.approx(1.0, rel=0.0, abs=1e-15)
        assert by_id["R2"].satisfaction == pytest.approx(0.0, rel=0.0, abs=0.0)
        # Jain over [1, 0]: 1 / (2 * 1) = 0.5.
        assert report.jain == pytest.approx(0.5, rel=0.0, abs=1e-15)
        assert report.max_min == pytest.approx(0.0, rel=0.0, abs=0.0)

    def test_over_service_is_capped_for_the_index_but_reported_uncapped(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=1, max_contacts=3),)
        windows = tuple(window("SC-1", 300.0 * k, 300.0 * k + 100.0) for k in range(3))
        problem = build_problem((station(),), requests, windows, 100.0)
        report = analyse(problem, solve_exact(problem))
        outcome = report.outcomes[0]
        assert outcome.granted_contacts == 3
        assert outcome.satisfaction == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert outcome.satisfaction_uncapped == pytest.approx(3.0, rel=0.0, abs=1e-15)

    def test_a_request_with_no_declared_minimum_has_no_demand(self):
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=0, max_contacts=1),
            ContactRequest("R2", "M2", "SC-2", 2, 100.0, min_contacts=1, max_contacts=1),
        )
        windows = (window("SC-1", 0.0, 100.0), window("SC-2", 400.0, 500.0))
        problem = build_problem((station(),), requests, windows, 100.0)
        report = analyse(problem, Assignment("none", STATUS_HEURISTIC, (), 0.0))
        by_id = {o.request_id: o for o in report.outcomes}
        assert by_id["R1"].has_demand is False
        assert by_id["R1"].starved is False
        assert by_id["R2"].starved is True
        # Only R2 enters the indices, so the max-min share is R2's zero.
        assert report.max_min == pytest.approx(0.0, rel=0.0, abs=0.0)

    def test_contacts_short_and_seconds_short_are_never_negative(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=1, max_contacts=3),)
        windows = tuple(window("SC-1", 300.0 * k, 300.0 * k + 100.0) for k in range(3))
        problem = build_problem((station(),), requests, windows, 100.0)
        outcome = analyse(problem, solve_exact(problem)).outcomes[0]
        assert outcome.contacts_short == 0
        assert outcome.seconds_short == pytest.approx(0.0, rel=0.0, abs=0.0)


class TestMissionAggregation:
    def test_mission_satisfaction_sums_seconds_not_ratios(self):
        # M1 owns two requests: one needing 100 s and one needing 900 s. Serving
        # only the small one gives a mission satisfaction of 100/1000 = 0.1, not
        # the 0.5 an average of ratios would report.
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=1, max_contacts=1),
            ContactRequest("R2", "M1", "SC-2", 1, 900.0, min_contacts=1, max_contacts=1),
        )
        windows = (window("SC-1", 0.0, 100.0), window("SC-2", 0.0, 900.0))
        problem = build_problem((station(),), requests, windows, 100.0)
        chosen = tuple(c.index for c in problem.candidates_of("R1"))[:1]
        report = analyse(problem, Assignment("x", STATUS_HEURISTIC, chosen, 0.0))
        assert report.mission_satisfaction["M1"] == pytest.approx(0.1, rel=1e-15, abs=0.0)

    def test_per_mission_and_per_request_indices_can_differ(self, instance_seed7):
        report = analyse(instance_seed7.problem, solve_exact(instance_seed7.problem))
        assert report.jain != report.jain_by_mission


class TestPriorityInversions:
    @pytest.fixture
    def inverted(self):
        # Antenna A1 has one slot at t=0 visible to SC-1 (priority 1 request) and
        # three slots visible to SC-2 (priority 3 request). The priority-1
        # request's only placement conflicts with the priority-3 request's first,
        # and the solver prefers the three cheap contacts in weighted seconds.
        requests = (
            ContactRequest("R-HIGH", "M1", "SC-1", 1, 100.0, min_contacts=1, max_contacts=1),
            ContactRequest("R-LOW", "M2", "SC-2", 3, 100.0, min_contacts=1, max_contacts=1),
        )
        windows = (window("SC-1", 0.0, 100.0), window("SC-2", 0.0, 100.0))
        problem = build_problem((station(),), requests, windows, 100.0)
        # Force the inversion by selecting the LOW request's candidate by hand.
        low = problem.candidates_of("R-LOW")[0].index
        return problem, Assignment("forced", STATUS_HEURISTIC, (low,), 0.25)

    def test_the_inversion_is_detected(self, inverted):
        problem, assignment = inverted
        report = analyse(problem, assignment)
        assert report.inversion_count == 1
        assert report.comparable_pairs == 1
        assert report.inversion_rate == pytest.approx(1.0, rel=0.0, abs=0.0)

    def test_the_inversion_names_both_requests_the_right_way_round(self, inverted):
        problem, assignment = inverted
        inversion = analyse(problem, assignment).inversions[0]
        assert inversion.higher_request == "R-HIGH"
        assert inversion.lower_request == "R-LOW"
        assert inversion.higher_priority == 1
        assert inversion.lower_priority == 3

    def test_the_gap_is_the_satisfaction_difference(self, inverted):
        problem, assignment = inverted
        inversion = analyse(problem, assignment).inversions[0]
        assert inversion.gap == pytest.approx(1.0, rel=0.0, abs=1e-15)

    def test_the_aggregate_promise_is_broken_too(self, inverted):
        problem, assignment = inverted
        report = analyse(problem, assignment)
        assert report.priority_monotone is False
        assert report.delivered_by_priority[1] == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert report.delivered_by_priority[3] == pytest.approx(1.0, rel=0.0, abs=1e-15)

    def test_equal_priorities_are_not_comparable(self):
        requests = (
            ContactRequest("R1", "M1", "SC-1", 2, 100.0, 1, 1),
            ContactRequest("R2", "M2", "SC-2", 2, 100.0, 1, 1),
        )
        windows = (window("SC-1", 0.0, 100.0), window("SC-2", 0.0, 100.0))
        problem = build_problem((station(),), requests, windows, 100.0)
        report = analyse(problem, solve_exact(problem))
        assert report.comparable_pairs == 0
        assert report.inversion_count == 0
        assert report.inversion_rate == pytest.approx(0.0, rel=0.0, abs=0.0)

    def test_a_tolerance_sized_difference_is_not_an_inversion(self):
        # Two requests whose satisfactions differ by less than the tolerance must
        # not be counted. Asserted with an explicit tolerance rather than by
        # relying on exact float equality, which is the defect class this
        # mission has shipped twice.
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 1),
            ContactRequest("R2", "M2", "SC-2", 2, 100.0, 1, 1),
        )
        windows = (window("SC-1", 0.0, 100.0), window("SC-2", 400.0, 500.0))
        problem = build_problem((station(),), requests, windows, 100.0)
        report = analyse(problem, solve_exact(problem), tolerance=INVERSION_TOLERANCE)
        assert report.inversion_count == 0

    def test_worst_gap_is_zero_without_inversions(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 1),)
        windows = (window("SC-1", 0.0, 100.0),)
        problem = build_problem((station(),), requests, windows, 100.0)
        report = analyse(problem, solve_exact(problem))
        assert report.worst_inversion_gap == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert report.priority_monotone is True


class TestFairnessUnderRealMethods:
    def test_both_methods_produce_a_report(self, instance_seed7):
        problem = instance_seed7.problem
        for assignment in (schedule_greedy(problem), solve_exact(problem)):
            report = analyse(problem, assignment)
            assert len(report.outcomes) == len(problem.requests)
            assert 0.0 <= report.jain <= 1.0
            assert 0.0 <= report.max_min <= 1.0

    def test_enforcing_minimums_removes_starvation_or_is_infeasible(self, instance_seed7):
        problem = instance_seed7.problem
        fair = solve_exact(problem, policy=FairnessPolicy(enforce_min_contacts=True))
        if fair.status == "optimal":
            assert analyse(problem, fair).starved == ()
        else:
            assert fair.status == "infeasible"

    def test_the_report_carries_the_method_name(self, instance_seed7):
        assignment = schedule_greedy(instance_seed7.problem)
        assert analyse(instance_seed7.problem, assignment).method == assignment.method
