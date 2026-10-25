"""Requests, candidate generation, conflicts, feasibility and shortfalls.

The hand-built fixtures in ``conftest.py`` have round numbers so the expected
candidate sets and conflict pairs can be written out in full. Those expectations
are in this file as literals.
"""

from __future__ import annotations

import pytest

from contactsched.problem import (
    Candidate,
    ContactRequest,
    build_candidates,
    build_problem,
    check_feasible,
    priority_weight,
    shortfalls,
)
from contactsched.visibility import ContactWindow


class TestPriorityWeight:
    def test_priority_one_has_unit_weight(self):
        assert priority_weight(1) == pytest.approx(1.0, rel=0.0, abs=0.0)

    def test_default_base_halves_each_step(self):
        assert priority_weight(2) == pytest.approx(0.5, rel=0.0, abs=0.0)
        assert priority_weight(3) == pytest.approx(0.25, rel=0.0, abs=0.0)
        assert priority_weight(4) == pytest.approx(0.125, rel=0.0, abs=0.0)

    def test_a_larger_base_separates_priorities_further(self):
        assert priority_weight(2, base=10.0) == pytest.approx(0.1, rel=1e-15, abs=0.0)

    def test_weight_is_strictly_decreasing_in_priority(self):
        weights = [priority_weight(p) for p in range(1, 9)]
        assert weights == sorted(weights, reverse=True)
        assert len(set(weights)) == len(weights)

    @pytest.mark.parametrize("priority", [0, -1])
    def test_non_positive_priority_raises(self, priority):
        with pytest.raises(ValueError, match="priority must be 1 or greater"):
            priority_weight(priority)

    @pytest.mark.parametrize("base", [1.0, 0.5, 0.0, -2.0])
    def test_base_at_or_below_one_raises(self, base):
        with pytest.raises(ValueError, match="weight base"):
            priority_weight(2, base=base)


class TestContactRequest:
    def test_requested_seconds_uses_the_minimum_not_the_maximum(self):
        request = ContactRequest("R", "M", "S", 1, 300.0, min_contacts=3, max_contacts=5)
        assert request.requested_seconds == pytest.approx(900.0, rel=0.0, abs=0.0)

    def test_zero_minimum_means_no_declared_demand(self):
        request = ContactRequest("R", "M", "S", 1, 300.0, min_contacts=0, max_contacts=2)
        assert request.requested_seconds == pytest.approx(0.0, rel=0.0, abs=0.0)

    def test_allows_any_station_when_unrestricted(self):
        request = ContactRequest("R", "M", "S", 1, 300.0)
        assert request.allows("ANYTHING")

    def test_allows_only_the_declared_stations(self):
        request = ContactRequest("R", "M", "S", 1, 300.0, stations=("A1",))
        assert request.allows("A1")
        assert not request.allows("A2")

    @pytest.mark.parametrize(
        "overrides, match",
        [
            (dict(request_id=""), "request_id"),
            (dict(priority=0), "priority"),
            (dict(duration_s=0.0), "duration_s"),
            (dict(duration_s=-5.0), "duration_s"),
            (dict(min_contacts=-1), "min_contacts"),
            (dict(max_contacts=0), "max_contacts"),
            (dict(min_contacts=3, max_contacts=2), "below min_contacts"),
        ],
    )
    def test_invalid_requests_raise(self, overrides, match):
        base = dict(
            request_id="R",
            mission="M",
            spacecraft="S",
            priority=1,
            duration_s=300.0,
            min_contacts=1,
            max_contacts=2,
        )
        base.update(overrides)
        with pytest.raises(ValueError, match=match):
            ContactRequest(**base)


class TestBuildCandidates:
    def test_candidate_count_is_the_hand_counted_value(self, toy_requests, toy_windows):
        # R1 (SC-1, 200 s): A1[0,1000] latest start 800 -> 0, 400, 800;
        #                   A1[4000,4400] latest start 4200 -> 4000, then the
        #                   window-end alignment adds 4200. Five candidates.
        # R2 (SC-2, 200 s): A1[200,1200] latest start 1000 -> 200, 600, 1000;
        #                   A2[100,900] latest start 700 -> 100, 500, plus the
        #                   alignment at 700. Six candidates.
        candidates = build_candidates(toy_requests, toy_windows, slot_step_s=400.0)
        assert len(candidates) == 11
        assert sum(1 for c in candidates if c.request_id == "R1") == 5
        assert sum(1 for c in candidates if c.request_id == "R2") == 6

    def test_candidate_start_times_are_the_hand_listed_grid(self, toy_requests, toy_windows):
        candidates = build_candidates(toy_requests, toy_windows, slot_step_s=400.0)
        r1 = sorted(c.start_s for c in candidates if c.request_id == "R1")
        assert r1 == [0.0, 400.0, 800.0, 4000.0, 4200.0]
        r2 = sorted(c.start_s for c in candidates if c.request_id == "R2")
        assert r2 == [100.0, 200.0, 500.0, 600.0, 700.0, 1000.0]

    def test_indices_are_contiguous_from_zero(self, toy_requests, toy_windows):
        candidates = build_candidates(toy_requests, toy_windows, slot_step_s=400.0)
        assert [c.index for c in candidates] == list(range(len(candidates)))

    def test_every_candidate_fits_inside_its_window(self, toy_requests, toy_windows):
        candidates = build_candidates(toy_requests, toy_windows, slot_step_s=400.0)
        spans = {
            (w.station, w.spacecraft): [] for w in toy_windows
        }
        for window in toy_windows:
            spans[(window.station, window.spacecraft)].append((window.rise_s, window.set_s))
        for candidate in candidates:
            assert any(
                start <= candidate.start_s and candidate.end_s <= end
                for start, end in spans[(candidate.station, candidate.spacecraft)]
            )

    def test_duration_is_the_requested_duration(self, toy_requests, toy_windows):
        for candidate in build_candidates(toy_requests, toy_windows, slot_step_s=400.0):
            assert candidate.duration_s == pytest.approx(200.0, rel=0.0, abs=0.0)

    def test_a_window_shorter_than_the_request_yields_nothing(self):
        request = ContactRequest("R", "M", "S", 1, 600.0)
        window = ContactWindow("A", "S", "M", 0.0, 400.0, 30.0, 200.0, 5.0)
        assert build_candidates((request,), (window,), slot_step_s=100.0) == ()

    def test_a_window_exactly_as_long_as_the_request_yields_one(self):
        request = ContactRequest("R", "M", "S", 1, 400.0)
        window = ContactWindow("A", "S", "M", 0.0, 400.0, 30.0, 200.0, 5.0)
        candidates = build_candidates((request,), (window,), slot_step_s=100.0)
        assert len(candidates) == 1
        assert candidates[0].start_s == pytest.approx(0.0, rel=0.0, abs=0.0)

    def test_alignment_can_be_switched_off(self):
        request = ContactRequest("R", "M", "S", 1, 200.0)
        window = ContactWindow("A", "S", "M", 0.0, 550.0, 30.0, 200.0, 5.0)
        with_align = build_candidates(
            (request,), (window,), slot_step_s=200.0, align_to_window_end=True
        )
        without = build_candidates(
            (request,), (window,), slot_step_s=200.0, align_to_window_end=False
        )
        assert len(with_align) == len(without) + 1
        assert max(c.start_s for c in with_align) == pytest.approx(350.0, rel=0.0, abs=1e-9)

    def test_station_restriction_is_honoured(self, toy_windows):
        request = ContactRequest("R", "M", "SC-2", 1, 200.0, stations=("A2",))
        candidates = build_candidates((request,), toy_windows, slot_step_s=400.0)
        assert {c.station for c in candidates} == {"A2"}

    def test_non_positive_slot_step_raises(self, toy_requests, toy_windows):
        with pytest.raises(ValueError, match="slot_step_s must be positive"):
            build_candidates(toy_requests, toy_windows, slot_step_s=0.0)

    def test_weight_follows_the_priority(self, toy_requests, toy_windows):
        candidates = build_candidates(toy_requests, toy_windows, slot_step_s=400.0)
        for candidate in candidates:
            expected = 1.0 if candidate.request_id == "R1" else 0.5
            assert candidate.weight == pytest.approx(expected, rel=0.0, abs=0.0)


class TestConflicts:
    def test_same_antenna_overlapping_conflicts(self, toy_problem):
        a = Candidate(0, "R1", "M1", "SC-1", "A1", 0.0, 200.0, 1.0, 10.0)
        b = Candidate(1, "R2", "M2", "SC-2", "A1", 100.0, 300.0, 0.5, 10.0)
        assert toy_problem.station_conflict(a, b)
        assert toy_problem.conflicts(a, b)

    def test_turnaround_separates_touching_contacts(self, toy_problem):
        # A1 has a 100 s turnaround, so a contact starting exactly at the end of
        # another still conflicts, and one starting 100 s later does not.
        a = Candidate(0, "R1", "M1", "SC-1", "A1", 0.0, 200.0, 1.0, 10.0)
        touching = Candidate(1, "R2", "M2", "SC-2", "A1", 200.0, 400.0, 0.5, 10.0)
        separated = Candidate(2, "R2", "M2", "SC-2", "A1", 300.0, 500.0, 0.5, 10.0)
        assert toy_problem.station_conflict(a, touching)
        assert not toy_problem.station_conflict(a, separated)

    def test_conflict_is_symmetric(self, toy_problem):
        a = Candidate(0, "R1", "M1", "SC-1", "A1", 0.0, 200.0, 1.0, 10.0)
        b = Candidate(1, "R2", "M2", "SC-2", "A1", 150.0, 350.0, 0.5, 10.0)
        assert toy_problem.station_conflict(a, b) == toy_problem.station_conflict(b, a)

    def test_a_candidate_does_not_conflict_with_itself(self, toy_problem):
        a = Candidate(0, "R1", "M1", "SC-1", "A1", 0.0, 200.0, 1.0, 10.0)
        assert not toy_problem.conflicts(a, a)

    def test_different_antennas_do_not_conflict_on_the_antenna(self, toy_problem):
        a = Candidate(0, "R1", "M1", "SC-1", "A1", 0.0, 200.0, 1.0, 10.0)
        b = Candidate(1, "R2", "M2", "SC-2", "A2", 0.0, 200.0, 0.5, 10.0)
        assert not toy_problem.station_conflict(a, b)
        assert not toy_problem.conflicts(a, b)

    def test_same_spacecraft_on_two_antennas_conflicts(self, toy_problem):
        a = Candidate(0, "R1", "M1", "SC-1", "A1", 0.0, 200.0, 1.0, 10.0)
        b = Candidate(1, "R9", "M1", "SC-1", "A2", 100.0, 300.0, 1.0, 10.0)
        assert toy_problem.spacecraft_conflict(a, b)

    def test_zero_turnaround_antenna_allows_back_to_back(self, toy_problem):
        a = Candidate(0, "R1", "M1", "SC-1", "A2", 0.0, 200.0, 1.0, 10.0)
        b = Candidate(1, "R2", "M2", "SC-2", "A2", 200.0, 400.0, 0.5, 10.0)
        assert not toy_problem.station_conflict(a, b)

    def test_turnaround_lookup_defaults_to_zero_for_an_unknown_antenna(self, toy_problem):
        assert toy_problem.turnaround_s("NOT-A-STATION") == pytest.approx(
            0.0, rel=0.0, abs=0.0
        )


class TestProblemAccessors:
    def test_missions_are_sorted_and_unique(self, toy_problem):
        assert toy_problem.missions == ("M1", "M2")

    def test_request_by_id_covers_every_request(self, toy_problem):
        assert set(toy_problem.request_by_id) == {"R1", "R2"}

    def test_candidates_of_filters_by_request(self, toy_problem):
        assert all(c.request_id == "R1" for c in toy_problem.candidates_of("R1"))

    def test_duplicate_request_ids_raise(self, toy_stations, toy_windows):
        duplicated = (
            ContactRequest("R1", "M1", "SC-1", 1, 200.0),
            ContactRequest("R1", "M2", "SC-2", 2, 200.0),
        )
        with pytest.raises(ValueError, match="duplicate request_id"):
            build_problem(toy_stations, duplicated, toy_windows)


class TestCheckFeasible:
    def test_the_empty_selection_is_feasible(self, toy_problem):
        assert check_feasible(toy_problem, ()) == []

    def test_a_single_candidate_is_feasible(self, toy_problem):
        assert check_feasible(toy_problem, (0,)) == []

    def test_a_repeated_index_is_reported(self, toy_problem):
        problems = check_feasible(toy_problem, (0, 0))
        assert any("more than once" in p for p in problems)

    def test_an_overlapping_pair_is_reported(self, toy_problem):
        # R1 at A1 t=0 and R2 at A1 t=200: A1's 100 s turnaround makes these
        # conflict, so this NEGATIVE CONTROL must fire. Anchored on the true
        # candidate start times rather than on an offset.
        first = next(
            c for c in toy_problem.candidates if c.request_id == "R1" and c.start_s == 0.0
        )
        second = next(
            c
            for c in toy_problem.candidates
            if c.request_id == "R2" and c.station == "A1" and c.start_s == 200.0
        )
        problems = check_feasible(toy_problem, (first.index, second.index))
        assert any("double-booked" in p for p in problems)

    def test_exceeding_max_contacts_is_reported(self, toy_problem):
        # R1 has max_contacts=2; selecting its three A1 candidates in the first
        # window breaks both the ceiling and the antenna exclusion.
        indices = tuple(
            c.index
            for c in toy_problem.candidates
            if c.request_id == "R1" and c.start_s in (0.0, 400.0, 800.0)
        )
        problems = check_feasible(toy_problem, indices)
        assert any("above its" in p for p in problems)

    def test_a_candidate_outside_every_window_is_reported(self, toy_stations, toy_requests):
        # Build a problem whose window list omits the window a candidate came
        # from, so the containment check has something to catch.
        window = ContactWindow("A1", "SC-1", "M1", 0.0, 1000.0, 45.0, 500.0, 10.0)
        problem = build_problem(toy_stations, toy_requests[:1], (window,), slot_step_s=400.0)
        problem.windows = (ContactWindow("A1", "SC-1", "M1", 5000.0, 6000.0, 45.0, 5500.0, 10.0),)
        problems = check_feasible(problem, (0,))
        assert any("not contained in any visibility window" in p for p in problems)


class TestShortfalls:
    def test_every_request_appears_even_when_satisfied(self, toy_problem):
        result = shortfalls(toy_problem, ())
        assert set(result) == {"R1", "R2"}

    def test_nothing_selected_means_the_full_shortfall(self, toy_problem):
        result = shortfalls(toy_problem, ())
        assert result["R1"] == (0, 1, pytest.approx(200.0, rel=0.0, abs=0.0))
        assert result["R2"] == (0, 2, pytest.approx(400.0, rel=0.0, abs=0.0))

    def test_meeting_the_minimum_clears_the_shortfall(self, toy_problem):
        index = toy_problem.candidates_of("R1")[0].index
        result = shortfalls(toy_problem, (index,))
        assert result["R1"] == (1, 0, pytest.approx(0.0, rel=0.0, abs=0.0))

    def test_shortfall_is_never_negative(self, toy_problem):
        indices = tuple(c.index for c in toy_problem.candidates_of("R1")[:2])
        granted, short, seconds = shortfalls(toy_problem, indices)["R1"]
        assert granted == 2
        assert short == 0
        assert seconds == pytest.approx(0.0, rel=0.0, abs=0.0)
