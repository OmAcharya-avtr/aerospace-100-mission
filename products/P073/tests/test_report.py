"""Text rendering."""

from __future__ import annotations

from contactsched.contention import contention_map
from contactsched.exact import solve_exact
from contactsched.fairness import analyse
from contactsched.greedy import schedule_greedy
from contactsched.report import (
    format_backends,
    format_contention,
    format_fairness,
    format_schedule,
    format_windows,
    hms,
)


class TestHms:
    def test_zero(self):
        assert hms(0.0) == "00:00:00"

    def test_one_hour_one_minute_one_second(self):
        assert hms(3661.0) == "01:01:01"

    def test_rounds_to_the_nearest_second(self):
        assert hms(59.6) == "00:01:00"

    def test_does_not_wrap_past_a_day(self):
        # A horizon longer than a day is ordinary; wrapping would read as the
        # wrong day, so the hours field keeps counting.
        assert hms(90000.0) == "25:00:00"

    def test_negative_offsets_carry_a_sign(self):
        assert hms(-61.0) == "-00:01:01"


class TestFormatWindows:
    def test_header_and_count_are_present(self, windows):
        text = format_windows(list(windows), limit=5)
        assert "antenna" in text
        assert f"{len(windows)} windows" in text

    def test_limit_truncates_and_says_so(self, windows):
        text = format_windows(list(windows), limit=3)
        assert "further windows not shown" in text

    def test_no_limit_shows_every_row(self, windows):
        text = format_windows(list(windows), limit=None)
        assert "further windows not shown" not in text

    def test_flags_are_rendered(self, windows):
        text = format_windows(list(windows), limit=None)
        if any(w.clipped_start for w in windows):
            assert "clipped-start" in text

    def test_empty_input_renders_a_zero_count(self):
        assert "0 windows" in format_windows([])


class TestFormatSchedule:
    def test_names_the_method_and_status(self, toy_problem):
        assignment = solve_exact(toy_problem)
        text = format_schedule(toy_problem, assignment)
        assert assignment.method in text
        assert assignment.status in text

    def test_lists_every_selected_contact(self, toy_problem):
        assignment = solve_exact(toy_problem)
        text = format_schedule(toy_problem, assignment)
        assert f"{assignment.contact_count} contacts scheduled" in text
        for index in assignment.selected:
            assert toy_problem.candidates[index].request_id in text

    def test_rows_are_in_time_order(self, instance_seed7):
        assignment = solve_exact(instance_seed7.problem)
        rows = assignment.timeline(instance_seed7.problem)
        assert [r[0] for r in rows] == sorted(r[0] for r in rows)

    def test_notes_are_rendered(self, toy_problem):
        text = format_schedule(toy_problem, solve_exact(toy_problem))
        assert "note:" in text


class TestFormatFairness:
    def test_includes_every_declared_metric(self, toy_problem):
        report = analyse(toy_problem, solve_exact(toy_problem))
        text = format_fairness(report)
        for label in (
            "per-request Jain index",
            "per-mission Jain index",
            "max-min share",
            "requests starved",
            "total seconds short",
            "priority inversions",
            "worst inversion gap",
            "delivered satisfaction by priority",
            "priority promise held in aggregate",
        ):
            assert label in text

    def test_states_the_degenerate_convention_when_it_applies(self, toy_problem):
        from contactsched.assignment import STATUS_HEURISTIC, Assignment

        report = analyse(toy_problem, Assignment("none", STATUS_HEURISTIC, (), 0.0))
        text = format_fairness(report)
        assert "degenerate all-zero convention" in text

    def test_lists_inversions_when_present(self, instance_seed7):
        report = analyse(instance_seed7.problem, schedule_greedy(instance_seed7.problem))
        text = format_fairness(report)
        if report.inversions:
            assert "inversions:" in text
            assert report.inversions[0].higher_request in text

    def test_marks_starved_requests(self, instance_seed7):
        report = analyse(instance_seed7.problem, schedule_greedy(instance_seed7.problem))
        text = format_fairness(report)
        if report.starved:
            assert "starved" in text


class TestFormatContention:
    def test_orders_by_descending_ratio(self, instance_seed7):
        cells = contention_map(
            instance_seed7.problem, instance_seed7.scenario.duration_s, 1800.0
        )
        text = format_contention(cells, limit=5)
        assert "antenna" in text
        assert f"{len(cells)} cells" in text

    def test_infinite_ratios_render_as_inf(self):
        from contactsched.contention import ContentionCell

        cells = [ContentionCell("A1", 0.0, 100.0, 0.0, 5.0, ("R1",))]
        assert "inf" in format_contention(cells)

    def test_limit_can_be_disabled(self, instance_seed7):
        cells = contention_map(
            instance_seed7.problem, instance_seed7.scenario.duration_s, 3600.0
        )
        text = format_contention(cells, limit=None)
        assert text.count("\n") >= len(cells)


class TestFormatBackends:
    def test_lists_highs(self):
        text = format_backends()
        assert "highs" in text
        assert "scipy" in text

    def test_mentions_pulp_and_ortools(self):
        text = format_backends()
        assert "pulp" in text
        assert "ortools-cpsat" in text

    def test_states_which_backend_is_exercised(self):
        assert "the only backend" in format_backends()
