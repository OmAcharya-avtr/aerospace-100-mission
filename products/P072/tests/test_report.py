"""Text rendering. Nothing in the library prints, so every function returns a string."""

from __future__ import annotations

from downlinkpick.exact import exact_select
from downlinkpick.marginal import left_behind_report, marginal_value_of_contact
from downlinkpick.report import (
    format_full_report,
    format_invariants,
    format_left_behind,
    format_marginals,
    format_plan,
    format_selection,
)
from downlinkpick.selection import Selection, SelectionProblem


def test_format_plan_has_one_line_per_window_plus_a_total(
    tiny_problem: SelectionProblem,
) -> None:
    text = format_plan(tiny_problem)
    lines = text.splitlines()
    assert len(lines) == len(tiny_problem.plan) + 2
    assert "window" in lines[0]
    assert "w1" in text
    assert "w2" in text
    assert "total" in lines[-1]


def test_format_selection_lists_every_assignment(tiny_problem: SelectionProblem) -> None:
    selection = exact_select(tiny_problem)
    text = format_selection(tiny_problem, selection)
    for assignment in selection.assignments:
        assert assignment.product in text
    assert "value source declared" in text
    assert "committed" in text
    assert "spare" in text


def test_format_left_behind_reports_reasons(tiny_problem: SelectionProblem) -> None:
    rows = left_behind_report(tiny_problem, exact_select(tiny_problem))
    text = format_left_behind(rows)
    assert "bulk" in text
    assert "oversized" in text
    assert "value left behind by reason" in text


def test_format_left_behind_on_an_empty_report() -> None:
    assert "nothing left behind" in format_left_behind(())


def test_format_marginals_includes_the_crossing_column(
    tiny_problem: SelectionProblem,
) -> None:
    marginals = marginal_value_of_contact(tiny_problem, delta_s=10.0)
    text = format_marginals(marginals)
    assert "crossings" in text
    assert "finite-difference step" in text
    assert "w1" in text


def test_format_marginals_on_an_empty_tuple() -> None:
    assert format_marginals(()) == "no windows priced"


def test_format_invariants_reports_both_residuals(tiny_problem: SelectionProblem) -> None:
    text = format_invariants(tiny_problem, exact_select(tiny_problem))
    assert "all hold" in text
    assert "worst capacity overshoot" in text
    assert "value residual" in text


def test_format_invariants_reports_a_violation(tiny_problem: SelectionProblem) -> None:
    text = format_invariants(tiny_problem, Selection((), 99.0, "corrupt"))
    assert "violated" in text
    assert "VIOLATION" in text


def test_format_full_report_contains_every_block(tiny_problem: SelectionProblem) -> None:
    selection = exact_select(tiny_problem)
    marginals = marginal_value_of_contact(tiny_problem, base=selection, delta_s=10.0)
    rows = left_behind_report(tiny_problem, selection)
    text = format_full_report(tiny_problem, selection, marginals, rows)
    for heading in (
        "contact plan",
        "selection",
        "left behind",
        "marginal value of contact time",
        "invariants",
    ):
        assert heading in text


def test_no_print_in_the_library() -> None:
    # The CLI is the only module permitted to write to a stream. A print in the
    # library makes the package unusable inside anything that captures stdout.
    import pathlib

    source = pathlib.Path(__file__).resolve().parent.parent / "src" / "downlinkpick"
    offenders = []
    for path in sorted(source.glob("*.py")):
        if path.name == "__main__.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("print(") or " print(" in stripped:
                offenders.append(f"{path.name}:{number}")
    assert offenders == []
