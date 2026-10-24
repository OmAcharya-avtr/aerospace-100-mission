"""Report formatting: strings, not prints, and every section present."""

from __future__ import annotations

from pointahead.angle import decompose_point_ahead, light_time_pair, score_closed_form
from pointahead.budget import build_budget, verify_sensitivities
from pointahead.report import (
    format_budget,
    format_decomposition,
    format_light_time,
    format_point_ahead,
    format_score,
    format_sensitivity_checks,
)


def test_point_ahead_report_names_the_method_and_the_frame(crossing_link) -> None:
    from pointahead.angle import reference_point_ahead

    text = format_point_ahead(reference_point_ahead(crossing_link))
    assert "point-ahead (reference)" in text
    assert crossing_link.frame in text
    assert "urad" in text


def test_decomposition_report_lists_all_three_lines(crossing_link) -> None:
    text = format_decomposition(decompose_point_ahead(crossing_link))
    for line in ("light_time_lead", "stellar_aberration", "velocity_aberration"):
        assert line in text
    assert "closure vs reference" in text
    assert "frame-dependent split" in text


def test_score_report_shows_all_three_metrics(crossing_link) -> None:
    text = format_score(score_closed_form(crossing_link))
    assert "magnitude difference" in text
    assert "offset difference (closed)" in text
    assert "transmit direction (open)" in text


def test_budget_report_lists_every_term_and_the_beam(crossing_link, default_inputs) -> None:
    text = format_budget(build_budget(crossing_link, default_inputs))
    for term in (
        "relative_velocity_knowledge",
        "ephemeris_staleness_velocity",
        "ephemeris_staleness_geometry",
        "light_time_iteration_residual",
        "aberration_model_error",
        "angle_model_error",
    ):
        assert term in text
    assert "total (root sum of squares)" in text
    assert "total (linear sum)" in text
    assert "dominant term" in text
    assert "divergence half-angle" in text
    assert "pointing loss at total" in text


def test_sensitivity_report_marks_pass_and_comparison(crossing_link, default_inputs) -> None:
    text = format_sensitivity_checks(verify_sensitivities(crossing_link, default_inputs))
    assert "analytic" in text
    assert "numeric" in text
    assert "(equality)" in text
    assert "(upper_bound)" in text


def test_light_time_report_prints_every_iterate(crossing_link) -> None:
    receive, transmit = light_time_pair(crossing_link)
    text = format_light_time("x", receive, transmit)
    assert "receive" in text
    assert "transmit" in text
    for index in range(len(receive.iterates)):
        assert f"iterate[{index}]" in text


def test_reports_are_strings_and_print_nothing(crossing_link, default_inputs, capsys) -> None:
    format_budget(build_budget(crossing_link, default_inputs))
    format_decomposition(decompose_point_ahead(crossing_link))
    format_score(score_closed_form(crossing_link))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
