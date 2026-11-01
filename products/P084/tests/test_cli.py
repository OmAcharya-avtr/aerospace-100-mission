"""Tests for the command line interface, in process, through build_parser."""

from __future__ import annotations

import pytest

from transfergap.__main__ import build_parser, main


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert "transfergap 0.1.0" in capsys.readouterr().out


def test_no_command_is_an_error():
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([])
    assert exit_info.value.code == 2


def test_bounds_command_prints_every_method(capsys):
    assert main(["bounds", "--n", "60", "--seed", "202"]) == 0
    out = capsys.readouterr().out
    for method in (
        "hoeffding",
        "empirical_bernstein",
        "bootstrap_percentile",
        "normal_tolerance",
        "split_conformal",
        "wilks_tolerance",
    ):
        assert method in out
    assert "dist-free" in out


def test_bounds_command_says_what_is_unavailable_at_a_small_n(capsys):
    """At n = 15 neither Wilks nor split conformal exists, and the CLI says so."""
    assert main(["bounds", "--n", "15", "--seed", "101"]) == 0
    out = capsys.readouterr().out
    assert "[not available] wilks_tolerance_upper" in out
    assert "[not available] split_conformal_upper" in out


def test_plan_command_prints_the_additional_counts(capsys):
    assert main(["plan", "--n", "15", "--target-slack", "2.0"]) == 0
    out = capsys.readouterr().out
    assert "MORE" in out
    assert "hoeffding" in out
    assert "existence" in out


def test_exchangeability_command_runs_both_statistics(capsys):
    assert main(["exchangeability", "--n", "40", "--permutations", "199"]) == 0
    out = capsys.readouterr().out
    assert "trend" in out
    assert "half_split" in out


def test_learned_command_reports_coverage_and_the_safety_statement(capsys):
    assert main(["learned", "--n", "60", "--seed", "202"]) == 0
    out = capsys.readouterr().out
    assert "achieved coverage" in out
    assert "This model is not certified for operational flight use." in out


def test_truth_command_reports_a_monte_carlo_standard_error(capsys):
    assert main(["truth", "--draws", "20000"]) == 0
    out = capsys.readouterr().out
    assert "MC standard error" in out
    assert "clip fraction at R" in out


def test_header_carries_the_run_provenance(capsys):
    main(["bounds", "--n", "25", "--seed", "9", "--campaign", "swept"])
    first = capsys.readouterr().out.splitlines()[0]
    assert "transfergap 0.1.0" in first
    assert "n 25" in first
    assert "campaign swept" in first
    assert "seed 9" in first


def test_bad_confidence_returns_status_two(capsys):
    assert main(["bounds", "--n", "40", "--confidence", "1.5"]) == 2
    assert "error:" in capsys.readouterr().err
