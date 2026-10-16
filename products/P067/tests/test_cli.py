"""Command line interface: every subcommand runs and prints what it claims to."""

from __future__ import annotations

import io

import pytest

from acqsearch.__main__ import build_parser, main


def run(argv: list[str]) -> str:
    """Run the CLI and capture its output."""
    buffer = io.StringIO()
    assert main(argv, out=buffer) == 0
    return buffer.getvalue()


def test_patterns_lists_all_four_patterns():
    text = run(["patterns"])
    for name in ("spiral", "raster", "lissajous", "random_restart"):
        assert name in text
    assert "cells 89" in text


def test_expected_prints_the_three_estimates():
    text = run(["expected", "--trials", "4000"])
    assert "analytic E[T | in grid]" in text
    assert "Monte Carlo, same assumptions" in text
    assert "Monte Carlo, continuous geometry" in text


def test_expected_accepts_another_pattern():
    text = run(["expected", "--pattern", "raster", "--trials", "2000"])
    assert "pattern raster" in text


def test_dwell_sweep_reports_a_grid_minimum():
    text = run(["dwell", "--points", "9", "--dwell-min-ms", "0.4", "--dwell-max-ms", "4.0"])
    assert "grid minimum" in text
    assert len(text.strip().splitlines()) == 11


def test_coverage_compares_the_closed_form_with_monte_carlo():
    text = run(["coverage", "--turns", "2", "--samples", "4000"])
    assert "closed form applies: True" in text
    assert "closed_form" in text and "monte_carlo" in text


def test_benchmark_runs_end_to_end_and_names_a_winner():
    text = run(["benchmark", "--scenarios", "20", "--train-episodes", "1", "--targets", "1"])
    assert "winner" in text
    assert "bayesian_greedy" in text
    assert "learned" in text
    assert "paired learned - bayesian_greedy" in text


def test_scenarios_writes_a_file(tmp_path):
    out = tmp_path / "scenarios.json"
    text = run(["scenarios", "--scenarios", "12", "--output", str(out)])
    assert out.exists()
    assert "wrote 12 scenarios" in text


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_unknown_command_exits_non_zero():
    with pytest.raises(SystemExit):
        main(["nonsense"])


def test_physics_arguments_change_the_answer():
    narrow = run(["patterns", "--sigma-urad", "80"])
    wide = run(["patterns", "--sigma-urad", "220"])
    assert narrow != wide
