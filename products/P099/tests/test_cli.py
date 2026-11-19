"""The command-line interface, exercised in process."""

from __future__ import annotations

import os

import pytest

from passyield.__main__ import build_parser, main


def _home_prefix() -> str:
    """``<sep>home<sep>``, assembled rather than written as a literal.

    An absolute home-directory path in any committed file fails release-gate
    check 8. The scanner that enforces that lives in
    ``tests/test_determinism.py`` and assembles its needle from ``os.sep`` so
    that its own source does not match; this helper does the same, so a test
    that CHECKS for an absolute path does not become one.
    """
    return os.sep + "home" + os.sep




def test_help_exits_zero(capsys):
    """``--help`` prints and exits 0, which is what the release gate runs."""
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(["--help"])
    assert caught.value.code == 0
    assert "ground-station" in capsys.readouterr().out


def test_help_states_the_research_grade_limit(capsys):
    """The interface's own description carries the required statement."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    text = capsys.readouterr().out
    assert "not flight-qualified" in text
    assert "not certified" in text


def test_no_subcommand_is_an_error():
    """A subcommand is required."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_decompose_prints_the_four_terms(capsys):
    """``decompose`` prints the shortfall and its four named terms."""
    assert main(["decompose", "--pass-id", "3"]) == 0
    text = capsys.readouterr().out
    for token in ("SHORTFALL", "geometry", "acquisition", "re-sync", "idle", "residual"):
        assert token in text
    assert "baseline 1 predicts a shortfall of 0.0 Mbit" in text


def test_distribution_prints_quantiles_for_both_methods(capsys):
    """``distribution`` prints the convolution and the Monte Carlo beside it."""
    assert main(["distribution", "--pass-id", "3", "--monte-carlo-draws", "2000"]) == 0
    text = capsys.readouterr().out
    assert "numerical-convolution" in text
    assert "monte-carlo" in text
    assert "truncated mass" in text
    assert text.count("quantile      shortfall, Mbit") == 2


def test_distribution_backlog_aware_variant_runs(capsys):
    """The strengthened variant is reachable from the interface."""
    assert main([
        "distribution", "--pass-id", "3", "--monte-carlo-draws", "2000", "--backlog-aware"
    ]) == 0
    assert "deterministic dry time" in capsys.readouterr().out


def test_accounted_prints_the_resampling_unit(capsys):
    """``accounted`` prints baseline 3 and names its resampling unit."""
    assert main(["accounted", "--resamples", "200"]) == 0
    text = capsys.readouterr().out
    assert "resampling unit" in text
    assert "'pass'" in text
    assert "bootstrap interval" in text


def test_unknown_pass_id_is_an_error_with_the_available_range(capsys):
    """A bad pass id exits 2 and says what the record holds."""
    assert main(["decompose", "--pass-id", "10000000"]) == 2
    assert "no pass with pass_id" in capsys.readouterr().err


def test_missing_record_is_an_error_without_an_absolute_path(capsys):
    """A missing record exits 2 and prints no absolute path."""
    assert main(["--record", "nowhere.csv", "decompose"]) == 2
    error = capsys.readouterr().err
    assert "nowhere.csv" in error
    assert _home_prefix() not in error
