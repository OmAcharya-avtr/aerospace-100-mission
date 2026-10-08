"""Tests for the ``python -m twininvalidate`` command line interface."""

from __future__ import annotations

import pytest

from twininvalidate.__main__ import build_parser, main

SUBCOMMANDS = ["twin", "scenarios", "calibrate", "curve", "benchmark", "ambiguity"]


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "twininvalidate" in out
    for name in SUBCOMMANDS:
        assert name in out


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_every_subcommand_has_help(name, capsys):
    with pytest.raises(SystemExit) as exc:
        main([name, "--help"])
    assert exc.value.code == 0
    assert capsys.readouterr().out


def test_help_carries_the_safety_statement(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "not flight-qualified" in out
    assert "not certified" in out
    assert "not approved" in out


def test_no_subcommand_is_an_error():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


def test_twin_subcommand(capsys):
    assert main(["twin"]) == 0
    out = capsys.readouterr().out
    assert "innovation variance S" in out
    assert "predictor gain K" in out
    assert "spectral radius" in out


def test_scenarios_subcommand(capsys):
    assert main(["scenarios"]) == 0
    out = capsys.readouterr().out
    for name in ("parameter_step", "slow_ramp", "noise_variance"):
        assert name in out


def test_calibrate_subcommand(capsys):
    assert main(["calibrate", "--runs", "20", "--samples", "500", "--target", "200"]) == 0
    out = capsys.readouterr().out
    assert "target ARL0 = 200" in out
    assert "per 1000 h" in out
    assert "CUSUM" in out and "EWMA" in out and "GLR" in out


def test_curve_subcommand(capsys):
    assert (
        main(
            [
                "curve",
                "parameter_step",
                "--runs",
                "20",
                "--samples",
                "400",
                "--points",
                "3",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "detection-delay against false-alarm curve" in out
    assert "FA/1000h" in out


def test_curve_rejects_an_unknown_scenario():
    with pytest.raises(SystemExit):
        main(["curve", "nonsense"])


def test_ambiguity_subcommand(capsys):
    assert main(["ambiguity", "--runs", "5", "--samples", "500"]) == 0
    out = capsys.readouterr().out
    assert "indistinguishable from the residual alone" in out
    assert "never its cause" in out


def test_benchmark_subcommand(capsys):
    assert main(["benchmark", "--runs", "25", "--samples", "600", "--target", "200"]) == 0
    out = capsys.readouterr().out
    assert "matched in-control" in out
    assert "OUT OF DIST" in out
    assert "wall clock" in out


def test_parser_builds_and_requires_a_command():
    parser = build_parser()
    assert parser.prog == "python -m twininvalidate"
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_errors_are_reported_as_exit_code_two(capsys):
    # A calibration bank far too small to resolve the requested ARL0 raises a
    # ValueError inside the subcommand, which the CLI must turn into exit 2
    # with a readable message rather than a traceback.
    with pytest.raises(SystemExit) as exc:
        main(["calibrate", "--runs", "2", "--samples", "20", "--target", "1.0"])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
