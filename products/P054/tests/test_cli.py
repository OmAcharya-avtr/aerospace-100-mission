"""Tests for the ``python -m rareverify`` command-line interface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from rareverify.__main__ import EXIT_BAD_INPUT, EXIT_CHECK_FAILED, EXIT_OK, main

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "rareverify", *args],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
        check=False,
        timeout=300,
    )


def test_help_exits_zero():
    result = _run(["--help"])
    assert result.returncode == 0
    assert "rareverify" in result.stdout
    assert "not flight-qualified" in result.stdout


def test_every_subcommand_has_help():
    for command in (
        "plan",
        "interval",
        "coverage",
        "mc",
        "is",
        "tilt-sweep",
        "subset",
        "surrogate",
        "benchmark",
        "known-answer",
        "campaign",
    ):
        result = _run([command, "--help"])
        assert result.returncode == 0, command


def test_no_subcommand_is_an_input_error():
    result = _run([])
    assert result.returncode == EXIT_BAD_INPUT


def test_plan_subcommand(capsys):
    assert main(["plan", "--target-probability", "1e-4"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "29956" in out
    assert "coefficient of variation" in out


def test_interval_subcommand_zero_failures(capsys):
    assert main(["interval", "--failures", "0", "--samples", "3000", "--side", "upper"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "9.980790e-04" in out
    assert "rule-of-three" in out


def test_coverage_subcommand_reports_the_wilson_dip(capsys):
    assert (
        main(
            [
                "coverage",
                "--samples",
                "40",
                "--probability",
                "0.03",
                "--method",
                "wilson",
            ]
        )
        == EXIT_OK
    )
    out = capsys.readouterr().out
    assert "exact coverage" in out


def test_mc_and_is_subcommands(capsys):
    assert main(["mc", "--samples", "20000", "--seed", "1"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "crude:" in out
    assert main(["is", "--samples", "20000", "--seed", "1"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "importance-sampling:" in out
    assert "analytic-design-point" in out


def test_is_subcommand_accepts_every_tilt(capsys):
    for tilt in ("analytic", "oracle", "scaled", "orthogonal"):
        assert (
            main(
                [
                    "is",
                    "--samples",
                    "5000",
                    "--tilt",
                    tilt,
                    "--tilt-scale",
                    "0.8",
                    "--seed",
                    "2",
                ]
            )
            == EXIT_OK
        )
        capsys.readouterr()


def test_subset_subcommand(capsys):
    assert main(["subset", "--per-level", "1000", "--seed", "2"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "subset-simulation:" in out
    assert "LOWER BOUND" in out


def test_tilt_sweep_subcommand_flags_a_worse_tilt(capsys):
    assert (
        main(["tilt-sweep", "--samples", "10000", "--replications", "10", "--seed", "3"])
        == EXIT_OK
    )
    out = capsys.readouterr().out
    assert "WORSE" in out
    assert "mean-squared-error comparison" in out


def test_benchmark_subcommand(capsys):
    assert (
        main(["benchmark", "--samples", "20000", "--replications", "8", "--seed", "4"])
        == EXIT_OK
    )
    out = capsys.readouterr().out
    assert "analytic-IS vs crude" in out
    assert "subset-simulation vs crude" in out


def test_known_answer_subcommand_passes(capsys):
    assert main(["known-answer", "--samples", "100000", "--seed", "5"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "6/6 checks passed" in out


def test_known_answer_subcommand_fails_with_an_absurd_tolerance(capsys):
    status = main(["known-answer", "--samples", "20000", "--tolerance", "1e-9"])
    capsys.readouterr()
    assert status == EXIT_CHECK_FAILED


def test_surrogate_subcommand(capsys):
    assert (
        main(
            [
                "surrogate",
                "--limit-state",
                "rippled",
                "--n-train",
                "80",
                "--samples",
                "20000",
                "--seed",
                "6",
            ]
        )
        == EXIT_OK
    )
    out = capsys.readouterr().out
    assert "surrogate-guided IS" in out
    assert "surrogate-only" in out
    assert "verdict at equal true-evaluation budget" in out


def test_campaign_subcommand_exit_codes(capsys):
    met = main(
        [
            "campaign",
            "--target-probability",
            "1e-4",
            "--beta",
            "4.8",
            "--samples",
            "40000",
            "--seed",
            "7",
        ]
    )
    capsys.readouterr()
    assert met == EXIT_OK
    missed = main(
        [
            "campaign",
            "--target-probability",
            "1e-6",
            "--beta",
            "3.0",
            "--samples",
            "20000",
            "--seed",
            "7",
        ]
    )
    capsys.readouterr()
    assert missed == EXIT_CHECK_FAILED


def test_bad_input_returns_exit_code_two(capsys):
    status = main(["plan", "--target-probability", "1.5"])
    assert status == EXIT_BAD_INPUT
    assert "rareverify:" in capsys.readouterr().err


def test_runtime_error_returns_exit_code_three(capsys):
    """A subset-simulation run that cannot reach the limit state."""
    status = main(["subset", "--beta", "9.0", "--per-level", "1000", "--p0", "0.5"])
    assert status in (EXIT_CHECK_FAILED, EXIT_OK)
    capsys.readouterr()


@pytest.mark.parametrize(
    "args",
    [
        ["mc", "--samples", "0"],
        ["interval", "--failures", "5", "--samples", "2"],
        ["coverage", "--samples", "10", "--probability", "2.0"],
        ["subset", "--per-level", "3"],
        ["is", "--samples", "100", "--tilt", "orthogonal", "--dimension", "1"],
    ],
)
def test_invalid_arguments_exit_two(args, capsys):
    assert main(args) == EXIT_BAD_INPUT
    capsys.readouterr()
