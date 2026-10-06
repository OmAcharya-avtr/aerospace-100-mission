"""The command-line interface, exercised in-process and in a clean subprocess."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

import slotsync
from slotsync.__main__ import build_parser, main


def _child_environment() -> dict[str, str]:
    """Environment for a clean subprocess, with the package importable.

    The gate runs pytest without ``PYTHONPATH``; ``[tool.pytest.ini_options]``
    puts ``src`` on the *pytest* process's path but a child process does not
    inherit that, so the path is derived here from the imported package's own
    location rather than written down anywhere.
    """
    root = pathlib.Path(slotsync.__file__).resolve().parent.parent
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(root) if not existing else str(root) + os.pathsep + existing
    )
    return environment


def test_help_exits_zero_in_a_clean_subprocess() -> None:
    """The packaging contract: ``python -m slotsync --help`` must work from a cold start."""
    result = subprocess.run(
        [sys.executable, "-m", "slotsync", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=_child_environment(),
    )
    assert result.returncode == 0, result.stderr
    assert "slot and symbol timing recovery" in result.stdout.lower()


def test_version_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "slotsync", "--version"],
        capture_output=True,
        text=True,
        check=False,
        env=_child_environment(),
    )
    assert result.returncode == 0, result.stderr
    assert "slotsync 0.1.0" in result.stdout


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_scurve_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["scurve", "--detector", "mueller-muller", "--pulse", "tri"]) == 0
    out = capsys.readouterr().out
    assert "K_d" in out
    assert "mueller-muller" in out


def test_scurve_reports_a_zero_gain_rather_than_hiding_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["scurve", "--detector", "early-late", "--pulse", "rect"]) == 0
    assert "carries no timing information" in capsys.readouterr().out


def test_loop_subcommand_round_trips(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["loop", "--bandwidth", "0.01", "--damping", "0.7", "--gain", "1.5"]) == 0
    out = capsys.readouterr().out
    assert "recovered B_n" in out
    assert "0.01" in out


def test_jitter_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "jitter",
                "--detector",
                "mueller-muller",
                "--pulse",
                "nyq-rc",
                "--symbols",
                "8000",
                "--open-loop-samples",
                "20000",
                "--max-lag",
                "8",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "jitter var, closed form (white)" in out
    assert "jitter var, Monte Carlo" in out


def test_jitter_subcommand_refuses_a_zero_gain(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["jitter", "--detector", "early-late", "--pulse", "rect", "--symbols", "4000"]) == 1
    assert "gain is zero" in capsys.readouterr().out


def test_slip_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "slip",
                "--detector",
                "mueller-muller",
                "--points",
                "3",
                "--open-loop-samples",
                "20000",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "slip/symbol" in out
    assert out.count("\n") >= 5


def test_ppm_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "ppm",
                "--order",
                "4",
                "--symbols",
                "4000",
                "--open-loop-symbols",
                "2000",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "K_d_per_slot" in out
    assert "slot error rate (closed loop)" in out
    assert "B_n per slot" in out


def test_unknown_subcommand_exits_non_zero() -> None:
    with pytest.raises(SystemExit):
        main(["frequency"])
