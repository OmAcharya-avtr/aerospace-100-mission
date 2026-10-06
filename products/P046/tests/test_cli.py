"""CLI surface: every subcommand runs, and --help exits 0 in a clean subprocess."""

import os
import pathlib
import subprocess
import sys

import pytest

import interleavekit
from interleavekit.__main__ import main

#: Directory holding the installed or source-tree package, resolved at runtime so
#: no absolute path is ever written into a tracked file.
_SRC_DIR = str(pathlib.Path(interleavekit.__file__).resolve().parent.parent)


def _run(args):
    """Run the CLI in-process and return the exit status."""
    return main(args)


def test_help_exits_zero_in_a_clean_subprocess():
    """The gate runs this with no PYTHONPATH beyond src/; it must exit 0."""
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "interleavekit", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "permutation" in proc.stdout
    assert "design" in proc.stdout


def test_version_exits_zero():
    with pytest.raises(SystemExit) as exc:
        _run(["--version"])
    assert exc.value.code == 0


def test_no_subcommand_is_an_error():
    with pytest.raises(SystemExit) as exc:
        _run([])
    assert exc.value.code != 0


@pytest.mark.parametrize("kind", ["block", "helical", "srandom", "convolutional"])
def test_permutation_subcommand(kind, capsys):
    assert _run(["permutation", kind, "--length", "32", "--spread", "3"]) == 0
    out = capsys.readouterr().out
    assert out.strip()


@pytest.mark.parametrize("kind", ["block", "helical", "srandom", "convolutional"])
def test_metrics_subcommand(kind, capsys):
    assert (
        _run(["metrics", kind, "--length", "64", "--spread", "4", "--bursts", "6"]) == 0
    )
    out = capsys.readouterr().out
    assert "largest fully dispersed burst" in out


def test_metrics_reports_minimum_spread_for_a_permutation(capsys):
    assert _run(["metrics", "block", "--depth", "4", "--span", "4"]) == 0
    out = capsys.readouterr().out
    assert "minimum spread: 5" in out
    assert "normalised dispersion" in out


def test_metrics_says_spread_is_undefined_for_the_convolutional_bank(capsys):
    assert _run(["metrics", "convolutional", "--registers", "4", "--slope", "1"]) == 0
    assert "not defined" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["block", "helical", "srandom", "convolutional"])
def test_cost_subcommand(kind, capsys):
    assert _run(["cost", kind, "--length", "64", "--spread", "4"]) == 0
    out = capsys.readouterr().out
    assert "pair latency" in out
    assert "ms at" in out


def test_compare_subcommand(capsys):
    assert _run(["compare", "--depth", "8", "--span", "8", "--spread", "5"]) == 0
    out = capsys.readouterr().out
    assert "BlockInterleaver(depth=8, span=8)" in out
    assert "ConvolutionalInterleaver" in out


def test_design_subcommand(capsys):
    assert _run(["design", "8", "--max-block-symbols", "256"]) == 0
    out = capsys.readouterr().out
    assert "cheapest parameters that fully disperse a burst of 8 symbols" in out


def test_invalid_parameters_return_status_two(capsys):
    assert _run(["cost", "block", "--depth", "0"]) == 2
    assert "error:" in capsys.readouterr().err


def test_infeasible_srandom_returns_status_two(capsys):
    assert _run(["cost", "srandom", "--length", "32", "--spread", "20"]) == 2
    assert "error:" in capsys.readouterr().err
