"""The command-line interface, including exit statuses and a clean subprocess."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from simplexguard.__main__ import main

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_help_exits_zero_in_a_clean_subprocess():
    result = subprocess.run(
        [sys.executable, "-m", "simplexguard", "--help"],
        capture_output=True,
        text=True,
        env=_env(),
        check=False,
    )
    assert result.returncode == 0
    assert "runtime-assurance" in result.stdout.lower()
    assert "not flight-qualified" in result.stdout


def test_no_subcommand_is_an_error():
    result = subprocess.run(
        [sys.executable, "-m", "simplexguard"],
        capture_output=True,
        text=True,
        env=_env(),
        check=False,
    )
    assert result.returncode == 2


def test_plant_subcommand(capsys):
    assert main(["plant"]) == 0
    out = capsys.readouterr().out
    assert "declared W (box)" in out
    assert "sample interval     0.05 s" in out


def test_invariant_subcommand(capsys):
    assert main(["invariant"]) == 0
    out = capsys.readouterr().out
    assert "converged                 True" in out
    assert "robustly_invariant" in out


def test_guard_subcommand(capsys):
    assert main(["guard"]) == 0
    assert "Chebyshev radius of S-W" in capsys.readouterr().out


def test_run_subcommand(capsys):
    assert main(["run", "--steps", "300"]) == 0
    out = capsys.readouterr().out
    for name in ("guarded", "unguarded", "baseline"):
        assert name in out


def test_accounting_subcommand(capsys):
    assert main(["accounting", "--steps", "400"]) == 0
    out = capsys.readouterr().out
    assert "1  switch rate" in out
    assert "6  invariant-set exits" in out


def test_bound_sweep_subcommand(capsys):
    assert main(["bound-sweep", "--episodes", "2", "--steps", "300",
                 "--scales", "1.0", "4.0"]) == 0
    out = capsys.readouterr().out
    assert "rho= 1.000" in out
    assert "invariance certificate was lost" in out


def test_predict_subcommand(capsys):
    assert main(["predict", "--episodes", "6", "--steps", "250", "--lead", "4",
                 "--trees", "30"]) == 0
    out = capsys.readouterr().out
    assert "exact one-step guard condition" in out
    assert "learned forest" in out
    assert "learned lead times" in out


def test_empty_invariant_set_exits_three(capsys):
    assert main(["invariant", "--baseline-r", "400"]) == 3
    assert "robust invariant set emptied" in capsys.readouterr().err


def test_bad_input_exits_two(capsys):
    assert main(["plant", "--dt", "-1"]) == 2
    assert "error:" in capsys.readouterr().err


def test_recursion_cap_exits_three(capsys):
    assert main(["invariant", "--max-iterations", "2"]) == 3
    assert "did not converge" in capsys.readouterr().err


def test_vertex_sampler_through_the_cli(capsys):
    assert main(["accounting", "--steps", "300", "--disturbance-mode", "vertex"]) == 0
    assert "constraint violations" in capsys.readouterr().out


def test_minimum_dwell_through_the_cli(capsys):
    assert main(["accounting", "--steps", "300", "--min-baseline-dwell", "5"]) == 0
    out = capsys.readouterr().out
    assert "steps held by min-dwell" in out


@pytest.mark.parametrize("argv", [["plant", "--angle-limit", "0"], ["run", "--steps", "0"]])
def test_argument_validation_exits_two(argv, capsys):
    assert main(argv) == 2
    assert "error:" in capsys.readouterr().err
