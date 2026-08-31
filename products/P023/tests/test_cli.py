"""CLI tests, in-process and through a subprocess."""

import subprocess
import sys

import pytest

from alloclab.cli import _build, main


def test_config_subcommand_prints_the_matrix(capsys):
    assert main(["config", "--config", "pyramid"]) == 0
    out = capsys.readouterr().out
    assert "EffectorSet: 4 effectors" in out
    assert "rw1" in out


def test_ams_subcommand_prints_matching_volumes(capsys):
    assert main(["ams", "--config", "triad"]) == 0
    out = capsys.readouterr().out
    assert "vertices" in out
    assert "8" in out
    assert "hull volume" in out
    assert "closed-form volume" in out


def test_ams_bruteforce_matches_pairwise(capsys):
    main(["ams", "--config", "pyramid", "--method", "pairwise"])
    a = capsys.readouterr().out
    main(["ams", "--config", "pyramid", "--method", "bruteforce"])
    b = capsys.readouterr().out
    assert a.split("method=")[1].split("\n")[0] == "pairwise"
    assert [ln.split(":")[1] for ln in a.splitlines()[1:3]] == [
        ln.split(":")[1] for ln in b.splitlines()[1:3]
    ]


def test_allocate_subcommand_succeeds_for_a_feasible_command(capsys):
    code = main(["allocate", "--torque", "0.1", "0.0", "0.0", "--method", "qp"])
    out = capsys.readouterr().out
    assert code == 0
    assert "status" in out
    assert "exact" in out or "saturated" in out


def test_allocate_subcommand_returns_one_for_an_infeasible_command(capsys):
    code = main(["allocate", "--torque", "50", "0", "0", "--method", "qp"])
    out = capsys.readouterr().out
    assert code == 1
    assert "infeasible" in out


def test_allocate_with_failures_reports_the_degraded_set(capsys):
    code = main(["allocate", "--torque", "0.02", "0.0", "0.0", "--failed", "0", "--method", "qp"])
    out = capsys.readouterr().out
    assert code == 0
    assert "failed=[0]" in out
    assert "attainable by degraded set : True" in out
    assert "AMS volume ratio" in out


@pytest.mark.parametrize("method", ["pinv", "wpinv", "rpi", "lp", "qp"])
def test_every_method_runs_from_the_cli(capsys, method):
    main(["allocate", "--torque", "0.05", "0.01", "-0.02", "--method", method])
    out = capsys.readouterr().out
    assert "residual norm" in out


def test_unknown_config_is_rejected():
    with pytest.raises(ValueError, match="unknown config"):
        _build("orbit")


def test_missing_subcommand_exits_with_usage_error():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_module_entry_point_runs_in_a_subprocess():
    """Uses the PYTHONPATH exported by the repository-root conftest.py."""
    proc = subprocess.run(
        [sys.executable, "-m", "alloclab", "ams", "--config", "triad"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "hull volume" in proc.stdout


def test_module_entry_point_propagates_the_infeasible_exit_code():
    proc = subprocess.run(
        [sys.executable, "-m", "alloclab", "allocate", "--torque", "99", "0", "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "infeasible" in proc.stdout
