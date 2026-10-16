"""The command line interface, exercised as real subprocesses.

Exit codes are a documented contract, so they are measured by running the
program rather than by calling ``main`` in process.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from conftest import MINIMAL, PLANTED, ROOT, WORKED

ENV_SRC = str(ROOT / "src")


def run(*args: str) -> subprocess.CompletedProcess[str]:
    """Run ``python -m watchdogspec`` with ``src`` on the path."""
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = ENV_SRC + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "watchdogspec", *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        check=False,
    )


def test_check_on_a_consistent_spec_exits_zero():
    result = run("check", str(MINIMAL))
    assert result.returncode == 0
    assert "CONSISTENT" in result.stdout


def test_check_on_the_planted_fixture_exits_one():
    result = run("check", str(PLANTED))
    assert result.returncode == 1
    assert "7 error, 5 warning" in result.stdout


def test_check_on_the_worked_example_exits_one():
    result = run("check", str(WORKED))
    assert result.returncode == 1
    assert "1 error, 1 warning" in result.stdout


def test_strict_makes_warnings_blocking():
    # The minimal spec has no finding at all, so --strict changes nothing
    # there; a warning-only spec is what --strict is for.
    assert run("check", str(MINIMAL), "--strict").returncode == 0
    assert run("check", str(WORKED), "--strict").returncode == 1


def test_json_output_parses():
    result = run("check", str(WORKED), "--format", "json")
    payload = json.loads(result.stdout)
    assert payload["exit_code"] == 1
    assert result.returncode == 1


def test_markdown_output_has_tables():
    result = run("check", str(WORKED), "--format", "markdown")
    assert "| task |" in result.stdout


def test_tolerance_option_is_honoured():
    # TASK-FDIR overruns its deadline by 578.5 ms. A comparison tolerance of
    # 1000 ms puts that slack inside the undecided band, so the WS003 error
    # becomes a WS004 warning and the exit code falls to 0. That is not a
    # recommendation to use such a tolerance; it is how the option behaves.
    wide = run("check", str(WORKED), "--tol-ms", "1000")
    assert wide.returncode == 0
    assert "WS004" in wide.stdout
    assert run("check", str(WORKED), "--tol-ms", "1e-9").returncode == 1


def test_latency_subcommand_prints_the_table_and_exits_zero():
    result = run("latency", str(WORKED))
    assert result.returncode == 0
    assert "WCDL" in result.stdout
    assert "TASK-FDIR" in result.stdout


def test_simulate_subcommand_agrees_with_the_closed_form():
    result = run("simulate", str(MINIMAL), "--seeds", "2", "--trials", "200")
    assert result.returncode == 0, result.stderr
    assert "bound violations        : 0" in result.stdout
    assert "closed form WCDL" in result.stdout


def test_simulate_subcommand_accepts_one_task():
    result = run("simulate", str(MINIMAL), "--task", "TASK-HK", "--seeds", "1", "--trials", "100")
    assert result.returncode == 0
    assert "TASK-CTRL" not in result.stdout


def test_simulate_subcommand_rejects_an_unknown_task():
    result = run("simulate", str(MINIMAL), "--task", "absent")
    assert result.returncode == 2
    assert "no timed task" in result.stderr


def test_simulate_subcommand_rejects_zero_seeds():
    result = run("simulate", str(MINIMAL), "--seeds", "0")
    assert result.returncode == 2


def test_mermaid_subcommand_emits_a_flowchart():
    result = run("mermaid", str(WORKED))
    assert result.returncode == 0
    assert result.stdout.startswith("flowchart LR")
    assert "TASK-GNC" in result.stdout


def test_codes_subcommand_lists_the_catalogue():
    result = run("codes")
    assert result.returncode == 0
    assert result.stdout.count("WS0") == 12


def test_missing_file_exits_two():
    result = run("check", "does-not-exist.yaml")
    assert result.returncode == 2
    assert "no such specification file" in result.stderr


def test_malformed_file_exits_two(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("spec_version: 1\n", encoding="utf-8")
    result = run("check", str(bad))
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_no_subcommand_exits_two():
    assert run().returncode == 2


@pytest.mark.parametrize("command", ["check", "latency", "simulate", "mermaid", "codes"])
def test_every_subcommand_has_help(command):
    result = run(command, "--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout
