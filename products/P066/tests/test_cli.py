"""CLI behaviour, including exit status, exercised as real subprocesses."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "examples" / "cases"


def run(*args: str) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "HOME": str(ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "budgetroll", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        timeout=120,
    )


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("clean_mass.yaml", 0),
        ("cubesat_link.yaml", 0),
        ("cubesat_power.yaml", 1),
        ("planted_findings.yaml", 1),
    ],
)
def test_check_exit_status(case, expected):
    result = run("check", str(CASES / case))
    assert result.returncode == expected, result.stderr


@pytest.mark.parametrize(
    "command", ["check", "rollup", "sensitivity", "compare"]
)
def test_every_command_runs_on_the_clean_case(command):
    result = run(command, str(CASES / "clean_mass.yaml"))
    assert result.returncode == 0, result.stderr
    assert "not flight-qualified" in result.stdout


@pytest.mark.parametrize("fmt", ["text", "json", "markdown", "full"])
def test_rollup_formats(fmt):
    result = run("rollup", "--format", fmt, str(CASES / "clean_mass.yaml"))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_rollup_json_is_parseable():
    result = run("rollup", "--format", "json", str(CASES / "cubesat_link.yaml"))
    payload = json.loads(result.stdout)
    assert payload["budget"]["id"] == "cubesat_sband_downlink"


def test_codes_lists_ten_codes():
    result = run("codes")
    assert result.returncode == 0
    assert sum(1 for line in result.stdout.splitlines() if "  BR0" in line) == 10


def test_missing_file_exits_two():
    result = run("check", str(CASES / "does_not_exist.yaml"))
    assert result.returncode == 2
    assert "no such budget declaration" in result.stderr


def test_malformed_file_exits_two(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("policies: [\n", encoding="utf-8")
    result = run("check", str(bad))
    assert result.returncode == 2
    assert "not valid YAML" in result.stderr


def test_no_subcommand_exits_two():
    result = run()
    assert result.returncode == 2


def test_unknown_subcommand_exits_two():
    result = run("explode", str(CASES / "clean_mass.yaml"))
    assert result.returncode == 2


def test_sensitivity_step_is_honoured():
    small = run("sensitivity", "--step", "0.01", str(CASES / "cubesat_power.yaml"))
    large = run("sensitivity", "--step", "0.20", str(CASES / "cubesat_power.yaml"))
    assert small.returncode == 1
    assert large.returncode == 1
    assert small.stdout != large.stdout


def test_compare_prints_four_rows():
    result = run("compare", str(CASES / "cubesat_power.yaml"))
    assert result.returncode == 1
    for label in ("linear", "root-sum-square", "declared hierarchy", "exact form"):
        assert label in result.stdout


def test_help_exits_zero():
    result = run("--help")
    assert result.returncode == 0
    assert "budgetroll" in result.stdout
