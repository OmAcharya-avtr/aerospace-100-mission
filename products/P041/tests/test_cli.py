"""CLI: every subcommand exits 0 in a clean subprocess and prints no absolute path."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")

COMMANDS: list[list[str]] = [
    ["--help"],
    ["channel", "--samples", "20000"],
    ["channel", "--samples", "20000", "--marginal", "gammagamma"],
    ["channel", "--samples", "20000", "--kernel", "gauss"],
    ["fade", "--samples", "60000"],
    ["fade", "--samples", "60000", "--kernel", "gauss"],
    ["code"],
    ["depth", "--codewords", "128", "--depths", "1,64"],
    ["preflight", "--depth", "256"],
    ["dryrun", "--depth", "256", "--blocks", "1"],
]


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "-m", "codedfade", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )


@pytest.mark.parametrize("args", COMMANDS, ids=[" ".join(a) for a in COMMANDS])
def test_command_exits_zero(args: list[str]) -> None:
    result = _run(args)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


@pytest.mark.parametrize("args", COMMANDS, ids=[" ".join(a) for a in COMMANDS])
def test_command_prints_no_absolute_path(args: list[str]) -> None:
    result = _run(args)
    assert "/home/" not in result.stdout
    assert "/Users/" not in result.stdout


def test_preflight_exits_nonzero_when_a_check_fails() -> None:
    result = _run(["preflight", "--depth", "2", "--tau", "2e-4"])
    assert result.returncode == 1
    assert "FAIL" in result.stdout


def test_unknown_command_exits_nonzero() -> None:
    assert _run(["nonsense"]).returncode != 0
