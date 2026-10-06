"""The command-line interface, exercised in clean subprocesses."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``python -m arqlonghaul`` with only ``src`` on the path."""
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "-m", "arqlonghaul", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )


def test_help_exits_zero() -> None:
    out = run(["--help"])
    assert out.returncode == 0
    assert "arqlonghaul" in out.stdout
    for cmd in ("link", "closedform", "window", "simulate", "harq", "crc"):
        assert cmd in out.stdout


def test_version_exits_zero() -> None:
    out = run(["--version"])
    assert out.returncode == 0
    assert "0.1.0" in out.stdout


def test_no_subcommand_is_an_error() -> None:
    out = run([])
    assert out.returncode != 0


@pytest.mark.parametrize("sub", ["link", "closedform", "window"])
def test_link_subcommands_on_a_preset(sub: str) -> None:
    out = run([sub, "--preset", "geo"])
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip()


def test_link_requires_a_preset_or_explicit_values() -> None:
    out = run(["link"])
    assert out.returncode == 2
    assert "preset" in out.stderr


def test_link_with_explicit_values() -> None:
    out = run(["link", "--rate-bps", "1e6", "--rtt-s", "2.0", "--frame-bits", "1000"])
    assert out.returncode == 0
    assert "N = 1 + RTT/T_f" in out.stdout


def test_closedform_omits_gbn_below_the_knee() -> None:
    out = run(["closedform", "--preset", "geo", "--window", "5"])
    assert out.returncode == 0
    assert "go_back_n omitted" in out.stderr
    assert "stop_and_wait" in out.stdout


def test_simulate_runs_the_three_protocols() -> None:
    out = run(
        ["simulate", "--n", "20", "--window", "20", "--fer", "0.05", "--slots", "20000"]
    )
    assert out.returncode == 0, out.stderr
    for name in ("stop_and_wait", "go_back_n", "selective_repeat"):
        assert name in out.stdout
    assert "independent" in out.stdout


def test_simulate_with_a_bursty_channel() -> None:
    out = run(
        [
            "simulate", "--n", "20", "--window", "20", "--fer", "0.05",
            "--burst", "25", "--slots", "20000",
        ]
    )
    assert out.returncode == 0, out.stderr
    assert "gilbert" in out.stdout.lower() or "GE " in out.stdout


def test_harq_with_crossover() -> None:
    out = run(["harq", "--crossover", "--rtt-symbols", "500"])
    assert out.returncode == 0, out.stderr
    assert "crossover D" in out.stdout
    assert "optimal first rate" in out.stdout


def test_harq_rejects_an_impossible_configuration() -> None:
    out = run(["harq", "--k", "200", "--n1", "100"])
    assert out.returncode == 2
    assert "error" in out.stderr


def test_crc_prints_the_catalogue_check_value() -> None:
    out = run(["crc", "--name", "crc-32"])
    assert out.returncode == 0
    assert "0xcbf43926" in out.stdout


def test_crc_unknown_name_exits_two() -> None:
    out = run(["crc", "--name", "crc-nonexistent"])
    assert out.returncode == 2
    assert "unknown CRC" in out.stderr


def test_no_absolute_paths_in_any_output() -> None:
    for args in (
        ["link", "--preset", "lunar"],
        ["window", "--preset", "geo"],
        ["harq"],
        ["crc"],
    ):
        out = run(args)
        combined = out.stdout + out.stderr
        assert "/home/" not in combined
        assert "/Users/" not in combined
