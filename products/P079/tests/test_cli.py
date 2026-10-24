"""The command-line interface. Every command must exit zero and print numbers."""

from __future__ import annotations

from pathlib import Path

import pytest

from timetransfer.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    out = capsys.readouterr().out
    assert "timetransfer 0.1.0" in out
    assert "Level 3, hardware-pending" in out


def test_info_names_the_absent_packages(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "allantools" in out
    assert "ntplib" in out
    assert "not a time-sync daemon" in out


def test_simulate(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["simulate", "--n", "60", "--asymmetry", "8e-4"]) == 0
    out = capsys.readouterr().out
    assert "offset uncertainty budget" in out
    assert "least squares (baseline 2a)" in out
    assert "kalman (baseline 2b)" in out
    assert "declared static bias" in out
    assert "WARNING" in out  # no asymmetry term declared


def test_simulate_with_a_declared_asymmetry_bound(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["simulate", "--n", "60", "--asymmetry-bound", "1.6e-3"]) == 0
    out = capsys.readouterr().out
    assert "declared_bound" in out
    assert "WARNING" not in out


def test_allan(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["allan", "--n", "400", "--noise-kind", "white_fm"]) == 0
    out = capsys.readouterr().out
    assert "noise kind white_fm (declared)" in out
    assert "adev" in out


def test_parity(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["parity", "--n", "80"]) == 0
    out = capsys.readouterr().out
    assert "timestamp bytes   : equal" in out
    assert "offset max |diff| : 0.000e+00 s" in out
    assert "is_hardware=False" in out


def test_preflight(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["preflight", "--n", "40"]) == 0
    out = capsys.readouterr().out
    assert "10/10 passed" in out


def test_record_then_dryrun(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "rec.json"
    assert main(["record", "--n", "80", "--loss", "0.05", str(target)]) == 0
    assert target.exists()
    capsys.readouterr()
    assert main(["dryrun", str(target)]) == 0
    out = capsys.readouterr().out
    assert "hardware operations  : 0 (must be 0)" in out
    assert "passed     : True" in out


def test_bench_leaves_the_hardware_rows_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "bench",
                "--n",
                "60",
                "--repeats",
                "2",
                "--note",
                "unit test; machine not quiesced",
                "--json",
                str(tmp_path / "b.json"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "LEVEL 4 HARDWARE TABLE -- EMPTY AND UNMEASURED" in out
    assert "UNSTABLE" in out
    assert (tmp_path / "b.json").exists()


def test_a_library_error_becomes_exit_status_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # min_exchanges is 8 inside the CLI, so four attempts under a total outage
    # cannot satisfy it. A loss probability just under one is the reachable
    # equivalent through the command line.
    assert main(["simulate", "--n", "10", "--loss", "0.999"]) == 2
    assert "InsufficientDataError" in capsys.readouterr().err


def test_an_unknown_command_exits_non_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["nonsense"])
    assert excinfo.value.code != 0


def test_a_missing_recording_is_a_user_error_not_a_crash(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["dryrun", "fixtures/does-not-exist.json"]) == 2
    assert "FileNotFoundError" in capsys.readouterr().err
