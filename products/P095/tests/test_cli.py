"""The command line, in process and as a subprocess.

``--help`` is checked in a genuinely clean subprocess, because that is the
check the release gate runs and the one an earlier product of this mission
shipped broken.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from boardprobe.__main__ import build_parser, fixture_path, main

ROOT = Path(__file__).resolve().parent.parent


def clean_environment() -> dict[str, str]:
    """An environment whose HOME and caches are outside the repository."""
    scratch = Path(tempfile.mkdtemp(prefix="boardprobe-test-"))
    environment = dict(os.environ)
    environment.update(
        {
            "HOME": str(scratch),
            "MPLCONFIGDIR": str(scratch / "mpl"),
            "MPLBACKEND": "Agg",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT / "src"),
        }
    )
    return environment


def test_help_exits_zero_in_a_clean_subprocess():
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "boardprobe", "--help"],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
        env=clean_environment(),
    )
    assert completed.returncode == 0
    assert "usage: python -m boardprobe" in completed.stdout
    assert "hardware-pending" in completed.stdout


def test_version_exits_zero_in_a_clean_subprocess():
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "boardprobe", "--version"],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
        env=clean_environment(),
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "boardprobe 0.1.0"


@pytest.mark.parametrize(
    "argv",
    [
        ["channels"],
        ["channels", "--references"],
        ["provenance"],
        ["attest"],
        ["sample", "--backend", "simulated", "--seed", "3"],
        ["sample", "--backend", "recorded"],
        ["sample", "--backend", "live"],
        ["replay", "--samples", "6"],
        ["benchmark", "--repeats", "8", "--overhead-repeats", "20"],
        ["benchmark", "--repeats", "8", "--environment", "none",
         "--overhead-repeats", "20"],
        ["dry-run"],
    ],
)
def test_every_subcommand_exits_zero(argv, capsys):
    assert main(argv) == 0
    captured = capsys.readouterr().out
    assert captured.strip()


def test_a_live_sample_serves_the_clock_and_reports_the_refusals(capsys):
    assert main(["sample", "--backend", "live"]) == 0
    out = capsys.readouterr().out
    assert "monotonic_clock" in out
    assert "channels refused" in out
    assert "MEASURED OFF-TARGET" in out


def test_a_live_environment_benchmark_fails_cleanly_with_exit_one(capsys):
    assert main(["benchmark", "--repeats", "4", "--environment", "live"]) == 1
    error = capsys.readouterr().err
    assert "could not run" in error
    assert "not reachable" in error


def test_the_provenance_report_prints_the_lattice_and_the_refusal_rule(capsys):
    assert main(["provenance"]) == 0
    out = capsys.readouterr().out
    assert "meet is the greatest lower bound" in out
    assert "MEASURED" in out
    assert "REFUSED, not taken as the identity" in out
    assert "No figure in this repository was measured" in out
    assert "Level 3, hardware-pending" in out


def test_the_channel_table_says_zero_channels_are_verified(capsys):
    assert main(["channels"]) == 0
    out = capsys.readouterr().out
    assert "channels verified on hardware: 0 of 14" in out
    assert "UNVERIFIED ON HARDWARE" in out


def test_the_dry_run_subcommand_reports_zero_device_accesses(capsys):
    assert main(["dry-run"]) == 0
    out = capsys.readouterr().out
    assert "device accesses against the real root : 0" in out
    assert "side-effect free                      : yes" in out


def test_a_bad_argument_exits_two():
    with pytest.raises(SystemExit) as error:
        main(["sample", "--backend", "psychic"])
    assert error.value.code == 2


def test_no_subcommand_exits_two():
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


def test_a_missing_trace_file_exits_one(capsys, tmp_path):
    assert main(["replay", "--trace", str(tmp_path / "absent.json")]) == 1
    assert "absent.json" in capsys.readouterr().err


def test_a_truncated_trace_file_exits_one(capsys, tmp_path):
    path = tmp_path / "truncated.json"
    path.write_text('{"schema": "boardprobe/trace/1", "board_mo')
    assert main(["replay", "--trace", str(path)]) == 1
    assert "not valid JSON" in capsys.readouterr().err


def test_fixture_path_resolves_relative_to_the_package():
    path = fixture_path("orin_nano_idle_simulated_trace.json")
    assert path.is_file()
    assert path.parent.name == "fixtures"


def test_a_missing_fixture_is_reported_with_a_relative_path():
    from boardprobe.trace import TraceError

    with pytest.raises(TraceError, match="fixtures/not_a_fixture.json"):
        fixture_path("not_a_fixture.json")


def test_the_parser_declares_every_documented_subcommand():
    parser = build_parser()
    actions = [
        action for action in parser._subparsers._group_actions  # noqa: SLF001
    ]
    names = set(actions[0].choices)
    assert names == {
        "channels", "sample", "benchmark", "replay", "provenance", "dry-run",
        "attest",
    }
