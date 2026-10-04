"""The command-line interface. ``--help`` must exit 0 on every subcommand."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hilforge.cli import build_parser, main

SRC = str(Path(__file__).resolve().parents[1] / "src")
SUBCOMMANDS = (
    "info",
    "preflight",
    "run",
    "dryrun",
    "parity",
    "replay",
    "bench",
    "overruns",
    "predict",
)


def _module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hilforge", *args],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin", "MPLBACKEND": "Agg"},
        timeout=300,
        check=False,
    )


def test_module_help_exits_zero():
    result = _module("--help")
    assert result.returncode == 0
    assert "hardware-in-the-loop" in result.stdout.lower()


def test_module_version_exits_zero():
    result = _module("--version")
    assert result.returncode == 0
    assert "hilforge 0.1.0" in result.stdout


@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_every_subcommand_help_exits_zero(command):
    result = _module(command, "--help")
    assert result.returncode == 0, result.stderr
    assert result.stdout


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_info_reports_the_validation_level(capsys):
    assert main(["info", "--clock-samples", "200"]) == 0
    out = capsys.readouterr().out
    assert "3, hardware-pending" in out
    assert "is_hardware=False" in out
    assert "no number here is a hardware measurement" in out


@pytest.mark.parametrize("backend", ["simulated", "device"])
def test_preflight_is_go_for_both_backends(capsys, backend):
    assert main(["preflight", "--backend", backend]) == 0
    assert "GO" in capsys.readouterr().out


def test_preflight_is_no_go_for_an_absent_device(capsys):
    assert main(["preflight", "--backend", "absent"]) == 2
    assert "NO-GO" in capsys.readouterr().out


def test_run_prints_the_summary_and_digest(capsys):
    assert main(["run", "--iterations", "100", "--inject", "ramp"]) == 0
    out = capsys.readouterr().out
    assert "data_digest" in out
    assert "full_digest" in out
    assert "cascade_overruns" in out


def test_run_with_cascade_limit_aborts_cleanly():
    # A ramp ending at 1.4 T trips a three-long cascade.
    from hilforge.errors import OverrunCascadeError

    with pytest.raises(OverrunCascadeError):
        main(["run", "--iterations", "400", "--inject", "ramp", "--cascade-limit", "3"])


def test_dryrun_returns_zero_and_reports_no_write(capsys):
    assert main(["dryrun", "--iterations", "100"]) == 0
    out = capsys.readouterr().out
    assert "no write reached hardware : True" in out
    assert "writes issued             : 0" in out


def test_parity_returns_zero_and_reports_identical_bytes(capsys):
    assert main(["parity", "--iterations", "150"]) == 0
    out = capsys.readouterr().out
    assert "signal bytes identical: True" in out
    assert "loopback stub, not hardware" in out


def test_replay_returns_zero(capsys):
    assert main(["replay", "--iterations", "120"]) == 0
    assert "identical            : True" in capsys.readouterr().out


def test_bench_writes_both_record_files(tmp_path, capsys):
    stem = tmp_path / "record"
    assert main(["bench", "--iterations", "200", "--note", "pytest", "--out", str(stem)]) == 0
    out = capsys.readouterr().out
    assert "Level 4 validation is NOT claimed" in out
    assert stem.with_suffix(".txt").exists()
    payload = json.loads(stem.with_suffix(".json").read_text())
    assert payload["is_hardware"] is False


def test_overruns_text_and_json(tmp_path, capsys):
    assert main(["overruns", "--iterations", "500", "--preset", "bursty"]) == 0
    text = capsys.readouterr().out
    assert "cascade overruns" in text
    assert main(["overruns", "--iterations", "500", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "direct_overrun_count" in payload


def test_overruns_reads_a_json_trace(tmp_path, capsys):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"trace": [0.004, 0.012, 0.003, 0.011]}))
    assert main(["overruns", "--trace", str(path), "--period", "0.010"]) == 0
    out = capsys.readouterr().out
    assert "direct overruns          : 2" in out


def test_overruns_reads_an_npy_trace(tmp_path, capsys):
    import numpy as np

    path = tmp_path / "trace.npy"
    np.save(path, np.array([0.004, 0.012, 0.003, 0.011]))
    assert main(["overruns", "--trace", str(path), "--period", "0.010"]) == 0
    assert "direct overruns          : 2" in capsys.readouterr().out


def test_predict_prints_all_three_predictors(capsys):
    assert main(["predict", "--iterations", "6000", "--preset", "bursty"]) == 0
    out = capsys.readouterr().out
    for name in ("fixed_threshold", "queueing_markov", "learned_hgb"):
        assert name in out
    assert "matched flag rate" in out


def test_predict_f1_operating_point(capsys):
    assert main(["predict", "--iterations", "6000", "--flag-rate", "0.0"]) == 0
    assert "F1-maximising" in capsys.readouterr().out
