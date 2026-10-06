"""CLI tests. The ``--help`` case runs in a clean subprocess on purpose."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from linkoutage.__main__ import build_parser, main

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "linkoutage", *args],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin", "MPLBACKEND": "Agg"},
        check=False,
    )


def test_help_exits_zero_in_a_clean_subprocess():
    result = _run(["--help"])
    assert result.returncode == 0
    assert "linkoutage" in result.stdout


def test_version_exits_zero_in_a_clean_subprocess():
    result = _run(["--version"])
    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_no_subcommand_is_an_error():
    result = _run([])
    assert result.returncode != 0


def test_stats_subcommand_runs(capsys):
    assert main(["stats", "--n-samples", "20000"]) == 0
    out = capsys.readouterr().out
    assert "level-crossing rate" in out
    assert "a[n] < T (strict)" in out


def test_stats_honours_definition_flags(capsys):
    assert main(["stats", "--n-samples", "20000", "--duration-convention", "interval_count"]) == 0
    assert "duration = (L - 1) / fs" in capsys.readouterr().out


def test_stats_honours_non_strict_flag(capsys):
    assert main(["stats", "--n-samples", "20000", "--non-strict"]) == 0
    assert "a[n] <= T (non-strict)" in capsys.readouterr().out


def test_fit_subcommand_runs(capsys):
    assert main(["fit", "--n-samples", "60000"]) == 0
    out = capsys.readouterr().out
    assert "Memoryless" in out
    assert "transition matrix" in out
    assert "Semi-Markov fit" in out
    assert "Markov-order" in out


def test_predict_subcommand_runs(capsys):
    assert (
        main(
            [
                "predict",
                "--n-samples",
                "400000",
                "--trees",
                "20",
                "--window",
                "200",
                "--horizon",
                "200",
                "--stride",
                "100",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "constant base rate" in out
    assert "analytic LCR" in out
    assert "logistic regression" in out
    assert "random forest" in out
    assert "Lowest Brier score" in out
    assert "Accuracy is not reported as a headline" in out


def test_stats_reads_a_numpy_file(tmp_path, capsys):
    path = tmp_path / "a.npy"
    np.save(path, np.array([1.0, 0.5, 0.5, 1.0, 1.0, 0.4, 1.0]))
    assert main(["stats", "--input", str(path), "--fs", "1.0", "--threshold", "0.6"]) == 0
    out = capsys.readouterr().out
    assert "down-crossings         : 2" in out
    assert "mean fade duration     : 1.5 s" in out


def test_stats_reads_a_text_file(tmp_path, capsys):
    path = tmp_path / "a.csv"
    path.write_text("1.0\n0.5\n0.5\n1.0\n1.0\n0.4\n1.0\n")
    assert main(["stats", "--input", str(path), "--fs", "1.0", "--threshold", "0.6"]) == 0
    assert "down-crossings         : 2" in capsys.readouterr().out


def test_stats_rejects_a_one_sample_file(tmp_path):
    path = tmp_path / "a.npy"
    np.save(path, np.array([1.0]))
    with pytest.raises(SystemExit, match="at least 2 samples"):
        main(["stats", "--input", str(path)])


def test_invalid_parameter_returns_exit_code_two(capsys):
    assert main(["stats", "--n-samples", "20000", "--si", "-1"]) == 2
    assert "error:" in capsys.readouterr().err


def test_parser_lists_every_subcommand():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions
    assert set(actions[0].choices) == {"stats", "fit", "crosscheck", "predict"}
