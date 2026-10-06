"""The CLI, exercised in-process and as a clean subprocess."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from coderateopt.__main__ import build_parser, main

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "coderateopt", *args],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin", "MPLBACKEND": "Agg"},
        check=False,
        timeout=180,
    )


def test_help_exits_zero_in_a_clean_subprocess():
    result = _run(["--help"])
    assert result.returncode == 0
    assert "coderateopt" in result.stdout


def test_version_exits_zero():
    result = _run(["--version"])
    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_solve_prints_the_mix(capsys):
    assert main(["solve", "--max-entries", "2", "--mode", "long_run"]) == 0
    out = capsys.readouterr().out
    assert "expected goodput" in out
    assert "worst-interval availability" in out
    assert "ook-" in out


def test_solve_honours_the_method_flag(capsys):
    assert main(["solve", "--method", "milp"]) == 0
    assert "canonical=False" in capsys.readouterr().out
    assert main(["solve", "--method", "exhaustive"]) == 0
    assert "canonical=True" in capsys.readouterr().out


def test_sensitivity_reports_the_knife_edge_verdict(capsys):
    assert main(["sensitivity", "--parameter", "scintillation_index"]) == 0
    out = capsys.readouterr().out
    assert "stable over" in out
    assert "knife-edge" in out


@pytest.mark.parametrize("parameter", ["scintillation_index", "margin_db", "availability_target"])
def test_sensitivity_accepts_every_parameter(parameter, capsys):
    assert main(["sensitivity", "--parameter", parameter]) == 0
    assert parameter in capsys.readouterr().out


def test_compare_prints_both_heuristics(capsys):
    assert main(["compare"]) == 0
    out = capsys.readouterr().out
    assert "highest_feasible_rate" in out
    assert "highest_rate_within_reserve" in out


def test_table_prints_every_entry(capsys):
    assert main(["table"]) == 0
    out = capsys.readouterr().out
    assert out.count("ook-") + out.count("qpsk-") + out.count("qam16-") >= 9


def test_fade_prints_quantiles_for_both_models(capsys):
    assert main(["fade"]) == 0
    assert "sigma_lnI" in capsys.readouterr().out
    assert main(["fade", "--fade-model", "gamma-gamma"]) == 0
    assert "alpha" in capsys.readouterr().out


def test_infeasible_instance_exits_three(capsys):
    assert main(["solve", "--margin-db", "0", "--target", "0.99999"]) == 3
    assert "infeasible" in capsys.readouterr().err


def test_bad_input_exits_two(capsys):
    assert main(["solve", "--target", "1.5"]) == 2
    assert "error" in capsys.readouterr().err
    assert main(["solve", "--scintillation-index", "0"]) == 2
    assert "error" in capsys.readouterr().err


def test_modcod_csv_is_read(tmp_path, capsys):
    path = tmp_path / "table.csv"
    path.write_text(
        "name,rate_bits_per_symbol,threshold_db\nslow,1.0,3.0\nfast,2.0,7.0\n",
        encoding="utf-8",
    )
    assert main(["solve", "--modcod-csv", str(path), "--target", "0.9"]) == 0
    out = capsys.readouterr().out
    assert "slow" in out or "fast" in out


def test_empty_modcod_csv_is_rejected(tmp_path, capsys):
    path = tmp_path / "empty.csv"
    path.write_text("name,rate_bits_per_symbol,threshold_db\n", encoding="utf-8")
    assert main(["solve", "--modcod-csv", str(path)]) == 2
    assert "no MODCOD rows" in capsys.readouterr().err


def test_missing_modcod_csv_is_reported(capsys):
    assert main(["solve", "--modcod-csv", "no_such_file.csv"]) == 2
    assert "error" in capsys.readouterr().err


def test_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
