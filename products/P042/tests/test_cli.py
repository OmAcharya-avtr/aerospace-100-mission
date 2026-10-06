"""Tests for ``python -m acmpilot``.

The ``--help`` and ``--version`` paths run in a clean subprocess with no
PYTHONPATH beyond ``src``, because the release gate invokes them that way and a
packaging mistake that only shows up out of process is exactly what this catches.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from acmpilot.__main__ import build_parser, main

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "acmpilot", *args],
        cwd=REPO,
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin", "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


class TestSubprocess:
    def test_help_exits_zero(self):
        result = _run("--help")
        assert result.returncode == 0
        assert "acmpilot" in result.stdout
        assert "feedback" in result.stdout

    def test_version_exits_zero(self):
        result = _run("--version")
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_no_subcommand_fails_with_usage(self):
        result = _run()
        assert result.returncode != 0
        assert "usage" in (result.stderr + result.stdout).lower()

    def test_unknown_subcommand_fails(self):
        result = _run("nonsense")
        assert result.returncode != 0

    @pytest.mark.parametrize(
        "command", ["thresholds", "simulate", "sweep", "predict"]
    )
    def test_each_subcommand_has_help(self, command):
        result = _run(command, "--help")
        assert result.returncode == 0

    def test_help_output_contains_no_absolute_paths(self):
        result = _run("--help")
        assert "/home/" not in result.stdout
        assert "/Users/" not in result.stdout


class TestParser:
    def test_version_action_present(self):
        parser = build_parser()
        actions = {a.dest for a in parser._actions}
        assert "version" in actions

    def test_subcommands_registered(self):
        parser = build_parser()
        sub = next(
            a for a in parser._actions if a.dest == "command"
        )
        assert set(sub.choices) == {"thresholds", "simulate", "sweep", "predict"}

    def test_simulate_defaults(self):
        args = build_parser().parse_args(["simulate"])
        assert args.tau_ms == 10.0
        assert args.tau_c_ms == 10.0
        assert args.marginal == "lognormal"
        assert args.n_slots == 20000

    def test_sweep_accepts_a_tau_list(self):
        args = build_parser().parse_args(["sweep", "--tau-ms-list", "0", "5", "20"])
        assert args.tau_ms_list == [0.0, 5.0, 20.0]

    def test_marginal_choice_enforced(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["simulate", "--marginal", "rayleigh"])

    def test_predict_defaults(self):
        args = build_parser().parse_args(["predict"])
        assert args.gate_k == 0.5
        assert args.n_lags == 8


@pytest.fixture(scope="module")
def cli_table(tmp_path_factory):
    """Measure the table once through the CLI and reuse the JSON file.

    Measuring costs about five seconds, and four CLI tests do not need four
    measurements.
    """
    out = tmp_path_factory.mktemp("cli") / "table.json"
    assert main(["thresholds", "--out", str(out)]) == 0
    return out


class TestInProcess:
    def test_thresholds_writes_json(self, cli_table):
        assert cli_table.exists()
        text = cli_table.read_text()
        assert "/home/" not in text
        assert "/Users/" not in text
        assert '"target_ber": 1e-06' in text

    def test_thresholds_prints_a_monotone_ladder(self, capsys):
        from acmpilot.modcod import ModcodTable

        assert main(["thresholds"]) == 0
        out = capsys.readouterr().out
        assert "monotone ladder: True" in out
        assert "16QAM RS(255,239)" in out
        assert ModcodTable  # imported for the assertion above to be meaningful

    def test_simulate_reports_all_three_baselines(self, cli_table, capsys):
        table = cli_table
        rc = main(
            [
                "simulate", "--table", str(table), "--n-slots", "3000",
                "--tau-ms", "5",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "fixed margin" in out
        assert "hysteresis" in out
        assert "upper bound" in out
        assert "goodput_bit_per_symbol" in out

    def test_sweep_emits_csv(self, cli_table, capsys):
        table = cli_table
        rc = main(
            [
                "sweep", "--table", str(table), "--n-slots", "2000", "--n-seeds", "2",
                "--tau-ms-list", "0", "10",
            ]
        )
        assert rc == 0
        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        header = lines[0].split(",")
        assert "goodput_bit_per_symbol" in header
        assert "policy" in header
        # 2 delays x 3 policies
        assert len(lines) == 1 + 6

    def test_predict_reports_calibration_and_goodput(self, cli_table, capsys):
        table = cli_table
        rc = main(
            [
                "predict", "--table", str(table), "--n-slots", "3000",
                "--train-slots", "1500", "--train-episodes", "1",
                "--n-estimators", "8", "--n-lags", "3", "--tau-ms", "5",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "coverage_80" in out
        assert "below_q10" in out
        assert "learned quantile GBR" in out
        assert "not learned" in out
        assert "upper bound" in out
