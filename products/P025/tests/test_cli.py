"""CLI: every subcommand runs in a subprocess and prints what it promises."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str, expect_ok: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, "-m", "fdiscope", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if expect_ok:
        assert proc.returncode == 0, proc.stderr
    return proc


class TestHelp:
    def test_no_command_is_an_error(self):
        proc = run_cli(expect_ok=False)
        assert proc.returncode != 0

    def test_help_lists_every_subcommand(self):
        out = run_cli("--help").stdout
        for name in ("design", "simulate", "signatures", "benchmark"):
            assert name in out

    def test_help_states_the_safety_position(self):
        out = run_cli("--help").stdout
        assert "not flight-qualified" in out


class TestDesign:
    def test_prints_both_thresholds_and_both_delay_expressions(self):
        out = run_cli("design").stdout
        for key in (
            "chi-squared test",
            "threshold",
            "steady-state residual mu",
            "Wald mean delay",
            "Siegmund mean delay",
            "achieved analytic ARL0",
        ):
            assert key in out

    def test_a_tighter_alpha_raises_the_threshold(self):
        loose = run_cli("design", "--alpha", "0.01").stdout
        tight = run_cli("design", "--alpha", "0.0001").stdout
        loose_value = float(
            [ln for ln in loose.splitlines() if "threshold" in ln][0].split("=")[1]
        )
        tight_value = float(
            [ln for ln in tight.splitlines() if "threshold" in ln][0].split("=")[1]
        )
        assert tight_value > loose_value

    def test_rejects_an_impossible_alpha(self):
        proc = run_cli("design", "--alpha", "2.0", expect_ok=False)
        assert proc.returncode == 2
        assert "error:" in proc.stderr


class TestSimulate:
    def test_healthy_run_reports_alarm_fractions(self):
        out = run_cli("simulate", "--fault", "none", "--steps", "800").stdout
        assert "chi2 alarm fraction" in out
        assert "CUSUM alarm fraction" in out
        assert "mean NIS before onset" in out

    @pytest.mark.parametrize(
        "fault",
        [
            "sensor_bias",
            "sensor_drift",
            "sensor_stuck",
            "sensor_dropout",
            "actuator_loss_of_effectiveness",
            "actuator_stuck",
            "actuator_runaway",
        ],
    )
    def test_every_fault_runs(self, fault):
        args = ["simulate", "--fault", fault, "--steps", "900", "--onset", "400"]
        if fault == "actuator_loss_of_effectiveness":
            args += ["--magnitude", "0.6", "--units", "physical"]
        elif fault == "actuator_runaway":
            args += ["--magnitude", "1e-4", "--units", "physical"]
        out = run_cli(*args).stdout
        assert f"fault                     = {fault}" in out
        assert "detection delay" in out or "never detected" in out

    def test_a_large_bias_is_detected(self):
        out = run_cli(
            "simulate", "--fault", "sensor_bias", "--magnitude", "8", "--channel", "1",
            "--steps", "900", "--onset", "400",
        ).stdout
        assert "never detected" not in out

    def test_rejects_an_unknown_fault(self):
        proc = run_cli("simulate", "--fault", "wheel_explosion", expect_ok=False)
        assert proc.returncode != 0

    def test_rejects_a_bad_channel(self):
        proc = run_cli("simulate", "--channel", "3", expect_ok=False)
        assert proc.returncode != 0


class TestSignatures:
    def test_prints_a_square_gram_and_the_worst_pair(self):
        out = run_cli("signatures", "--window", "40", "--onsets", "2").stdout
        assert "worst off-diagonal" in out
        assert "sensor_bias" in out and "actuator_runaway" in out
        assert out.count("1.0000") >= 7


class TestBenchmark:
    def test_reduced_benchmark_reports_every_method(self):
        out = run_cli(
            "benchmark", "--train", "16", "--test", "16", "--calib", "8", "--trees", "10"
        ).stdout
        assert "NOT the published numbers" in out
        for name in ("chi2_short", "chi2_long", "cusum", "glr", "learned"):
            assert name in out
