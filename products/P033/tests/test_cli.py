"""The CLI, exercised as a subprocess so the module entry point is real."""

from __future__ import annotations

import functools
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")


@functools.cache
def run(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the CLI in a subprocess.

    Cached on the argument tuple: a fresh interpreter costs about two seconds
    once ``onnxruntime`` and ``scikit-learn`` are imported, and several tests
    assert on different parts of the same invocation's output.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "edgeinfer", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )


class TestHelpAndVersion:
    def test_help_exits_zero(self) -> None:
        result = run("--help")
        assert result.returncode == 0
        assert "edgeinfer" in result.stdout

    def test_help_states_the_validation_level(self) -> None:
        result = run("--help")
        assert "Validation level: 3, hardware-pending" in result.stdout

    def test_help_makes_no_certification_claim(self) -> None:
        lowered = run("--help").stdout.lower()
        for banned in ("flight-safe", "mission-ready", "production-ready"):
            assert banned not in lowered
        assert "not flight-qualified and carries no certification" in lowered

    def test_version_exits_zero(self) -> None:
        result = run("--version")
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_no_subcommand_is_an_error(self) -> None:
        assert run().returncode != 0

    @pytest.mark.parametrize(
        "sub", ["analyse", "profile", "check", "feasibility", "env"]
    )
    def test_every_subcommand_has_help(self, sub: str) -> None:
        result = run(sub, "--help")
        assert result.returncode == 0


class TestAnalyse:
    def test_analyse_prints_counts_with_units(self) -> None:
        result = run("analyse", "--graph", "hand_cnn")
        assert result.returncode == 0
        assert "total MACs              : 684" in result.stdout
        assert "total FLOPs             : 1514" in result.stdout
        assert "compulsory traffic      : 1832 B" in result.stdout

    def test_analyse_cites_the_roofline_source(self) -> None:
        result = run("analyse", "--graph", "hand_mlp")
        assert "Williams, Waterman &" in result.stdout
        assert "2009" in result.stdout

    def test_analyse_says_the_device_peaks_are_declared(self) -> None:
        result = run("analyse", "--graph", "hand_mlp")
        assert "declared, not measured" in result.stdout

    def test_analyse_breaks_down_per_node(self) -> None:
        result = run("analyse", "--graph", "hand_cnn")
        assert "conv0" in result.stdout
        assert "memory" in result.stdout or "compute" in result.stdout

    def test_a_missing_onnx_file_exits_two(self) -> None:
        result = run("analyse", "--onnx", "/nonexistent/model.onnx")
        assert result.returncode == 2
        assert "no such ONNX file" in result.stderr

    def test_an_onnx_file_shipped_with_onnxruntime_can_be_analysed(
        self, tmp_path
    ) -> None:
        from edgeinfer.onnx_io import shipped_model_path

        result = run("analyse", "--onnx", shipped_model_path("mul_1.onnx"))
        assert result.returncode == 0
        assert "total FLOPs             : 6" in result.stdout


class TestProfile:
    def test_profile_prints_the_tail_before_the_median(self) -> None:
        result = run("profile", "--graph", "hand_mlp", "--repeats", "40", "--warmup", "4")
        assert result.returncode == 0
        assert result.stdout.index("p99") < result.stdout.index("p50 (median)")

    def test_profile_states_the_environment_and_repeat_count(self) -> None:
        result = run("profile", "--graph", "hand_mlp", "--repeats", "40")
        assert "n=40 repeats" in result.stdout
        assert "shared host" in result.stdout

    def test_profile_prints_an_uncertainty_budget_per_statistic(self) -> None:
        result = run("profile", "--graph", "hand_mlp", "--repeats", "40")
        assert result.stdout.count("u_c (combined, RSS)") == 3


class TestCheck:
    def test_a_loose_budget_passes_and_exits_zero(self) -> None:
        result = run(
            "check", "--graph", "hand_mlp", "--repeats", "40",
            "--budget-latency-ms", "100", "--budget-memory-kb", "4096",
        )
        assert result.returncode == 0
        assert "overall     : PASS" in result.stdout

    def test_an_impossible_latency_budget_fails_and_exits_one(self) -> None:
        result = run(
            "check", "--graph", "hand_mlp", "--repeats", "40",
            "--budget-latency-ms", "0.000001", "--budget-memory-kb", "4096",
        )
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_an_infeasible_budget_exits_two_without_measuring(self) -> None:
        result = run(
            "check", "--graph", "hand_mlp",
            "--budget-latency-ms", "2", "--budget-duty-cycle", "0.1",
            "--budget-period-ms", "10",
        )
        assert result.returncode == 2
        assert "INFEASIBLE BY CONSTRUCTION" in result.stdout
        assert "not measuring" in result.stdout

    def test_the_power_row_is_never_measured(self) -> None:
        result = run(
            "check", "--graph", "hand_mlp", "--repeats", "40",
            "--budget-latency-ms", "100", "--budget-memory-kb", "4096",
            "--budget-power-w", "7",
        )
        assert "average power" in result.stdout
        assert "NOT MEASURED" in result.stdout

    def test_the_report_carries_the_container_note(self) -> None:
        result = run(
            "check", "--graph", "hand_mlp", "--repeats", "40",
            "--budget-latency-ms", "100", "--budget-memory-kb", "4096",
        )
        assert "shared, single-CPU-core cloud container" in result.stdout


class TestFeasibilityCommand:
    def test_a_consistent_budget_exits_zero(self) -> None:
        result = run("feasibility", "--budget-latency-ms", "1")
        assert result.returncode == 0
        assert "self-consistent" in result.stdout

    def test_a_contradictory_budget_exits_two(self) -> None:
        result = run(
            "feasibility", "--budget-latency-ms", "1", "--budget-median-ms", "5"
        )
        assert result.returncode == 2
        assert "INFEASIBLE BY CONSTRUCTION" in result.stdout

    def test_it_says_feasibility_is_not_achievability(self) -> None:
        result = run("feasibility", "--budget-latency-ms", "1")
        assert "says nothing about whether any model meets them" in result.stdout

    def test_an_out_of_range_duty_cycle_exits_two(self) -> None:
        result = run("feasibility", "--budget-latency-ms", "1", "--budget-duty-cycle", "2")
        assert result.returncode == 2
        assert "duty_cycle" in result.stderr


class TestEnvCommand:
    def test_env_prints_the_core_count_and_the_clock(self) -> None:
        result = run("env", "--timer-samples", "200")
        assert result.returncode == 0
        assert "cpu_count_available" in result.stdout
        assert "clock_resolution_s" in result.stdout
        assert "timer pair cost" in result.stdout

    def test_env_prints_the_container_caveat(self) -> None:
        result = run("env", "--timer-samples", "200")
        assert "not measurements of any edge or flight target" in result.stdout
