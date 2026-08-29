"""CLI tests: ``python -m linkswitch ...``."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "linkswitch", *args],
        capture_output=True, text=True, cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )


class TestThresholdCommand:
    def test_runs_and_prints_thresholds(self):
        out = _run(["threshold"])
        assert out.returncode == 0, out.stderr
        assert "Optimal fixed switching threshold" in out.stdout
        assert "bounded-optimizer" in out.stdout
        assert "grid-search" in out.stdout

    def test_custom_params_change_output(self):
        out_default = _run(["threshold"])
        out_custom = _run(["threshold", "--margin-db", "10", "--sigma-i2", "0.5"])
        assert out_default.stdout != out_custom.stdout


class TestSimulateCommand:
    def test_fixed_policy(self):
        out = _run(["simulate", "--policy", "fixed", "--n-steps", "500", "--seed", "3"])
        assert out.returncode == 0, out.stderr
        assert "throughput_mbps" in out.stdout
        assert "switch_count" in out.stdout

    def test_hysteresis_policy(self):
        out = _run(["simulate", "--policy", "hysteresis", "--n-steps", "500"])
        assert out.returncode == 0, out.stderr
        assert "policy=hysteresis" in out.stdout

    def test_learned_policy_unsupported_here(self):
        out = _run(["simulate", "--policy", "learned"])
        assert out.returncode != 0
        assert "learned" in out.stderr

    def test_invalid_choice_rejected(self):
        out = _run(["simulate", "--policy", "quantum"])
        assert out.returncode != 0


class TestCompareCommand:
    def test_small_comparison_runs(self):
        out = _run([
            "compare", "--n-steps", "300", "--n-reps", "10", "--n-train-episodes", "4",
            "--horizon", "3",
        ])
        assert out.returncode == 0, out.stderr
        assert "fixed_threshold" in out.stdout
        assert "hysteresis" in out.stdout
        assert "learned" in out.stdout

    def test_bad_scenario_param_fails_cleanly(self):
        out = _run(["compare", "--sigma-i2", "-1.0"])
        assert out.returncode != 0
        assert "error" in out.stderr


class TestNoSubcommand:
    def test_requires_a_command(self):
        out = _run([])
        assert out.returncode != 0
