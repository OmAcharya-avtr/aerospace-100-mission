"""Command-line interface."""

import subprocess
import sys

import pytest

from cmgsteer.cli import main


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "cmgsteer", *args], capture_output=True, text=True, check=False
    )


class TestArrayCommand:
    def test_prints_a_summary(self, capsys):
        assert main(["array"]) == 0
        out = capsys.readouterr().out
        assert "CMGArray: 4 CMGs" in out
        assert "singularity measure m" in out

    def test_roof_config(self, capsys):
        assert main(["array", "--config", "roof"]) == 0
        assert "pair1a" in capsys.readouterr().out

    def test_failed_gimbal_is_reported(self, capsys):
        assert main(["array", "--failed", "0"]) == 0
        out = capsys.readouterr().out
        assert "LOCKED" in out
        assert "3 free" in out

    def test_explicit_deltas(self, capsys):
        assert main(["array", "--deltas", "90", "90", "90", "90"]) == 0
        assert "3.2" in capsys.readouterr().out


class TestSingularityCommand:
    def test_classifies_the_z_saturation(self, capsys):
        assert main(["singularity", "--direction", "0", "0", "1"]) == 0
        out = capsys.readouterr().out
        assert "kind                     : external" in out
        assert "passability              : elliptic" in out

    def test_internal_singularity_from_signs(self, capsys):
        assert (
            main(["singularity", "--direction", "0.2", "0.3", "0.9", "--signs", "1", "1", "-1", "-1"])
            == 0
        )
        assert "kind                     : internal" in capsys.readouterr().out

    def test_regular_configuration(self, capsys):
        assert main(["singularity", "--deltas", "0", "0", "0", "0"]) == 0
        assert "kind                     : none" in capsys.readouterr().out


class TestSteerCommand:
    def test_meets_an_easy_command(self, capsys):
        assert main(["steer", "--torque", "0.1", "-0.05", "0.2", "--method", "pinv"]) == 0
        assert "torque error norm" in capsys.readouterr().out

    def test_exits_one_when_the_command_cannot_be_met(self, capsys):
        code = main(
            [
                "steer",
                "--deltas", "90", "90", "90", "90",
                "--torque", "0", "0", "0.1",
                "--method", "sr",
                "--lam", "0.01",
            ]
        )
        assert code == 1
        assert "exceeds the" in capsys.readouterr().out

    @pytest.mark.parametrize("method", ["pinv", "sr", "gsr"])
    def test_every_method(self, capsys, method):
        main(["steer", "--torque", "0.05", "0.0", "0.0", "--method", method])
        assert method in capsys.readouterr().out

    def test_rate_limit_is_reported(self, capsys):
        main(["steer", "--torque", "3", "0", "0", "--max-gimbal-rate", "0.1"])
        assert "rate limited             : True" in capsys.readouterr().out


class TestManoeuvreCommand:
    def test_runs_and_summarises(self, capsys):
        assert main(["manoeuvre", "--seed", "11", "--index", "0"]) == 0
        out = capsys.readouterr().out
        assert "accumulated momentum error" in out
        assert "rate-limited steps" in out

    def test_gradient_null_motion_is_selectable(self, capsys):
        assert main(["manoeuvre", "--seed", "11", "--null", "gradient"]) == 0
        assert "null=gradient" in capsys.readouterr().out

    def test_measure_floor_forces_exit_one(self, capsys):
        code = main(["manoeuvre", "--seed", "11", "--measure-floor", "1e9"])
        assert code == 1
        assert "below the" in capsys.readouterr().out


class TestErrorHandling:
    def test_bad_delta_count_is_a_diagnostic_not_a_traceback(self, capsys):
        assert main(["array", "--deltas", "0", "0"]) == 2
        assert capsys.readouterr().err.startswith("error: ")

    def test_bad_failed_index(self, capsys):
        assert main(["array", "--failed", "17"]) == 2
        assert "out of range" in capsys.readouterr().err

    def test_direction_along_a_gimbal_axis(self, capsys):
        assert main(["singularity", "--direction", "0.8", "0", "0.6"]) == 2
        assert "parallel to gimbal axis" in capsys.readouterr().err

    def test_bad_skew_angle(self, capsys):
        assert main(["array", "--skew", "120"]) == 2
        assert "error:" in capsys.readouterr().err


class TestSubprocessEntryPoint:
    def test_module_runs_as_a_subprocess(self):
        proc = _run(["array"])
        assert proc.returncode == 0
        assert "CMGArray" in proc.stdout

    def test_missing_subcommand_exits_two(self):
        proc = _run([])
        assert proc.returncode == 2

    def test_help_works(self):
        proc = _run(["--help"])
        assert proc.returncode == 0
        assert "singularity" in proc.stdout
