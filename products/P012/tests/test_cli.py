"""Tests for the navbench command-line interface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from navbench.cli import build_parser, main

SRC = str(Path(__file__).resolve().parents[1] / "src")


class TestParser:
    def test_subcommand_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_unknown_subcommand_exits(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["nope"])

    @pytest.mark.parametrize(
        "cmd", ["riccati", "bench", "attitude", "consistency", "adaptive"]
    )
    def test_all_subcommands_parse(self, cmd):
        args = build_parser().parse_args([cmd])
        assert args.command == cmd
        assert callable(args.func)

    def test_version_exits_zero(self):
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args(["--version"])
        assert exc.value.code == 0


class TestRiccatiCommand:
    def test_default_run(self, capsys):
        assert main(["riccati"]) == 0
        out = capsys.readouterr().out
        assert "1.618033988" in out

    def test_json_output(self, capsys):
        assert main(["riccati", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["model"] == "random-walk"
        assert payload["p_prior"][0][0] == pytest.approx(1.618033988749895, abs=1e-9)

    @pytest.mark.parametrize("model", ["random-walk", "cv-cwna", "cv-dwna"])
    def test_each_model(self, model, capsys):
        assert main(["riccati", "--model", model, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["iterations"] > 0

    def test_invalid_q_exits_2(self, capsys):
        assert main(["riccati", "--q", "-1"]) == 2
        assert "error:" in capsys.readouterr().err

    def test_invalid_r_exits_2(self):
        assert main(["riccati", "--r", "0"]) == 2


class TestBenchCommand:
    def test_runs(self, capsys):
        assert main(["bench", "--steps", "40"]) == 0
        out = capsys.readouterr().out
        assert "EKF" in out and "UKF" in out and "mean NEES" in out

    def test_json(self, capsys):
        assert main(["bench", "--steps", "40", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        names = {s["name"] for s in payload["scores"]}
        assert names == {"EKF", "UKF", "KF (converted)"}

    def test_reproducible(self, capsys):
        main(["bench", "--steps", "40", "--seed", "5", "--json"])
        a = capsys.readouterr().out
        main(["bench", "--steps", "40", "--seed", "5", "--json"])
        assert capsys.readouterr().out == a

    def test_different_seed_differs(self, capsys):
        main(["bench", "--steps", "40", "--seed", "5", "--json"])
        a = capsys.readouterr().out
        main(["bench", "--steps", "40", "--seed", "6", "--json"])
        assert capsys.readouterr().out != a

    def test_bad_burn_in_exits_2(self):
        assert main(["bench", "--steps", "10", "--burn-in", "50"]) == 2


class TestAttitudeCommand:
    def test_runs(self, capsys):
        assert main(["attitude", "--steps", "120", "--burn-in", "20"]) == 0
        out = capsys.readouterr().out
        assert "attitude RMS error" in out and "CAVEAT" in out

    def test_json_fields(self, capsys):
        assert main(["attitude", "--steps", "120", "--burn-in", "20", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        for key in ("attitude_rms_rad", "mean_nis", "max_quat_norm_error"):
            assert key in payload
        assert payload["max_quat_norm_error"] < 1e-12

    def test_bad_sigma_exits_2(self):
        assert main(["attitude", "--steps", "40", "--sigma-st", "-1"]) == 2


class TestConsistencyCommand:
    def test_correct_filter_inside_bounds(self, capsys):
        assert main(["consistency", "--runs", "20", "--steps", "60", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        lo, hi = payload["nees_bounds"]
        assert lo <= payload["mean_anees"] <= hi

    def test_mismatched_filter_leaves_bounds(self, capsys):
        assert main([
            "consistency", "--runs", "20", "--steps", "60",
            "--q-mismatch", "0.02", "--json",
        ]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["mean_anees"] > payload["nees_bounds"][1]

    def test_text_output(self, capsys):
        assert main(["consistency", "--runs", "10", "--steps", "50"]) == 0
        assert "ANEES" in capsys.readouterr().out

    def test_bad_alpha_exits_2(self):
        assert main([
            "consistency", "--runs", "5", "--steps", "40", "--alpha-level", "2.0"
        ]) == 2


class TestAdaptiveCommand:
    def test_runs_small(self, capsys):
        assert main([
            "adaptive", "--train-runs", "12", "--test-runs", "5", "--steps", "250",
            "--members", "2", "--burn-in", "40", "--json",
        ]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert set(payload["results"]) == {"fixed", "mehra", "learned"}
        for vals in payload["results"].values():
            assert vals["position_rmse"] > 0.0

    def test_text_table(self, capsys):
        assert main([
            "adaptive", "--train-runs", "12", "--test-runs", "4", "--steps", "250",
            "--members", "2", "--burn-in", "40",
        ]) == 0
        out = capsys.readouterr().out
        assert "fixed" in out and "mehra" in out and "learned" in out


class TestModuleInvocation:
    def test_python_dash_m_runs(self):
        proc = subprocess.run(
            [sys.executable, "-m", "navbench", "riccati"],
            capture_output=True, text=True, env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
            timeout=120,
        )
        assert proc.returncode == 0
        assert "1.618033988" in proc.stdout

    def test_python_dash_m_bad_input_exit_2_no_traceback(self):
        proc = subprocess.run(
            [sys.executable, "-m", "navbench", "riccati", "--q", "-1"],
            capture_output=True, text=True, env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
            timeout=120,
        )
        assert proc.returncode == 2
        assert "Traceback" not in proc.stderr
        assert proc.stderr.startswith("error:")

    def test_help_exits_zero(self):
        proc = subprocess.run(
            [sys.executable, "-m", "navbench", "--help"],
            capture_output=True, text=True, env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
            timeout=120,
        )
        assert proc.returncode == 0
        assert "consistency" in proc.stdout
