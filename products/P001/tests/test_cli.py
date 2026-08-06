"""Integration tests: end-to-end CLI (scenario file -> report / sweep PNG)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from beamtwin.__main__ import main

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "link_10km.yaml"


@pytest.fixture()
def small_scenario(tmp_path):
    data = {
        "name": "cli-test",
        "link": {"range_km": 10.0, "attenuation_db_per_km": 2.5, "rx_sensitivity_dbm": -30.0},
        "channel": {"cn2": 5.0e-16, "pointing_jitter_urad": 5.0},
        "monte_carlo": {"n_samples": 5000, "seed": 1},
    }
    p = tmp_path / "cli.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


class TestRunCommand:
    def test_run_succeeds_and_prints_report(self, small_scenario, capsys):
        rc = main(["run", str(small_scenario), "--no-surrogate"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "LINK BUDGET" in out and "MONTE CARLO" in out

    def test_run_writes_json(self, small_scenario, tmp_path, capsys):
        out_json = tmp_path / "r.json"
        rc = main(["run", str(small_scenario), "--json", str(out_json), "--no-surrogate"])
        capsys.readouterr()
        assert rc == 0 and out_json.exists()
        data = json.loads(out_json.read_text())
        assert data["name"] == "cli-test"
        assert "margin_db" in data["budget"]

    def test_run_on_shipped_example(self, capsys):
        rc = main(["run", str(EXAMPLE), "--no-surrogate"])
        capsys.readouterr()
        assert rc == 0

    def test_run_missing_file_clean_error(self, tmp_path, capsys):
        rc = main(["run", str(tmp_path / "absent.yaml")])
        err = capsys.readouterr().err
        assert rc == 2
        assert err.startswith("error:") and "not found" in err

    def test_run_invalid_yaml_clean_error(self, tmp_path, capsys):
        p = tmp_path / "bad.yaml"
        p.write_text("link: {range_km: [unclosed", encoding="utf-8")
        rc = main(["run", str(p)])
        assert rc == 2
        assert "error:" in capsys.readouterr().err

    def test_run_invalid_value_clean_error(self, tmp_path, capsys):
        p = tmp_path / "neg.yaml"
        p.write_text(yaml.safe_dump({"link": {"range_km": -3.0}}), encoding="utf-8")
        rc = main(["run", str(p)])
        assert rc == 2
        assert "error:" in capsys.readouterr().err

    def test_run_reproducible_across_invocations(self, small_scenario, tmp_path, capsys):
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        main(["run", str(small_scenario), "--json", str(a), "--no-surrogate"])
        main(["run", str(small_scenario), "--json", str(b), "--no-surrogate"])
        capsys.readouterr()
        assert a.read_text() == b.read_text()


class TestSweepCommand:
    def test_sweep_produces_png(self, small_scenario, tmp_path, capsys):
        png = tmp_path / "sweep.png"
        rc = main(
            [
                "sweep", str(small_scenario), "--param", "range_km",
                "--start", "5", "--stop", "12", "--steps", "4",
                "--output", str(png), "--no-surrogate",
            ]
        )
        capsys.readouterr()
        assert rc == 0 and png.exists() and png.stat().st_size > 1000

    def test_sweep_writes_json_rows(self, small_scenario, tmp_path, capsys):
        png, js = tmp_path / "s.png", tmp_path / "s.json"
        main(
            [
                "sweep", str(small_scenario), "--param", "range_km",
                "--start", "5", "--stop", "12", "--steps", "4",
                "--output", str(png), "--json", str(js), "--no-surrogate",
            ]
        )
        capsys.readouterr()
        rows = json.loads(js.read_text())["rows"]
        assert len(rows) == 4
        # Fade probability must increase with range for a fixed link.
        assert rows[-1]["fade_probability"] >= rows[0]["fade_probability"]

    def test_sweep_over_cn2_log_spacing(self, small_scenario, tmp_path, capsys):
        png, js = tmp_path / "c.png", tmp_path / "c.json"
        rc = main(
            [
                "sweep", str(small_scenario), "--param", "cn2",
                "--start", "1e-16", "--stop", "1e-14", "--steps", "3",
                "--log", "--output", str(png), "--json", str(js), "--no-surrogate",
            ]
        )
        capsys.readouterr()
        assert rc == 0
        rows = json.loads(js.read_text())["rows"]
        assert rows[0]["value"] == pytest.approx(1e-16)
        assert rows[-1]["value"] == pytest.approx(1e-14)

    def test_sweep_over_jitter(self, small_scenario, tmp_path, capsys):
        png = tmp_path / "j.png"
        rc = main(
            [
                "sweep", str(small_scenario), "--param", "pointing_jitter_urad",
                "--start", "0", "--stop", "20", "--steps", "3",
                "--output", str(png), "--no-surrogate",
            ]
        )
        capsys.readouterr()
        assert rc == 0 and png.exists()

    def test_sweep_rejects_bad_step_count(self, small_scenario, capsys):
        rc = main(
            [
                "sweep", str(small_scenario), "--param", "range_km",
                "--start", "1", "--stop", "5", "--steps", "1", "--no-surrogate",
            ]
        )
        assert rc == 2
        assert "steps" in capsys.readouterr().err

    def test_sweep_rejects_inverted_range(self, small_scenario, capsys):
        rc = main(
            [
                "sweep", str(small_scenario), "--param", "range_km",
                "--start", "10", "--stop", "2", "--steps", "3", "--no-surrogate",
            ]
        )
        assert rc == 2
        assert "stop" in capsys.readouterr().err

    def test_sweep_rejects_unknown_param(self, small_scenario):
        with pytest.raises(SystemExit):
            main(
                [
                    "sweep", str(small_scenario), "--param", "not_a_param",
                    "--start", "1", "--stop", "5", "--steps", "3",
                ]
            )


class TestModuleInvocation:
    def test_python_dash_m_runs(self, small_scenario):
        proc = subprocess.run(
            [sys.executable, "-m", "beamtwin", "run", str(small_scenario), "--no-surrogate"],
            capture_output=True, text=True, cwd=ROOT, timeout=180,
        )
        assert proc.returncode == 0
        assert "BeamTwin report" in proc.stdout

    def test_python_dash_m_bad_file_exit_code_2_no_traceback(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, "-m", "beamtwin", "run", str(tmp_path / "x.yaml")],
            capture_output=True, text=True, cwd=ROOT, timeout=180,
        )
        assert proc.returncode == 2
        assert "Traceback" not in proc.stderr
        assert proc.stderr.startswith("error:")

    def test_no_subcommand_exits_nonzero(self):
        proc = subprocess.run(
            [sys.executable, "-m", "beamtwin"],
            capture_output=True, text=True, cwd=ROOT, timeout=180,
        )
        assert proc.returncode != 0
