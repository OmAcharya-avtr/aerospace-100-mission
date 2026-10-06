"""The CLI, including a clean-subprocess ``python -m photoncount --help``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from photoncount.__main__ import build_parser, main

SRC = Path(__file__).resolve().parents[1] / "src"


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    env["MPLBACKEND"] = "Agg"
    return env


def test_module_help_exits_zero_in_a_clean_subprocess():
    proc = subprocess.run(
        [sys.executable, "-m", "photoncount", "--help"],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
        check=False,
    )
    assert proc.returncode == 0
    assert "photon-counting" in proc.stdout.lower()
    assert "not flight-qualified" in proc.stdout


def test_module_version_exits_zero_in_a_clean_subprocess():
    proc = subprocess.run(
        [sys.executable, "-m", "photoncount", "--version"],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
        check=False,
    )
    assert proc.returncode == 0
    assert "0.1.0" in proc.stdout


def test_no_subcommand_is_an_error():
    proc = subprocess.run(
        [sys.executable, "-m", "photoncount"],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
        check=False,
    )
    assert proc.returncode != 0


def _run(capsys, argv: list[str]) -> dict:
    assert main(argv) == 0
    return json.loads(capsys.readouterr().out)


def test_ppm_subcommand(capsys):
    out = _run(capsys, ["ppm", "--order", "16", "--signal", "2"])
    assert out["symbol_error_probability"] == pytest.approx(0.1268768280, rel=1e-9)
    assert out["bits_per_symbol"] == 4.0
    assert "erasure_capacity_bits_per_symbol" in out


def test_ppm_subcommand_with_background(capsys):
    out = _run(capsys, ["ppm", "--order", "16", "--signal", "4", "--background", "0.2"])
    assert "hard_decision_capacity_bits_per_symbol" in out
    assert 0.0 < out["symbol_error_probability"] < 1.0


def test_deadtime_subcommand_forward(capsys):
    out = _run(capsys, ["deadtime", "--dead-time", "1e-6", "--true-rate", "1e5"])
    assert out["observed_rate_hz"] == pytest.approx(90483.74180359596, rel=1e-9)
    assert out["paralyzable_max_observed_rate_hz"] == pytest.approx(367879.4411714, rel=1e-9)


def test_deadtime_subcommand_warns_about_two_roots(capsys):
    out = _run(capsys, ["deadtime", "--dead-time", "1e-6", "--observed-rate", "3e5"])
    assert "two-valued" in out["branch_warning"]
    assert out["true_rate_lower_branch_hz"] < 1e6 < out["true_rate_upper_branch_hz"]


def test_deadtime_subcommand_reports_an_unreachable_rate(capsys):
    out = _run(capsys, ["deadtime", "--dead-time", "1e-6", "--observed-rate", "5e5"])
    assert out["observable"] is False
    assert "no true rate" in out["branch_warning"]


def test_webb_subcommand(capsys):
    out = _run(capsys, ["webb", "--mean", "100", "--excess-noise", "2"])
    assert out["moments"]["variance"] == pytest.approx(200.0)
    assert out["binned_pmf_sum"] == pytest.approx(1.0, abs=1e-9)
    assert out["total_variation_to_poisson"] > 0.0


def test_capacity_subcommand_background_free(capsys):
    out = _run(capsys, ["capacity", "--order", "16", "--signal", "2"])
    assert out["minimum_photons_per_bit_limit"] == 0.25
    assert out["photons_per_bit"] > 0.25


def test_capacity_subcommand_with_background(capsys):
    out = _run(
        capsys,
        ["capacity", "--order", "16", "--signal", "4", "--background", "0.2",
         "--monte-carlo", "4000", "--seed", "1"],
    )
    assert out["soft_decision"]["bits_per_symbol"] > out["hard_decision"]["bits_per_symbol"]


def test_acquire_subcommand_simulated(capsys, tmp_path):
    out = _run(
        capsys,
        ["acquire", "--rate", "1e5", "--dead-time", "1e-7", "--window", "1e-3",
         "--windows", "20", "--label", "cli", "--directory", str(tmp_path)],
    )
    assert out["preflight_passed"] is True
    assert out["record_file"] == "cli.json"
    assert out["is_measurement"] is False
    assert (tmp_path / "cli.json").exists()


def test_acquire_subcommand_dry_run(capsys, tmp_path):
    out = _run(
        capsys,
        ["acquire", "--window", "1e-3", "--windows", "5", "--dry-run",
         "--label", "dry", "--directory", str(tmp_path)],
    )
    assert out["mode"] == "dry_run"
    assert out["wall_seconds"] > 0.0


def test_acquire_subcommand_device_backend_reports_not_implemented(tmp_path, capsys):
    status = main(
        ["acquire", "--backend", "device", "--window", "1e-3", "--windows", "5",
         "--label", "dev", "--directory", str(tmp_path)]
    )
    payload = json.loads(capsys.readouterr().out)
    assert status == 2
    assert "physical photon-counting module" in payload["not_implemented"]


def test_acquire_subcommand_aborts_on_failed_preflight(tmp_path, capsys):
    status = main(
        ["acquire", "--rate", "1e5", "--dead-time", "1e-4", "--window", "1e-3",
         "--windows", "5", "--label", "bad", "--directory", str(tmp_path)]
    )
    payload = json.loads(capsys.readouterr().out)
    assert status == 1
    assert payload["preflight_passed"] is False
    assert "no acquisition attempted" in payload["aborted"]


def test_correct_subcommand_reports_every_closed_form(capsys):
    out = _run(
        capsys,
        ["correct", "--observed-rate", "9e4", "--dead-time", "1e-6",
         "--model", "paralyzable", "--afterpulse", "0.05"],
    )
    assert out["nonparalyzable_closed_form_hz"] > 9e4
    assert out["paralyzable_closed_form_hz"] > 9e4
    assert out["composed_closed_form_hz"] < out["matched_closed_form_hz"]


def test_correct_subcommand_reports_a_missing_model_file(capsys, tmp_path):
    out = _run(
        capsys,
        ["correct", "--observed-rate", "9e4", "--dead-time", "1e-6",
         "--model-file", str(tmp_path / "absent.joblib")],
    )
    assert "not found" in out["learned"]


def test_correct_subcommand_uses_a_saved_model(capsys, tmp_path):
    from photoncount.correction import RateCorrector
    from photoncount.dataset import generate_dataset

    path = tmp_path / "c.joblib"
    RateCorrector(n_estimators=25).fit(generate_dataset(120, 4)).save(path)
    out = _run(
        capsys,
        ["correct", "--observed-rate", "9e4", "--dead-time", "1e-6", "--model",
         "paralyzable", "--afterpulse", "0.05", "--fano", "0.9",
         "--model-file", str(path)],
    )
    assert out["learned_hz"] > 0.0
    assert out["learned_interval_hz"][0] <= out["learned_hz"] <= out["learned_interval_hz"][1]


def test_parser_exposes_every_subcommand():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert set(actions[0].choices) == {
        "ppm",
        "deadtime",
        "webb",
        "capacity",
        "acquire",
        "correct",
    }
