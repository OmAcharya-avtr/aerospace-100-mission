"""CLI behaviour, exercised through real subprocesses.

Exit statuses are asserted on ``returncode`` from ``subprocess.run`` rather
than by catching ``SystemExit``, so what is tested is what a shell would see.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")


def run_cli(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "-m", "calibaudit", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


class TestInvocation:
    def test_help_exits_zero(self):
        res = run_cli("--help")
        assert res.returncode == 0
        assert "calibaudit" in res.stdout

    def test_version_exits_zero_and_prints_the_version(self):
        res = run_cli("--version")
        assert res.returncode == 0
        assert "0.1.0" in res.stdout

    def test_no_subcommand_is_a_usage_error(self):
        res = run_cli()
        assert res.returncode == 2
        assert "required" in res.stderr.lower()

    def test_unknown_subcommand_is_a_usage_error(self):
        res = run_cli("calibrate-everything")
        assert res.returncode == 2

    def test_help_mentions_the_research_grade_scope(self):
        res = run_cli("--help")
        assert "not flight-qualified" in res.stdout

    @pytest.mark.parametrize(
        "sub", ["specs", "decompose", "ece", "ece-bias", "reliability", "recalibrate"]
    )
    def test_every_subcommand_has_help(self, sub):
        res = run_cli(sub, "--help")
        assert res.returncode == 0


class TestSpecs:
    def test_specs_lists_every_shipped_spec(self):
        res = run_cli("specs")
        assert res.returncode == 0
        for name in ("calibrated", "overconfident", "biased_high"):
            assert name in res.stdout

    def test_specs_json_is_parseable(self):
        res = run_cli("specs", "--json")
        assert res.returncode == 0
        payload = json.loads(res.stdout)
        assert len(payload["specs"]) == 6
        assert payload["specs"][0]["name"] == "calibrated"

    def test_specs_json_reports_zero_ece_for_calibrated_specs(self):
        payload = json.loads(run_cli("specs", "--json").stdout)
        for row in payload["specs"]:
            if row["calibrated"]:
                assert row["ece"] == 0.0


class TestDecompose:
    def test_decompose_runs_and_prints_both_decompositions(self):
        res = run_cli("decompose", "-n", "1000", "--seed", "56")
        assert res.returncode == 0
        assert "exact Murphy decomposition" in res.stdout
        assert "binned decomposition" in res.stdout

    def test_decompose_json_residuals_are_at_machine_precision(self):
        res = run_cli("decompose", "-n", "1000", "--seed", "56", "--json")
        assert res.returncode == 0
        payload = json.loads(res.stdout)
        assert abs(payload["binned"]["identity_residual"]) < 1e-14
        assert abs(payload["exact_on_rounded"]["three_term_residual"]) < 1e-14

    def test_decompose_reports_a_nonzero_three_term_residual_when_binned(self):
        payload = json.loads(
            run_cli("decompose", "-n", "4000", "--seed", "56", "--json").stdout
        )
        assert abs(payload["binned"]["three_term_residual"]) > 1e-6

    def test_decompose_accepts_equal_mass(self):
        res = run_cli("decompose", "-n", "800", "--strategy", "equal_mass")
        assert res.returncode == 0

    def test_decompose_rejects_an_unknown_spec(self):
        res = run_cli("decompose", "--spec", "perfect")
        assert res.returncode == 2

    def test_decompose_rejects_zero_bins(self):
        res = run_cli("decompose", "-n", "100", "--bins", "0")
        assert res.returncode == 1
        assert "at least 1" in res.stderr


class TestEce:
    def test_ece_reports_the_bias_on_a_calibrated_spec(self):
        res = run_cli("ece", "--spec", "calibrated", "-n", "1000", "--replicates", "100")
        assert res.returncode == 0
        assert "population ECE            : 0.000000000" in res.stdout
        assert "null mean (binning bias)" in res.stdout

    def test_ece_json_debiased_is_closer_to_zero_than_raw(self):
        payload = json.loads(
            run_cli(
                "ece", "--spec", "calibrated", "-n", "4000", "--replicates", "150", "--json"
            ).stdout
        )
        assert payload["population_ece"] == 0.0
        assert abs(payload["debiased"]) < abs(payload["raw"])

    def test_ece_p_value_is_small_for_a_biased_spec(self):
        payload = json.loads(
            run_cli(
                "ece", "--spec", "biased_high", "-n", "4000", "--replicates", "150", "--json"
            ).stdout
        )
        assert payload["p_value"] < 0.05


class TestEceBias:
    def test_bias_grid_runs_and_fits_a_power_law(self):
        res = run_cli(
            "ece-bias",
            "--bins-grid", "5", "10", "20",
            "--samples-grid", "200", "1000",
            "--replicates", "30",
        )
        assert res.returncode == 0
        assert "power-law fit" in res.stdout
        assert "theory: exponent 0.5" in res.stdout

    def test_bias_grid_json_has_one_row_per_cell(self):
        payload = json.loads(
            run_cli(
                "ece-bias",
                "--bins-grid", "5", "10",
                "--samples-grid", "400",
                "--replicates", "20",
                "--json",
            ).stdout
        )
        assert len(payload["rows"]) == 2
        assert all(r["bias"] > 0 for r in payload["rows"])

    def test_single_cell_reports_that_the_fit_is_unavailable(self):
        res = run_cli(
            "ece-bias", "--bins-grid", "10", "--samples-grid", "400", "--replicates", "10"
        )
        assert res.returncode == 0
        assert "power-law fit not available" in res.stdout


class TestReliability:
    def test_reliability_prints_a_banded_table(self):
        res = run_cli("reliability", "-n", "1000", "--bootstrap", "100")
        assert res.returncode == 0
        assert "band_lo" in res.stdout
        assert "pointwise" in res.stdout

    def test_reliability_json_bands_bracket_the_observed_frequency(self):
        payload = json.loads(
            run_cli("reliability", "-n", "1000", "--bootstrap", "100", "--json").stdout
        )
        for lo, obs, hi in zip(
            payload["lower"], payload["observed_frequency"], payload["upper"], strict=True
        ):
            assert lo <= obs + 1e-12
            assert hi >= obs - 1e-12

    def test_reliability_rejects_a_bad_level(self):
        res = run_cli("reliability", "-n", "200", "--level", "1.5")
        assert res.returncode == 1
        assert "level must lie" in res.stderr


class TestRecalibrate:
    def test_recalibrate_prints_the_baseline_first(self):
        res = run_cli("recalibrate", "-n", "2000", "--bootstrap", "100")
        assert res.returncode == 0
        lines = [ln for ln in res.stdout.splitlines() if ln.strip().startswith("raw")]
        assert lines, res.stdout

    def test_recalibrate_exits_zero_without_the_flag_even_when_nothing_helps(self):
        res = run_cli("recalibrate", "--spec", "calibrated", "-n", "200", "--bootstrap", "100")
        assert res.returncode == 0

    def test_require_improvement_exits_two_when_nothing_helps(self):
        res = run_cli(
            "recalibrate",
            "--spec", "calibrated",
            "-n", "200",
            "--bootstrap", "200",
            "--require-improvement",
        )
        assert res.returncode == 2
        assert "no method improved" in res.stderr

    def test_require_improvement_exits_zero_when_recalibration_helps(self):
        res = run_cli(
            "recalibrate",
            "--spec", "biased_high",
            "-n", "8000",
            "--bootstrap", "200",
            "--require-improvement",
        )
        assert res.returncode == 0

    def test_recalibrate_json_lists_the_improved_methods(self):
        payload = json.loads(
            run_cli(
                "recalibrate", "--spec", "biased_high", "-n", "8000",
                "--bootstrap", "100", "--json",
            ).stdout
        )
        assert "platt" in payload["improved"]
        assert payload["results"][0]["method"] == "raw"

    def test_recalibrate_rejects_an_impossible_split(self):
        res = run_cli("recalibrate", "-n", "10", "--test-fraction", "0.05")
        assert res.returncode == 1
        assert "at least 2" in res.stderr

    def test_methods_subset_is_honoured(self):
        payload = json.loads(
            run_cli(
                "recalibrate", "-n", "1000", "--methods", "raw", "platt",
                "--bootstrap", "50", "--json",
            ).stdout
        )
        assert [r["method"] for r in payload["results"]] == ["raw", "platt"]

    def test_ensemble_flag_populates_the_member_count(self):
        payload = json.loads(
            run_cli(
                "recalibrate", "-n", "1000", "--ensemble", "20",
                "--bootstrap", "50", "--json",
            ).stdout
        )
        platt = next(r for r in payload["results"] if r["method"] == "platt")
        assert platt["n_ensemble"] == 20
