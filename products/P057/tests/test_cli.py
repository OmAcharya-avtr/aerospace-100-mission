"""CLI tests. Every one runs the module in a clean subprocess and asserts on
the real exit status, because a CLI that imports in pytest's interpreter and
fails in a fresh one has shipped from this portfolio before.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")
TIMEOUT = 600


def run(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "-m", "conformalband", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=TIMEOUT,
    )


def test_help_exits_zero():
    result = run("--help")
    assert result.returncode == 0
    assert "conformalband" in result.stdout


def test_version_exits_zero():
    result = run("--version")
    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_no_subcommand_is_an_argparse_error():
    result = run()
    assert result.returncode == 2
    assert "required" in result.stderr


def test_unknown_subcommand_is_an_argparse_error():
    result = run("nonsense")
    assert result.returncode == 2


def test_info_exits_zero():
    result = run("info")
    assert result.returncode == 0
    assert "declared covariate shifts" in result.stdout
    assert "Mahalanobis" in result.stdout


def test_info_json_is_parseable():
    result = run("info", "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["version"] == "0.1.0"
    assert len(payload["shift_levels"]) == 4


def test_bound_exits_zero():
    result = run("bound", "--n", "500", "--alpha", "0.1")
    assert result.returncode == 0
    assert "0.900199600798" in result.stdout


def test_bound_json_values():
    result = run("bound", "--n", "100", "--alpha", "0.1", "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["rank"] == 91
    assert payload["exact"] == pytest.approx(91 / 101, rel=1e-12)


def test_bound_refuses_a_calibration_size_that_is_too_small():
    # The one deliberate non-zero exit: n = 5 cannot attain alpha = 0.1.
    result = run("bound", "--n", "5", "--alpha", "0.1")
    assert result.returncode == 2
    assert "refused" in result.stderr
    assert "n_calibration >= 9" in result.stderr


def test_baseline_exits_zero():
    result = run("baseline", "--n-fit", "400", "--n-calibration", "120", "--n-test", "150")
    assert result.returncode == 0
    assert "analytic baseline fitted coefficients" in result.stdout
    assert "physics" in result.stdout
    assert "learned" in result.stdout


def test_baseline_json_has_both_models():
    result = run(
        "baseline", "--n-fit", "400", "--n-calibration", "120", "--n-test", "150", "--json"
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert set(payload["models"]) == {"physics", "learned"}
    assert payload["models"]["physics"]["conformal_quantile_Wh"] > 0.0


def test_audit_exits_zero():
    result = run(
        "audit",
        "--replicates",
        "3",
        "--n-fit",
        "400",
        "--n-calibration",
        "120",
        "--n-test",
        "150",
        "--severities",
        "0",
        "2",
        "--models",
        "physics",
        "--methods",
        "split",
        "weighted_declared",
    )
    assert result.returncode == 0
    assert "weighted_declared" in result.stdout


def test_audit_json_rows():
    result = run(
        "audit",
        "--replicates",
        "3",
        "--n-fit",
        "400",
        "--n-calibration",
        "120",
        "--n-test",
        "150",
        "--severities",
        "0",
        "--models",
        "physics",
        "--methods",
        "split",
        "--json",
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert len(payload["rows"]) == 1
    assert payload["config"]["n_calibration"] == 120


def test_audit_fail_on_undercoverage_exits_two_when_a_method_breaks():
    # Severity 3 with no weighting is a gross failure: split conformal loses
    # several points of coverage, so the gate flag must fire.
    result = run(
        "audit",
        "--replicates",
        "12",
        "--n-fit",
        "600",
        "--n-calibration",
        "200",
        "--n-test",
        "300",
        "--severities",
        "3",
        "--models",
        "learned",
        "--methods",
        "split",
        "--fail-on-undercoverage",
    )
    assert result.returncode == 2
    assert "demonstrably below nominal" in result.stderr


def test_audit_fail_on_undercoverage_exits_zero_in_distribution():
    result = run(
        "audit",
        "--replicates",
        "6",
        "--n-fit",
        "400",
        "--n-calibration",
        "150",
        "--n-test",
        "200",
        "--severities",
        "0",
        "--models",
        "physics",
        "--methods",
        "split",
        "--fail-on-undercoverage",
    )
    assert result.returncode == 0


def test_audit_rejects_an_unknown_method():
    result = run("audit", "--methods", "nonsense")
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_breaking_point_exits_zero():
    result = run(
        "breaking-point",
        "--replicates",
        "3",
        "--n-calibration",
        "120",
        "--n-test",
        "150",
        "--model",
        "physics",
    )
    assert result.returncode == 0
    assert "breaking fraction" in result.stdout


def test_breaking_point_json_grid():
    result = run(
        "breaking-point",
        "--replicates",
        "2",
        "--n-calibration",
        "120",
        "--n-test",
        "100",
        "--model",
        "physics",
        "--json",
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert len(payload["rows"]) == 31
    assert payload["rows"][0]["fraction"] == 0.0


def test_strata_exits_zero():
    result = run("strata", "--replicates", "2", "--severity", "2", "--model", "physics")
    assert result.returncode == 0
    assert "per-tercile coverage" in result.stdout


def test_strata_json_keys():
    result = run(
        "strata", "--replicates", "2", "--severity", "0", "--model", "physics", "--json"
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert set(payload) == {"split", "mondrian"}
