"""The command-line interface, exercised in-process and as a subprocess."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from pointahead import __version__
from pointahead.__main__ import build_parser, main

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _run(arguments: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    environment = {"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"}
    return subprocess.run(
        [sys.executable, "-m", "pointahead", *arguments],
        capture_output=True,
        text=True,
        cwd=str(cwd or ROOT),
        env=environment,
        check=False,
        timeout=300,
    )


def test_version_flag_prints_the_package_version() -> None:
    result = _run(["--version"])
    assert result.returncode == 0
    assert __version__ in result.stdout


def test_info_reports_the_environment_and_the_safety_statement() -> None:
    result = _run(["info"])
    assert result.returncode == 0
    assert "pointahead" in result.stdout
    assert "numpy" in result.stdout
    assert "not flight-qualified" in result.stdout


def test_scenario_subcommand_prints_a_budget() -> None:
    result = _run(["scenario", "leo_crossing"])
    assert result.returncode == 0
    for fragment in (
        "error budget",
        "relative_velocity_knowledge",
        "ephemeris_staleness_velocity",
        "light_time_iteration_residual",
        "beam context",
        "analytic lines checked against numerical propagation",
    ):
        assert fragment in result.stdout


def test_budget_subcommand_on_a_shipped_case(case_dir: Path) -> None:
    result = _run(["budget", str(case_dir / "leo_crossing.yaml")])
    assert result.returncode == 0
    assert "leo_crossing" in result.stdout
    assert "dominant term" in result.stdout


def test_compare_subcommand_scores_the_baseline(case_dir: Path) -> None:
    result = _run(["compare", str(case_dir / "geo_relay_to_leo.yaml")])
    assert result.returncode == 0
    assert "baseline vs reference" in result.stdout
    assert "offset difference (closed)" in result.stdout
    assert "transmit direction (open)" in result.stdout


def test_lighttime_subcommand_prints_the_iteration_history(case_dir: Path) -> None:
    result = _run(["lighttime", str(case_dir / "leo_trailing.yaml")])
    assert result.returncode == 0
    assert "light-time solutions" in result.stdout
    assert "iterate[0]" in result.stdout
    assert "converged" in result.stdout


def test_a_relative_case_path_works_from_the_product_root(case_dir: Path) -> None:
    relative = case_dir.relative_to(ROOT) / "leo_crossing.yaml"
    result = _run(["budget", str(relative)])
    assert result.returncode == 0


def test_a_missing_case_file_exits_non_zero_without_a_traceback() -> None:
    result = _run(["budget", "no_such_case.yaml"])
    assert result.returncode != 0
    assert "case file not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_an_invalid_case_file_names_the_offending_key(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("label: x\nnonsense: 1\n")
    result = _run(["budget", str(bad)])
    assert result.returncode != 0
    assert "invalid case file" in result.stderr
    assert "nonsense" in result.stderr


def test_an_unknown_scenario_is_refused_by_argparse() -> None:
    result = _run(["scenario", "mars_relay"])
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_no_subcommand_is_an_error() -> None:
    result = _run([])
    assert result.returncode != 0


def test_parser_is_constructible_without_a_shell() -> None:
    parser = build_parser()
    arguments = parser.parse_args(["budget", "x.yaml"])
    assert arguments.command == "budget"
    assert arguments.case == "x.yaml"


def test_main_returns_zero_in_process() -> None:
    assert main(["info"]) == 0
    assert main(["scenario", "leo_trailing"]) == 0


def test_unknown_scenario_through_the_helper_raises_system_exit() -> None:
    from pointahead.__main__ import _case_from_scenario

    with pytest.raises(SystemExit, match="unknown scenario"):
        _case_from_scenario("mars_relay")
