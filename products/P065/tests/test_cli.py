"""CLI tests. Every assertion on an exit code runs a real subprocess."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hazardtrace.__main__ import EXIT_CANNOT_RUN, EXIT_FINDINGS, EXIT_OK, build_parser, main

ROOT = Path(__file__).resolve().parent.parent
PLANTED = "examples/cases/planted_gaps.yaml"
WORKED = "examples/cases/p024_bdot_detumble.yaml"
CLEAN = "examples/cases/complete_minimal.yaml"
PLANTED_REQ = "fixtures/planted/docs/REQUIREMENTS.md"
PLANTED_JUNIT = "fixtures/planted/junit.xml"


def run(*argv: str) -> subprocess.CompletedProcess[str]:
    """Run ``python -m hazardtrace`` as a subprocess from the repository root."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "-m", "hazardtrace", *argv],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=180,
    )


def test_help_exits_zero():
    assert run("--help").returncode == EXIT_OK


def test_version_prints_the_package_version():
    result = run("--version")
    assert result.returncode == EXIT_OK
    assert "hazardtrace 0.1.0" in result.stdout


def test_no_subcommand_is_a_usage_error():
    assert run().returncode == 2


def test_codes_lists_every_check():
    result = run("codes")
    assert result.returncode == EXIT_OK
    assert "HT016" in result.stdout


def test_check_on_a_clean_case_exits_zero():
    result = run("check", CLEAN)
    assert result.returncode == EXIT_OK
    assert "COMPLETE" in result.stdout
    assert "0 error, 0 warning" in result.stdout


def test_check_on_the_planted_case_exits_one():
    result = run("check", PLANTED, "-r", PLANTED_REQ, "-t", PLANTED_JUNIT)
    assert result.returncode == EXIT_FINDINGS
    assert "14 error, 2 warning" in result.stdout
    for code in (f"HT{n:03d}" for n in range(1, 17)):
        assert code in result.stdout


def test_check_on_the_worked_example_exits_one():
    result = run(
        "check", WORKED,
        "-r", "fixtures/p024_bdot/docs/REQUIREMENTS.md",
        "-t", "fixtures/p024_bdot/junit.xml",
    )
    assert result.returncode == EXIT_FINDINGS
    assert "11 error, 0 warning" in result.stdout


def test_strict_promotes_warnings_to_blocking():
    lenient = run("check", CLEAN)
    assert lenient.returncode == EXIT_OK
    # complete_minimal has no warnings, so --strict must not change its verdict.
    strict = run("check", CLEAN, "--strict")
    assert strict.returncode == EXIT_OK


def test_strict_changes_the_verdict_on_a_warning_only_case(tmp_path):
    case = tmp_path / "warning_only.yaml"
    case.write_text(
        "meta:\n  title: warning only\nlosses:\n  - id: L-1\n    description: d\n",
        encoding="utf-8",
    )
    assert run("check", str(case)).returncode == EXIT_OK
    assert run("check", str(case), "--strict").returncode == EXIT_FINDINGS


def test_json_output_is_parseable():
    import json

    result = run("check", PLANTED, "-r", PLANTED_REQ, "-t", PLANTED_JUNIT, "-f", "json")
    assert result.returncode == EXIT_FINDINGS
    payload = json.loads(result.stdout)
    assert len(payload["findings"]) == 16


def test_markdown_output():
    result = run("check", PLANTED, "-f", "markdown")
    assert "## Enumeration grid" in result.stdout


def test_grid_subcommand_exits_zero_even_with_gaps():
    result = run("grid", PLANTED)
    assert result.returncode == EXIT_OK
    assert "cells covered: 7/8" in result.stdout


def test_coverage_subcommand():
    result = run("coverage", WORKED, "-r", "fixtures/p024_bdot/docs/REQUIREMENTS.md")
    assert result.returncode == EXIT_OK
    assert "8/10 = 80.0000 %" in result.stdout


def test_diagram_writes_mermaid_and_png(tmp_path):
    mmd = tmp_path / "structure.mmd"
    png = tmp_path / "structure.png"
    result = run("diagram", WORKED, "--mermaid", str(mmd), "--png", str(png))
    assert result.returncode == EXIT_OK
    assert mmd.read_text(encoding="utf-8").startswith("flowchart TD")
    assert png.stat().st_size > 10_000


def test_diagram_to_stdout():
    result = run("diagram", WORKED, "--mermaid", "-")
    assert result.returncode == EXIT_OK
    assert result.stdout.startswith("flowchart TD")


def test_diagram_without_an_output_cannot_run():
    result = run("diagram", WORKED)
    assert result.returncode == EXIT_CANNOT_RUN
    assert "needs --mermaid or --png" in result.stderr


def test_missing_analysis_file_exits_two():
    result = run("check", "examples/cases/absent.yaml")
    assert result.returncode == EXIT_CANNOT_RUN
    assert "no such file" in result.stderr


def test_malformed_analysis_exits_two(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("meta:\n  title: t\nlosses:\n  - id: L-1\n", encoding="utf-8")
    result = run("check", str(bad))
    assert result.returncode == EXIT_CANNOT_RUN
    assert "losses[0]" in result.stderr


def test_missing_requirements_document_exits_two():
    result = run("check", CLEAN, "-r", "docs/absent.md")
    assert result.returncode == EXIT_CANNOT_RUN


def test_missing_junit_report_exits_two():
    result = run("check", CLEAN, "-t", "absent.xml")
    assert result.returncode == EXIT_CANNOT_RUN


def test_custom_requirement_pattern(tmp_path):
    doc = tmp_path / "req.md"
    doc.write_text("### SR-1 something\n", encoding="utf-8")
    case = tmp_path / "case.yaml"
    case.write_text(
        "meta:\n  title: t\nlosses:\n  - id: L-1\n    description: d\n"
        "hazards:\n  - id: H-1\n    description: d\n    losses: [L-1]\n"
        "constraints:\n  - id: SC-1\n    description: d\n    hazards: [H-1]\n"
        "    requirements: [SR-1]\n",
        encoding="utf-8",
    )
    result = run(
        "check", str(case), "-r", str(doc), "--requirement-pattern", r"SR-\d+"
    )
    assert result.returncode == EXIT_OK


def test_main_is_callable_in_process(capsys):
    assert main(["codes"]) == EXIT_OK
    assert "HT001" in capsys.readouterr().out


def test_main_returns_one_on_findings(capsys):
    code = main(["check", str(ROOT / PLANTED)])
    capsys.readouterr()
    assert code == EXIT_FINDINGS


def test_parser_rejects_an_unknown_format():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["check", "x.yaml", "-f", "latex"])
