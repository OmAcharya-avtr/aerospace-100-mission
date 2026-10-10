"""CLI tests.

Every case that exercises a non-zero exit status does so through
``subprocess.run`` and asserts on ``returncode``; nothing here lets a non-zero
status propagate out of the test process.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from traceaudit.__main__ import EXIT_FINDINGS, EXIT_OK, EXIT_USAGE, build_parser, main

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _run(args, cwd):
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "traceaudit", *args],
        capture_output=True, text=True, env=env, cwd=cwd, timeout=300, check=False,
    )


def _call(args):
    buffer = io.StringIO()
    code = main(args, out=buffer)
    return code, buffer.getvalue()


@pytest.fixture
def sample_args(sample_project):
    return [
        "audit",
        "-r", str(sample_project / "docs" / "REQUIREMENTS.md"),
        "-t", str(sample_project / "junit.xml"),
        "--test-root", str(sample_project / "tests"),
    ]


@pytest.mark.verifies("REQ-020")
def test_help_exits_zero_in_a_subprocess(product_root):
    out = _run(["--help"], cwd=product_root)
    assert out.returncode == 0
    assert "requirements-to-test traceability" in out.stdout


@pytest.mark.verifies("REQ-020")
def test_version_exits_zero_in_a_subprocess(product_root):
    out = _run(["--version"], cwd=product_root)
    assert out.returncode == 0
    assert out.stdout.strip().startswith("traceaudit 0.")


@pytest.mark.verifies("REQ-020")
def test_audit_on_the_defective_fixture_exits_one(product_root):
    out = _run(
        ["audit", "-r", "fixtures/sample_project/docs/REQUIREMENTS.md",
         "-t", "fixtures/sample_project/junit.xml",
         "--test-root", "fixtures/sample_project/tests"],
        cwd=product_root,
    )
    assert out.returncode == EXIT_FINDINGS
    assert "7 finding(s), 7 blocking; exit status 1" in out.stdout


@pytest.mark.verifies("REQ-020")
def test_audit_on_a_clean_input_exits_zero(tmp_path, product_root):
    (tmp_path / "REQUIREMENTS.md").write_text("## REQ-001 A\n", encoding="utf-8")
    (tmp_path / "j.xml").write_text(
        '<testsuite tests="1"><testcase classname="tests.test_a" name="test_a" '
        'file="tests/test_a.py"><properties>'
        '<property name="requirement_id" value="REQ-001"/>'
        "</properties></testcase></testsuite>",
        encoding="utf-8",
    )
    out = _run(["audit", "-r", str(tmp_path / "REQUIREMENTS.md"),
                "-t", str(tmp_path / "j.xml"), "--no-heuristic"], cwd=product_root)
    assert out.returncode == EXIT_OK
    assert "0 finding(s), 0 blocking; exit status 0" in out.stdout


@pytest.mark.verifies("REQ-020")
def test_a_missing_requirements_file_exits_two(product_root, tmp_path):
    out = _run(["audit", "-r", str(tmp_path / "absent.md"),
                "-t", "fixtures/sample_project/junit.xml"], cwd=product_root)
    assert out.returncode == EXIT_USAGE
    assert "requirements document not found" in out.stderr


@pytest.mark.verifies("REQ-020")
def test_a_missing_report_exits_two(product_root, tmp_path):
    out = _run(["audit", "-r", "fixtures/sample_project/docs/REQUIREMENTS.md",
                "-t", str(tmp_path / "absent.xml")], cwd=product_root)
    assert out.returncode == EXIT_USAGE
    assert "junit XML report not found" in out.stderr


@pytest.mark.verifies("REQ-020")
def test_an_unparseable_report_exits_two(product_root, tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<testsuite>", encoding="utf-8")
    out = _run(["audit", "-r", "fixtures/sample_project/docs/REQUIREMENTS.md",
                "-t", str(bad)], cwd=product_root)
    assert out.returncode == EXIT_USAGE
    assert "not parseable XML" in out.stderr


@pytest.mark.verifies("REQ-020")
def test_a_missing_subcommand_exits_two(product_root):
    out = _run([], cwd=product_root)
    assert out.returncode == EXIT_USAGE


@pytest.mark.verifies("REQ-019")
def test_ignoring_every_code_gives_exit_zero_on_the_defective_fixture(product_root):
    out = _run(
        ["audit", "-r", "fixtures/sample_project/docs/REQUIREMENTS.md",
         "-t", "fixtures/sample_project/junit.xml",
         "--test-root", "fixtures/sample_project/tests",
         "--ignore", "TA001", "--ignore", "TA002", "--ignore", "TA003",
         "--ignore", "TA004", "--ignore", "TA005", "--ignore", "TA006"],
        cwd=product_root,
    )
    assert out.returncode == EXIT_OK
    assert "7 finding(s), 0 blocking; exit status 0" in out.stdout


@pytest.mark.verifies("REQ-019")
def test_an_unknown_ignore_code_exits_two(product_root):
    out = _run(["audit", "-r", "fixtures/sample_project/docs/REQUIREMENTS.md",
                "-t", "fixtures/sample_project/junit.xml", "--ignore", "TA999"],
               cwd=product_root)
    assert out.returncode == EXIT_USAGE
    assert "unknown finding code(s) to ignore: TA999" in out.stderr


@pytest.mark.verifies("REQ-023")
def test_json_output_parses_and_carries_the_exit_code(sample_args):
    code, text = _call([*sample_args, "--json"])
    assert code == EXIT_FINDINGS
    data = json.loads(text)
    assert data["exit_code"] == 1
    assert data["counts_by_code"]["TA004"] == 2


@pytest.mark.verifies("REQ-024")
def test_markdown_matrix_output_is_a_table(sample_args):
    code, text = _call([*sample_args, "--markdown-matrix"])
    assert code == EXIT_FINDINGS
    assert text.splitlines()[0] == "| requirement | title | tests | outcomes |"


@pytest.mark.verifies("REQ-010")
def test_show_matrix_prints_the_mapping(sample_args):
    code, text = _call([*sample_args, "--show-matrix"])
    assert code == EXIT_FINDINGS
    assert "requirement -> tests" in text


@pytest.mark.verifies("REQ-001")
def test_requirements_subcommand_lists_declarations(product_root):
    code, text = _call(["requirements", "-r",
                        str(product_root / "docs" / "REQUIREMENTS.md")])
    assert code == EXIT_OK
    assert "26 declaration(s), 26 unique id(s)" in text


@pytest.mark.verifies("REQ-004")
def test_requirements_subcommand_reports_a_duplicate(sample_project):
    code, text = _call(["requirements", "-r",
                        str(sample_project / "docs" / "REQUIREMENTS.md")])
    assert code == EXIT_OK
    assert "duplicate: REQ-007 declared 2 times" in text


@pytest.mark.verifies("REQ-023")
def test_requirements_subcommand_json(sample_project):
    code, text = _call(["requirements", "-r",
                        str(sample_project / "docs" / "REQUIREMENTS.md"), "--json"])
    assert code == EXIT_OK
    data = json.loads(text)
    assert len(data["declarations"]) == 9
    assert data["duplicates"] == {"REQ-007": 2}


@pytest.mark.verifies("REQ-005")
def test_tests_subcommand_lists_outcomes_and_claims(sample_project):
    code, text = _call(["tests", "-t", str(sample_project / "junit.xml")])
    assert code == EXIT_OK
    assert "skipped  tests/test_monitor.py::test_desaturation_threshold" in text
    assert "xfailed  tests/test_monitor.py::test_hysteresis_band" in text


@pytest.mark.verifies("REQ-023")
def test_tests_subcommand_json(sample_project):
    code, text = _call(["tests", "-t", str(sample_project / "junit.xml"), "--json"])
    assert code == EXIT_OK
    data = json.loads(text)
    assert data["outcome_counts"]["xfailed"] == 1
    assert len(data["cases"]) == 8


@pytest.mark.verifies("REQ-008")
def test_tests_subcommand_accepts_a_collection_report(sample_project):
    code, text = _call(["tests", "-t", str(sample_project / "collect_only.txt")])
    assert code == EXIT_OK
    assert "collect-only" in text
    assert text.count("unknown ") == 8


@pytest.mark.verifies("REQ-018")
def test_assertions_subcommand_lists_verdicts(heuristic_corpus):
    code, text = _call(["assertions", "--test-root", str(heuristic_corpus)])
    assert code == EXIT_OK
    assert "NO ASSERTION" in text
    assert "HEURISTIC: calls are not followed" in text


@pytest.mark.verifies("REQ-023")
def test_assertions_subcommand_json(heuristic_corpus):
    code, text = _call(["assertions", "--test-root", str(heuristic_corpus), "--json"])
    assert code == EXIT_OK
    data = json.loads(text)
    tests = [f for f in data["functions"] if f["function"].startswith("test_")]
    assert len(tests) == 23
    assert sum(1 for f in tests if not f["has_assertion"]) == 8


@pytest.mark.verifies("REQ-013")
def test_codes_subcommand_lists_every_code():
    code, text = _call(["codes"])
    assert code == EXIT_OK
    for taco in ("TA001", "TA002", "TA003", "TA004", "TA005", "TA006"):
        assert taco in text
    assert "[heuristic]" in text


@pytest.mark.verifies("REQ-021")
def test_config_template_subcommand_round_trips():
    from traceaudit import TraceConfig

    code, text = _call(["config-template"])
    assert code == EXIT_OK
    assert TraceConfig.from_dict(json.loads(text)) == TraceConfig.default()


@pytest.mark.verifies("REQ-021")
def test_a_configuration_file_is_honoured(tmp_path, sample_project):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"ignored_codes": ["TA006"], "heuristic_assertions": True}),
                   encoding="utf-8")
    code, text = _call([
        "audit",
        "-r", str(sample_project / "docs" / "REQUIREMENTS.md"),
        "-t", str(sample_project / "junit.xml"),
        "--test-root", str(sample_project / "tests"),
        "-c", str(cfg),
    ])
    assert code == EXIT_FINDINGS
    assert "7 finding(s), 6 blocking" in text


@pytest.mark.verifies("REQ-021")
def test_an_unknown_configuration_key_in_a_file_exits_two(tmp_path, product_root):
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"nope": 1}', encoding="utf-8")
    out = _run(["audit", "-r", "fixtures/sample_project/docs/REQUIREMENTS.md",
                "-t", "fixtures/sample_project/junit.xml", "-c", str(cfg)],
               cwd=product_root)
    assert out.returncode == EXIT_USAGE
    assert "unknown configuration key(s): nope" in out.stderr


@pytest.mark.verifies("REQ-001")
def test_several_requirements_documents_are_merged(tmp_path, product_root):
    (tmp_path / "a.md").write_text("## REQ-001 A\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("## REQ-002 B\n", encoding="utf-8")
    code, text = _call(["requirements", "-r", str(tmp_path / "a.md"),
                        "-r", str(tmp_path / "b.md")])
    assert code == EXIT_OK
    assert "2 declaration(s), 2 unique id(s)" in text
    _ = product_root


@pytest.mark.verifies("REQ-022")
def test_relative_to_keeps_absolute_paths_out_of_the_output(product_root, sample_project):
    code, text = _call([
        "audit",
        "-r", str(sample_project / "docs" / "REQUIREMENTS.md"),
        "-t", str(sample_project / "junit.xml"),
        "--relative-to", str(product_root),
        "--no-heuristic",
    ])
    assert code == EXIT_FINDINGS
    assert "fixtures/sample_project/docs/REQUIREMENTS.md" in text
    assert str(product_root) not in text


@pytest.mark.verifies("REQ-020")
def test_the_parser_exposes_every_subcommand():
    parser = build_parser()
    actions = [a for a in parser._subparsers._group_actions if hasattr(a, "choices")]
    assert set(actions[0].choices) == {
        "audit", "requirements", "tests", "assertions", "codes", "config-template"
    }
