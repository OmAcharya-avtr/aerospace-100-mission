"""Tests for ``python -m assuregraph``, including the nonzero exit on an incomplete case."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import pytest
from conftest import COMPLETE_CASE, CYCLIC_CASE, INCOMPLETE_CASE, REPO_ROOT, document, edge, node

from assuregraph.__main__ import EXIT_INCOMPLETE, EXIT_OK, EXIT_USAGE, build_parser, main


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0


@pytest.mark.parametrize("command", ["check", "mermaid", "summary"])
def test_subcommand_help_exits_zero(command: str) -> None:
    with pytest.raises(SystemExit) as info:
        main([command, "--help"])
    assert info.value.code == 0


def test_version_exits_zero() -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0


def test_no_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as info:
        main([])
    assert info.value.code == EXIT_USAGE


def test_check_on_the_complete_case_exits_zero() -> None:
    code, out, err = _run(["check", COMPLETE_CASE])
    assert code == EXIT_OK
    assert "COMPLETE" in out
    assert err == ""


def test_check_on_the_incomplete_case_exits_one() -> None:
    # This is the documented behaviour a CI job keys on.
    code, out, _ = _run(["check", INCOMPLETE_CASE])
    assert code == EXIT_INCOMPLETE
    assert "INCOMPLETE" in out


def test_check_on_the_cyclic_case_exits_one() -> None:
    code, out, _ = _run(["check", CYCLIC_CASE])
    assert code == EXIT_INCOMPLETE
    assert "cycle" in out


def test_strict_turns_a_warning_into_a_nonzero_exit(tmp_path, artifact, write_case) -> None:
    artifact("a.txt")
    doc = document(
        [node("G1", "goal"), node("Sn1", "solution", evidence={"path": str(tmp_path / "a.txt")})],
        [edge("G1", "Sn1")],
        top_goals=["G1"],
    )
    path = write_case(doc)
    assert _run(["check", path])[0] == EXIT_OK
    assert _run(["check", path, "--strict"])[0] == EXIT_INCOMPLETE


def test_missing_file_is_exit_two_not_one() -> None:
    # A malformed or unreadable case is a different problem from an incomplete
    # one, and the exit codes keep them apart.
    code, out, err = _run(["check", "/nonexistent/case.yaml"])
    assert code == EXIT_USAGE
    assert out == ""
    assert "cannot read case file" in err


def test_malformed_case_is_exit_two(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nnodes: []\n", encoding="utf-8")
    code, _, err = _run(["check", str(path)])
    assert code == EXIT_USAGE
    assert "at least one Goal" in err


def test_invalid_yaml_is_exit_two(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: [oops\n", encoding="utf-8")
    assert _run(["check", str(path)])[0] == EXIT_USAGE


def test_json_output_parses_and_carries_the_exit_code() -> None:
    code, out, _ = _run(["check", INCOMPLETE_CASE, "--json"])
    payload = json.loads(out)
    assert payload["exit_code"] == code == EXIT_INCOMPLETE
    assert payload["counts"]["error"] == 6
    assert payload["counts"]["warning"] == 1


def test_json_and_markdown_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as info:
        main(["check", COMPLETE_CASE, "--json", "--markdown"])
    assert info.value.code == EXIT_USAGE


def test_markdown_output_is_markdown() -> None:
    _, out, _ = _run(["check", COMPLETE_CASE, "--markdown"])
    assert out.startswith("# assuregraph report")


def test_mermaid_exits_zero_even_on_an_incomplete_case() -> None:
    code, out, _ = _run(["mermaid", INCOMPLETE_CASE])
    assert code == EXIT_OK
    assert out.startswith("flowchart TD")


def test_mermaid_direction_and_fence() -> None:
    _, out, _ = _run(["mermaid", COMPLETE_CASE, "--direction", "LR", "--fence"])
    assert out.startswith("```mermaid\nflowchart LR")


def test_mermaid_highlight_marks_findings() -> None:
    _, out, _ = _run(["mermaid", INCOMPLETE_CASE, "--highlight"])
    assert "gsnFinding;" in out.split("classDef gsnFinding")[-1]


def test_mermaid_writes_to_a_file(tmp_path) -> None:
    target = tmp_path / "case.mmd"
    code, out, _ = _run(["mermaid", COMPLETE_CASE, "-o", str(target)])
    assert code == EXIT_OK
    assert out == ""
    assert target.read_text(encoding="utf-8").startswith("flowchart TD")


def test_mermaid_output_to_an_unwritable_path_is_exit_two(tmp_path) -> None:
    code, _, err = _run(["mermaid", COMPLETE_CASE, "-o", str(tmp_path / "no" / "dir" / "x.mmd")])
    assert code == EXIT_USAGE
    assert err != ""


def test_summary_prints_the_coverage_table_only() -> None:
    code, out, _ = _run(["summary", COMPLETE_CASE])
    assert code == EXIT_OK
    assert "Coverage, with denominators" in out
    assert "Verdict" not in out


def test_summary_markdown() -> None:
    _, out, _ = _run(["summary", COMPLETE_CASE, "--markdown"])
    assert "| prefix | GSN element | count |" in out


def test_epilog_carries_the_scope_statement() -> None:
    assert "not a safe system" in (build_parser().epilog or "")


def test_scope_statement_is_in_the_help_text() -> None:
    assert "not flight-qualified" in build_parser().format_help()


def test_subprocess_help_exits_zero() -> None:
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO_ROOT, "src"))
    result = subprocess.run(
        [sys.executable, "-m", "assuregraph", "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "assurance case" in result.stdout


def test_subprocess_exit_codes_on_the_shipped_cases() -> None:
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO_ROOT, "src"))
    codes = {}
    for label, path in (("complete", COMPLETE_CASE), ("incomplete", INCOMPLETE_CASE)):
        result = subprocess.run(
            [sys.executable, "-m", "assuregraph", "check", path],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        codes[label] = result.returncode
    assert codes == {"complete": 0, "incomplete": 1}
