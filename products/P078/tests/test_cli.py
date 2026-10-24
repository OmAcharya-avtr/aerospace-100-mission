"""The CLI, including the exit codes, which are the contract a pipeline reads."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import CASE_DIR

from cmdguard.__main__ import (
    EXIT_CLEAN,
    EXIT_GATE_REFUSED,
    EXIT_INPUT_ERROR,
    EXIT_VIOLATIONS,
    main,
)


def _case(name: str) -> str:
    return str(CASE_DIR / f"{name}.yaml")


def test_version_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert "cmdguard 0.1.0" in capsys.readouterr().out


def test_info_lists_kinds_families_and_features(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == EXIT_CLEAN
    out = capsys.readouterr().out
    assert "irreversible_unverified" in out
    assert "swap_enabler" in out
    assert "draw_fraction_of_ceiling" in out
    assert "exit codes: 0 nothing found, 1 violations, 2 gate refused, 3 input error" in out
    assert "not a statement\nthat a sequence is safe to uplink" in out or (
        "safe to uplink" in out
    )


def test_info_on_a_case_prints_the_declarations(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", _case("irreversible_before_verify")]) == EXIT_CLEAN
    out = capsys.readouterr().out
    assert "irreversible commands    : 1" in out


def test_check_clean_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", _case("nominal_science_pass")]) == EXIT_CLEAN
    assert "violations            : 0" in capsys.readouterr().out


def test_check_with_violations_exits_one(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", _case("late_payload_enable")]) == EXIT_VIOLATIONS
    out = capsys.readouterr().out
    assert "precondition_false" in out
    assert "because step 5" in out


def test_check_gate_refusal_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", _case("irreversible_before_verify")]) == EXIT_GATE_REFUSED
    out = capsys.readouterr().out
    assert "REFUSED" in out


def test_check_state_at(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", _case("late_payload_enable"), "--state-at", "5"]) == EXIT_VIOLATIONS
    assert "state after step 5" in capsys.readouterr().out


def test_check_state_at_out_of_range_exits_three() -> None:
    assert main(["check", _case("late_payload_enable"), "--state-at", "99"]) == EXIT_INPUT_ERROR


def test_check_block_policy(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", _case("late_payload_enable"), "--policy", "block"]) == EXIT_VIOLATIONS
    assert "effect policy         : block" in capsys.readouterr().out


def test_lint_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["lint", _case("late_payload_enable")]) == EXIT_VIOLATIONS
    out = capsys.readouterr().out
    assert "static lint (per_command)" in out


def test_lint_clean_message(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["lint", _case("irreversible_after_verify"), "--strength", "with_timing"])
    out = capsys.readouterr().out
    if code == EXIT_CLEAN:
        assert "The lint found nothing" in out
    else:
        assert "findings:" in out


def test_compare_names_what_the_lint_missed(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare", _case("resource_and_exclusion")]) == EXIT_VIOLATIONS
    out = capsys.readouterr().out
    assert "lint miss rate on this case" in out
    assert "missed by the lint:" in out


def test_gate_subcommand_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["gate", _case("irreversible_before_verify")]) == EXIT_GATE_REFUSED
    assert "REFUSE" in capsys.readouterr().out
    assert main(["gate", _case("irreversible_after_verify")]) == EXIT_CLEAN
    assert "NO_REFUSAL_FOUND" in capsys.readouterr().out


def test_missing_file_exits_three(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(tmp_path / "nope.yaml")]) == EXIT_INPUT_ERROR
    assert "cannot load case" in capsys.readouterr().err


def test_broken_declaration_exits_three(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", _case("missing_declarations")]) == EXIT_INPUT_ERROR
    assert "cannot load case" in capsys.readouterr().err


def test_rank_validates_arguments() -> None:
    assert main(["rank", _case("nominal_science_pass"), "--k", "0"]) == EXIT_INPUT_ERROR
    assert main(["rank", _case("nominal_science_pass"), "--train-seeds", "0"]) == EXIT_INPUT_ERROR


def test_rank_prints_a_ranking(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["rank", _case("resource_and_exclusion"), "--k", "4", "--train-seeds", "4"])
    out = capsys.readouterr().out
    assert code == EXIT_VIOLATIONS
    assert "rank step        t_s command                risk  spread  involved" in out
    assert "not\na calibrated confidence interval" in out or "calibrated" in out


def test_benchmark_runs_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["benchmark", "--train-seeds", "4", "--test-seeds", "3"]) == EXIT_CLEAN
    out = capsys.readouterr().out
    assert "static lint miss rate (per_command):" in out
    assert "precision at k on held-out cases:" in out
    assert "95% CI" in out
