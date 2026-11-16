"""The command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from seedaudit.__main__ import main


def test_power_prints_the_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["power"]) == 0
    out = capsys.readouterr().out
    assert "detection power at p =" in out
    assert "p_max(n) clean" in out
    assert "0.776393202250" in out


def test_power_accepts_a_rate_and_an_alpha(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["power", "--rate", "0.5", "--alpha", "0.01"]) == 0
    out = capsys.readouterr().out
    assert "0.5" in out


def test_power_states_the_audit_model(capsys: pytest.CaptureFixture[str]) -> None:
    main(["power"])
    assert "escaped-rate upper bound" in capsys.readouterr().out


def test_static_reports_coverage_and_findings(
    polluting_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["static", str(polluting_tree)]) == 0
    out = capsys.readouterr().out
    assert "files parsed:" in out
    assert "self_polluting.py" in out


def test_static_flags_the_self_polluting_corpus_script(
    polluting_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["static", str(polluting_tree)])
    assert "write_under_glob" in capsys.readouterr().out


def test_arrangements_classifies_the_self_polluting_check(
    polluting_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "arrangements",
            str(polluting_tree),
            "self_polluting.py",
            "--artifact",
            "self_polluting_output.txt",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "classification: self_polluting_check" in out
    assert "escaped-rate upper bound" in out


def test_arrangements_is_silent_on_a_control(
    polluting_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["arrangements", str(polluting_tree), "deterministic_control.py"])
    assert "classification: no_evidence" in capsys.readouterr().out


def test_corpus_prints_a_row_per_rate(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["corpus", "--runs", "8", "--repeats", "2"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n") >= 8
    assert "derived" in out
    assert "0.022727" in out


def test_audit_reports_the_headline(
    fixture_repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["audit", str(fixture_repository), "--workers", "1", "--timeout", "60"]) == 0
    out = capsys.readouterr().out
    assert "HEADLINE" in out
    assert "COVERAGE" in out


def test_audit_exit_status_is_zero_even_with_a_finding(
    fixture_repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A finding is not an error. Exiting non-zero on one would push a user
    # towards not running the tool.
    code = main(["audit", str(fixture_repository), "--workers", "1", "--timeout", "60"])
    capsys.readouterr()
    assert code == 0


def test_audit_honours_the_limit(
    fixture_repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "audit",
            str(fixture_repository),
            "--workers",
            "1",
            "--timeout",
            "60",
            "--limit",
            "1",
        ]
    )
    assert "candidates attempted: 1" in capsys.readouterr().out


def test_no_subcommand_is_an_error() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_an_unknown_subcommand_is_an_error() -> None:
    with pytest.raises(SystemExit):
        main(["nonsense"])
