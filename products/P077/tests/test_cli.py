"""The command line, exercised through ``main`` with explicit argument lists."""

from __future__ import annotations

import pytest

from relfiledelivery.__main__ import build_parser, main


def test_version_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "relfiledelivery 0.1.0" in capsys.readouterr().out


def test_info_names_the_mature_alternatives(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "cfdp-py" in out
    assert "spacepackets" in out
    assert "NOT a CFDP implementation" in out
    assert "Not flight-qualified" in out


def test_info_lists_every_mode_and_every_modelling_choice(
    capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    for mode in ("unacknowledged", "deferred_nak", "immediate_nak"):
        assert mode in out
    for choice in ("M1", "M2", "M3", "M4", "M5", "M6", "M7"):
        assert choice in out


def test_analytic_reports_the_closed_form(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["analytic", "--file-bytes", "20480", "--loss", "0.1"]) == 0
    out = capsys.readouterr().out
    assert "E[rounds]" in out
    assert "series tail bound" in out


def test_analytic_refuses_an_outage_link() -> None:
    with pytest.raises(ValueError, match="assumes no outages"):
        main(["analytic", "--mean-up-s", "10", "--mean-down-s", "5"])


def test_simulate_reports_one_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["simulate", "--file-bytes", "20480", "--seed", "99"]) == 0
    out = capsys.readouterr().out
    assert "seed                    : 99" in out
    assert "source digest" in out


def test_distribution_appends_the_closed_form_check_in_the_clean_regime(
    capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["distribution", "--file-bytes", "20480", "--runs", "60"]) == 0
    out = capsys.readouterr().out
    assert "tail, stated separately" in out
    assert "difference / std error" in out


def test_distribution_omits_the_closed_form_check_with_outages(
    capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "distribution",
            "--file-bytes", "20480",
            "--runs", "40",
            "--mean-up-s", "6",
            "--mean-down-s", "3",
        ]
    )
    assert code == 0
    assert "difference / std error" not in capsys.readouterr().out


def test_compare_reports_every_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare", "--file-bytes", "20480", "--runs", "40"]) == 0
    out = capsys.readouterr().out
    for mode in ("unacknowledged", "deferred_nak", "immediate_nak"):
        assert mode in out


def test_windows_can_be_declared_on_the_command_line(
    capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "simulate",
            "--file-bytes", "20480",
            "--windows", "4",
            "--window-duration-s", "30",
            "--window-period-s", "60",
        ]
    )
    assert code == 0
    assert "4 windows" in capsys.readouterr().out


def test_outage_arguments_must_be_given_together() -> None:
    with pytest.raises(ValueError, match="must be given together"):
        main(["simulate", "--mean-up-s", "10"])


def test_an_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown transfer mode"):
        main(["simulate", "--mode", "selective_repeat"])


def test_a_missing_subcommand_exits_non_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code != 0


def test_parser_declares_every_subcommand() -> None:
    parser = build_parser()
    actions = [
        action for action in parser._subparsers._actions if hasattr(action, "choices")
    ]
    choices: set[str] = set()
    for action in actions:
        if action.choices:
            choices.update(action.choices)
    assert {"info", "analytic", "simulate", "distribution", "compare", "benchmark"} <= choices


def test_benchmark_runs_at_a_tiny_size(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["benchmark", "--train-rows", "60", "--test-rows", "40", "--seed", "5150"])
    assert code == 0
    out = capsys.readouterr().out
    assert "predictor benchmark" in out
    assert "learned_quantile" in out
    assert "analytic" in out
