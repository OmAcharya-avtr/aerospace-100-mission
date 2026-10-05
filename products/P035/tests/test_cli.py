"""Tests for ``python -m telemetryool``.

Every subcommand is exercised at a size that keeps this file inside a few
seconds; the CLI's own defaults are larger and are measured in ``validation/``.
"""

from __future__ import annotations

import csv

import pytest

from telemetryool.__main__ import build_parser, main


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "telemetryool 0.1.0" in capsys.readouterr().out


def test_design_prints_all_three_methods(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["design", "--alpha", "0.05", "--window", "100"]) == 0
    out = capsys.readouterr().out
    assert "cusum" in out and "ewma" in out and "ool" in out
    assert "ARL0" in out
    assert "alpha_W = 0.050000" in out
    # The precision note must state a window count, so the reader cannot take a
    # design value on trust.
    assert "1900 independent windows" in out


def test_arl_cusum_prints_both_derivations(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["arl", "cusum", "--delta", "1.0", "--k", "0.5", "--h", "6.35"]) == 0
    out = capsys.readouterr().out
    assert "Brook & Evans 1972" in out
    assert "Siegmund 1985" in out
    assert "relative difference" in out


def test_arl_ewma_says_there_is_no_closed_form(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["arl", "ewma", "--lam", "0.2", "--limit", "3.0"]) == 0
    out = capsys.readouterr().out
    assert "Lucas & Saccucci 1990" in out
    assert "no closed form" in out


def test_far_reports_design_and_measurement(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["far", "cusum", "--windows", "2000", "--seed", "11"]) == 0
    out = capsys.readouterr().out
    assert "design  alpha_W" in out
    assert "measured alpha_W" in out
    assert "binomial SE" in out
    assert "Clopper-Pearson" in out


def test_far_warns_when_the_design_hypothesis_is_violated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["far", "ewma", "--windows", "1000", "--rho", "0.5", "--seed", "12"]) == 0
    out = capsys.readouterr().out
    assert "violates the design hypothesis" in out


def _write_csv(path, rows, header) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def test_check_reports_the_alarm_transitions(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Values 0, 3, 3, 0, 0 with soft limits +/- 2, persistence_soft 2 and
    clear_persistence 2: SOFT_ALARM raises at index 2 and clears at index 4."""
    csv_path = tmp_path / "hk.csv"
    _write_csv(csv_path, [[v] for v in (0.0, 3.0, 3.0, 0.0, 0.0)], ["BUS_V"])
    code = main(
        [
            "check", str(csv_path), "--column", "BUS_V",
            "--soft-low", "-2", "--soft-high", "2",
            "--hard-low", "-5", "--hard-high", "5",
            "--persistence-soft", "2", "--clear-persistence", "2",
            "--units", "V",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "RAISE" in out and "CLEAR" in out
    assert "final state: NOMINAL" in out


def test_check_with_validity_and_mode_columns(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "hk2.csv"
    _write_csv(
        csv_path,
        [[0.0, 1, "SAFE"], [9.0, 0, "SAFE"], [9.0, 1, "SAFE"]],
        ["T", "VALID", "MODE"],
    )
    code = main(
        [
            "check", str(csv_path), "--column", "T",
            "--valid-column", "VALID", "--mode-column", "MODE",
            "--hard-high", "5", "--all-samples",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "HARD_ALARM" in out
    # the invalid sample reports no level
    assert out.count("     -") >= 1


def test_check_reports_no_transitions_when_everything_is_in_limit(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = tmp_path / "quiet.csv"
    _write_csv(csv_path, [[0.0], [0.1]], ["X"])
    assert main(["check", str(csv_path), "--column", "X", "--hard-high", "9"]) == 0
    assert "no alarm transitions" in capsys.readouterr().out


def test_check_rejects_a_missing_column(tmp_path) -> None:
    csv_path = tmp_path / "bad.csv"
    _write_csv(csv_path, [[1.0]], ["A"])
    with pytest.raises(SystemExit, match="no column"):
        main(["check", str(csv_path), "--column", "B"])


def test_check_rejects_an_empty_file(tmp_path) -> None:
    csv_path = tmp_path / "empty.csv"
    _write_csv(csv_path, [], ["A"])
    with pytest.raises(SystemExit, match="no data rows"):
        main(["check", str(csv_path), "--column", "A"])


def test_changepoint_finds_an_injected_step(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    series = rng.standard_normal(200)
    series[120:] += 4.0
    csv_path = tmp_path / "cp.csv"
    _write_csv(csv_path, [[v] for v in series], ["X"])
    code = main(
        [
            "changepoint", str(csv_path), "--column", "X",
            "--alpha", "0.05", "--simulations", "2000", "--min-segment", "20",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "detected change points" in out
    assert "120" in out
    assert "calibrated per segment" in out


def test_compare_runs_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "compare", "--channels", "3", "--window", "60",
            "--train-windows", "200", "--cal-windows", "1000",
            "--measure-windows", "1000", "--scenario-windows", "300",
            "--confusion", "--seed", "5",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Window false-alarm probability" in out
    assert "Confusion matrix" in out
    assert "decorrelate" in out
    assert "comb SE" in out
