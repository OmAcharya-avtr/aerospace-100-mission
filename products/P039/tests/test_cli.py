"""The ``python -m latencynet`` command line."""

from __future__ import annotations

import pytest

from latencynet.__main__ import build_parser, main


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    assert "sum-of-stages" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert "latencynet 0.1.0" in capsys.readouterr().out


def test_subcommand_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_analytic_subcommand(capsys):
    code = main(["analytic", "--mean-us", "60", "180", "30", "--std-us", "12", "40", "6"])
    out = capsys.readouterr().out
    assert code == 0
    # Hand-calculated: 60 + 180 + 30 = 270 us exactly.
    assert "270.000 us" in out
    # sqrt(12^2 + 40^2 + 6^2) = sqrt(144 + 1600 + 36) = sqrt(1780) = 42.190 us.
    assert "42.190 us" in out
    assert "90 % PI" in out


def test_analytic_with_covariance(capsys):
    code = main(
        [
            "analytic",
            "--mean-us", "60", "180",
            "--std-us", "12", "40",
            "--rho", "0.8",
            "--use-covariance",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "analytic_sum_cov" in out
    # Means still add to 240 us exactly; dependence changes only the spread.
    assert "240.000 us" in out


def test_analytic_rejects_mismatched_lengths():
    with pytest.raises(SystemExit, match="same number of values"):
        main(["analytic", "--mean-us", "60", "180", "--std-us", "12"])


def test_sample_subcommand(capsys):
    code = main(
        ["sample", "--mean-us", "60", "180", "30", "--std-us", "12", "40", "6", "--n", "4000"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "injected mean" in out
    assert "sampled  mean" in out
    assert "rel diff" in out


def test_tail_subcommand(capsys):
    code = main(["tail", "--p", "0.999", "--n", "20000"])
    assert code == 0
    out = capsys.readouterr().out
    assert "order-statistic tail uncertainty" in out
    # Smallest interior n for p = 0.999 is ceil(1/0.001) = 1000.
    assert "1000" in out


def test_compare_subcommand(capsys):
    code = main(
        [
            "compare",
            "--regime", "independent",
            "--n-train", "26",
            "--n-calibration", "10",
            "--n-test", "10",
            "--n-probe", "32",
            "--n-reference", "2000",
        ]
    )
    assert code == 0
    assert "winner by mean absolute log error" in capsys.readouterr().out


def test_bad_input_returns_exit_code_two(capsys):
    code = main(["analytic", "--mean-us", "-1", "--std-us", "1"])
    assert code == 2
    assert "error:" in capsys.readouterr().err
