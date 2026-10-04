"""CLI tests. Every subcommand must exit 0 and print the numbers it promises."""

from __future__ import annotations

import pytest

from rtclock.__main__ import build_parser, main


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_version_exits_zero():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0


def test_no_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([])
    assert exc.value.code == 2


def test_resolution_subcommand(capsys):
    assert main(["resolution", "--samples", "3000"]) == 0
    out = capsys.readouterr().out
    assert "measured observable tick" in out
    assert "standard duration uncert." in out


def test_resolution_on_perf_counter(capsys):
    assert main(["resolution", "--samples", "2000", "--clock", "perf_counter"]) == 0
    assert "perf_counter" in capsys.readouterr().out


def test_resolution_rejects_too_few_samples(capsys):
    assert main(["resolution", "--samples", "1"]) == 1
    assert "samples must be >= 2" in capsys.readouterr().err


def test_bound_subcommand_prints_the_textbook_values(capsys):
    assert main(["bound", "1", "2", "3", "10"]) == 0
    out = capsys.readouterr().out
    assert "1.000000000000" in out
    assert "0.828427124746" in out
    assert "0.779763149685" in out
    assert "0.717734625363" in out
    assert "0.693147180560" in out  # ln 2


def test_bound_rejects_zero(capsys):
    assert main(["bound", "0"]) == 1
    assert "n must be >= 1" in capsys.readouterr().err


def test_analyse_subcommand_reproduces_the_hand_computation(capsys):
    assert main(["analyse", "t1:7:3", "t2:12:3", "t3:20:5"]) == 0
    out = capsys.readouterr().out
    assert "0.928571428571" in out
    assert "20.000000000" in out
    assert "all deadlines met" in out


def test_analyse_with_blocking_reports_the_miss(capsys):
    code = main(
        [
            "analyse",
            "t1:7:3",
            "t2:12:3",
            "t3:20:5",
            "t4:50:5",
            "--critical-section",
            "t4:s1:1.0",
            "--critical-section",
            "t3:s1:0.001",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "21.000000000" in out
    assert "at least one deadline missed" in out


def test_analyse_with_deadline_monotonic_priorities(capsys):
    assert main(["analyse", "a:20:1:2", "b:5:1:5", "--priority", "dm"]) == 0
    out = capsys.readouterr().out
    assert "deadline-monotonic" in out
    assert "bounds not valid" in out  # D < T, so the utilization bounds are skipped


def test_analyse_rejects_a_malformed_task_spec(capsys):
    assert main(["analyse", "t1:7"]) == 1
    assert "task spec must be" in capsys.readouterr().err


def test_analyse_rejects_a_non_numeric_field(capsys):
    assert main(["analyse", "t1:seven:3"]) == 1
    assert "non-numeric field" in capsys.readouterr().err


def test_analyse_rejects_a_malformed_critical_section(capsys):
    assert main(["analyse", "t1:7:3", "--critical-section", "t1:s1"]) == 1
    assert "must be TASK:SEM:DURATION_S" in capsys.readouterr().err


def test_percentile_subcommand_defaults(capsys):
    assert main(["percentile", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]) == 0
    out = capsys.readouterr().out
    assert "nearest_rank" in out
    assert "p50" in out


def test_percentile_subcommand_both_methods(capsys):
    args = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "--p", "50"]
    assert main(["percentile", *args, "--method", "nearest_rank"]) == 0
    assert "5" in capsys.readouterr().out
    assert main(["percentile", *args, "--method", "linear"]) == 0
    assert "5.5" in capsys.readouterr().out


def test_percentile_rejects_out_of_range(capsys):
    assert main(["percentile", "1", "2", "--p", "101"]) == 1
    assert "p must be in" in capsys.readouterr().err


def test_drift_subcommand_matches_its_own_closed_form(capsys):
    assert main(
        ["drift", "--period", "0.01", "--iterations", "1000", "--skew-ppm", "100",
         "--mode", "relative"]
    ) == 0
    out = capsys.readouterr().out
    assert "9.990000" in out  # 999 * 10 ms * 1e-4 = 9.99e-4 s
    assert "closed-form drift" in out


def test_drift_subcommand_absolute_mode_is_bounded(capsys):
    assert main(
        ["drift", "--period", "0.01", "--iterations", "1000", "--skew-ppm", "100"]
    ) == 0
    out = capsys.readouterr().out
    assert "9.999000" in out  # 1e-6 / 1.0001 = 9.999000099990e-7 s
    assert "drift slope" in out


def test_drift_subcommand_with_zero_iterations(capsys):
    assert main(["drift", "--period", "0.01", "--iterations", "0"]) == 0
    assert "final drift" in capsys.readouterr().out


def test_drift_subcommand_rejects_a_bad_period(capsys):
    assert main(["drift", "--period", "0"]) == 1
    assert "period_s must be" in capsys.readouterr().err
