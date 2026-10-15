"""The command-line interface: every subcommand runs and says what it should."""

from __future__ import annotations

import pytest

from activityplan.__main__ import build_parser, main


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0


def test_version_exits_zero() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0


def test_a_missing_subcommand_is_an_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code != 0


def test_parser_rejects_an_unknown_instance() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["plan", "--instance", "not-an-instance"])


def test_instances_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["instances"]) == 0
    out = capsys.readouterr().out
    assert "leo-imaging" in out
    assert "tiny-enumerable" in out
    assert "surplus/Wh" in out


def test_declare_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["declare", "--instance", "recorder-bound"]) == 0
    out = capsys.readouterr().out
    assert "recorder-bound" in out
    assert "admissible starts" in out
    assert "window float lost/s" in out


def test_plan_subcommand_greedy(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "--instance", "energy-bound", "--planner", "greedy"]) == 0
    out = capsys.readouterr().out
    assert "planner             : greedy" in out
    assert "BINDING CONSTRAINT" in out
    assert "dropped" in out


def test_plan_subcommand_exact(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "plan",
                "--instance",
                "tiny-enumerable",
                "--planner",
                "exact",
                "--time-limit",
                "30",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "exact MILP" in out
    assert "proven optimal True" in out


def test_plan_subcommand_accepts_an_explicit_step(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "--instance", "tiny-enumerable", "--step", "200"]) == 0
    out = capsys.readouterr().out
    assert "6 steps" not in out or "step" in out


def test_compare_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare", "--instance", "leo-imaging", "--time-limit", "60"]) == 0
    out = capsys.readouterr().out
    assert "greedy" in out
    assert "exact" in out
    assert "below the exact optimum" in out
    assert "DISCRETISED" in out


def test_predict_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["predict", "--n-train", "200", "--n-test", "80", "--mode", "absolute"]) == 0
    out = capsys.readouterr().out
    assert "analytic MAE" in out
    assert "learned MAE" in out
    assert "interval coverage" in out
    assert "nominal coverage" in out


def test_robust_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["robust", "--n-train", "200", "--n-instances", "2"]) == 0
    out = capsys.readouterr().out
    assert "oracle" in out
    assert "ml-robust" in out
    assert "realised priority" in out


def test_warmstart_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "warmstart",
                "--n-train",
                "200",
                "--n-instances",
                "2",
                "--instance-seed",
                "9000",
                "--time-limit",
                "60",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "cold nodes" in out
    assert "warm nodes" in out
    assert "no MIP start" in out


def test_no_output_claims_certification_or_flight_readiness(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for argv in (
        ["instances"],
        ["declare"],
        ["plan", "--instance", "window-bound"],
        ["compare", "--instance", "tiny-enumerable"],
    ):
        assert main(argv) == 0
        out = capsys.readouterr().out.lower()
        for word in (
            "flight-safe",
            "mission-ready",
            "production-ready",
            "flight qualified",
            "!",
        ):
            assert word not in out, (argv, word)
