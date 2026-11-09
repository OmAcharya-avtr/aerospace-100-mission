"""The command line, exercised in process. The subprocess run is validation/validate_cli.py."""

from __future__ import annotations

import pytest

from twindiverge.__main__ import build_parser, main


def test_the_parser_declares_every_subcommand():
    parser = build_parser()
    text = parser.format_help()
    for command in (
        "telemetry",
        "identify",
        "nullspace",
        "attribute",
        "profile",
        "crb",
        "sweep",
        "learned",
    ):
        assert command in text


def test_version_exits_zero():
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0


def test_no_subcommand_exits_two():
    with pytest.raises(SystemExit) as raised:
        main([])
    assert raised.value.code == 2


def test_telemetry_lists_the_declared_sets(capsys):
    assert main(["telemetry"]) == 0
    out = capsys.readouterr().out
    assert "rate_full" in out
    assert "DECLARED, not measured" in out
    assert "Not flight-qualified" in out


def test_identify_prints_the_refusal_and_the_combinations(capsys):
    assert main(["identify", "--telemetry", "rate_short"]) == 0
    out = capsys.readouterr().out
    assert "REFUSED" in out
    assert "identifiable combinations" in out
    assert "refusal is an answer, not an error" in out


def test_nullspace_prints_the_principal_angles(capsys):
    assert main(["nullspace", "--telemetry", "rate_coast"]) == 0
    out = capsys.readouterr().out
    assert "principal angles" in out
    assert "derived dimension  : 2" in out
    assert "computed dimension : 2" in out


def test_nullspace_of_a_full_rank_set_says_there_is_nothing_to_compare(capsys):
    assert main(["nullspace", "--telemetry", "rate_torque_short"]) == 0
    out = capsys.readouterr().out
    assert "both subspaces are empty" in out


def test_attribute_prints_both_baselines_and_the_equivalence_family(capsys):
    assert main(
        ["attribute", "--telemetry", "rate_short", "--seed", "9001", "--kind", "single"]
    ) == 0
    out = capsys.readouterr().out
    assert "baseline 1, the gate" in out
    assert "baseline 2, attribution" in out
    assert "NOT UNIQUE" in out
    assert "what this product delivers" in out


def test_attribute_on_an_equivalent_case_still_refuses(capsys):
    assert main(
        [
            "attribute",
            "--telemetry",
            "rate_short",
            "--seed",
            "9002",
            "--kind",
            "equivalent",
        ]
    ) == 0
    out = capsys.readouterr().out
    assert "true label                         : MULTI" in out
    assert "REFUSED" in out


def test_profile_prints_the_override(capsys):
    assert main(
        ["profile", "--telemetry", "rate_short", "--seed", "9003", "--points", "3"]
    ) == 0
    out = capsys.readouterr().out
    assert "profiles overriding covariance" in out
    assert "the covariance is wrong and the profile is the reference" in out


def test_crb_prints_the_bound_against_the_spread(capsys):
    assert main(
        ["crb", "--telemetry", "rate_torque_short", "--realisations", "4"]
    ) == 0
    out = capsys.readouterr().out
    assert "measured sd" in out
    assert "Neither computation uses" in out


def test_sweep_separates_structural_from_practical(capsys):
    assert main(["sweep"]) == 0
    out = capsys.readouterr().out
    assert "The rank and the refusal list do not move" in out


def test_learned_prints_the_model_card_statement(capsys):
    assert main(
        [
            "learned",
            "--telemetry",
            "rate_short",
            "--train-runs",
            "40",
            "--test-runs",
            "8",
            "--null-runs",
            "8",
        ]
    ) == 0
    out = capsys.readouterr().out
    assert "This model is not certified for operational flight use." in out
    assert "population equivalent" in out
    assert "computation budget" in out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["identify", "--telemetry", "invented"], "unknown telemetry set"),
        (["profile", "--points", "4"], "odd integer"),
        (["crb", "--realisations", "2"], "at least 4"),
        (["learned", "--train-runs", "5"], "--train-runs must be at least 10"),
        (["learned", "--test-runs", "1"], "--test-runs and --null-runs"),
    ],
)
def test_actionable_errors(argv: list[str], message: str):
    with pytest.raises(SystemExit) as raised:
        main(argv)
    assert message in str(raised.value)


def test_an_unknown_subcommand_exits_two():
    with pytest.raises(SystemExit) as raised:
        main(["invent"])
    assert raised.value.code == 2
