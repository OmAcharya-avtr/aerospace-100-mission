"""Command-line interface tests, including the exit statuses."""

from __future__ import annotations

import pytest

from twincalib.__main__ import build_parser, main


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "identifiability verdict" in out or "identifiability\nverdict" in out


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "0.1.0" in capsys.readouterr().out


def test_missing_subcommand_exits_two():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_unknown_design_exits_two():
    with pytest.raises(SystemExit) as excinfo:
        main(["identify", "--design", "nope"])
    assert excinfo.value.code == 2


def test_designs_lists_all_five(capsys):
    assert main(["designs"]) == 0
    out = capsys.readouterr().out
    for name in (
        "baseline-eclipse",
        "full-sun",
        "lumped-nodes",
        "coarse-sampled",
        "weak-excitation",
    ):
        assert name in out


def test_identify_on_the_good_design_exits_zero(capsys):
    assert main(["identify", "--design", "baseline-eclipse", "--exact"]) == 0
    out = capsys.readouterr().out
    assert "full numerical rank 6" in out
    assert "parameter correlation" in out


def test_identify_on_the_degenerate_design_exits_one(capsys):
    # Exit status 1 is a result, not an error: the design is not identifiable.
    assert main(["identify", "--design", "full-sun", "--exact"]) == 1
    out = capsys.readouterr().out
    assert "NOT IDENTIFIABLE" in out
    assert "d ln q_sun" in out


def test_fit_prints_the_verdict_above_the_estimate(capsys):
    assert main(["fit", "--design", "baseline-eclipse", "--exact", "--show-truth"]) == 0
    out = capsys.readouterr().out
    assert out.index("VERDICT") < out.index("weighted least-squares estimate")
    assert "rel. error" in out


def test_fit_on_the_degenerate_design_exits_one(capsys):
    assert main(["fit", "--design", "full-sun", "--exact"]) == 1
    assert "NOT IDENTIFIABLE" in capsys.readouterr().out


def test_profile_on_the_degenerate_design_exits_one_and_says_open(capsys):
    assert (
        main(
            [
                "profile",
                "--design",
                "full-sun",
                "--exact",
                "--points",
                "2",
            ]
        )
        == 1
    )
    out = capsys.readouterr().out
    assert "OPEN" in out
    assert "NOT determined" in out


def test_profile_on_the_good_design_exits_zero(capsys):
    assert (
        main(
            [
                "profile",
                "--design",
                "coarse-sampled",
                "--exact",
                "--points",
                "3",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "every profile closed" in out


def test_residual_runs_and_reports_the_three_subspaces(capsys):
    assert (
        main(
            [
                "residual",
                "--design",
                "baseline-eclipse",
                "--n-train",
                "2",
                "--n-test",
                "1",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "VERDICT" in out
    assert "ABSORPTION" in out
    assert "parameter drift" in out
    assert "fitted residual kernel" in out


def test_parser_exposes_every_subcommand():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert len(actions) == 1
    assert set(actions[0].choices) == {
        "designs",
        "identify",
        "fit",
        "profile",
        "residual",
    }
