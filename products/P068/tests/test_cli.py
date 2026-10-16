"""Command line interface."""

from __future__ import annotations

import csv

import pytest

from degradepredict.__main__ import build_parser, main


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "not flight-qualified" in out


def test_missing_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        main([])


def test_parser_exposes_the_three_subcommands():
    parser = build_parser()
    actions = [a for a in parser._subparsers._group_actions if hasattr(a, "choices")]
    assert set(actions[0].choices) == {"generate", "fit", "benchmark"}


def test_generate_writes_a_csv_with_one_row_per_sample(tmp_path, capsys):
    out = tmp_path / "pop.csv"
    assert main(["generate", "--n", "5", "--seed", "3", "--out", str(out)]) == 0
    with out.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) > 50
    assert {"traj_id", "family", "crossing_time_days", "event", "capacity_percent_bol"} <= set(
        rows[0]
    )
    assert len({r["traj_id"] for r in rows}) == 5
    printed = capsys.readouterr().out
    assert "censored fraction" in printed


def test_generate_is_reproducible(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    main(["generate", "--n", "4", "--seed", "9", "--out", str(a)])
    main(["generate", "--n", "4", "--seed", "9", "--out", str(b)])
    assert a.read_text() == b.read_text()


def test_fit_prints_all_three_baselines(capsys):
    assert main(["fit", "--index", "1", "--seed", "20261016"]) == 0
    out = capsys.readouterr().out
    assert "linear" in out
    assert "exponential" in out
    assert "powerlaw-fade" in out
    assert "Fieller" in out


def test_benchmark_prints_the_split_columns(capsys):
    assert main(["benchmark", "--n-train", "220", "--n-test", "40", "--seed", "5"]) == 0
    out = capsys.readouterr().out
    assert "cov cmpl" in out
    assert "cov cens" in out
    assert "learned-ipcw" in out


@pytest.mark.parametrize("level", ["0", "1", "1.5"])
def test_level_outside_the_unit_interval_is_rejected(level):
    with pytest.raises(SystemExit):
        main(["fit", "--level", level])
