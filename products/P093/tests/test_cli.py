"""The command-line interface, including every exit status it declares."""

from __future__ import annotations

import pytest

from powerenvelope.__main__ import EXIT_ABSENT, EXIT_USAGE, main

from fixtures import synthetic_campaign


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "powerenvelope" in capsys.readouterr().out


def test_no_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


@pytest.mark.parametrize("command", ["manifest", "headline", "report"])
def test_simulate_runs_each_short_subcommand(capsys, command):
    assert main(["--simulate", "--simulate-cycles", "6", command]) == 0
    captured = capsys.readouterr().out
    assert "SYNTHETIC CAMPAIGN" in captured
    assert "Nothing below was measured" in captured


def test_simulate_learned_runs_and_scores_both_splits(capsys):
    assert main(["--simulate", "--simulate-cycles", "10", "learned"]) == 0
    captured = capsys.readouterr().out
    assert "held-out duty" in captured
    assert "held-out frame_dim" in captured
    assert "learned forest" in captured
    assert "analytic duty-cycle" in captured
    assert "constant (null)" in captured


def test_headline_prints_the_measurement_model_and_both_variants(capsys):
    assert main(["--simulate", "--simulate-cycles", "4", "headline"]) == 0
    captured = capsys.readouterr().out
    assert "MEASUREMENT MODEL" in captured
    assert "measured-active" in captured
    assert "nominal-rail" in captured
    assert "HEADLINE understatement factor F" in captured


def test_report_prints_every_baseline(capsys):
    assert main(["--simulate", "--simulate-cycles", "6", "report"]) == 0
    captured = capsys.readouterr().out
    for fragment in (
        "measured background floor",
        "baseline 1, energy per inference",
        "baseline 2, the analytic duty-cycle model",
        "measured rail-accounting ratio against load",
        "baseline 3, ground truth",
        "between-run spread",
        "the headline",
        "permissive-gate control",
        "RAIL-ACCOUNTING RATIO, not a converter efficiency",
        "BACKGROUND FLOOR ON A SHARED MACHINE",
    ):
        assert fragment in captured


def test_absent_traces_exit_three_and_do_not_fall_back(capsys, tmp_path):
    status = main(["--traces", str(tmp_path / "nothing"), "report"])
    assert status == EXIT_ABSENT
    captured = capsys.readouterr()
    assert "CAMPAIGN_COMPLETE is absent" in captured.err
    assert "SYNTHETIC" not in captured.out


def test_incomplete_campaign_exits_three(capsys, tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "partial", complete=False)
    assert main(["--traces", str(directory), "manifest"]) == EXIT_ABSENT
    assert "writes that marker last" in capsys.readouterr().err


def test_a_complete_directory_on_disk_is_analysed(capsys, tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "ok", cycles=4)
    assert main(["--traces", str(directory), "manifest"]) == 0
    assert "cells loaded" in capsys.readouterr().out


def test_an_unknown_cell_exits_three_naming_what_is_present(capsys, tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "ok", cycles=4)
    assert main(["--traces", str(directory), "headline", "--cell", "w64_d999"]) == EXIT_ABSENT
    assert "cells present" in capsys.readouterr().err


def test_report_on_a_campaign_too_short_for_a_window_says_so(capsys):
    assert main(["--simulate", "--simulate-cycles", "3", "report"]) == 0
    assert "NO WINDOW DATASET" in capsys.readouterr().out


def test_a_corrupt_header_exits_two(capsys, tmp_path):
    directory = synthetic_campaign.write_campaign_dir(tmp_path / "bad", cycles=4)
    target = directory / "power_w64_d050.csv"
    lines = target.read_text().splitlines()
    target.write_text("a,b,c,d,e,f,g,h\n" + "\n".join(lines[1:]) + "\n")
    assert main(["--traces", str(directory), "manifest"]) == EXIT_USAGE
    assert "INPUT ERROR" in capsys.readouterr().err


def test_exit_statuses_are_the_declared_constants():
    assert EXIT_ABSENT == 3
    assert EXIT_USAGE == 2
