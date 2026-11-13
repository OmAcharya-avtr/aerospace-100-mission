"""The ``python -m deadlinerisk`` command line, called in process."""

from __future__ import annotations

from pathlib import Path

import pytest

from deadlinerisk.__main__ import main

from fixtures import synthetic_trace


@pytest.fixture(scope="module")
def traces(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("cli-campaign")
    return synthetic_trace.write_synthetic_campaign(target)


def test_manifest_subcommand_prints_the_board_and_the_probe_table(
    traces: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(["--traces", str(traces), "--origin", "synthetic", "manifest"])
    output = capsys.readouterr().out
    assert status == 0
    assert "INPUT ORIGIN: synthetic" in output
    assert "probe table" in output
    assert "deadline D" in output


def test_account_subcommand_reports_the_reconstruction(
    traces: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(
        ["--traces", str(traces), "--origin", "synthetic", "account", "--run", "gt_long"]
    )
    output = capsys.readouterr().out
    assert status == 0
    assert "releases = sum(1+drop)" in output
    assert "executed-only miss rate" in output


def test_account_subcommand_covers_every_run_by_default(
    traces: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(["--traces", str(traces), "--origin", "synthetic", "account"])
    output = capsys.readouterr().out
    assert status == 0
    assert output.count("run                     :") == 10


def test_headline_subcommand_reports_the_factor_and_a_verdict(
    traces: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(
        [
            "--traces",
            str(traces),
            "--origin",
            "synthetic",
            "headline",
            "--run",
            "gt_long",
            "--resamples",
            "200",
        ]
    )
    output = capsys.readouterr().out
    assert status == 0
    assert "understatement factor" in output
    assert "verdict" in output
    assert "block length (releases)" in output


def test_evt_subcommand_prints_the_stability_table(
    traces: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(["--traces", str(traces), "--origin", "synthetic", "evt"])
    output = capsys.readouterr().out
    assert status == 0
    assert "mod_beta_ms" in output
    assert "declared rule" in output


def test_learned_subcommand_scores_both_splits(
    traces: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(["--traces", str(traces), "--origin", "synthetic", "learned"])
    output = capsys.readouterr().out
    assert status == 0
    assert "held-out workload mixes" in output
    assert "held-out time" in output
    assert "constant rate" in output
    assert "speed scaling" in output


def test_absent_traces_exit_three(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(["--traces", str(tmp_path / "nothing"), "manifest"])
    assert status == 3
    assert "TRACES ABSENT" in capsys.readouterr().err


def test_unknown_run_exits_three_and_lists_what_is_present(
    traces: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(
        ["--traces", str(traces), "--origin", "synthetic", "headline", "--run", "nope"]
    )
    assert status == 3
    error = capsys.readouterr().err
    assert "nope" in error
    assert "gt_long" in error


def test_bad_holdout_fraction_exits_two(
    traces: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(
        [
            "--traces",
            str(traces),
            "--origin",
            "synthetic",
            "learned",
            "--holdout-fraction",
            "1.0",
        ]
    )
    assert status == 2
    assert "INPUT ERROR" in capsys.readouterr().err


def test_no_subcommand_is_a_usage_error(traces: Path) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--traces", str(traces)])
    assert info.value.code == 2
