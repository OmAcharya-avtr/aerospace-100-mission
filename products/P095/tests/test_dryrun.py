"""The dry run: it walks everything, touches no board, and writes no file."""

from __future__ import annotations

import pytest

from boardprobe.dryrun import declared_paths, dry_run
from boardprobe.orin_nano import ORIN_NANO


def test_the_dry_run_touches_no_device_and_writes_no_file(idle_trace, tmp_path):
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "existing.txt").write_text("already here\n")
    report = dry_run(ORIN_NANO, idle_trace, seed=1, watch=watched)
    assert report.device_accesses == ()
    assert report.files_added == ()
    assert report.side_effect_free
    assert report.files_before == ("existing.txt",)
    assert report.files_after == ("existing.txt",)


def test_it_exercises_every_named_code_path(idle_trace, tmp_path):
    report = dry_run(ORIN_NANO, idle_trace, seed=1, watch=tmp_path)
    assert report.path_count >= 15
    assert len(set(report.paths)) == report.path_count
    assert declared_paths(report) == tuple(report.paths)


def test_the_live_refusal_leg_is_counted_separately_and_not_hidden(
    idle_trace, tmp_path
):
    """Attempts against a root asserted absent are reported, not swallowed."""
    report = dry_run(ORIN_NANO, idle_trace, seed=1, watch=tmp_path)
    board_channels = sum(
        1 for channel in ORIN_NANO.channels if channel.locator != "CLOCK_MONOTONIC_RAW"
    )
    assert len(report.absent_root_accesses) == board_channels
    assert len(report.refusals) >= board_channels
    assert report.device_accesses == ()
    assert report.absent_root


def test_it_records_the_labels_it_was_granted_and_refused(idle_trace, tmp_path):
    report = dry_run(ORIN_NANO, idle_trace, seed=1, watch=tmp_path)
    granted = " ".join(report.labels_granted)
    refused = " ".join(report.labels_refused)
    assert "SYNTHETIC" in granted
    assert "MEASURED on simulated inputs refused" in refused
    assert "tampered reading refused" in refused
    assert "meet_all([]) refused" in refused


def test_a_recorded_trace_replays_as_replayed_in_the_walk(recorded_trace, tmp_path):
    report = dry_run(ORIN_NANO, recorded_trace, seed=1, watch=tmp_path)
    assert any("REPLAYED" in entry for entry in report.labels_granted)
    assert report.side_effect_free


def test_an_absent_root_that_exists_is_refused(idle_trace, tmp_path):
    present = tmp_path / "present"
    present.mkdir()
    with pytest.raises(RuntimeError, match="cannot contain a device node"):
        dry_run(ORIN_NANO, idle_trace, watch=tmp_path, absent_root=present)


def test_two_dry_runs_at_the_same_seed_report_the_same_paths(idle_trace, tmp_path):
    """A scanner run twice from a clean tree must agree with itself."""
    first = dry_run(ORIN_NANO, idle_trace, seed=5, watch=tmp_path)
    second = dry_run(ORIN_NANO, idle_trace, seed=5, watch=tmp_path)
    assert first.paths == second.paths
    assert first.files_added == second.files_added == ()
    assert len(first.absent_root_accesses) == len(second.absent_root_accesses)
