"""Preflight, capture and backout: the deployment procedure as runnable code."""

from __future__ import annotations

import json

import numpy as np
import pytest

from photoncount.hal import (
    AcquisitionRequest,
    BackendMode,
    DeviceBackend,
    SimulatedBackend,
)
from photoncount.ops import (
    IN_PROGRESS_SUFFIX,
    PreflightCheck,
    PreflightReport,
    RunJournal,
    RunRecord,
    backout_incomplete_runs,
    environment_summary,
    read_run_record,
    run_preflight,
    write_run_record,
)
from photoncount.simulate import DetectorSpec

EXPECTED_CHECKS = (
    "backend_described",
    "dead_time_declared",
    "window_exceeds_dead_time",
    "rate_below_saturation",
    "inversion_branch_unambiguous",
    "afterpulsing_declared",
    "directory_writable",
    "backend_self_test",
)


def _backend(**kwargs) -> SimulatedBackend:
    spec = DetectorSpec(
        dead_time_s=kwargs.pop("dead_time_s", 1e-7),
        model=kwargs.pop("model", "paralyzable"),
        afterpulse_probability=kwargs.pop("afterpulse_probability", 0.04),
        afterpulse_mean_delay_s=kwargs.pop("afterpulse_mean_delay_s", 4e-7),
    )
    return SimulatedBackend(spec, kwargs.pop("rate", 1e5), np.random.default_rng(0))


def test_preflight_runs_every_check_in_order(tmp_path):
    report = run_preflight(_backend(), AcquisitionRequest(1e-3, 10), 1e5, tmp_path)
    assert report.names() == EXPECTED_CHECKS


def test_preflight_passes_on_a_sane_configuration(tmp_path):
    report = run_preflight(_backend(), AcquisitionRequest(1e-3, 10), 1e5, tmp_path)
    assert report.passed is True
    assert report.failures == ()


def test_short_window_fails_preflight(tmp_path):
    """A window only 10 dead times long makes per-window statistics meaningless."""
    report = run_preflight(_backend(), AcquisitionRequest(1e-6, 10), 1e5, tmp_path)
    failed = {c.name for c in report.failures}
    assert "window_exceeds_dead_time" in failed
    assert report.passed is False


def test_saturated_paralyzable_rate_fails_preflight(tmp_path):
    """At or above 1/tau the observed rate falls with illumination: refuse to run."""
    report = run_preflight(_backend(), AcquisitionRequest(1e-3, 10), 2e7, tmp_path)
    failed = {c.name for c in report.failures}
    assert "rate_below_saturation" in failed


def test_near_saturation_warns_but_does_not_block(tmp_path):
    report = run_preflight(_backend(), AcquisitionRequest(1e-3, 10), 5e6, tmp_path)
    names = {c.name for c in report.warnings}
    assert "rate_below_saturation" in names
    assert report.passed is True


def test_branch_ambiguity_warns_near_the_maximum(tmp_path):
    # tau = 1e-7, 1/tau = 1e7; at n = 1e7 the observed rate is at the maximum.
    report = run_preflight(_backend(), AcquisitionRequest(1e-3, 10), 0.9e7, tmp_path)
    warned = {c.name for c in report.warnings}
    assert "inversion_branch_unambiguous" in warned


def test_nonparalyzable_branch_check_always_passes(tmp_path):
    report = run_preflight(
        _backend(model="nonparalyzable"), AcquisitionRequest(1e-3, 10), 1e5, tmp_path
    )
    check = next(c for c in report.checks if c.name == "inversion_branch_unambiguous")
    assert check.status == "PASS"


def test_zero_afterpulsing_warns(tmp_path):
    report = run_preflight(
        _backend(afterpulse_probability=0.0), AcquisitionRequest(1e-3, 10), 1e5, tmp_path
    )
    warned = {c.name for c in report.warnings}
    assert "afterpulsing_declared" in warned


def test_ideal_detector_skips_the_dead_time_constraints(tmp_path):
    report = run_preflight(
        _backend(dead_time_s=0.0, afterpulse_probability=0.0),
        AcquisitionRequest(1e-6, 10),
        1e9,
        tmp_path,
    )
    assert report.passed is True


def test_device_backend_self_test_is_a_warning_not_a_failure(tmp_path):
    """A device with no hardware yet must not look like a broken device."""
    device = DeviceBackend(dead_time_s=45e-9, dead_time_model="paralyzable",
                           afterpulse_probability=0.01)
    report = run_preflight(device, AcquisitionRequest(1e-3, 10), 1e5, tmp_path)
    check = next(c for c in report.checks if c.name == "backend_self_test")
    assert check.status == "WARN"
    assert "hardware" in check.detail
    assert report.passed is True


def test_unwritable_directory_fails_preflight(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory\n", encoding="utf-8")
    report = run_preflight(_backend(), AcquisitionRequest(1e-3, 10), 1e5, blocker)
    failed = {c.name for c in report.failures}
    assert "directory_writable" in failed


def test_report_helpers():
    checks = (
        PreflightCheck("a", "PASS", 1.0, "ok"),
        PreflightCheck("b", "WARN", 2.0, "hmm"),
        PreflightCheck("c", "FAIL", 3.0, "no"),
    )
    report = PreflightReport(checks)
    assert report.passed is False
    assert [c.name for c in report.warnings] == ["b"]
    assert [c.name for c in report.failures] == ["c"]
    assert len(report.as_dicts()) == 3


def test_environment_summary_contains_no_filesystem_path():
    summary = environment_summary()
    assert set(summary) == {
        "python_version",
        "platform",
        "machine",
        "cpu_count",
        "photoncount_version",
    }
    joined = " ".join(summary.values())
    assert "/" not in joined.replace("/proc", "")


def test_run_record_round_trip(tmp_path):
    backend = _backend()
    request = AcquisitionRequest(1e-3, 8, "unit")
    report = run_preflight(backend, request, 1e5, tmp_path)
    with backend:
        acq = backend.acquire(request, BackendMode.SIMULATION)
    record = RunRecord.from_acquisition(acq, request, backend, report, seed=0)
    name = write_run_record(record, tmp_path)
    assert name == "unit.json"
    assert "/" not in name
    loaded = read_run_record(name, tmp_path)
    assert loaded["simulated"] is True
    assert loaded["mode"] == "simulation"
    assert len(loaded["counts"]) == 8
    assert loaded["preflight_passed"] is True
    assert loaded["seed"] == 0


def test_run_record_json_has_no_nan_tokens(tmp_path):
    """A dry-run record has a NaN rate; JSON must carry null, not the NaN token."""
    backend = _backend()
    request = AcquisitionRequest(1e-3, 4, "dry")
    report = run_preflight(backend, request, 1e5, tmp_path)
    with backend:
        acq = backend.acquire(request, BackendMode.DRY_RUN)
    record = RunRecord.from_acquisition(acq, request, backend, report)
    text = record.to_json()
    assert "NaN" not in text
    assert "Infinity" not in text
    json.loads(text)


def test_read_run_record_rejects_a_path(tmp_path):
    with pytest.raises(ValueError, match="bare name"):
        read_run_record("sub/run.json", tmp_path)


def test_journal_marks_and_clears_a_run(tmp_path):
    journal = RunJournal("night", tmp_path)
    name = journal.begin({"window_s": 1e-3, "n_windows": 4})
    assert name == f"night{IN_PROGRESS_SUFFIX}"
    assert journal.is_in_progress is True
    backend = _backend()
    request = AcquisitionRequest(1e-3, 4, "night")
    report = run_preflight(backend, request, 1e5, tmp_path)
    with backend:
        acq = backend.acquire(request)
    journal.commit(RunRecord.from_acquisition(acq, request, backend, report))
    assert journal.is_in_progress is False
    assert (tmp_path / "night.json").exists()


def test_journal_refuses_to_overwrite_evidence_of_a_failure(tmp_path):
    journal = RunJournal("night", tmp_path)
    journal.begin({"window_s": 1e-3})
    with pytest.raises(FileExistsError, match="did not finish"):
        journal.begin({"window_s": 1e-3})


def test_backout_renames_in_progress_journals(tmp_path):
    RunJournal("a", tmp_path).begin({"x": 1})
    RunJournal("b", tmp_path).begin({"x": 2})
    out = backout_incomplete_runs(tmp_path)
    assert [row["label"] for row in out] == ["a", "b"]
    for row in out:
        assert "/" not in row["from"] and "/" not in row["to"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.aborted.json", "b.aborted.json"]
    payload = json.loads((tmp_path / "a.aborted.json").read_text(encoding="utf-8"))
    assert payload["state"] == "aborted"
    assert "aborted_unix" in payload


def test_backout_is_idempotent(tmp_path):
    RunJournal("a", tmp_path).begin({"x": 1})
    assert len(backout_incomplete_runs(tmp_path)) == 1
    assert backout_incomplete_runs(tmp_path) == []


def test_backout_leaves_completed_records_alone(tmp_path):
    (tmp_path / "done.json").write_text("{}\n", encoding="utf-8")
    RunJournal("half", tmp_path).begin({"x": 1})
    backout_incomplete_runs(tmp_path)
    assert (tmp_path / "done.json").read_text(encoding="utf-8") == "{}\n"


def test_backout_survives_an_unreadable_journal(tmp_path):
    (tmp_path / f"broken{IN_PROGRESS_SUFFIX}").write_text("{not json", encoding="utf-8")
    out = backout_incomplete_runs(tmp_path)
    assert out[0]["label"] == "broken"
    payload = json.loads((tmp_path / "broken.aborted.json").read_text(encoding="utf-8"))
    assert payload["note"] == "journal unreadable"


def test_backout_on_a_missing_directory_returns_empty(tmp_path):
    assert backout_incomplete_runs(tmp_path / "nope") == []


def test_journal_rejects_a_path_label(tmp_path):
    with pytest.raises(ValueError, match="bare name"):
        RunJournal("a/b", tmp_path)
