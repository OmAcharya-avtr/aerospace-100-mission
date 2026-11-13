"""Loader behaviour, input validation and the measurement-model invariants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from deadlinerisk.provenance import Origin
from deadlinerisk.schema import JOB_COLUMNS, SchemaError
from deadlinerisk.trace import (
    TraceDataAbsentError,
    campaign_is_complete,
    load_campaign,
    load_job_trace,
    load_monitor_trace,
)

from fixtures import synthetic_trace

HEADER = ",".join(JOB_COLUMNS)


def _write(path: Path, body: str) -> Path:
    path.write_text(HEADER + "\n" + body)
    return path


def test_loads_the_synthetic_campaign(campaign) -> None:
    assert campaign.provenance.origin is Origin.SYNTHETIC
    assert campaign.provenance.directory == "synthetic"
    assert "gt_long" in campaign.jobs
    assert "gt_long" in campaign.monitors
    assert campaign.period_s == pytest.approx(0.010, rel=1e-12, abs=0.0)
    assert campaign.deadline_s == pytest.approx(0.010, rel=1e-12, abs=0.0)
    assert campaign.complete is True


def test_monitor_temperature_is_converted_to_degrees(campaign) -> None:
    monitor = campaign.monitors["gt_long"]
    # The fixture writes tj around 41000 + 950*workers millidegrees at 4 CPU
    # workers, so degrees Celsius must be in the tens, not the tens of thousands.
    assert 20.0 < float(monitor.tj_degC.mean()) < 120.0


def test_stressor_counts_come_from_the_manifest(campaign) -> None:
    assert campaign.stressors("mix_c6") == (6, 0)
    assert campaign.stressors("mix_mem2") == (0, 2)
    assert campaign.stressors("mix_c2mem1") == (2, 1)


def test_unknown_run_has_its_name_in_the_error(campaign) -> None:
    with pytest.raises(KeyError, match="not_a_run"):
        campaign.stressors("not_a_run")


def test_absent_directory_refuses_without_substituting(tmp_path: Path) -> None:
    with pytest.raises(TraceDataAbsentError, match="does not exist"):
        load_campaign(tmp_path / "nothing-here")


def test_absent_manifest_refuses(tmp_path: Path) -> None:
    (tmp_path / "jobs_x.csv").write_text(HEADER + "\n")
    with pytest.raises(TraceDataAbsentError, match="manifest.json is absent"):
        load_campaign(tmp_path)


def test_required_run_absent_names_every_missing_file(
    synthetic_directory: Path,
) -> None:
    with pytest.raises(TraceDataAbsentError) as info:
        load_campaign(
            synthetic_directory,
            origin=Origin.SYNTHETIC,
            require=("gt_long", "mix_c99", "mix_c100"),
            directory_label="synthetic",
        )
    message = str(info.value)
    assert "jobs_mix_c99.csv" in message
    assert "jobs_mix_c100.csv" in message
    assert "Nothing was substituted" in message


def test_campaign_is_complete_tracks_the_marker(tmp_path: Path) -> None:
    assert campaign_is_complete(tmp_path) is False
    (tmp_path / "CAMPAIGN_COMPLETE").write_text("")
    assert campaign_is_complete(tmp_path) is True


def test_manifest_without_period_is_an_input_error(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"deadline_s": 0.01}))
    with pytest.raises(ValueError, match="period_s"):
        load_campaign(tmp_path)


@pytest.mark.parametrize(
    ("period", "deadline", "pattern"),
    [
        (0.0, 0.010, "period_s must be finite"),
        (-1.0, 0.010, "period_s must be finite"),
        (float("nan"), 0.010, "period_s must be finite"),
        (0.010, 0.0, "deadline_s must be finite"),
        (0.010, float("inf"), "deadline_s must be finite"),
        (0.010, 0.020, "exceeds period_s"),
    ],
)
def test_period_and_deadline_validation(
    tmp_path: Path, period: float, deadline: float, pattern: str
) -> None:
    path = _write(tmp_path / "jobs_x.csv", "0,0.0,0.0,0.004,0.004,0.004,0,0\n")
    with pytest.raises(ValueError, match=pattern):
        load_job_trace(path, "x", period, deadline)


def test_wrong_header_is_a_schema_error(tmp_path: Path) -> None:
    path = tmp_path / "jobs_x.csv"
    path.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(SchemaError):
        load_job_trace(path, "x", 0.010, 0.010)


def test_short_row_is_rejected_with_its_line_number(tmp_path: Path) -> None:
    path = _write(tmp_path / "jobs_x.csv", "0,0.0,0.0,0.004\n")
    with pytest.raises(ValueError, match="line 2"):
        load_job_trace(path, "x", 0.010, 0.010)


def test_non_increasing_job_index_is_rejected(tmp_path: Path) -> None:
    body = "0,0.0,0.0,0.004,0.004,0.004,0,0\n0,0.010,0.0,0.014,0.004,0.004,0,0\n"
    path = _write(tmp_path / "jobs_x.csv", body)
    with pytest.raises(ValueError, match="strictly increasing"):
        load_job_trace(path, "x", 0.010, 0.010)


def test_negative_execution_time_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "jobs_x.csv", "0,0.0,0.0,0.004,-0.004,0.004,0,0\n")
    with pytest.raises(ValueError, match="exec_s has a negative value"):
        load_job_trace(path, "x", 0.010, 0.010)


def test_missed_flag_outside_zero_one_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "jobs_x.csv", "0,0.0,0.0,0.004,0.004,0.004,2,0\n")
    with pytest.raises(ValueError, match="missed must be 0 or 1"):
        load_job_trace(path, "x", 0.010, 0.010)


def test_negative_dropped_before_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "jobs_x.csv", "0,0.0,0.0,0.004,0.004,0.004,0,-1\n")
    with pytest.raises(ValueError, match="dropped_before has a negative value"):
        load_job_trace(path, "x", 0.010, 0.010)


def test_empty_file_is_absent_not_malformed(tmp_path: Path) -> None:
    path = tmp_path / "jobs_x.csv"
    path.write_text("")
    with pytest.raises(TraceDataAbsentError, match="is empty"):
        load_job_trace(path, "x", 0.010, 0.010)


def test_missing_monitor_file_names_itself(tmp_path: Path) -> None:
    with pytest.raises(TraceDataAbsentError, match="mon_absent.csv"):
        load_monitor_trace(tmp_path / "mon_absent.csv", "absent")


def test_response_equals_end_minus_release_on_the_fixture(campaign) -> None:
    # The measurement model's definition, checked against the written columns.
    # Exact in exact arithmetic; the committed CSV rounds to 1 ns and 1 us, so
    # the tolerance is the rounding of the two columns and not a fudge.
    for jobs in campaign.jobs.values():
        implied = jobs.end_s - jobs.release_s
        assert np.allclose(implied, jobs.response_s, rtol=0.0, atol=2e-6)


def test_missed_flag_agrees_with_the_deadline_comparison(campaign) -> None:
    # A miss is response > D. The flag is written by the generator and
    # recomputed here. Rows whose response is within the CSV's own rounding of
    # the deadline are excluded, because the boolean of a strict threshold
    # comparison is the one thing floating point cannot promise.
    for jobs in campaign.jobs.values():
        resolved = np.abs(jobs.response_s - jobs.deadline_s) > 2e-6
        recomputed = (jobs.response_s[resolved] > jobs.deadline_s).astype(np.int64)
        assert np.array_equal(recomputed, jobs.missed[resolved])


def test_loading_an_incomplete_campaign_still_works(tmp_path: Path) -> None:
    # The campaign writes its manifest after each run, so a directory read
    # mid-campaign has a manifest with complete=False and a subset of the runs.
    synthetic_trace.write_synthetic_campaign(
        tmp_path,
        plan=(synthetic_trace.RunSpec("mix_c0", 0, 0, 5.0),),
        complete=False,
    )
    partial = load_campaign(tmp_path, origin=Origin.SYNTHETIC, directory_label="partial")
    assert partial.complete is False
    assert set(partial.jobs) == {"mix_c0"}
