"""The benchmark harness, and the guards that keep the hardware rows empty."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from timetransfer.bench import (
    HARDWARE_STAGES,
    BenchmarkRecord,
    HardwareRow,
    hardware_rows,
    run_benchmark,
    time_stage,
)
from timetransfer.errors import ConfigurationError, HardwareClaimError


def _record(**overrides) -> BenchmarkRecord:
    return run_benchmark(
        {"sum": lambda: float(np.sum(np.arange(1000)))},
        label=overrides.pop("label", "unit"),
        environment_note=overrides.pop("environment_note", "unit test"),
        backend_kind="simulated",
        backend_driver="",
        n_repeats=2,
        clock_samples=100,
    )


def test_hardware_table_has_five_rows_all_empty() -> None:
    rows = hardware_rows()
    assert len(rows) == 5
    assert [row.stage for row in rows] == [stage for stage, _ in HARDWARE_STAGES]
    assert all(row.value is None for row in rows)
    assert all(row.status == "unmeasured" for row in rows)
    assert all(row.reason for row in rows)
    assert all("--" in row.as_row() for row in rows)


def test_hardware_rows_require_a_reason() -> None:
    with pytest.raises(ConfigurationError, match="reason"):
        hardware_rows("   ")


def test_record_refuses_a_hardware_row_that_carries_a_value() -> None:
    filled = (
        HardwareRow(
            stage="exchange_round_trip_s",
            description="x",
            units="s",
            reason="r",
            value=None,
            status="measured",
        ),
    )
    with pytest.raises(HardwareClaimError, match="status other than 'unmeasured'"):
        BenchmarkRecord(
            label="x",
            environment_note="note",
            backend_kind="simulated",
            backend_driver="",
            is_hardware=False,
            stages=(),
            tracemalloc_peak_bytes=0.0,
            maxrss_bytes=0.0,
            maxrss_note="n",
            clock_reported_resolution_ns=1,
            clock_measured_resolution_ns=1,
            clock_call_overhead_ns=0.0,
            offset_quantisation_uncertainty_s=0.0,
            hardware=filled,
        )


def test_record_refuses_to_be_constructed_as_hardware() -> None:
    with pytest.raises(HardwareClaimError, match="no backend in"):
        BenchmarkRecord(
            label="x",
            environment_note="note",
            backend_kind="device",
            backend_driver="ixgbe-ptp",
            is_hardware=True,
            stages=(),
            tracemalloc_peak_bytes=0.0,
            maxrss_bytes=0.0,
            maxrss_note="n",
            clock_reported_resolution_ns=1,
            clock_measured_resolution_ns=1,
            clock_call_overhead_ns=0.0,
            offset_quantisation_uncertainty_s=0.0,
        )


def test_record_requires_an_environment_note() -> None:
    with pytest.raises(ConfigurationError, match="environment_note is mandatory"):
        run_benchmark(
            {"noop": lambda: None},
            label="x",
            environment_note="  ",
            backend_kind="simulated",
            backend_driver="",
            n_repeats=1,
            clock_samples=50,
        )


def test_promote_to_hardware_always_refuses() -> None:
    record = _record()
    with pytest.raises(HardwareClaimError, match="cannot be presented as a hardware"):
        record.promote_to_hardware()


def test_record_text_declares_the_table_unstable_and_the_rows_unmeasured() -> None:
    text = _record().as_text()
    assert "UNSTABLE" in text
    assert "LEVEL 4 HARDWARE TABLE -- EMPTY AND UNMEASURED" in text
    assert text.count("unmeasured") >= 5
    assert "is_hardware          : False" in text
    assert "allantools" in text
    assert "ntplib" in text


def test_record_measures_host_values() -> None:
    record = _record()
    assert record.stages[0].n == 2
    assert record.stages[0].mean_s >= 0.0
    assert record.tracemalloc_peak_bytes > 0.0
    assert record.maxrss_bytes > 0.0
    assert record.clock_measured_resolution_ns >= 1
    assert record.offset_quantisation_uncertainty_s > 0.0
    assert record.host["numpy"]
    assert record.host["allantools"] == "not installed"


def test_record_json_round_trip(tmp_path: Path) -> None:
    record = _record()
    path = record.write(tmp_path / "bench.json")
    payload = json.loads(path.read_text())
    assert payload["is_hardware"] is False
    assert len(payload["hardware"]) == 5
    assert all(row["value"] is None for row in payload["hardware"])
    assert all(row["status"] == "unmeasured" for row in payload["hardware"])


def test_time_stage_statistics_are_ordered() -> None:
    timing = time_stage("spin", lambda: sum(range(2000)), n_repeats=9)
    assert timing.n == 9
    assert timing.min_s <= timing.median_s <= timing.max_s
    assert timing.min_s <= timing.mean_s <= timing.max_s
    assert timing.p95_s <= timing.max_s
    assert timing.stdev_s >= 0.0


def test_time_stage_single_repeat_has_zero_stdev() -> None:
    timing = time_stage("one", lambda: None, n_repeats=1)
    assert timing.stdev_s == 0.0


def test_time_stage_and_run_benchmark_validate() -> None:
    with pytest.raises(ConfigurationError, match="n_repeats"):
        time_stage("x", lambda: None, n_repeats=0)
    with pytest.raises(ConfigurationError, match="at least one stage"):
        run_benchmark({}, label="x", environment_note="n", backend_kind="simulated",
                      backend_driver="")
