"""Schema declaration and header validation."""

from __future__ import annotations

import pytest

from deadlinerisk.schema import (
    JOB_COLUMNS,
    MONITOR_COLUMNS,
    N_CORES,
    SchemaError,
    validate_job_header,
    validate_monitor_header,
)


def test_job_columns_are_the_eight_the_campaign_writes() -> None:
    assert JOB_COLUMNS == (
        "job_index",
        "release_s",
        "start_delay_s",
        "end_s",
        "exec_s",
        "response_s",
        "missed",
        "dropped_before",
    )


def test_monitor_columns_carry_one_clock_per_core() -> None:
    clock_columns = [name for name in MONITOR_COLUMNS if name.startswith("clk")]
    # Six per-core columns plus clk_mean_khz and clk_min_khz.
    assert len(clock_columns) == N_CORES + 2
    assert MONITOR_COLUMNS[-1] == "load1"


def test_valid_headers_pass() -> None:
    validate_job_header(list(JOB_COLUMNS))
    validate_monitor_header(list(MONITOR_COLUMNS))


def test_whitespace_in_a_header_is_tolerated() -> None:
    validate_job_header([f" {name} " for name in JOB_COLUMNS])


def test_missing_column_is_named_in_the_message() -> None:
    with pytest.raises(SchemaError, match="dropped_before"):
        validate_job_header(list(JOB_COLUMNS[:-1]))


def test_extra_column_is_named_in_the_message() -> None:
    with pytest.raises(SchemaError, match="surprise"):
        validate_job_header([*JOB_COLUMNS, "surprise"])


def test_reordered_header_is_rejected_with_its_own_reason() -> None:
    reordered = [JOB_COLUMNS[1], JOB_COLUMNS[0], *JOB_COLUMNS[2:]]
    with pytest.raises(SchemaError, match="wrong order"):
        validate_job_header(reordered)
