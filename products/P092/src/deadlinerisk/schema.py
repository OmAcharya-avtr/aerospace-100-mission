"""Column schemas for the campaign's CSV files, and the validation of them.

The campaign writer is ``data/campaign.py``. Its two table shapes are declared
here once, so a loader cannot silently accept a file whose columns have moved.
A file with the right column names in the wrong order is rejected: the loader
indexes by name, but a reordered header is evidence that the file was not
written by the committed campaign script and nothing downstream should assume
otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Number of CPU cores the campaign monitor samples clocks from.
N_CORES = 6

#: Columns of ``jobs_<run>.csv``, one row per RELEASED-AND-EXECUTED job.
JOB_COLUMNS: tuple[str, ...] = (
    "job_index",
    "release_s",
    "start_delay_s",
    "end_s",
    "exec_s",
    "response_s",
    "missed",
    "dropped_before",
)

#: Columns of ``mon_<run>.csv``, one row per 1 Hz monitor sample.
MONITOR_COLUMNS: tuple[str, ...] = (
    "t_s",
    "total_busy_j",
    "task_busy_j",
    "stress_busy_j",
    "foreign_busy_j",
    "foreign_pct",
    "task_pct",
    "stress_pct",
    *(f"clk{core}_khz" for core in range(N_CORES)),
    "clk_mean_khz",
    "clk_min_khz",
    "tj_mdegC",
    "load1",
)


class SchemaError(ValueError):
    """A CSV file's header or values do not match the declared schema."""


def _check_header(found: Sequence[str], expected: tuple[str, ...], what: str) -> None:
    found = tuple(name.strip() for name in found)
    if found == expected:
        return
    missing = [name for name in expected if name not in found]
    extra = [name for name in found if name not in expected]
    if missing or extra:
        raise SchemaError(
            f"{what} header does not match the declared schema. "
            f"missing={missing} unexpected={extra}. "
            f"Expected exactly, in order: {list(expected)}"
        )
    raise SchemaError(
        f"{what} header has the right names in the wrong order. "
        f"got={list(found)} expected={list(expected)}. The loader indexes by name, "
        "but a reordered header means the file was not written by the committed "
        "campaign script, so nothing downstream may assume its semantics."
    )


def validate_job_header(found: Sequence[str]) -> None:
    """Raise :class:`SchemaError` unless ``found`` is exactly :data:`JOB_COLUMNS`."""
    _check_header(found, JOB_COLUMNS, "jobs CSV")


def validate_monitor_header(found: Sequence[str]) -> None:
    """Raise :class:`SchemaError` unless ``found`` is exactly :data:`MONITOR_COLUMNS`."""
    _check_header(found, MONITOR_COLUMNS, "monitor CSV")
