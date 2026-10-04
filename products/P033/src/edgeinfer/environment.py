"""Capture of the machine a number was measured on.

Every latency figure this package produces is written out with the
environment it came from. The reason is specific: this repository's own
numbers were measured in a **shared single-core cloud container**, which is
not a representative edge target, and a reader who sees a latency without its
environment will reasonably assume otherwise.

:func:`describe_environment` returns a one-line string for report headers and
:func:`environment_record` returns the same information as a dict for JSON
output. Neither reads anything off the network and neither requires a
non-stdlib dependency.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import asdict, dataclass

__all__ = [
    "JETSON_ORIN_NANO_COLUMN",
    "EnvironmentRecord",
    "clock_resolution_s",
    "describe_environment",
    "environment_record",
]

#: The Jetson Orin Nano result column is declared here and deliberately left
#: empty. Filling it requires running ``validation/validate_performance.py``
#: on the device itself and pasting its raw output; no number in this
#: repository may be extrapolated into it, taken from a vendor datasheet, or
#: produced by the simulated backend.
JETSON_ORIN_NANO_COLUMN: str = ""


@dataclass(frozen=True)
class EnvironmentRecord:
    """Machine and interpreter identification for a measurement."""

    platform: str
    machine: str
    processor: str
    python_version: str
    cpu_count_logical: int | None
    cpu_count_available: int | None
    clock_name: str
    clock_resolution_s: float
    clock_monotonic: bool
    shared_host: bool
    note: str

    def one_line(self) -> str:
        """Single-line description for a report header."""
        shared = "shared" if self.shared_host else "unshared (unverified)"
        return (
            f"{self.platform} {self.machine}, {self.cpu_count_available} CPU core(s) "
            f"available of {self.cpu_count_logical} logical, {shared} host, "
            f"Python {self.python_version}, clock {self.clock_name} "
            f"resolution {self.clock_resolution_s:.3g} s. {self.note}"
        )

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable dict."""
        return asdict(self)


def clock_resolution_s(clock: str = "perf_counter") -> float:
    """Resolution of a :mod:`time` clock [s], from :func:`time.get_clock_info`.

    The resolution is the quantisation step of the timer and is carried into
    the uncertainty budget in :mod:`edgeinfer.uncertainty` as a rectangular
    error term.
    """
    return float(time.get_clock_info(clock).resolution)


def _available_cpus() -> int | None:
    sched = getattr(os, "sched_getaffinity", None)
    if sched is not None:
        try:
            return len(sched(0))
        except OSError:
            pass
    return os.cpu_count()


def environment_record(shared_host: bool = True, note: str = "") -> EnvironmentRecord:
    """Capture the current environment.

    Parameters
    ----------
    shared_host
        Whether other workloads are known to share the host. Defaults to
        ``True`` because that is the honest default for a cloud container and
        because the wrong direction of this flag makes a tail latency look
        like a property of the code.
    note
        Extra text appended to the one-line description.
    """
    info = time.get_clock_info("perf_counter")
    return EnvironmentRecord(
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor() or "unknown",
        python_version=sys.version.split()[0],
        cpu_count_logical=os.cpu_count(),
        cpu_count_available=_available_cpus(),
        clock_name="perf_counter",
        clock_resolution_s=float(info.resolution),
        clock_monotonic=bool(info.monotonic),
        shared_host=shared_host,
        note=note,
    )


def describe_environment(shared_host: bool = True, note: str = "") -> str:
    """One-line environment description for a report header."""
    return environment_record(shared_host=shared_host, note=note).one_line()
