"""Peak-memory measurement.

Two mechanisms, with different meanings, and the difference matters:

``tracemalloc`` (:func:`measure_peak_python_bytes`)
    Counts bytes allocated **through the CPython allocator**, which is what
    NumPy array allocation goes through. It gives an exact peak of Python-side
    allocation, reproducible to the byte, and it sees nothing allocated by a
    C++ runtime's own allocator. ``onnxruntime`` allocates its arena outside
    CPython, so a ``tracemalloc`` figure for an ``onnxruntime`` session is a
    measurement of the input/output marshalling, not of the model's working
    set. :mod:`tracemalloc` documentation, CPython standard library.

Resident set size (:func:`resident_set_bytes`)
    Read from ``/proc/self/status`` (``VmHWM``, the high-water mark of
    resident set size, in kB --- Linux ``proc(5)`` manual page). It sees every
    allocator but it is a whole-process figure: it includes the interpreter,
    NumPy, and anything else resident, so only the *increase* across a
    measured region is informative, and even that is confounded by the fact
    that a page returned to the allocator is not returned to the kernel.

The peak-memory validation in this repository therefore uses ``tracemalloc``
against a **known allocation pattern** whose byte count is derived by hand
(``validation/validate_memory.py``). That is a check of the measurement path,
not a claim that ``onnxruntime``'s peak working set has been measured. What is
missing to measure a real model's peak working set on the target is stated in
``README.md`` under Limitations.
"""

from __future__ import annotations

import gc
import tracemalloc
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

__all__ = [
    "PeakMemory",
    "measure_peak_python_bytes",
    "resident_set_bytes",
    "resident_set_high_water_bytes",
    "track_peak_python_bytes",
]

T = TypeVar("T")

_STATUS = Path("/proc/self/status")


@dataclass(frozen=True)
class PeakMemory:
    """Result of a peak-memory measurement.

    Attributes
    ----------
    peak_bytes
        Peak allocated bytes observed inside the measured region [B].
    baseline_bytes
        Bytes already allocated when the region began [B].
    peak_above_baseline_bytes
        ``peak_bytes - baseline_bytes`` [B]: the region's own contribution.
    method
        Verbatim measurement-method string for a report.
    """

    peak_bytes: int
    baseline_bytes: int
    peak_above_baseline_bytes: int
    method: str

    def summary_lines(self) -> list[str]:
        """Human-readable result, every number with its unit."""
        return [
            f"peak (absolute)         : {self.peak_bytes} B",
            f"baseline at entry       : {self.baseline_bytes} B",
            f"peak above baseline     : {self.peak_above_baseline_bytes} B",
            f"method                  : {self.method}",
        ]


def resident_set_bytes() -> int | None:
    """Current resident set size from ``/proc/self/status`` [B], or ``None``.

    ``None`` on a platform without ``/proc``. ``VmRSS`` is reported in kB by
    the kernel (Linux ``proc(5)``), and is converted here with 1 kB = 1024 B,
    which is the kernel's own convention for this field.
    """
    return _status_field("VmRSS")


def resident_set_high_water_bytes() -> int | None:
    """Peak resident set size (``VmHWM``) from ``/proc/self/status`` [B].

    Process lifetime high-water mark; it never decreases, so it is only
    informative when sampled before and after a region and even then it is a
    lower bound on that region's peak.
    """
    return _status_field("VmHWM")


def _status_field(key: str) -> int | None:
    try:
        text = _STATUS.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(key + ":"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


@contextmanager
def track_peak_python_bytes(collect_first: bool = True):
    """Context manager yielding a one-element list that receives a :class:`PeakMemory`.

    Uses :mod:`tracemalloc`. If tracing is already active the existing trace is
    left running and the measurement is taken relative to the current
    snapshot, so nesting is safe.

    Parameters
    ----------
    collect_first
        Run :func:`gc.collect` before taking the baseline, so that garbage from
        earlier work does not inflate it. Defaults to ``True`` for
        reproducibility.
    """
    if collect_first:
        gc.collect()
    already = tracemalloc.is_tracing()
    if not already:
        tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    holder: list[PeakMemory] = []
    try:
        yield holder
    finally:
        _current, peak = tracemalloc.get_traced_memory()
        if not already:
            tracemalloc.stop()
        holder.append(
            PeakMemory(
                peak_bytes=int(peak),
                baseline_bytes=int(baseline),
                peak_above_baseline_bytes=int(peak - baseline),
                method=(
                    "tracemalloc.get_traced_memory peak after reset_peak; counts CPython "
                    "allocator traffic only, not a C++ runtime's own arena; gc.collect "
                    f"before baseline = {collect_first}"
                ),
            )
        )


def measure_peak_python_bytes(
    fn: Callable[..., T], *args: Any, collect_first: bool = True, **kwargs: Any
) -> tuple[T, PeakMemory]:
    """Call ``fn(*args, **kwargs)`` and return its result with the peak memory.

    Returns
    -------
    (result, peak)
        ``result`` is whatever ``fn`` returned; ``peak`` is a
        :class:`PeakMemory`.

    Notes
    -----
    The returned object is itself still allocated when the peak is read, so a
    function that returns a large array has that array inside its own peak.
    That is the intended accounting for an inference call, whose output tensor
    is part of its working set.
    """
    with track_peak_python_bytes(collect_first=collect_first) as holder:
        result = fn(*args, **kwargs)
    return result, holder[0]
