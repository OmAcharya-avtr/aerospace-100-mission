"""Performance benchmark harness (Level 3 requirement).

Every number this module produces is a WORKSTATION/CONTAINER number, measured
on whatever machine ran it, and the environment record is emitted with the
results so a figure can never be read without knowing where it came from.

Measurement method
------------------
* Wall-clock via ``time.perf_counter`` -- the highest-resolution monotonic
  clock Python exposes.  Its resolution is measured, not assumed, and is
  reported as ``clock_resolution_s``.
* Each stage is repeated ``repeats`` times.  The MEDIAN and the MINIMUM are
  both reported: the minimum is the cleanest estimate of the work itself on a
  shared machine, the median is closer to what a user experiences, and the
  maximum exposes interference.  The mean alone is not reported, because on a
  contended single-core container it is dominated by scheduling noise.
* No warm-up run is discarded silently.  ``repeats`` includes the first call,
  and the spread between minimum and maximum shows the warm-up cost.

Peak memory is sampled with ``tracemalloc``, which counts Python-level
allocations only.  NumPy array data IS counted (it is allocated through the
Python allocator), but memory held by C extensions outside it -- including
parts of the SGP4 C++ propagator -- is not.  The figure is therefore a lower
bound and is labelled as such in the output.
"""

from __future__ import annotations

import platform
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = ["BenchmarkResult", "Environment", "environment", "clock_resolution_s",
           "benchmark"]


@dataclass(frozen=True)
class Environment:
    """Where a benchmark number came from."""

    python_version: str
    platform: str
    machine: str
    processor: str
    numpy_version: str
    clock_resolution_s: float
    note: str = ("workstation/container measurement; not a measurement on target "
                 "flight or edge hardware")

    def format_block(self) -> str:
        """Multi-line environment record (str)."""
        return "\n".join(f"{k}: {v}" for k, v in asdict(self).items())


def clock_resolution_s(n_probe: int = 2000) -> float:
    """Measured resolution of ``time.perf_counter`` [s].

    Takes ``n_probe`` back-to-back readings and returns the smallest non-zero
    difference.  Returns ``perf_counter`` 's advertised resolution if every
    difference is zero, which happens on very coarse clocks.
    """
    if n_probe < 10:
        raise ValueError(f"n_probe must be >= 10, got {n_probe}")
    samples = np.array([time.perf_counter() for _ in range(n_probe)])
    diffs = np.diff(samples)
    nonzero = diffs[diffs > 0.0]
    if nonzero.size == 0:  # pragma: no cover - only on a pathological clock
        return float(time.get_clock_info("perf_counter").resolution)
    return float(nonzero.min())


def environment() -> Environment:
    """Capture the environment record."""
    return Environment(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor() or "unknown",
        numpy_version=np.__version__,
        clock_resolution_s=clock_resolution_s(),
    )


@dataclass
class BenchmarkResult:
    """Timing and memory for one named stage.

    Attributes
    ----------
    name : stage label.
    repeats : number of timed calls, including the first.
    min_s, median_s, max_s : wall-clock statistics [s].
    peak_python_bytes : ``tracemalloc`` peak during a single call [B], a LOWER
        BOUND (see the module docstring).
    samples_s : every timing sample [s].
    """

    name: str
    repeats: int
    min_s: float
    median_s: float
    max_s: float
    peak_python_bytes: int
    samples_s: list[float] = field(default_factory=list)

    def format_row(self) -> str:
        """One fixed-width line."""
        return (f"{self.name:<34}{self.repeats:>5}{self.min_s * 1e3:>12.3f}"
                f"{self.median_s * 1e3:>12.3f}{self.max_s * 1e3:>12.3f}"
                f"{self.peak_python_bytes / 1024.0:>14.1f}")

    @staticmethod
    def header() -> str:
        """Matching header for :meth:`format_row`."""
        return (f"{'stage':<34}{'rep':>5}{'min[ms]':>12}{'med[ms]':>12}"
                f"{'max[ms]':>12}{'peakPy[KiB]':>14}")


def benchmark(name: str, fn: Callable[[], object], repeats: int = 3,
              measure_memory: bool = True) -> BenchmarkResult:
    """Time ``fn`` ``repeats`` times and optionally record its peak allocation.

    ``fn`` takes no arguments and its return value is discarded.  Memory is
    measured in a separate, additional call so the timing samples are not
    perturbed by ``tracemalloc`` overhead; that call is NOT included in the
    timing statistics.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    peak = 0
    if measure_memory:
        tracemalloc.start()
        try:
            fn()
            _, peak_b = tracemalloc.get_traced_memory()
            peak = int(peak_b)
        finally:
            tracemalloc.stop()
    arr = np.asarray(samples)
    return BenchmarkResult(name=name, repeats=repeats, min_s=float(arr.min()),
                           median_s=float(np.median(arr)), max_s=float(arr.max()),
                           peak_python_bytes=peak, samples_s=samples)
