"""The benchmark harness: a host table that is measured, a hardware table that is not.

What this records and why the method is in the file
---------------------------------------------------
A timing number without its measurement method and its environment is not
evidence. Every record this module writes carries the clock used, its reported
and measured resolution, the measured cost of a timing-call pair, the host
platform and library versions, a mandatory free-text environment note, and
:attr:`~timetransfer.hal.BackendInfo.is_hardware`.

Method
------
Each stage is bracketed by two ``time.perf_counter_ns`` reads. A stage
duration is therefore a difference of two integer nanosecond timestamps,
converted to seconds once. Stages are repeated ``n_repeats`` times and every
repetition contributes one sample; the table reports the median and the 95th
percentile as well as the mean, because an estimator's tail is what an
acquisition window cares about and a mean destroys it.

Memory: ``tracemalloc`` peak for Python-level allocation during the run, and
``resource.getrusage(RUSAGE_SELF).ru_maxrss`` for the process high-water mark.
``ru_maxrss`` is kibibytes on Linux and bytes on macOS; the record states which
conversion was applied and derives it from ``sys.platform`` rather than
assuming.

**The hardware rows are empty and marked unmeasured**
-----------------------------------------------------
:data:`HARDWARE_STAGES` names the five figures that Level 4 requires from the
Jetson Orin Nano itself. :func:`hardware_rows` returns one row per stage with
**every numeric field ``None``** and a status of ``"unmeasured"`` plus the
reason. They are not filled from simulation, and
:meth:`BenchmarkRecord.promote_to_hardware` raises
:class:`~timetransfer.errors.HardwareClaimError` unless the record was produced
on a backend whose ``is_hardware`` is true -- which no backend in this
repository is.

This is deliberate and it is the reason this product is labelled **Level 3,
hardware-pending** and never Level 4. A simulated timing figure presented as a
board measurement is the single most damaging thing this package could emit,
so the guard is an exception and the empty rows are in the published table
rather than omitted from it.

Timings are unstable
--------------------
Every number in the host table moves between runs on a shared two-core
container. The table is declared unstable where it is printed, and no ratio
derived from it appears in prose anywhere in this repository.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .errors import ConfigurationError, HardwareClaimError
from .timebase import NS_PER_S, clock_report

__all__ = [
    "HARDWARE_STAGES",
    "BenchmarkRecord",
    "HardwareRow",
    "StageTiming",
    "hardware_rows",
    "run_benchmark",
    "time_stage",
]

HARDWARE_STAGES: tuple[tuple[str, str], ...] = (
    (
        "exchange_round_trip_s",
        "wall-clock time for one two-way exchange driven by a real timestamping device",
    ),
    (
        "timestamp_resolution_ns",
        "resolution the board's timestamping path actually resolves, measured on the board",
    ),
    (
        "estimator_latency_s",
        "time to run the full estimation path on the board's CPU for one window",
    ),
    (
        "peak_rss_bytes",
        "process high-water memory on the board under the acquisition workload",
    ),
    (
        "cpu_utilisation_fraction",
        "fraction of one Orin Nano core consumed by a continuous 1 Hz acquisition",
    ),
)
"""The five figures Level 4 requires from the Jetson Orin Nano, with what each means."""


@dataclass(frozen=True)
class StageTiming:
    """Timing samples for one named stage, in seconds.

    Attributes
    ----------
    stage:
        Stage name.
    n:
        Repetitions measured.
    mean_s, median_s, p95_s, min_s, max_s, stdev_s:
        Statistics over the repetitions [s].
    """

    stage: str
    n: int
    mean_s: float
    median_s: float
    p95_s: float
    min_s: float
    max_s: float
    stdev_s: float

    def as_row(self) -> str:
        """One fixed-width table row."""
        return (
            f"{self.stage:>26} {self.n:>6d} {self.mean_s * 1e3:>11.4f} "
            f"{self.median_s * 1e3:>11.4f} {self.p95_s * 1e3:>11.4f} "
            f"{self.min_s * 1e3:>11.4f} {self.max_s * 1e3:>11.4f}"
        )


@dataclass(frozen=True)
class HardwareRow:
    """One Level 4 figure that has not been measured.

    Every numeric field is ``None``. This is the point of the class: a row that
    cannot hold a number cannot be filled in from simulation by accident.

    Attributes
    ----------
    stage:
        The figure's name, from :data:`HARDWARE_STAGES`.
    description:
        What the figure means.
    value:
        Always ``None``.
    units:
        The units the figure would be in.
    status:
        Always ``"unmeasured"``.
    reason:
        Why it is unmeasured.
    """

    stage: str
    description: str
    units: str
    reason: str
    value: None = None
    status: str = "unmeasured"

    def as_row(self) -> str:
        """One fixed-width table row, with an em dash where the number would go."""
        return f"{self.stage:>26} {'--':>14} {self.units:>10} {self.status:>12}"


_HARDWARE_UNITS: dict[str, str] = {
    "exchange_round_trip_s": "s",
    "timestamp_resolution_ns": "ns",
    "estimator_latency_s": "s",
    "peak_rss_bytes": "B",
    "cpu_utilisation_fraction": "1",
}

_HARDWARE_REASON = (
    "no Jetson Orin Nano was reachable from the environment this product was built in; "
    "filling this row from simulation would be a false hardware claim"
)


def hardware_rows(reason: str = _HARDWARE_REASON) -> tuple[HardwareRow, ...]:
    """The Level 4 hardware table, every row empty and marked unmeasured.

    Parameters
    ----------
    reason:
        Why the rows are empty, recorded in each one.
    """
    if not str(reason).strip():
        raise ConfigurationError("hardware rows must carry a reason for being unmeasured")
    return tuple(
        HardwareRow(
            stage=stage,
            description=description,
            units=_HARDWARE_UNITS[stage],
            reason=str(reason),
        )
        for stage, description in HARDWARE_STAGES
    )


def time_stage(stage: str, fn: Callable[[], object], *, n_repeats: int = 5) -> StageTiming:
    """Time ``fn`` ``n_repeats`` times and summarise the samples.

    Parameters
    ----------
    stage:
        Name recorded in the result.
    fn:
        Zero-argument callable; its return value is discarded.
    n_repeats:
        Repetitions, >= 1.

    Returns
    -------
    StageTiming
    """
    if int(n_repeats) < 1:
        raise ConfigurationError(f"n_repeats must be >= 1, got {n_repeats!r}")
    samples_ns: list[int] = []
    for _ in range(int(n_repeats)):
        start = time.perf_counter_ns()
        fn()
        samples_ns.append(time.perf_counter_ns() - start)
    samples = np.asarray(samples_ns, dtype=np.float64) / NS_PER_S
    return StageTiming(
        stage=str(stage),
        n=int(n_repeats),
        mean_s=float(np.mean(samples)),
        median_s=float(np.median(samples)),
        p95_s=float(np.percentile(samples, 95.0)),
        min_s=float(np.min(samples)),
        max_s=float(np.max(samples)),
        stdev_s=float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0,
    )


def _maxrss_bytes() -> tuple[float, str]:
    """Process high-water memory [B] and the conversion used."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(raw), "ru_maxrss interpreted as bytes (darwin)"
    return float(raw) * 1024.0, "ru_maxrss interpreted as kibibytes (linux)"


def _cpu_count() -> str:
    """CPUs this process may run on, as a string."""
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        return str(len(getter(0)))
    return str(os.cpu_count() or "unknown")


def _host() -> dict[str, str]:
    """Platform and library versions, measured in this process."""
    import importlib.metadata as md

    def version(name: str) -> str:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return "not installed"

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count_available": _cpu_count(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit-learn": version("scikit-learn"),
        "allantools": version("allantools"),
        "ntplib": version("ntplib"),
    }


@dataclass
class BenchmarkRecord:
    """One benchmark run: the host table measured, the hardware table empty.

    Attributes
    ----------
    label:
        Short name for the run.
    environment_note:
        Mandatory free text describing the machine's state. A benchmark without
        it is not interpretable, so an empty note is refused.
    backend_kind, backend_driver, is_hardware:
        Backend identity.
    stages:
        The measured host timings.
    tracemalloc_peak_bytes, maxrss_bytes, maxrss_note:
        Measured memory and the conversion applied.
    clock_reported_resolution_ns, clock_measured_resolution_ns,
    clock_call_overhead_ns, offset_quantisation_uncertainty_s:
        The timebase measurement and its derived error term.
    host:
        Platform and versions.
    hardware:
        The Level 4 rows, every value ``None`` and status ``"unmeasured"``.
    method:
        How each table was measured, carried with the numbers.
    caveat:
        The standing warning about what this record does and does not establish.
    """

    label: str
    environment_note: str
    backend_kind: str
    backend_driver: str
    is_hardware: bool
    stages: tuple[StageTiming, ...]
    tracemalloc_peak_bytes: float
    maxrss_bytes: float
    maxrss_note: str
    clock_reported_resolution_ns: int
    clock_measured_resolution_ns: int
    clock_call_overhead_ns: float
    offset_quantisation_uncertainty_s: float
    host: dict[str, str] = field(default_factory=dict)
    hardware: tuple[HardwareRow, ...] = field(default_factory=hardware_rows)
    method: dict[str, str] = field(default_factory=dict)
    caveat: str = ""

    def __post_init__(self) -> None:
        if not str(self.environment_note).strip():
            raise ConfigurationError(
                "environment_note is mandatory: a timing table without the state of the "
                "machine it was measured on is not interpretable"
            )
        if self.is_hardware:
            raise HardwareClaimError(
                "BenchmarkRecord was constructed with is_hardware=true, but no backend in "
                "this repository is hardware. If this is a real board, the record must be "
                "produced by a driver whose name is not a stub."
            )
        if any(row.value is not None or row.status != "unmeasured" for row in self.hardware):
            raise HardwareClaimError(
                "a hardware row carries a value or a status other than 'unmeasured'; "
                "hardware rows are empty until measured on the board"
            )

    def promote_to_hardware(self) -> None:
        """Refuse to label this record a hardware measurement.

        Always raises :class:`~timetransfer.errors.HardwareClaimError` for a
        record produced on a non-hardware backend, which is every record this
        repository can produce. The method exists so that the refusal is
        testable rather than a convention.
        """
        raise HardwareClaimError(
            f"record {self.label!r} was produced on backend "
            f"{self.backend_kind}/{self.backend_driver or 'n/a'} with is_hardware=False; "
            f"it cannot be presented as a hardware measurement. Level 4 requires the five "
            f"figures in HARDWARE_STAGES measured on the Jetson Orin Nano itself."
        )

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable dictionary."""
        return asdict(self)

    def as_text(self) -> str:
        """The two tables and the provenance, as committed to ``validation/``."""
        lines = [
            f"benchmark record: {self.label}",
            f"backend              : {self.backend_kind}/{self.backend_driver or 'n/a'}",
            f"is_hardware          : {self.is_hardware}",
            f"environment note     : {self.environment_note}",
            "",
            "host timing table (UNSTABLE: wall-clock on a shared container; "
            "every value moves between runs)",
            f"{'stage':>26} {'n':>6} {'mean ms':>11} {'median ms':>11} "
            f"{'p95 ms':>11} {'min ms':>11} {'max ms':>11}",
        ]
        lines.extend(stage.as_row() for stage in self.stages)
        lines.extend(
            [
                "",
                "host resources",
                f"  tracemalloc peak   : {self.tracemalloc_peak_bytes:,.0f} B",
                f"  maxrss             : {self.maxrss_bytes:,.0f} B ({self.maxrss_note})",
                "",
                "host timebase",
                f"  reported resolution: {self.clock_reported_resolution_ns} ns",
                f"  measured resolution: {self.clock_measured_resolution_ns} ns",
                f"  call-pair overhead : {self.clock_call_overhead_ns:.1f} ns",
                f"  u_q(offset)        : {self.offset_quantisation_uncertainty_s:.6e} s",
                "",
                "LEVEL 4 HARDWARE TABLE -- EMPTY AND UNMEASURED",
                f"{'figure':>26} {'value':>14} {'units':>10} {'status':>12}",
            ]
        )
        lines.extend(row.as_row() for row in self.hardware)
        lines.extend(
            [
                f"  reason: {self.hardware[0].reason}" if self.hardware else "",
                "",
                "host",
            ]
        )
        lines.extend(f"  {key:<20}: {value}" for key, value in sorted(self.host.items()))
        if self.method:
            lines.append("")
            lines.append("method")
            lines.extend(f"  {key:<20}: {value}" for key, value in sorted(self.method.items()))
        if self.caveat:
            lines.extend(["", f"caveat: {self.caveat}"])
        return "\n".join(line for line in lines if line is not None)

    def write(self, path: str | Path) -> Path:
        """Write :meth:`as_dict` to ``path`` as JSON and return the path."""
        target = Path(path)
        target.write_text(json.dumps(self.as_dict(), indent=1, sort_keys=True, default=str) + "\n")
        return target


def run_benchmark(
    stages: dict[str, Callable[[], object]],
    *,
    label: str,
    environment_note: str,
    backend_kind: str,
    backend_driver: str,
    n_repeats: int = 5,
    clock_samples: int = 2000,
) -> BenchmarkRecord:
    """Time every stage, measure memory and the clock, and build the record.

    Parameters
    ----------
    stages:
        Mapping from stage name to a zero-argument callable.
    label:
        Short name for the run.
    environment_note:
        Mandatory description of the machine's state.
    backend_kind, backend_driver:
        Backend identity, recorded with the numbers.
    n_repeats:
        Repetitions per stage, >= 1.
    clock_samples:
        Read pairs for the clock measurement, >= 2.

    Returns
    -------
    BenchmarkRecord
        With the host table measured and the hardware table empty.
    """
    if not stages:
        raise ConfigurationError("run_benchmark needs at least one stage")
    clock = clock_report(int(clock_samples))
    started_tracemalloc = not tracemalloc.is_tracing()
    if started_tracemalloc:
        tracemalloc.start()
    timings: list[StageTiming] = []
    for name, fn in stages.items():
        timings.append(time_stage(name, fn, n_repeats=int(n_repeats)))
    _, peak = tracemalloc.get_traced_memory()
    if started_tracemalloc:
        tracemalloc.stop()
    maxrss, maxrss_note = _maxrss_bytes()
    return BenchmarkRecord(
        label=str(label),
        environment_note=str(environment_note),
        backend_kind=str(backend_kind),
        backend_driver=str(backend_driver),
        is_hardware=False,
        stages=tuple(timings),
        tracemalloc_peak_bytes=float(peak),
        maxrss_bytes=maxrss,
        maxrss_note=maxrss_note,
        clock_reported_resolution_ns=int(clock.reported_resolution_ns),
        clock_measured_resolution_ns=int(clock.measured_resolution_ns),
        clock_call_overhead_ns=float(clock.call_overhead_ns),
        offset_quantisation_uncertainty_s=float(clock.offset_quantisation_uncertainty_s),
        host=_host(),
        hardware=hardware_rows(),
        method={
            "timing": "two perf_counter_ns reads per repetition, integer ns differenced once",
            "memory": "tracemalloc peak over the whole run plus ru_maxrss high-water",
            "clock": "smallest non-zero gap over read pairs; median zero-interval reading",
            "hardware": "not measured; rows are empty by construction",
        },
        caveat=(
            "nothing in this record is a hardware measurement. The backend's is_hardware "
            "flag is False, the Level 4 table is empty, and this product is labelled "
            "Level 3, hardware-pending."
        ),
    )
