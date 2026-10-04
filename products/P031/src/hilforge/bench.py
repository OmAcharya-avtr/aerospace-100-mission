"""The benchmark harness: latency, memory, throughput, with its method stated.

What this records, and why the method is in the file
----------------------------------------------------
A benchmark number without its measurement method and environment is not
evidence. Every record this module writes carries:

* the clock used, its reported and measured resolution, and the measured cost
  of a timing-call pair;
* the resolution-derived standard uncertainty on a duration
  (``delta / sqrt(6)``, see :mod:`hilforge.timebase`);
* the uncertainty budget on the mean, combined per JCGM 100:2008;
* the host: platform, Python, NumPy, CPU count, and a free-text
  ``environment_note`` the caller must supply;
* whether the backend was hardware (:attr:`hilforge.hal.BackendInfo.is_hardware`).

Measurement method
------------------
Latency: each loop stage is bracketed by two ``time.perf_counter_ns`` reads
(:class:`hilforge.timebase.MonotonicTimebase`), so a stage duration is a
difference of two integer nanosecond timestamps converted once to float
seconds. No averaging over repetitions is done inside a stage: every iteration
contributes one sample to each stage's histogram, because the distribution
tail is the thing a deadline cares about and an average destroys it.

Memory: ``tracemalloc`` peak for Python-level allocation during the run, and
``resource.getrusage(RUSAGE_SELF).ru_maxrss`` for the process high-water mark.
``ru_maxrss`` is in kibibytes on Linux and bytes on macOS; the record states
which conversion was applied and it is derived from ``sys.platform``, not
assumed.

Throughput: completed iterations divided by the wall-clock span of the run,
measured with the same clock.

What a record does not establish
--------------------------------
Nothing here measures hardware unless the backend was hardware. A record with
``is_hardware = false`` is a host-side measurement of the harness. Promoting
such a number to a hardware claim is the defect this field exists to prevent.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .errors import ConfigurationError
from .loop import STAGES, HilLoop, LoopConfig, RunRecord
from .timebase import MonotonicTimebase, clock_report
from .timing import TimingUncertainty, timing_uncertainty

__all__ = ["BenchmarkRecord", "run_benchmark", "write_record"]


def _cpu_count() -> str:
    """CPUs this process may run on, as a string, or ``"unknown"``."""
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        return str(len(getter(0)))
    return str(os.cpu_count() or "unknown")


def _maxrss_bytes() -> tuple[float, str]:
    """Process high-water memory [B] and the conversion used."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(raw), "ru_maxrss interpreted as bytes (darwin)"
    return float(raw) * 1024.0, "ru_maxrss interpreted as kibibytes (linux)"


@dataclass
class BenchmarkRecord:
    """One benchmark run, with its method and environment.

    Every field is either measured in this process or supplied by the caller.
    Nothing is inferred from a datasheet.
    """

    label: str
    backend_kind: str
    backend_driver: str
    is_hardware: bool
    environment_note: str
    period_s: float
    n_iterations: int
    n_completed: int
    dry_run: bool
    wall_clock_s: float
    throughput_iter_s: float
    stage_summary_s: dict[str, dict[str, float]]
    total_summary_s: dict[str, float]
    direct_overruns: int
    cascade_overruns: int
    clock_reported_resolution_s: float
    clock_measured_resolution_s: float
    clock_call_overhead_s: float
    duration_resolution_uncertainty_s: float
    mean_uncertainty: dict[str, float]
    tracemalloc_peak_bytes: float
    maxrss_bytes: float
    maxrss_note: str
    host: dict[str, str] = field(default_factory=dict)
    method: dict[str, str] = field(default_factory=dict)
    caveat: str = ""

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable dictionary."""
        return asdict(self)

    def as_text(self) -> str:
        """Human-readable record, suitable for a ``validation/*.txt`` file."""
        lines = [
            f"HilForge benchmark record — {self.label}",
            "=" * 72,
            "",
            "ENVIRONMENT AND MEASUREMENT METHOD",
            f"  environment note      : {self.environment_note}",
            (
                f"  backend               : {self.backend_kind} "
                f"(driver={self.backend_driver or 'n/a'})"
            ),
            f"  is_hardware           : {self.is_hardware}",
        ]
        for key in sorted(self.host):
            lines.append(f"  {key:<21} : {self.host[key]}")
        for key in sorted(self.method):
            lines.append(f"  {key:<21} : {self.method[key]}")
        lines += [
            "",
            "CLOCK",
            f"  reported resolution   : {self.clock_reported_resolution_s:.6e} s",
            f"  measured resolution   : {self.clock_measured_resolution_s:.6e} s",
            f"  call-pair overhead    : {self.clock_call_overhead_s:.6e} s (bias)",
            (
                f"  u(duration) from res. : "
                f"{self.duration_resolution_uncertainty_s:.6e} s = delta/sqrt(6)"
            ),
            "",
            "RUN",
            f"  period                : {self.period_s:.6e} s",
            f"  iterations requested  : {self.n_iterations}",
            f"  iterations completed  : {self.n_completed}",
            f"  dry run               : {self.dry_run}",
            f"  wall clock            : {self.wall_clock_s:.6f} s",
            f"  throughput            : {self.throughput_iter_s:.2f} iterations/s",
            f"  direct overruns       : {self.direct_overruns}",
            f"  cascade overruns      : {self.cascade_overruns}",
            "",
            "LATENCY PER STAGE [s] (exact percentiles over every sample)",
            (
                f"  {'stage':<10} {'count':>8} {'mean':>12} {'p50':>12} "
                f"{'p90':>12} {'p99':>12} {'max':>12}"
            ),
        ]
        for stage in list(STAGES) + ["total"]:
            s = self.stage_summary_s.get(stage)
            if s is None:
                continue
            lines.append(
                f"  {stage:<10} {int(s['count']):>8} {s['mean_s']:>12.6e} "
                f"{s['p50_s']:>12.6e} {s['p90_s']:>12.6e} {s['p99_s']:>12.6e} "
                f"{s['max_s']:>12.6e}"
            )
        lines += [
            "",
            "UNCERTAINTY ON THE MEAN TOTAL DURATION (JCGM 100:2008 §5.1.2)",
        ]
        for key in (
            "n",
            "mean_s",
            "stdev_s",
            "u_statistical_s",
            "u_resolution_s",
            "combined_s",
            "expanded_k2_s",
        ):
            if key in self.mean_uncertainty:
                lines.append(f"  {key:<21} : {self.mean_uncertainty[key]:.9e}")
        lines += [
            "",
            "MEMORY",
            f"  tracemalloc peak      : {self.tracemalloc_peak_bytes / 1e6:.3f} MB",
            (
                f"  process max RSS       : {self.maxrss_bytes / 1e6:.3f} MB "
                f"({self.maxrss_note})"
            ),
            "",
            "CAVEAT",
        ]
        lines += [f"  {line}" for line in self.caveat.splitlines()]
        return "\n".join(lines)


_DEFAULT_CAVEAT = (
    "These are host-side measurements of the harness, taken on a shared,\n"
    "single-core cloud container with four other build agents running\n"
    "concurrently. Absolute latencies are therefore pessimistic and their\n"
    "scatter is dominated by contention outside this process. They are not\n"
    "hardware numbers and must not be quoted as such.\n"
    "Level 4 validation is NOT claimed and is not reachable from this record:\n"
    "it requires this same harness run on a Jetson Orin Nano with its raw\n"
    "output captured, and until that exists the product is Level 3,\n"
    "hardware-pending."
)


def run_benchmark(
    backend,
    *,
    period_s: float = 0.010,
    n_iterations: int = 2000,
    label: str = "simulated-backend",
    environment_note: str,
    dry_run: bool = False,
    record_signals: bool = False,
    clock_samples: int = 2000,
    caveat: str | None = None,
) -> tuple[BenchmarkRecord, RunRecord]:
    """Run the loop with measured timing and build a :class:`BenchmarkRecord`.

    Parameters
    ----------
    backend:
        Backend to benchmark. Its
        :attr:`~hilforge.hal.BackendInfo.is_hardware` flag is copied into the
        record verbatim.
    period_s:
        Loop period [s], > 0. Note that this harness does **not** sleep to the
        period: it runs the loop as fast as it can and accounts deadlines
        against ``period_s``, because sleeping would measure the sleep rather
        than the work.
    n_iterations:
        Iterations, >= 2.
    label:
        Name of the record.
    environment_note:
        Required free-text description of the machine and its load. There is
        no default, because a benchmark whose environment nobody wrote down is
        not evidence.
    dry_run:
        Benchmark the dry-run path instead of the write path.
    record_signals:
        Keep per-iteration arrays. Off by default for a long run.
    clock_samples:
        Samples for the clock characterisation, >= 2.
    caveat:
        Override the default caveat text. The default states the shared
        single-core container and that Level 4 is not claimed.

    Returns
    -------
    tuple
        ``(BenchmarkRecord, RunRecord)``.
    """
    if not (period_s > 0.0):
        raise ConfigurationError(f"period_s must be > 0, got {period_s!r}")
    if n_iterations < 2:
        raise ConfigurationError(f"n_iterations must be >= 2, got {n_iterations!r}")
    if not environment_note.strip():
        raise ConfigurationError(
            "environment_note is required: a benchmark without its environment "
            "recorded is not evidence"
        )
    from .timing import PeriodSpec

    clock = clock_report(clock_samples)
    cfg = LoopConfig(
        period=PeriodSpec(period_s=period_s),
        n_iterations=n_iterations,
        dry_run=dry_run,
        record_signals=record_signals,
    )
    loop = HilLoop(backend, cfg, measurement_clock=MonotonicTimebase())
    tracemalloc.start()
    t0 = time.perf_counter()
    run = loop.run()
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    maxrss, maxrss_note = _maxrss_bytes()

    stage_summary = {
        name: hist.summary()
        for name, hist in run.stage_histograms.items()
        if len(hist)
    }
    totals = run.durations_s()
    unc: TimingUncertainty | None = None
    if totals.size >= 2:
        unc = timing_uncertainty(
            totals,
            resolution_s=clock.measured_resolution_s,
            call_overhead_s=clock.call_overhead_s,
        )
    return (
        BenchmarkRecord(
            label=label,
            backend_kind=run.backend_kind,
            backend_driver=run.backend_driver,
            is_hardware=run.is_hardware,
            environment_note=environment_note.strip(),
            period_s=period_s,
            n_iterations=n_iterations,
            n_completed=run.n_completed,
            dry_run=dry_run,
            wall_clock_s=wall,
            throughput_iter_s=run.n_completed / wall if wall > 0 else float("nan"),
            stage_summary_s=stage_summary,
            total_summary_s=stage_summary.get("total", {}),
            direct_overruns=run.overruns.direct_count if run.overruns else -1,
            cascade_overruns=run.overruns.cascade_count if run.overruns else -1,
            clock_reported_resolution_s=clock.reported_resolution_s,
            clock_measured_resolution_s=clock.measured_resolution_s,
            clock_call_overhead_s=clock.call_overhead_s,
            duration_resolution_uncertainty_s=clock.duration_uncertainty_s,
            mean_uncertainty=(
                {
                    "n": float(unc.n),
                    "mean_s": unc.mean_s,
                    "stdev_s": unc.stdev_s,
                    "u_statistical_s": unc.u_statistical_s,
                    "u_resolution_s": unc.u_resolution_s,
                    "combined_s": unc.combined_s,
                    "expanded_k2_s": unc.expanded_k2_s,
                }
                if unc is not None
                else {}
            ),
            tracemalloc_peak_bytes=float(peak),
            maxrss_bytes=maxrss,
            maxrss_note=maxrss_note,
            host={
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "cpu_count_os": _cpu_count(),
            },
            method={
                "clock": "time.perf_counter_ns via MonotonicTimebase",
                "latency": "one sample per stage per iteration, no intra-stage averaging",
                "percentiles": "exact over stored samples, Hyndman & Fan type 1",
                "memory": "tracemalloc peak + resource.getrusage(RUSAGE_SELF)",
                "throughput": "completed iterations / wall-clock span",
                "pacing": "none; the loop is not slept to the period",
            },
            caveat=caveat if caveat is not None else _DEFAULT_CAVEAT,
        ),
        run,
    )


def write_record(record: BenchmarkRecord, path: str | Path) -> tuple[Path, Path]:
    """Write ``record`` as both ``.txt`` and ``.json``.

    Returns the two paths written. The text form is for a reader, the JSON for
    a diff between runs.
    """
    p = Path(path)
    txt = p.with_suffix(".txt")
    js = p.with_suffix(".json")
    txt.write_text(record.as_text() + "\n", encoding="utf-8")
    js.write_text(json.dumps(record.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return txt, js
