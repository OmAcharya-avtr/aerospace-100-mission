"""Check 8 — the performance benchmark harness and its record.

Requirement (Level 3): "a performance benchmark". The harness is
:func:`hilforge.bench.run_benchmark`; this script exercises it across the
cases that matter and writes the machine-readable record alongside the text.

Every record states its measurement method, the clock it used and that clock's
measured resolution, the host, and whether the backend was hardware. None of
the runs below was on hardware, and each record says so in its own
``is_hardware`` field and in its caveat.

Cases
-----
1. Simulated backend, write path.
2. Loopback device backend, write path — the same loop through the driver
   shim, which is where a real board's driver would sit.
3. Simulated backend, dry-run path — the rehearsal, whose cost is what you pay
   to rehearse.
4. A short run and a long run, to show how the percentile tail moves with
   sample count on a contended host.

Run: ``python validation/performance_benchmark.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))

from hilforge.backends import make_backend_pair
from hilforge.bench import run_benchmark, write_record

PERIOD = 0.010
SEED = 20261004
ENVIRONMENT = (
    "shared single-core cloud container (1 CPU allowed to this process, "
    "7.8 GiB RAM, four other build agents running concurrently); "
    "NOT a Jetson Orin Nano and NOT any other target hardware"
)


def main() -> int:
    print("HilForge validation 8 — performance benchmark")
    print("=" * 76)
    print()
    cases = [
        ("simulated-write", "simulated", 20_000, False),
        ("loopback-device-write", "device", 20_000, False),
        ("simulated-dryrun", "simulated", 20_000, True),
        ("simulated-short", "simulated", 1_000, False),
    ]
    records = []
    for label, which, iterations, dry in cases:
        sim, dev = make_backend_pair(seed=SEED, sample_dt_s=PERIOD)
        backend = sim if which == "simulated" else dev
        record, run = run_benchmark(
            backend,
            period_s=PERIOD,
            n_iterations=iterations,
            label=label,
            environment_note=ENVIRONMENT,
            dry_run=dry,
            record_signals=False,
        )
        records.append(record)
        print(record.as_text())
        print()
        print(f"  writes issued / rehearsed : {run.writes_issued} / "
              f"{run.rehearsed_writes}")
        print(f"  dropped / rejected        : {run.dropped_samples} / "
              f"{run.rejected_writes}")
        print()
        print("-" * 76)
        print()

    main_record = records[0]
    txt, js = write_record(main_record, HERE / "benchmark_record")
    print(f"machine-readable record written: {js.name}")
    print(f"text record written           : {txt.name}")
    print()
    print("COMPARISON ACROSS CASES")
    print(f"  {'label':<24} {'n':>7} {'p50 [s]':>12} {'p99 [s]':>12} "
          f"{'max [s]':>12} {'iter/s':>10} {'peak [MB]':>10}")
    for record in records:
        total = record.total_summary_s
        print(f"  {record.label:<24} {record.n_completed:>7} "
              f"{total['p50_s']:>12.6e} {total['p99_s']:>12.6e} "
              f"{total['max_s']:>12.6e} {record.throughput_iter_s:>10.1f} "
              f"{record.tracemalloc_peak_bytes / 1e6:>10.3f}")
    print()
    print("  Every row is a host-side measurement on a shared, contended")
    print("  single-core container. The p99 and max columns are dominated by")
    print("  scheduler preemption outside this process, which is why they are")
    print("  reported rather than averaged away, and why none of them is")
    print("  usable as a hardware figure. Level 4 is not claimed: it requires")
    print("  this harness run on a Jetson Orin Nano with its raw output")
    print("  captured, and that has not happened.")
    hw = [r.is_hardware for r in records]
    print()
    print(f"  is_hardware across all cases : {hw}")
    return 0 if not any(hw) else 1


if __name__ == "__main__":
    raise SystemExit(main())
