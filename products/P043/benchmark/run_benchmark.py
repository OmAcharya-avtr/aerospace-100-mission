"""Latency, memory and throughput harness, with its method in its own output.

Run from this directory::

    PYTHONPATH=../src python3 run_benchmark.py

It writes ``benchmark_results.md`` beside itself. The file records the
measurement method and the environment as well as the numbers, because a timing
number without both is not a measurement of anything.

**These numbers are from whatever machine the script ran on.** In this
repository that is a shared two-core cloud container running four other build
agents at the same time, which is stated in the output file. Nothing here is a
Jetson Orin Nano measurement, and no Level 4 claim may be built on it. Closing
Level 4 means running this same script on the board and keeping its output.

What is measured
----------------
``acquisition``
    Wall-clock time of one :meth:`SimulatedBackend.acquire` window at a
    representative loading, repeated; reported as p50/p90/p99 and mean. This is
    the loop a device backend would replace, so its *shape* is what transfers,
    not its value.
``inference_single``
    One row through the fitted rate corrector, the per-sample latency a
    real-time correction would pay.
``inference_batch``
    1000 rows at once, reported as both total and per-row, because the batched
    per-row cost is what a post-processing pipeline sees.
``ppm_exact``
    One evaluation of the exact symbol error probability (P4) at M = 256.
``soft_metrics``
    Bit LLRs for 10000 PPM symbols, reported as symbols per second.
``simulator``
    Event throughput of the detector simulator, in registered events per second.

Memory is reported two ways: ``tracemalloc`` peak, which is Python allocations
only and excludes NumPy's own buffers in some paths, and the process maximum
resident set size from ``resource.getrusage``, which is the whole process and
never decreases. Both are stated because neither alone is honest.
"""

from __future__ import annotations

import gc
import json
import platform
import resource
import sys
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ""):
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

import photoncount  # noqa: E402
from photoncount.correction import RateCorrector  # noqa: E402
from photoncount.dataset import generate_dataset  # noqa: E402
from photoncount.hal import AcquisitionRequest, BackendMode, SimulatedBackend  # noqa: E402
from photoncount.ppm import (  # noqa: E402
    PPMConfig,
    bit_llrs,
    sample_counts,
    symbol_error_probability,
)
from photoncount.simulate import DetectorSpec, simulate_run  # noqa: E402

OUT = Path(__file__).resolve().parent / "benchmark_results.md"
MODEL = Path(__file__).resolve().parents[1] / "validation" / "rate_corrector.joblib"

SPEC = DetectorSpec(
    dead_time_s=5e-8,
    model="paralyzable",
    afterpulse_probability=0.05,
    afterpulse_mean_delay_s=2e-7,
)
INCIDENT_HZ = 4e6
WINDOW_S = 1e-3


def _time(fn, repeats: int, warmup: int = 3) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    gc.collect()
    samples = np.empty(repeats)
    for i in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples[i] = time.perf_counter() - t0
    return {
        "repeats": float(repeats),
        "mean_ms": float(samples.mean() * 1e3),
        "p50_ms": float(np.percentile(samples, 50) * 1e3),
        "p90_ms": float(np.percentile(samples, 90) * 1e3),
        "p99_ms": float(np.percentile(samples, 99) * 1e3),
        "min_ms": float(samples.min() * 1e3),
        "max_ms": float(samples.max() * 1e3),
    }


def _peak_kib(fn) -> float:
    gc.collect()
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024.0


def environment() -> dict[str, str]:
    import scipy
    import sklearn

    return {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unreported",
        "cpu_count": str(__import__("os").cpu_count()),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "photoncount": photoncount.__version__,
        "measured_on": (
            "shared cloud build container, two cores, running concurrently with "
            "other build jobs. NOT a Jetson Orin Nano and NOT a flight-representative "
            "board."
        ),
    }


def main() -> int:
    env = environment()
    rng = np.random.default_rng(20261006)
    results: dict[str, dict[str, float]] = {}

    backend = SimulatedBackend(SPEC, INCIDENT_HZ, rng)
    backend.open()
    request = AcquisitionRequest(WINDOW_S, 1, "bench")
    results["acquisition_one_window"] = _time(
        lambda: backend.acquire(request, BackendMode.SIMULATION), repeats=60
    )
    probe = backend.acquire(AcquisitionRequest(WINDOW_S, 20, "bench"), BackendMode.SIMULATION)
    observed_rate = probe.observed_rate_hz
    backend.close()

    if MODEL.exists():
        corrector = RateCorrector.load(MODEL)
        model_source = "validation/rate_corrector.joblib"
    else:
        corrector = RateCorrector(n_estimators=120).fit(generate_dataset(1500, 7))
        model_source = "trained in-line (1500 rows, 120 trees); run validate_correction.py"
    bench_set = generate_dataset(1000, 4242)
    single = bench_set.features[:1]
    single_tau = bench_set.dead_time_s[:1]
    results["inference_single_row"] = _time(
        lambda: corrector.predict_rate(single, single_tau), repeats=100
    )
    results["inference_batch_1000"] = _time(
        lambda: corrector.predict_rate(bench_set.features, bench_set.dead_time_s), repeats=20
    )

    cfg_big = PPMConfig(256, 4.0, 0.05)
    results["ppm_exact_error_M256"] = _time(
        lambda: symbol_error_probability(cfg_big), repeats=50
    )

    cfg = PPMConfig(64, 3.0, 0.05)
    symbols = sample_counts(cfg, rng.integers(0, 64, size=10_000), rng)
    results["bit_llrs_10000_symbols"] = _time(lambda: bit_llrs(symbols, cfg), repeats=30)

    results["simulator_one_window"] = _time(
        lambda: simulate_run(INCIDENT_HZ, WINDOW_S, SPEC, rng), repeats=60
    )

    throughput = {
        "observed_rate_hz": observed_rate,
        "registered_events_per_window": observed_rate * WINDOW_S,
        "simulated_events_per_second_of_cpu": (
            observed_rate * WINDOW_S / (results["simulator_one_window"]["p50_ms"] / 1e3)
        ),
        "acquisition_realtime_factor": (
            WINDOW_S / (results["acquisition_one_window"]["p50_ms"] / 1e3)
        ),
        "inference_rows_per_second_batched": (
            1000.0 / (results["inference_batch_1000"]["p50_ms"] / 1e3)
        ),
        "ppm_symbols_per_second_soft_metrics": (
            10_000.0 / (results["bit_llrs_10000_symbols"]["p50_ms"] / 1e3)
        ),
    }

    memory = {
        "acquisition_one_window_peak_kib": _peak_kib(
            lambda: simulate_run(INCIDENT_HZ, WINDOW_S, SPEC, rng)
        ),
        "inference_batch_1000_peak_kib": _peak_kib(
            lambda: corrector.predict_rate(bench_set.features, bench_set.dead_time_s)
        ),
        "bit_llrs_10000_symbols_peak_kib": _peak_kib(lambda: bit_llrs(symbols, cfg)),
        "process_max_rss_kib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }

    lines: list[str] = []
    lines.append("# photoncount benchmark results\n")
    lines.append(
        "**These numbers are not a Level 4 measurement.** They were produced on the "
        "machine described below. Level 4 for this product requires this same script "
        "run on a Jetson Orin Nano with its raw output kept; no figure here may stand "
        "in for that.\n"
    )
    lines.append("## Environment\n")
    lines.append("| key | value |")
    lines.append("|---|---|")
    for key, value in env.items():
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    lines.append("## Method\n")
    lines.append(
        "Each timed operation is run 3 times to warm up, then repeated with "
        "`time.perf_counter()` around each call and `gc.collect()` once before the "
        "measured loop. Percentiles are over the repeats, not over an internal loop, "
        "so a single slow call shows up in p99 rather than being averaged away. "
        "Memory is reported both as the `tracemalloc` peak of the operation (Python "
        "allocations only) and as the process-wide maximum resident set size from "
        "`resource.getrusage`, which never decreases and therefore bounds the whole "
        "run rather than the operation.\n"
    )
    lines.append(
        f"Workload: a paralyzable detector with tau = {SPEC.dead_time_s * 1e9:.0f} ns, "
        f"afterpulse probability {SPEC.afterpulse_probability:.3f} with a "
        f"{SPEC.afterpulse_mean_delay_s * 1e9:.0f} ns mean delay, illuminated at "
        f"{INCIDENT_HZ:.3g} counts/s (n tau = {INCIDENT_HZ * SPEC.dead_time_s:.3f}), "
        f"read out in {WINDOW_S * 1e3:.1f} ms windows. Rate corrector: {model_source}.\n"
    )
    lines.append("## Latency\n")
    lines.append("| operation | repeats | mean (ms) | p50 (ms) | p90 (ms) | p99 (ms) | "
                 "min (ms) | max (ms) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, row in results.items():
        lines.append(
            f"| `{name}` | {int(row['repeats'])} | {row['mean_ms']:.4f} | "
            f"{row['p50_ms']:.4f} | {row['p90_ms']:.4f} | {row['p99_ms']:.4f} | "
            f"{row['min_ms']:.4f} | {row['max_ms']:.4f} |"
        )
    lines.append("")
    lines.append("## Throughput\n")
    lines.append("| quantity | value |")
    lines.append("|---|---:|")
    for name, value in throughput.items():
        lines.append(f"| `{name}` | {value:.6g} |")
    lines.append("")
    lines.append(
        "`acquisition_realtime_factor` is the ratio of simulated window length to the "
        "wall-clock time taken to produce it. Above 1 means the simulator runs faster "
        "than real time at this loading; it falls as the loading rises, because the "
        "event loop is linear in events.\n"
    )
    lines.append("## Memory\n")
    lines.append("| quantity | KiB |")
    lines.append("|---|---:|")
    for name, value in memory.items():
        lines.append(f"| `{name}` | {value:.1f} |")
    lines.append("")
    lines.append("## What is still missing for Level 4\n")
    lines.append(
        "- Latency, memory and throughput from a Jetson Orin Nano, produced by this "
        "script on that board.\n"
        "- A measured dead time, dead-time model, afterpulse probability and delay "
        "distribution, dark-count rate, timestamp resolution and jitter from the "
        "actual detector, replacing the declared values in "
        "`photoncount.hal.DeviceBackend`.\n"
        "- A `DeviceBackend` implementation of `open`, `self_test` and `acquire` "
        "against that hardware.\n"
        "\nNo extrapolation from this container, and no vendor datasheet, substitutes "
        "for any of the three.\n"
    )
    OUT.write_text("\n".join(lines), encoding="utf-8")

    print("run_benchmark.py")
    print(f"   wrote benchmark/{OUT.name}")
    print(json.dumps({"environment": env, "throughput": throughput}, indent=2))
    for name, row in results.items():
        print(f"   {name:28s} p50 {row['p50_ms']:9.4f} ms  p99 {row['p99_ms']:9.4f} ms")
    for name, value in memory.items():
        print(f"   {name:34s} {value:10.1f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
