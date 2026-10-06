"""Benchmark harness: latency, memory and throughput, with the method and environment.

What this measures and what it does not
---------------------------------------
This script measures **this package, on whatever machine runs it**. Run on a cloud
build container it records a cloud build container. Nothing it produces is
evidence about a Jetson Orin Nano, and no number from it may be used to support a
validation-level-4 claim for this product. Closing level 4 requires running this
same script on the Jetson and keeping its output file.

Measurement method, stated here and repeated into the output file
----------------------------------------------------------------
* **Clock.** ``time.perf_counter_ns``, the highest-resolution monotonic clock the
  interpreter offers. Resolution is measured at run time and reported.
* **Warm-up.** Each stage is run once and discarded before timing, so the first
  call's import and allocation costs do not enter the figures.
* **Repeats.** Each stage is timed ``repeats`` times; the output reports the
  minimum, median, mean and maximum, and the interquartile range. The **median**
  is the headline figure, because this container shares two cores with sibling
  processes and the mean is contaminated by scheduling.
* **Throughput.** Symbols processed divided by the median stage duration, so the
  throughput figure and the latency figure are consistent with one another.
* **Memory.** ``tracemalloc`` peak during one untimed stage execution (tracing
  perturbs timing, so it is never on while timing), plus the interleaver's own
  analytic storage requirement from equations (24)-(25) for comparison. The
  resident-set size of the process is **not** reported, because on a shared
  container it is dominated by the interpreter and the imported libraries rather
  than by anything this package does.
* **Contention.** ``os.cpu_count`` and the 1-minute load average are recorded, so
  a reader can see whether the machine was busy.

Output
------
Writes ``benchmark_results.json`` next to this script and prints the same content.
The file contains no filesystem path, by design.

Run: ``PYTHONPATH=../src python3 run_benchmark.py``
Runtime: about 60 s on one core.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from codedfade.channel import ChannelConfig, generate_amplitude
from codedfade.fade import fade_statistics
from codedfade.hal import ModemConfig, ModemSession, RunMode, SimulatedModemBackend
from codedfade.interleave import BlockInterleaver
from codedfade.link import CodedLink
from codedfade.reedsolomon import ReedSolomon

REPEATS = 7
SYMBOL_RATE_HZ = 1.0e6
DEPTH = 256
CODE = (31, 21, 5)
PATH_SAMPLES = 400_000
LINK_CODEWORDS = 256


def clock_resolution_ns(trials: int = 2000) -> int:
    """Smallest non-zero difference between consecutive clock reads, nanoseconds."""
    best = None
    prev = time.perf_counter_ns()
    for _ in range(trials):
        now = time.perf_counter_ns()
        delta = now - prev
        prev = now
        if delta > 0 and (best is None or delta < best):
            best = delta
    return int(best if best is not None else 0)


def time_stage(fn: Callable[[], Any], repeats: int = REPEATS) -> dict[str, float]:
    """Warm up once, then time ``repeats`` executions. Durations in seconds."""
    fn()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e9)
    samples.sort()
    q1 = samples[len(samples) // 4]
    q3 = samples[(3 * len(samples)) // 4]
    return {
        "repeats": float(repeats),
        "min_s": samples[0],
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "max_s": samples[-1],
        "iqr_s": q3 - q1,
    }


def peak_memory_bytes(fn: Callable[[], Any]) -> int:
    """tracemalloc peak during one untimed execution, bytes."""
    tracemalloc.start()
    try:
        fn()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)


def environment() -> dict[str, Any]:
    try:
        load1 = os.getloadavg()[0]
    except OSError:  # pragma: no cover - not all platforms
        load1 = float("nan")
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unavailable",
        "cpu_count": os.cpu_count(),
        "load_average_1min": load1,
        "numpy_version": np.__version__,
        "pointer_bits": sys.maxsize.bit_length() + 1,
        "clock": "time.perf_counter_ns",
        "clock_resolution_ns": clock_resolution_ns(),
        "note": (
            "These figures describe the machine that ran this script. They are not "
            "Jetson Orin Nano measurements and cannot support a validation-level-4 "
            "claim for this product."
        ),
    }


def method() -> dict[str, Any]:
    return {
        "clock": "time.perf_counter_ns (monotonic, highest available resolution)",
        "warmup": "one untimed execution per stage before timing",
        "repeats": REPEATS,
        "headline_statistic": (
            "median, because this host shares 2 cores with sibling processes and the "
            "mean is contaminated by scheduling"
        ),
        "throughput": "symbols processed / median stage duration",
        "memory": (
            "tracemalloc peak during one untimed execution; tracing is never on while "
            "timing. Resident-set size is deliberately not reported: on a shared "
            "container it is dominated by the interpreter and imported libraries."
        ),
        "interleaver_memory_model": (
            "2*D*S symbols at m bits per symbol, equations (24)-(25), reported "
            "alongside the measured peak for comparison"
        ),
        "excluded": [
            "process start-up and import time",
            "first-call allocation (removed by the warm-up)",
            "any hardware-specific path (no device backend exists)",
        ],
    }


def main() -> None:
    code = ReedSolomon(*CODE)
    channel = ChannelConfig(0.6, 2.0e-4, SYMBOL_RATE_HZ, "lognormal", "exp", seed=5)
    interleaver = BlockInterleaver(DEPTH, code.n)
    backend = SimulatedModemBackend()
    modem = ModemConfig(
        n=code.n, k=code.k, m=code.m, depth=DEPTH, symbol_rate_hz=SYMBOL_RATE_HZ
    )
    backend.open(modem, RunMode.SIMULATION)
    rng = np.random.default_rng(0)
    message_block = rng.integers(0, 1 << code.m, size=(DEPTH, code.k)).astype(np.int64)
    encoded_block = backend.encode(message_block)
    single_message = message_block[0]
    single_codeword = code.encode(single_message)
    corrupted = single_codeword.copy()
    for p in rng.choice(code.n, code.t, replace=False):
        corrupted[p] ^= int(rng.integers(1, 1 << code.m))
    interleave_payload = np.arange(DEPTH * code.n * 8, dtype=np.int64)
    amplitude = generate_amplitude(channel, PATH_SAMPLES)

    stages: dict[str, dict[str, Any]] = {}

    def add(name: str, fn: Callable[[], Any], symbols: int) -> None:
        timing = time_stage(fn)
        timing["symbols"] = float(symbols)
        timing["throughput_symbols_per_s"] = symbols / timing["median_s"]
        timing["peak_memory_bytes"] = float(peak_memory_bytes(fn))
        stages[name] = timing

    add(
        "channel_path_generation",
        lambda: generate_amplitude(channel, PATH_SAMPLES),
        PATH_SAMPLES,
    )
    add(
        "fade_statistics",
        lambda: fade_statistics(amplitude, 0.6, SYMBOL_RATE_HZ),
        PATH_SAMPLES,
    )
    add("rs_encode_one_codeword", lambda: code.encode(single_message), code.n)
    add("rs_decode_clean_codeword", lambda: code.decode(single_codeword), code.n)
    add("rs_decode_t_errors", lambda: code.decode(corrupted), code.n)
    add(
        "block_interleave",
        lambda: interleaver.interleave(interleave_payload),
        interleave_payload.size,
    )
    add(
        "block_deinterleave",
        lambda: interleaver.deinterleave(interleave_payload),
        interleave_payload.size,
    )
    add("hal_encode_block", lambda: backend.encode(message_block), DEPTH * code.n)
    add("hal_decode_block", lambda: backend.decode(encoded_block), DEPTH * code.n)
    add(
        "link_depth_point",
        lambda: CodedLink(code, channel, 14.0).run(DEPTH, LINK_CODEWORDS),
        LINK_CODEWORDS * code.n,
    )

    cost = interleaver.cost(SYMBOL_RATE_HZ, bits_per_symbol=code.m)

    dry = ModemSession(SimulatedModemBackend(), modem, RunMode.DRY_RUN, channel)
    dry.start()
    dry.transfer(2, seed=1)
    dry_capture = dry.finish().to_dict()

    results = {
        "product": "P041 codedfade",
        "version": "0.1.0",
        "validation_level": "3, hardware-pending",
        "configuration": {
            "code": f"RS({code.n},{code.k}) over GF(2^{code.m}), t={code.t}",
            "code_rate": code.rate,
            "interleaver_depth": DEPTH,
            "interleaver_span": code.n,
            "symbol_rate_hz": SYMBOL_RATE_HZ,
            "channel": {
                "marginal": channel.marginal,
                "kernel": channel.kernel,
                "scintillation_index": channel.scintillation_index,
                "correlation_time_s": channel.correlation_time_s,
                "samples_per_correlation_time": channel.samples_per_correlation_time,
                "seed": channel.seed,
            },
            "path_samples": PATH_SAMPLES,
            "link_codewords": LINK_CODEWORDS,
        },
        "measurement_method": method(),
        "environment": environment(),
        "stages": stages,
        "interleaver_cost_model": {
            "latency_symbols": cost.latency_symbols,
            "latency_ms": cost.latency_ms,
            "memory_symbols": cost.memory_symbols,
            "memory_bytes": cost.memory_bytes,
        },
        "dry_run_capture": dry_capture,
        "level_4_gap": (
            "Measured timing and resource use from a Jetson Orin Nano. No simulated "
            "backend, extrapolation, vendor datasheet or workstation run substitutes "
            "for it. This product is labelled 'Level 3, hardware-pending' and is never "
            "labelled Level 4."
        ),
    }

    out = Path(__file__).resolve().parent / "benchmark_results.json"
    text = json.dumps(results, indent=2, sort_keys=True)
    out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
