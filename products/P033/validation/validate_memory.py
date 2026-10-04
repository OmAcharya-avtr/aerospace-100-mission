#!/usr/bin/env python3
"""Validation 3 --- peak-memory measurement against known allocation patterns.

Three things are checked, in order of what they establish:

(a) **The measurement instrument.** ``tracemalloc`` is pointed at allocation
    patterns whose byte count is known exactly by hand --- a single array, two
    simultaneously live arrays, two sequentially live arrays, and a nested
    region --- and the measured peak is compared against the hand count.

(b) **The analytic liveness model.** The peak activation bytes that
    :func:`edgeinfer.analytic.peak_activation_bytes` computes for a graph are
    compared against a NumPy *execution* of the same dataflow in which every
    tensor is a real array, measured with the instrument validated in (a).
    This is the check that the liveness analysis releases tensors at the right
    step.

(c) **What is NOT measured.** ``onnxruntime`` allocates its arena outside the
    CPython allocator, so ``tracemalloc`` cannot see a session's working set.
    This is demonstrated rather than asserted away: the measured figure for an
    ``onnxruntime`` call is printed next to the analytic peak for the same
    graph so the reader can see that the two are measuring different things.

References:
  - CPython standard library, ``tracemalloc``: counts allocations made through
    the CPython allocator.
  - Linux ``proc(5)``: ``VmRSS`` and ``VmHWM`` in ``/proc/self/status``.
  - Aho, Lam, Sethi & Ullman (2006), Compilers, 2nd ed., section 8.4:
    liveness analysis.

Runtime: about 10 s on one CPU core.
"""

from __future__ import annotations

import sys

import numpy as np

from edgeinfer.analytic import analytic_estimate, peak_activation_bytes
from edgeinfer.backends import OnnxRuntimeBackend
from edgeinfer.dataset import hand_counted_cnn, hand_counted_mlp, random_mlp
from edgeinfer.environment import describe_environment
from edgeinfer.memtrace import (
    measure_peak_python_bytes,
    resident_set_bytes,
    resident_set_high_water_bytes,
    track_peak_python_bytes,
)
from edgeinfer.roofline import DeviceModel

#: A float64 array of n elements requests 8n bytes plus a NumPy object header
#: of a few hundred bytes. The allowance below is the header, not a fudge
#: factor on the payload: the payload is asserted exactly as a lower bound.
HEADER_ALLOWANCE_BYTES = 1024


def _report(label: str, measured: int, expected: int) -> bool:
    ok = expected <= measured <= expected + HEADER_ALLOWANCE_BYTES
    error = measured - expected
    print(
        f"  {label:<40} measured {measured:>10} B  hand {expected:>10} B  "
        f"delta {error:>+6} B  {'PASS' if ok else 'FAIL'}"
    )
    return ok


def instrument_checks() -> list[bool]:
    print("\n(a) the measurement instrument against hand-counted allocations")
    print("-" * 78)
    results: list[bool] = []

    n = 1_000_000
    _r, peak = measure_peak_python_bytes(lambda: np.zeros(n, dtype=np.float64))
    results.append(
        _report("one float64 array, 1e6 elements", peak.peak_above_baseline_bytes, n * 8)
    )

    def two_live() -> int:
        a = np.zeros(1_000_000, dtype=np.float64)
        b = np.zeros(2_000_000, dtype=np.float64)
        return int(a[0] + b[0])

    _r, peak = measure_peak_python_bytes(two_live)
    results.append(
        _report(
            "two arrays live together, 1e6 + 2e6",
            peak.peak_above_baseline_bytes,
            3_000_000 * 8,
        )
    )

    def two_sequential() -> int:
        a = np.zeros(1_000_000, dtype=np.float64)
        value = int(a[0])
        del a
        b = np.zeros(1_000_000, dtype=np.float64)
        return value + int(b[0])

    _r, peak = measure_peak_python_bytes(two_sequential)
    ok = 1_000_000 * 8 <= peak.peak_above_baseline_bytes < 1_500_000 * 8
    results.append(ok)
    print(
        f"  {'two arrays in sequence, first released':<40} "
        f"measured {peak.peak_above_baseline_bytes:>10} B  "
        f"hand {1_000_000 * 8:>10} B  (not 16 MB)  {'PASS' if ok else 'FAIL'}"
    )

    _r, peak32 = measure_peak_python_bytes(lambda: np.zeros(500_000, dtype=np.float32))
    _r, peak64 = measure_peak_python_bytes(lambda: np.zeros(500_000, dtype=np.float64))
    ratio = peak64.peak_above_baseline_bytes / peak32.peak_above_baseline_bytes
    ok = abs(ratio - 2.0) < 0.01
    results.append(ok)
    print(
        f"  {'float64 / float32 peak ratio':<40} measured {ratio:>10.4f}    "
        f"hand {2.0:>10.4f}              {'PASS' if ok else 'FAIL'}"
    )

    with track_peak_python_bytes() as outer:
        with track_peak_python_bytes() as inner:
            _buffer = np.zeros(300_000, dtype=np.float64)
        del _buffer
    ok = (
        inner[0].peak_above_baseline_bytes >= 300_000 * 8
        and outer[0].peak_bytes >= inner[0].peak_bytes - HEADER_ALLOWANCE_BYTES
    )
    results.append(ok)
    print(
        f"  {'nested regions agree on the inner peak':<40} "
        f"inner {inner[0].peak_above_baseline_bytes:>10} B  "
        f"hand {300_000 * 8:>10} B              {'PASS' if ok else 'FAIL'}"
    )

    print(f"\n  instrument method: {peak.method}")
    rss = resident_set_bytes()
    hwm = resident_set_high_water_bytes()
    print(
        f"  process RSS now {rss} B, lifetime high water {hwm} B "
        "(from /proc/self/status; whole-process figures, informative only as a "
        "difference)"
    )
    return results


def _execute_dataflow_with_real_arrays(model) -> int:
    """Execute a graph's dataflow in NumPy, holding every tensor as a real
    array, and measure the peak with the validated instrument.

    Only the operators the two hand-counted graphs use are implemented here.
    Tensors are deleted exactly when the liveness analysis says they die, so
    the measurement is of the *same* schedule the analytic model assumes --- a
    different schedule would be a different peak, which is the point.
    """
    graph = model.graph.infer_shapes()
    weight_names = {spec.name for spec in graph.initializers}
    remaining = {k: v for k, v in graph.consumers().items() if k not in weight_names}
    live: dict[str, np.ndarray] = {}

    with track_peak_python_bytes() as holder:
        for spec in graph.inputs:
            live[spec.name] = np.zeros(spec.shape, dtype=np.float32)
        for node in graph.nodes:
            out_spec = graph.spec(node.outputs[0])
            # The allocation is what is being measured; the arithmetic is not,
            # so every output is simply an array of the inferred shape.
            live[node.outputs[0]] = np.zeros(out_spec.shape, dtype=np.float32)
            for name in node.inputs:
                if not name or name in weight_names:
                    continue
                remaining[name] -= 1
                if remaining[name] <= 0 and name not in graph.outputs:
                    live.pop(name, None)
        live.clear()
    return holder[0].peak_above_baseline_bytes


def liveness_checks() -> list[bool]:
    print("\n(b) the analytic liveness model against a measured execution")
    print("-" * 78)
    results: list[bool] = []
    for model, label in (
        (hand_counted_mlp(), "hand MLP"),
        (hand_counted_cnn(), "hand CNN"),
    ):
        analytic_peak, _ = peak_activation_bytes(model.graph)
        measured_peak = _execute_dataflow_with_real_arrays(model)
        # Every intermediate is a separate NumPy array with its own header, so
        # the measured figure exceeds the payload by one header per live
        # tensor. The payload must be a lower bound on the measurement.
        n_tensors = len(model.graph.nodes) + len(model.graph.inputs)
        allowance = HEADER_ALLOWANCE_BYTES * n_tensors
        ok = analytic_peak <= measured_peak <= analytic_peak + allowance
        results.append(ok)
        print(
            f"  {label:<12} analytic peak activations {analytic_peak:>8} B  "
            f"measured {measured_peak:>8} B  "
            f"allowance {allowance:>6} B  {'PASS' if ok else 'FAIL'}"
        )

    print("\n  a larger graph, where the payload dominates the headers:")
    rng = np.random.default_rng(20260401)
    model = random_mlp(rng, "wide", n_in=512, widths=(2048, 2048), n_out=64)
    analytic_peak, _ = peak_activation_bytes(model.graph)
    measured_peak = _execute_dataflow_with_real_arrays(model)
    relative = (measured_peak - analytic_peak) / analytic_peak
    ok = analytic_peak <= measured_peak and relative < 0.05
    results.append(ok)
    print(
        f"  {'512-2048-2048-64 MLP':<40} analytic {analytic_peak:>8} B  "
        f"measured {measured_peak:>8} B  "
        f"relative excess {relative * 100:>5.2f} %  {'PASS' if ok else 'FAIL'}"
    )
    return results


def onnxruntime_disclosure() -> None:
    print("\n(c) what tracemalloc does NOT see: the onnxruntime arena")
    print("-" * 78)
    rng = np.random.default_rng(1)
    model = random_mlp(rng, "arena", n_in=256, widths=(1024, 1024), n_out=64)
    device = DeviceModel("declared", 2e9, 8e9)
    estimate = analytic_estimate(model.graph, device)
    backend = OnnxRuntimeBackend(model.model_bytes, model.input_feed(0))
    backend.prepare()
    try:
        _result, peak = measure_peak_python_bytes(backend.infer)
    finally:
        backend.close()
    print(
        f"  analytic peak memory (weights + activations) : "
        f"{estimate.peak_memory_bytes:>10} B"
    )
    print(f"      of which resident weights                : {estimate.weight_bytes:>10} B")
    print(
        f"      of which peak activations                : "
        f"{estimate.peak_activation_bytes:>10} B"
    )
    print(
        f"  tracemalloc peak across one session.run()    : "
        f"{peak.peak_above_baseline_bytes:>10} B"
    )
    print(
        "  These are NOT the same quantity and neither validates the other. The\n"
        "  tracemalloc figure is the Python-side marshalling of the output tensor;\n"
        "  onnxruntime's own arena, holding the weights and every intermediate, is\n"
        "  invisible to it. Measuring a real session's peak working set needs an\n"
        "  allocator-level instrument on the target, which this repository does not\n"
        "  have. That gap is listed under Limitations in README.md."
    )


def main() -> int:
    print("=" * 78)
    print("P033 edgeinfer -- Validation 3: peak-memory measurement")
    print("=" * 78)
    print(f"environment: {describe_environment(note='')}")
    print(
        "WORKSTATION NUMBERS, shared single-CPU-core cloud container. Byte counts "
        "in\nsections (a) and (b) are allocator arithmetic and do not depend on the "
        "host;\nthe RSS figures do."
    )

    results = instrument_checks()
    results += liveness_checks()
    onnxruntime_disclosure()

    print("\n" + "=" * 78)
    passed = sum(results)
    print(f"RESULT: {passed}/{len(results)} checks passed")
    print("=" * 78)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
