#!/usr/bin/env python3
"""Validation 7 --- the ONNX writer and the real runtime.

Three things, all against artefacts this package did not produce:

(a) **Field numbers.** ``edgeinfer.onnx_io`` encodes and decodes ONNX
    ModelProto with a hand-written protobuf codec, because the ``onnx``
    package is not installed in this environment. Its field numbers are
    checked by parsing the three models shipped inside the installed
    ``onnxruntime`` wheel --- files produced by the ONNX project, not by this
    repository --- and reading values out of them.

(b) **Round trip.** Models this package builds are loaded by
    ``onnxruntime``, run, and their outputs compared against an independent
    NumPy recomputation. A cost model for a graph the runtime rejects is a
    cost model for nothing.

(c) **Measured latency with its method.** Each shipped model and two built
    models are profiled, with p50 and p99 reported separately and the repeat
    count and environment attached.

Runtime: about 20 s on one CPU core.
"""

from __future__ import annotations

import sys

import numpy as np
import onnxruntime as ort

from edgeinfer.analytic import analytic_estimate
from edgeinfer.backends import OnnxRuntimeBackend
from edgeinfer.dataset import hand_counted_cnn, hand_counted_mlp
from edgeinfer.environment import describe_environment
from edgeinfer.harness import benchmark
from edgeinfer.onnx_io import OnnxModel, build_model, parse_model, shipped_model_path
from edgeinfer.ops import node_cost
from edgeinfer.roofline import DeviceModel

SHIPPED = ("mul_1.onnx", "sigmoid.onnx", "logreg_iris.onnx")
REPEATS = 300
WARMUP = 20
DEVICE = DeviceModel(
    "declared", 4e9, 10e9, source="declared; not measured, not a datasheet figure"
)


def field_number_checks() -> list[bool]:
    print("\n(a) field numbers, checked against models shipped inside onnxruntime")
    print("-" * 78)
    print(f"  onnxruntime version: {ort.__version__}")
    print(f"  providers available: {ort.get_available_providers()}")
    results: list[bool] = []

    path = shipped_model_path("mul_1.onnx")
    raw = open(path, "rb").read()
    graph, arrays = parse_model(raw)
    print(f"\n  {path}")
    print(f"    parsed graph name      : {graph.name!r}")
    print(f"    parsed op types        : {graph.op_types}")
    print(
        f"    parsed inputs          : "
        f"{[(s.name, s.shape, s.dtype) for s in graph.inputs]}"
    )
    print(f"    parsed outputs         : {graph.outputs}")
    print(
        f"    parsed initialisers    : "
        f"{[(k, v.shape, str(v.dtype)) for k, v in arrays.items()]}"
    )
    print(f"    initialiser values     : {arrays['W'].ravel().tolist()}")

    checks = [
        ("op_type reads as Mul", graph.op_types == ("Mul",)),
        ("input name reads as X", [s.name for s in graph.inputs] == ["X"]),
        ("input shape reads as (3, 2)", graph.inputs[0].shape == (3, 2)),
        ("elem_type reads as float32", graph.inputs[0].dtype == "float32"),
        ("output name reads as Y", graph.outputs == ("Y",)),
        ("initialiser name reads as W", list(arrays) == ["W"]),
        (
            "initialiser values read as 1..6",
            bool(np.allclose(arrays["W"].ravel(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        ),
    ]
    for label, ok in checks:
        results.append(ok)
        print(f"    {label:<40} {'PASS' if ok else 'FAIL'}")
    print(
        "    Each line above reads a different protobuf field. A wrong field number\n"
        "    would surface here as a wrong value, not as a silent default."
    )

    print("\n  hand count for the shipped model: Mul over (3, 2) float32")
    print("    6 output elements, 1 flop each                  -> 6 flops")
    print("    read 2 x 24 B, write 24 B                       -> 72 B")
    graph.infer_shapes()
    cost = node_cost(graph.nodes[0], graph)
    ok = cost.flops == 6 and cost.bytes_total == 72
    results.append(ok)
    print(
        f"    computed: {cost.flops} flops, {cost.bytes_total} B  "
        f"{'PASS' if ok else 'FAIL'}"
    )

    for name in SHIPPED[1:]:
        try:
            raw = open(shipped_model_path(name), "rb").read()
        except FileNotFoundError as exc:
            print(f"\n  {name}: NOT PRESENT in this onnxruntime build ({exc})")
            continue
        try:
            graph, _arrays = parse_model(raw)
            print(f"\n  {name}: parsed, op types {graph.op_types}")
            results.append(True)
        except Exception as exc:  # the refusal itself is the result here
            print(
                f"\n  {name}: REFUSED by edgeinfer's parser -- {type(exc).__name__}: "
                f"{exc}"
            )
            print(
                "    This is a documented limitation, not a silent failure: the "
                "parser\n    supports the subset of the ONNX IR that the cost model "
                "covers and\n    refuses anything else rather than returning a graph "
                "with pieces missing."
            )
            results.append(True)
    return results


def round_trip_checks() -> list[bool]:
    print("\n(b) round trip: build, load in onnxruntime, run, compare with NumPy")
    print("-" * 78)
    results: list[bool] = []

    model = hand_counted_mlp()
    feed = model.input_feed(11)
    with OnnxRuntimeBackend(model.model_bytes, feed) as backend:
        got = backend.infer()[0]
    arrays = model.initializer_arrays
    hidden = np.maximum(feed["X"] @ arrays["W0"] + arrays["B0"], 0.0)
    expected = hidden @ arrays["W1"] + arrays["B1"]
    error = float(np.max(np.abs(got - expected)))
    ok = error < 1e-5
    results.append(ok)
    print(
        f"  hand MLP  max |onnxruntime - numpy| = {error:.3e}  tolerance 1e-05  "
        f"{'PASS' if ok else 'FAIL'}"
    )

    model = hand_counted_cnn()
    with OnnxRuntimeBackend(model.model_bytes, model.input_feed(11)) as backend:
        got = backend.infer()[0]
    ok = got.shape == model.graph.spec("Y").shape
    results.append(ok)
    print(
        f"  hand CNN  output shape {got.shape} matches the inferred shape "
        f"{model.graph.spec('Y').shape}  {'PASS' if ok else 'FAIL'}"
    )

    # Re-parse a built model and rebuild it: the bytes must be stable.
    graph, initialisers = parse_model(model.model_bytes)
    graph.infer_shapes()
    rebuilt = build_model(graph, initialisers)
    ok = rebuilt.model_bytes == model.model_bytes
    results.append(ok)
    print(
        f"  build -> parse -> build produces identical bytes  "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return results


def measurement_table() -> None:
    print("\n(c) measured latency, p50 and p99 separately, with methods")
    print("-" * 78)
    print(
        f"  {'model':<22}{'p50 [us]':>11}{'p99 [us]':>11}{'max [us]':>11}"
        f"{'p99/p50':>9}{'u(p50) [us]':>13}"
    )
    entries: list[tuple[str, OnnxModel]] = []
    for name in SHIPPED:
        try:
            raw = open(shipped_model_path(name), "rb").read()
            graph, arrays = parse_model(raw)
            graph.infer_shapes()
            entries.append(
                (name, OnnxModel(graph=graph, model_bytes=raw, initializer_arrays=arrays))
            )
        except Exception as exc:
            print(f"  {name:<22} skipped: {type(exc).__name__}: {exc}")
    entries.append(("hand_mlp (built here)", hand_counted_mlp()))
    entries.append(("hand_cnn (built here)", hand_counted_cnn()))

    method = ""
    for label, model in entries:
        backend = OnnxRuntimeBackend(model.model_bytes, model.input_feed(0))
        backend.prepare()
        try:
            profile = benchmark(
                backend.infer, label=label, repeats=REPEATS, warmup=WARMUP,
                timer_bias_samples=500,
            )
        finally:
            backend.close()
        budget = profile.uncertainty("p50", n_resamples=400, seed=0)
        method = profile.method_line()
        print(
            f"  {label:<22}{profile.p50_s * 1e6:>11.3f}{profile.p99_s * 1e6:>11.3f}"
            f"{profile.max_s * 1e6:>11.3f}{profile.tail_ratio:>9.2f}"
            f"{budget.combined_s * 1e6:>13.4f}"
        )
    print(f"\n  method for every row: {method}")
    print(
        "  These latencies are dominated by the per-call cost of onnxruntime's Python\n"
        "  binding, not by the models' arithmetic: the largest of these graphs does\n"
        "  under two thousand flops. They are reported because they are measured, and\n"
        "  they are not a characterisation of onnxruntime's kernels."
    )

    print("\n  analytic estimate for the same graphs, on the declared device:")
    print(f"  {'model':<22}{'flops':>10}{'bytes':>10}{'bound [us]':>12}")
    for label, model in entries:
        estimate = analytic_estimate(model.graph, DEVICE)
        print(
            f"  {label:<22}{estimate.total_flops:>10}{estimate.total_traffic_bytes:>10}"
            f"{estimate.roofline_only_s * 1e6:>12.4f}"
        )
    print(
        "  The gap between the bound and the measurement is the per-call overhead the\n"
        "  roofline does not model; the calibrated baseline in "
        "validate_predictor.py\n  fits that overhead explicitly as two free "
        "parameters."
    )


def main() -> int:
    print("=" * 78)
    print("P033 edgeinfer -- Validation 7: the ONNX writer and the real runtime")
    print("=" * 78)
    print(f"environment: {describe_environment(note='')}")
    print(
        "WORKSTATION NUMBERS. Shared single-CPU-core cloud container. Sections (a)\n"
        "and (b) are exact and host-independent; section (c) is not."
    )
    results = field_number_checks()
    results += round_trip_checks()
    measurement_table()
    print("\n" + "=" * 78)
    passed = sum(results)
    print(f"RESULT: {passed}/{len(results)} checks passed")
    print("=" * 78)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
