#!/usr/bin/env python3
"""Validation 6 --- the benchmark harness and the results file.

Two jobs:

(a) **Benchmark `edgeinfer` itself.** Screening candidate models is only worth
    doing if the screen is cheaper than the thing it replaces. The cost of an
    analytic estimate, a feature extraction, a four-parameter calibration, a
    forest fit and a measured profile are all timed and reported with their
    repeat counts.

(b) **Write the results file.** ``validation/performance_results.md`` is the
    output artefact that carries the environment in its own text and the
    target-device column left empty. This is the file that would be filled in
    from a run on the device.

What would make this repository's validation level 4 --- and does not exist
--------------------------------------------------------------------------
Level 4 requires measured timing and resource use **from the Jetson Orin Nano
itself**: this script run on the device, its raw stdout captured, and the
device column of the results table filled from that output. Nothing in this
repository may be extrapolated into that column, taken from a vendor
datasheet, or produced by the simulated backend. Until a session is given raw
output from the device, the column stays empty and this product is
``Level 3, hardware-pending``.

Runtime: about 45 s on one CPU core.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from edgeinfer.analytic import analytic_estimate, node_cost_arrays
from edgeinfer.backends import OnnxRuntimeBackend
from edgeinfer.dataset import generate_population
from edgeinfer.environment import describe_environment, environment_record
from edgeinfer.features import population_features
from edgeinfer.harness import benchmark
from edgeinfer.memtrace import measure_peak_python_bytes
from edgeinfer.predictor import LatencyPredictor
from edgeinfer.report import ResultRow, write_results_file
from edgeinfer.roofline import DeviceModel, calibrate_device

N_MODELS = 120
SEED = 20260401
REPEATS = 60
WARMUP = 8
OUTPUT = Path(__file__).resolve().parent / "performance_results.md"

DEVICE = DeviceModel(
    "declared single CPU core",
    peak_flops=4.0e9,
    peak_bandwidth_bytes_s=10.0e9,
    source="declared; not measured and not a datasheet figure",
)

#: Per-script ceiling from this repository's compute budget. Every validation
#: script and every example must finish inside it on one shared CPU core.
SCRIPT_BUDGET_S = 180.0


def _time_repeated(label: str, fn, repeats: int) -> tuple[float, float, str]:
    """Return (median, p99, method) seconds for ``repeats`` calls of ``fn``."""
    profile = benchmark(
        fn, label=label, repeats=repeats, warmup=1, timer_bias_samples=200
    )
    return profile.p50_s, profile.p99_s, profile.method_line()


def main() -> int:
    t_script = time.perf_counter()
    print("=" * 78)
    print("P033 edgeinfer -- Validation 6: performance benchmark")
    print("=" * 78)
    print(f"environment: {describe_environment(note='')}")
    print(
        "WORKSTATION NUMBERS, shared single-CPU-core cloud container with other "
        "build\njobs running concurrently. No figure below is a measurement of any "
        "edge target."
    )

    rows: list[ResultRow] = []

    print("\n(a) cost of generating and characterising the model population")
    print("-" * 78)
    t0 = time.perf_counter()
    models, _summaries = generate_population(N_MODELS, seed=SEED)
    generation_s = time.perf_counter() - t0
    graphs = [m.graph for m in models]
    print(
        f"  generate {N_MODELS} ONNX models        : {generation_s:.3f} s "
        f"({generation_s / N_MODELS * 1e3:.3f} ms per model, 1 repeat)"
    )
    rows.append(
        ResultRow(
            f"generate {N_MODELS}-model population", "s", generation_s, None,
            f"wall clock, 1 repeat, seed {SEED}",
        )
    )

    one = graphs[len(graphs) // 2]
    median_s, p99_s, method = _time_repeated(
        "analytic estimate", lambda: analytic_estimate(one, DEVICE), 200
    )
    print(
        f"  analytic estimate, one graph        : p50 {median_s * 1e6:.2f} us, "
        f"p99 {p99_s * 1e6:.2f} us"
    )
    rows.append(
        ResultRow("analytic estimate, one graph (p50)", "s", median_s, None, method)
    )
    rows.append(
        ResultRow("analytic estimate, one graph (p99)", "s", p99_s, None, method)
    )

    t0 = time.perf_counter()
    features = population_features(graphs)
    featurise_s = time.perf_counter() - t0
    print(
        f"  featurise {N_MODELS} graphs             : {featurise_s:.3f} s "
        f"({featurise_s / N_MODELS * 1e6:.1f} us per graph, 1 repeat)"
    )
    rows.append(
        ResultRow(
            f"featurise {N_MODELS} graphs", "s", featurise_s, None, "wall clock, 1 repeat"
        )
    )

    print("\n(b) cost of the two predictors")
    print("-" * 78)
    node_flops, node_bytes, graph_index = node_cost_arrays(graphs)
    rng = np.random.default_rng(SEED)
    pseudo_measured = np.exp(rng.uniform(np.log(5e-6), np.log(1e-3), size=N_MODELS))
    t0 = time.perf_counter()
    fitted = calibrate_device(
        node_flops, node_bytes, graph_index, pseudo_measured, "bench-fit", grid=20
    )
    calibration_s = time.perf_counter() - t0
    print(
        f"  calibrate 4 parameters, grid 20x20  : {calibration_s:.3f} s "
        "(400 NNLS solves, 1 repeat)"
    )
    print(f"    (the fit itself is not the point here; its label is {fitted.source})")
    rows.append(
        ResultRow(
            "calibrate analytic baseline (4 params, grid 20x20)", "s",
            calibration_s, None, "wall clock, 1 repeat, 400 NNLS solves",
        )
    )

    t0 = time.perf_counter()
    forest = LatencyPredictor(n_estimators=120, seed=SEED).fit(features, pseudo_measured)
    training_s = time.perf_counter() - t0
    print(f"  train forest, 120 trees, n={N_MODELS}      : {training_s:.3f} s (1 repeat)")
    rows.append(
        ResultRow(
            "train random forest (120 trees)", "s", training_s, None,
            f"wall clock, 1 repeat, n={N_MODELS} training rows",
        )
    )

    median_s, p99_s, method = _time_repeated(
        "forest predict, 120 rows", lambda: forest.predict(features), 100
    )
    print(
        f"  forest predict, {N_MODELS} rows          : p50 {median_s * 1e3:.3f} ms, "
        f"p99 {p99_s * 1e3:.3f} ms"
    )
    rows.append(
        ResultRow(
            f"forest predict, {N_MODELS} rows (p50)", "s", median_s, None, method
        )
    )

    median_s, p99_s, method = _time_repeated(
        "forest predict with uncertainty",
        lambda: forest.predict_with_uncertainty(features[:1]),
        100,
    )
    print(
        f"  forest predict + uncertainty, 1 row : p50 {median_s * 1e3:.3f} ms, "
        f"p99 {p99_s * 1e3:.3f} ms"
    )
    rows.append(
        ResultRow(
            "forest predict with uncertainty, 1 row (p50)", "s", median_s, None, method
        )
    )

    print("\n(c) cost of measuring, which the analytic estimate replaces")
    print("-" * 78)
    model = models[len(models) // 2]
    t0 = time.perf_counter()
    backend = OnnxRuntimeBackend(model.model_bytes, model.input_feed(0))
    backend.prepare()
    session_s = time.perf_counter() - t0
    try:
        t0 = time.perf_counter()
        profile = benchmark(
            backend.infer, label=model.graph.name, repeats=REPEATS, warmup=WARMUP,
            timer_bias_samples=200,
        )
        profile_s = time.perf_counter() - t0
        _result, peak = measure_peak_python_bytes(backend.infer)
    finally:
        backend.close()
    print(f"  build one InferenceSession          : {session_s * 1e3:.2f} ms (1 repeat)")
    print(
        f"  measured profile, {REPEATS} repeats       : {profile_s * 1e3:.2f} ms "
        f"(including {WARMUP} warm-up calls)"
    )
    print(f"  the model's own p50 / p99           : {profile.p50_s * 1e6:.3f} us / "
          f"{profile.p99_s * 1e6:.3f} us")
    print(f"  tracemalloc peak over one run       : {peak.peak_above_baseline_bytes} B")
    rows.append(
        ResultRow(
            "build one onnxruntime InferenceSession", "s", session_s, None,
            "wall clock, 1 repeat",
        )
    )
    rows.append(
        ResultRow(
            f"measured profile of one model ({REPEATS} repeats)", "s", profile_s, None,
            f"wall clock, 1 repeat, {REPEATS} timed calls plus {WARMUP} warm-up",
        )
    )

    print("\n(d) the screening argument")
    print("-" * 78)
    t0 = time.perf_counter()
    for graph in graphs:
        analytic_estimate(graph, fitted)
    analytic_all_s = time.perf_counter() - t0
    # Screening a candidate by measurement costs a session build plus a timed
    # profile; the session build dominates and must be counted.
    measure_per_candidate_s = session_s + profile_s
    measure_all_s = measure_per_candidate_s * N_MODELS
    print(
        f"  analytic estimate for all {N_MODELS}       : {analytic_all_s:.4f} s "
        f"({analytic_all_s / N_MODELS * 1e6:.1f} us per candidate, measured)"
    )
    print(
        f"  measure one candidate               : {measure_per_candidate_s * 1e3:.2f} ms "
        f"= {session_s * 1e3:.2f} ms session build + {profile_s * 1e3:.2f} ms profile"
    )
    print(
        f"  measure all {N_MODELS} (extrapolated)     : {measure_all_s:.3f} s "
        "(one candidate's cost times 120)"
    )
    speedup = measure_all_s / max(analytic_all_s, 1e-12)
    print(
        f"  ratio                               : {speedup:.0f}x cheaper to estimate "
        "than to measure"
    )
    print(
        "  The session build dominates the measured path, which is why it is counted\n"
        "  here: screening a candidate by measurement means loading it. This ratio is\n"
        "  the only extrapolation in this file and it extrapolates this repository's\n"
        "  own cost, not any device's performance."
    )
    rows.append(
        ResultRow(
            f"analytic estimate for all {N_MODELS} graphs", "s", analytic_all_s, None,
            "wall clock, 1 repeat",
        )
    )
    rows.append(
        ResultRow(
            "measure one candidate (session build + 60-repeat profile)", "s",
            measure_per_candidate_s, None,
            "wall clock, 1 repeat; sum of the two rows above",
        )
    )

    elapsed = time.perf_counter() - t_script
    budget_ok = elapsed < SCRIPT_BUDGET_S
    print("\n(e) compute budget")
    print("-" * 78)
    print(
        f"  this script took {elapsed:.1f} s against a per-script ceiling of "
        f"{SCRIPT_BUDGET_S:.0f} s  {'PASS' if budget_ok else 'FAIL'}"
    )
    rows.append(
        ResultRow(
            "this validation script, end to end", "s", elapsed, None,
            "wall clock, 1 repeat",
        )
    )

    written = write_results_file(
        OUTPUT,
        "P033 edgeinfer -- performance results",
        tuple(rows),
        environment=environment_record(shared_host=True, note=""),
        extra_sections=(
            (
                "What the empty column needs",
                "The Jetson Orin Nano column is empty because no measurement from "
                "that device exists. To fill it: run `validation/"
                "validate_performance.py` on the device, capture its raw stdout, and "
                "transcribe the figures from that output into this column. No number "
                "may be extrapolated from the host column, taken from a vendor "
                "datasheet, or produced by the simulated backend. Until then this "
                "product is `Level 3, hardware-pending` and the mission's Level 4 "
                "count stays at zero.",
            ),
            (
                "How to read these numbers",
                "Every row is the cost of *using* edgeinfer, not the latency of any "
                "model. The one model latency that appears in the text above "
                f"(p50 {profile.p50_s * 1e6:.3f} us, p99 {profile.p99_s * 1e6:.3f} us "
                f"over {REPEATS} repeats) is a property of this shared container and "
                "of onnxruntime's CPU execution provider, and it is not quoted in "
                "the table for that reason.",
            ),
        ),
    )
    print(f"\n  results file written: {written}")

    print("\n" + "=" * 78)
    print(f"RESULT: {'1/1' if budget_ok else '0/1'} checks passed "
          "(the compute-budget ceiling)")
    print("=" * 78)
    return 0 if budget_ok else 1


if __name__ == "__main__":
    sys.exit(main())
