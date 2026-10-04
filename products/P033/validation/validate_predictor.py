#!/usr/bin/env python3
"""Validation 5 --- the analytic baseline against the learned predictor.

This is the comparison the product exists to report honestly. The analytic
cost model was implemented and validated first (``validate_opcounts.py``); the
learned predictor is measured against it here, on the same held-out models,
and the result is reported whichever way it falls.

Campaign
--------
1. Generate a reproducible population of small ONNX graphs
   (:func:`edgeinfer.dataset.generate_population`, fixed seed).
2. Measure each one with ``onnxruntime`` on this host, recording p50 and p99
   separately with their repeat count.
3. Split the population into train and held-out test with a fixed seed.
4. Fit the analytic baseline's four device parameters on the training set
   (:func:`edgeinfer.roofline.calibrate_device`): peak compute, peak
   bandwidth, per-node dispatch overhead, constant per-call overhead.
5. Fit a random forest on ``log10`` latency from graph features on the same
   training set.
6. Evaluate both on the held-out set: median and p90 of the absolute relative
   error, Spearman rank correlation, and whether either advantage survives a
   paired bootstrap.
7. Repeat for the p99 target, because a control loop is sized by its tail.
8. Report the forest's uncertainty output and its empirical coverage.
9. Report peak memory separately: it is exact integer arithmetic from the
   graph, so there is nothing for a learned model to improve on, and the
   contrast is measured rather than asserted.

What this comparison can and cannot show
----------------------------------------
Both predictors see the same features and the same training set. The
population is synthetic, small and narrow (MLPs and small CNNs, under about
40 MFLOP), the measurements come from a shared single-core cloud container,
and the target is ``onnxruntime``'s CPU execution provider. A result here
transfers to a different model family, a different runtime or a different
device only as a hypothesis.

References:
  - Williams, Waterman & Patterson (2009), Comm. ACM 52(4), 65-76: roofline.
  - Breiman (2001), Machine Learning 45(1), 5-32: random forests; the
    per-tree spread used as the uncertainty output.
  - Efron & Tibshirani (1993), An Introduction to the Bootstrap, ch. 6 and
    ch. 13: the bootstrap, and the paired comparison of two error measures.

Runtime: about 60 s on one CPU core.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from edgeinfer.analytic import analytic_estimate, node_cost_arrays
from edgeinfer.backends import OnnxRuntimeBackend
from edgeinfer.dataset import generate_population
from edgeinfer.environment import describe_environment
from edgeinfer.features import FEATURE_NAMES, population_features
from edgeinfer.harness import benchmark
from edgeinfer.predictor import (
    LatencyPredictor,
    compare_predictors,
    evaluate_predictions,
    split_indices,
)
from edgeinfer.roofline import DeviceModel, calibrate_device

N_MODELS = 120
POPULATION_SEED = 20260401
SPLIT_SEED = 20260401
FOREST_SEED = 20260401
REPEATS = 60
WARMUP = 8
TEST_FRACTION = 0.30
CALIBRATION_GRID = 20

#: The declared (uncalibrated) device, for the pure roofline bound. The peaks
#: are a plausible declaration for a single modern CPU core, NOT a measurement
#: and NOT a datasheet figure for anything.
DECLARED_DEVICE = DeviceModel(
    "declared single CPU core",
    peak_flops=4.0e9,
    peak_bandwidth_bytes_s=10.0e9,
    source="declared for this validation; not measured, not a datasheet figure",
)

#: Peaks chosen far above anything one core of this host delivers, so that the
#: pure roofline must be a genuine lower bound on every measured latency. Used
#: only to check the bound property; it describes no real device.
GENEROUS_DEVICE = DeviceModel(
    "deliberately over-declared device",
    peak_flops=200.0e9,
    peak_bandwidth_bytes_s=400.0e9,
    source="over-declared on purpose to test the bound property; not a device",
)

#: Independent measurement passes in the stability section. Each pass
#: re-measures the whole population and refits both predictors from scratch.
STABILITY_PASSES = 5


def measure_population(models) -> tuple[np.ndarray, np.ndarray, str]:
    """Measure p50 and p99 for every model. Returns (p50, p99, method)."""
    p50 = np.empty(len(models))
    p99 = np.empty(len(models))
    method = ""
    t0 = time.perf_counter()
    for i, model in enumerate(models):
        backend = OnnxRuntimeBackend(model.model_bytes, model.input_feed(0))
        backend.prepare()
        try:
            profile = benchmark(
                backend.infer,
                label=model.graph.name,
                repeats=REPEATS,
                warmup=WARMUP,
                timer_bias_samples=200,
            )
        finally:
            backend.close()
        p50[i] = profile.p50_s
        p99[i] = profile.p99_s
        method = profile.method_line()
    print(f"  measured {len(models)} models in {time.perf_counter() - t0:.1f} s")
    print(f"  method: {method}")
    return p50, p99, method


def fit_baseline(
    graphs, train: np.ndarray, measured: np.ndarray, label: str
) -> DeviceModel:
    node_flops, node_bytes, graph_index = node_cost_arrays(graphs)
    mask = np.isin(graph_index, train)
    remap = {g: i for i, g in enumerate(train)}
    t0 = time.perf_counter()
    fitted = calibrate_device(
        node_flops[mask],
        node_bytes[mask],
        np.asarray([remap[g] for g in graph_index[mask]]),
        measured[train],
        f"container-fit ({label})",
        grid=CALIBRATION_GRID,
    )
    print(
        f"  calibrated in {time.perf_counter() - t0:.1f} s at grid "
        f"{CALIBRATION_GRID}x{CALIBRATION_GRID}"
    )
    print(f"    peak compute          : {fitted.peak_flops / 1e9:.4f} GFLOP/s (fitted)")
    print(
        f"    peak bandwidth        : "
        f"{fitted.peak_bandwidth_bytes_s / 1e9:.4f} GB/s (fitted)"
    )
    print(
        f"    per-node overhead     : {fitted.overhead_per_node_s * 1e6:.4f} us (fitted)"
    )
    print(f"    fixed overhead        : {fitted.fixed_overhead_s * 1e6:.4f} us (fitted)")
    print(
        f"    ridge point           : "
        f"{fitted.ridge_point_flops_per_byte:.4f} FLOP/B (derived)"
    )
    print(f"    source label          : {fitted.source}")
    return fitted


def compare_target(
    graphs,
    features: np.ndarray,
    measured: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    target_label: str,
) -> tuple[bool, str, dict]:
    """Run the whole comparison for one target statistic."""
    print(f"\n--- target: {target_label} ---")
    print(
        f"  measured range: {measured.min() * 1e6:.3f} to {measured.max() * 1e6:.3f} us "
        f"({measured.max() / measured.min():.0f}x)"
    )

    generous_bound = np.asarray(
        [analytic_estimate(graphs[i], GENEROUS_DEVICE).roofline_only_s for i in test]
    )
    violations = int(np.sum(generous_bound > measured[test]))
    print(
        f"\n  (i-a) is the roofline a bound at all? Declared peaks "
        f"{GENEROUS_DEVICE.peak_flops / 1e9:.0f} GFLOP/s and"
    )
    print(
        f"        {GENEROUS_DEVICE.peak_bandwidth_bytes_s / 1e9:.0f} GB/s, chosen "
        "well above anything one core of this host"
    )
    print("        delivers, so the bound must lie BELOW every measured latency.")
    print(
        f"        bound above the measurement for {violations}/{len(test)} models "
        f"{'PASS' if violations == 0 else 'FAIL'}"
    )
    ratio = measured[test] / generous_bound
    print(
        f"        measured / bound: median {np.median(ratio):.1f}x, "
        f"min {ratio.min():.1f}x, max {ratio.max():.1f}x"
    )

    uncalibrated = np.asarray(
        [analytic_estimate(graphs[i], DECLARED_DEVICE).roofline_only_s for i in test]
    )
    above = int(np.sum(uncalibrated > measured[test]))
    print(
        f"\n  (i-b) the same bound on the declared device "
        f"({DECLARED_DEVICE.peak_flops / 1e9:.1f} GFLOP/s,"
    )
    print(
        f"        {DECLARED_DEVICE.peak_bandwidth_bytes_s / 1e9:.1f} GB/s, no "
        "overhead terms)"
    )
    metrics_bound = evaluate_predictions(uncalibrated, measured[test], "roofline bound")
    for line in metrics_bound.summary_lines():
        print(f"        {line}")
    print(
        f"        bound above the measurement for {above}/{len(test)} models"
    )
    print(
        "        interpretation: a roofline bound is a bound for the device it\n"
        "        DECLARES, not for the machine it is evaluated on. These declared\n"
        "        peaks are slower than this host's effective rate, so the bound sits\n"
        "        above the measurement for most models. That is a statement about the\n"
        "        declaration, not an error in the model, and it is the reason the\n"
        "        baseline below is fitted rather than declared."
    )

    print("\n  (ii) analytic baseline, four parameters fitted on the training set")
    fitted = fit_baseline(graphs, train, measured, target_label)
    analytic_test = np.asarray(
        [analytic_estimate(graphs[i], fitted).latency_s for i in test]
    )
    metrics_a = evaluate_predictions(
        analytic_test, measured[test], "analytic roofline (calibrated)"
    )
    for line in metrics_a.summary_lines():
        print(f"      {line}")

    print("\n  (iii) learned random forest on the same training set")
    t0 = time.perf_counter()
    forest = LatencyPredictor(
        n_estimators=120, max_depth=8, min_samples_leaf=2, seed=FOREST_SEED,
        target_name=target_label,
    ).fit(features[train], measured[train])
    print(f"      trained in {time.perf_counter() - t0:.2f} s, 120 trees, depth <= 8")
    learned_test = forest.predict(features[test])
    metrics_l = evaluate_predictions(
        learned_test, measured[test], "learned random forest"
    )
    for line in metrics_l.summary_lines():
        print(f"      {line}")

    print("\n  (iv) head to head on the held-out set")
    metrics_a2, metrics_l2, verdict = compare_predictors(
        analytic_test, learned_test, measured[test]
    )
    print(f"      {verdict}")
    print(
        f"      p90 |rel error|: analytic {metrics_a2.p90_abs_rel_error * 100:.2f} % "
        f"vs learned {metrics_l2.p90_abs_rel_error * 100:.2f} %"
    )
    print(
        f"      Spearman rho   : analytic {metrics_a2.spearman:.4f} "
        f"vs learned {metrics_l2.spearman:.4f}"
    )

    print("\n  (v) the forest's uncertainty output and its empirical coverage")
    predictions = forest.predict_with_uncertainty(features[test])
    relative_u = np.asarray([p.relative_uncertainty for p in predictions])
    values = np.asarray([p.value for p in predictions])
    uncertainties = np.asarray([p.uncertainty for p in predictions])
    inside_1 = float(np.mean(np.abs(values - measured[test]) <= uncertainties))
    inside_2 = float(np.mean(np.abs(values - measured[test]) <= 2.0 * uncertainties))
    print(
        f"      median relative ensemble spread : {np.median(relative_u) * 100:.2f} %"
    )
    print(f"      truth inside +/- 1 u            : {inside_1 * 100:.1f} % of models")
    print(f"      truth inside +/- 2 u            : {inside_2 * 100:.1f} % of models")
    print(
        "      For a calibrated Gaussian interval these would be about 68 % and 95 %.\n"
        "      The ensemble spread measures disagreement between trees, not the\n"
        "      irreducible measurement noise, so it is NOT a calibrated interval and\n"
        "      the figures above are the measured evidence for that statement."
    )

    top = sorted(forest.feature_importances.items(), key=lambda kv: -kv[1])[:6]
    print("\n  (vi) forest impurity importances (biased; diagnostic only)")
    for name, value in top:
        print(f"      {name:<24} {value:.4f}")

    analytic_wins = "analytic roofline (calibrated) wins" in verdict
    return analytic_wins, verdict, {
        "analytic_median_rel": metrics_a.median_abs_rel_error,
        "learned_median_rel": metrics_l.median_abs_rel_error,
        "analytic_p90_rel": metrics_a.p90_abs_rel_error,
        "learned_p90_rel": metrics_l.p90_abs_rel_error,
        "analytic_spearman": metrics_a.spearman,
        "learned_spearman": metrics_l.spearman,
        "bound_median_rel": metrics_bound.median_abs_rel_error,
    }


def memory_contrast(graphs, features: np.ndarray, train, test) -> None:
    print("\n--- target: analytic peak memory [B] ---")
    truth = np.asarray(
        [analytic_estimate(graphs[i], DECLARED_DEVICE).peak_memory_bytes for i in range(
            len(graphs)
        )],
        dtype=float,
    )
    print(
        f"  range: {truth.min():.0f} to {truth.max():.0f} B "
        f"({truth.max() / truth.min():.0f}x)"
    )
    print(
        "  The analytic peak memory is EXACT by construction: it is integer byte\n"
        "  arithmetic over the graph's tensor shapes plus a liveness analysis, and\n"
        "  validate_memory.py checks it against a measured execution. Its error\n"
        "  against itself is zero, so there is no accuracy for a learned model to\n"
        "  add. What follows is the cost of using one anyway."
    )
    # Remove the two features that hand the answer over, so the task is a real
    # prediction rather than a lookup.
    drop = [
        FEATURE_NAMES.index("log10_peak_act_bytes"),
        FEATURE_NAMES.index("log10_weight_bytes"),
    ]
    keep = [i for i in range(len(FEATURE_NAMES)) if i not in drop]
    ablated = features[:, keep]
    forest = LatencyPredictor(
        n_estimators=120, seed=FOREST_SEED, target_name="peak memory [B]"
    )
    # The predictor validates its input width against FEATURE_NAMES, so the
    # ablated matrix is padded with zeros rather than silently reshaped.
    padded = np.zeros_like(features)
    padded[:, : len(keep)] = ablated
    forest.fit(padded[train], truth[train])
    predicted = forest.predict(padded[test])
    metrics = evaluate_predictions(
        predicted, truth[test], "learned forest, memory features ablated"
    )
    for line in metrics.summary_lines():
        print(f"      {line}")
    print(
        f"      analytic error on the same models: 0.00 % (exact arithmetic)\n"
        f"      so the learned model is worse by "
        f"{metrics.median_abs_rel_error * 100:.2f} pp at the median and\n"
        f"      {metrics.p90_abs_rel_error * 100:.2f} pp at p90, and it needs the "
        "population to have been\n      measured first. For peak memory the analytic "
        "model wins outright."
    )


def stability_study(graphs, features: np.ndarray, models, train, test) -> dict:
    """Repeat measure-fit-compare several times, from scratch each time.

    The measured latency of a model on a shared single-core host is not a
    fixed quantity: it moves with whatever else is running. A single
    comparison therefore cannot establish which predictor is better, only
    which was better on one pass. This section runs the whole pipeline
    ``STABILITY_PASSES`` times and reports the direction and the margin each
    time.
    """
    print("\n(4) run-to-run stability of the comparison")
    print("-" * 78)
    print(
        f"  {STABILITY_PASSES} independent passes. Each re-measures all "
        f"{len(models)} models, refits the\n  analytic baseline's four parameters "
        "and refits the forest, on the same\n  train/test split."
    )
    outcomes: dict[str, list[tuple[float, float, str]]] = {"p50": [], "p99": []}
    for pass_index in range(STABILITY_PASSES):
        p50, p99, _m = measure_population(models)
        node_flops, node_bytes, graph_index = node_cost_arrays(graphs)
        mask = np.isin(graph_index, train)
        remap = {g: i for i, g in enumerate(train)}
        sub_index = np.asarray([remap[g] for g in graph_index[mask]])
        for key, measured in (("p50", p50), ("p99", p99)):
            fitted = calibrate_device(
                node_flops[mask], node_bytes[mask], sub_index, measured[train],
                "stability-fit", grid=CALIBRATION_GRID,
            )
            analytic_test = np.asarray(
                [analytic_estimate(graphs[i], fitted).latency_s for i in test]
            )
            forest = LatencyPredictor(
                n_estimators=120, seed=FOREST_SEED + pass_index
            ).fit(features[train], measured[train])
            learned_test = forest.predict(features[test])
            m_a, m_l, verdict = compare_predictors(
                analytic_test, learned_test, measured[test]
            )
            if "no measurable advantage" in verdict:
                direction = "tie"
            elif "learned random forest wins" in verdict:
                direction = "learned"
            else:
                direction = "analytic"
            outcomes[key].append(
                (m_a.median_abs_rel_error, m_l.median_abs_rel_error, direction)
            )
        print(
            f"    pass {pass_index + 1}: p50 -> {outcomes['p50'][-1][2]:<8} "
            f"(analytic {outcomes['p50'][-1][0] * 100:6.2f} %, learned "
            f"{outcomes['p50'][-1][1] * 100:6.2f} %)   "
            f"p99 -> {outcomes['p99'][-1][2]:<8} "
            f"(analytic {outcomes['p99'][-1][0] * 100:7.2f} %, learned "
            f"{outcomes['p99'][-1][1] * 100:7.2f} %)"
        )

    summary: dict[str, dict] = {}
    for key, rows in outcomes.items():
        directions = [row[2] for row in rows]
        analytic_errors = np.asarray([row[0] for row in rows])
        learned_errors = np.asarray([row[1] for row in rows])
        counts = {d: directions.count(d) for d in ("learned", "analytic", "tie")}
        summary[key] = {
            "counts": counts,
            "analytic_median": float(np.median(analytic_errors)),
            "analytic_range": (float(analytic_errors.min()), float(analytic_errors.max())),
            "learned_median": float(np.median(learned_errors)),
            "learned_range": (float(learned_errors.min()), float(learned_errors.max())),
        }
        print(
            f"\n  {key}: learned won {counts['learned']}/{STABILITY_PASSES}, "
            f"analytic won {counts['analytic']}/{STABILITY_PASSES}, "
            f"tie {counts['tie']}/{STABILITY_PASSES}"
        )
        print(
            f"       analytic median |rel error| across passes: "
            f"{analytic_errors.min() * 100:.2f} % to {analytic_errors.max() * 100:.2f} %"
        )
        print(
            f"       learned  median |rel error| across passes: "
            f"{learned_errors.min() * 100:.2f} % to {learned_errors.max() * 100:.2f} %"
        )
    return summary


def main() -> int:
    print("=" * 78)
    print("P033 edgeinfer -- Validation 5: analytic baseline vs learned predictor")
    print("=" * 78)
    print(f"environment: {describe_environment(note='')}")
    print(
        "WORKSTATION NUMBERS. Every measured latency below comes from a shared,\n"
        "single-CPU-core cloud container with other build jobs running "
        "concurrently.\nNo number here is a measurement of any edge target, and the "
        "Jetson Orin Nano\ncolumn of every results table in this repository is empty "
        "for that reason."
    )
    print(
        f"\npopulation: {N_MODELS} models, seed {POPULATION_SEED}; "
        f"split {1 - TEST_FRACTION:.0%}/{TEST_FRACTION:.0%}, seed {SPLIT_SEED}"
    )

    models, summaries = generate_population(N_MODELS, seed=POPULATION_SEED)
    graphs = [m.graph for m in models]
    families: dict[str, int] = {}
    for summary in summaries:
        families[summary.family] = families.get(summary.family, 0) + 1
    print(f"families: {families}")
    node_counts = [len(g.nodes) for g in graphs]
    print(
        f"nodes per graph: min {min(node_counts)}, median "
        f"{int(np.median(node_counts))}, max {max(node_counts)}"
    )

    print("\n(1) measurement campaign")
    print("-" * 78)
    p50, p99, _method = measure_population(models)

    features = population_features(graphs)
    train, test = split_indices(N_MODELS, TEST_FRACTION, seed=SPLIT_SEED)
    print(f"  train {len(train)} models, held-out test {len(test)} models")

    print("\n(2) comparison, one measurement pass in detail")
    print("-" * 78)
    _wins_p50, verdict_p50, stats_p50 = compare_target(
        graphs, features, p50, train, test, "p50 latency [s]"
    )
    _wins_p99, verdict_p99, stats_p99 = compare_target(
        graphs, features, p99, train, test, "p99 latency [s]"
    )

    print("\n(3) peak memory")
    print("-" * 78)
    memory_contrast(graphs, features, train, test)

    stability = stability_study(graphs, features, models, train, test)

    print("\n" + "=" * 78)
    print("HEADLINE RESULT")
    print("=" * 78)
    print("One measurement pass, in detail:")
    print(f"  p50 latency : {verdict_p50}")
    print(f"  p99 latency : {verdict_p99}")
    print("  peak memory : analytic wins outright (exact arithmetic vs a fitted model)")
    print()
    print(f"Across {STABILITY_PASSES} independent passes, which is the figure to "
          "believe:")
    for key in ("p50", "p99"):
        counts = stability[key]["counts"]
        print(
            f"  {key} latency : learned won {counts['learned']}, analytic won "
            f"{counts['analytic']}, tie {counts['tie']}"
        )
        print(
            f"               analytic median |rel error| "
            f"{stability[key]['analytic_range'][0] * 100:.2f}-"
            f"{stability[key]['analytic_range'][1] * 100:.2f} %, "
            f"learned {stability[key]['learned_range'][0] * 100:.2f}-"
            f"{stability[key]['learned_range'][1] * 100:.2f} %"
        )
    print()
    print(
        "No retuning was done to change any of these outcomes. The analytic model's\n"
        "four parameters were fitted once per pass, on the training split, with the\n"
        "grid size declared in this script; the forest's hyperparameters are the\n"
        "package defaults. Both predictors saw the same features and the same\n"
        "training set, and the analytic model was written and validated first."
    )
    print()
    print("summary table from the detailed pass (held-out, |relative error|):")
    print(
        f"  {'target':<22}{'analytic med':>14}{'learned med':>13}"
        f"{'analytic p90':>14}{'learned p90':>13}"
    )
    for label, stats in (("p50 latency", stats_p50), ("p99 latency", stats_p99)):
        print(
            f"  {label:<22}{stats['analytic_median_rel'] * 100:>13.2f}%"
            f"{stats['learned_median_rel'] * 100:>12.2f}%"
            f"{stats['analytic_p90_rel'] * 100:>13.2f}%"
            f"{stats['learned_p90_rel'] * 100:>12.2f}%"
        )
    print(
        f"\n  Spearman rho, p50 target: analytic "
        f"{stats_p50['analytic_spearman']:.4f}, learned "
        f"{stats_p50['learned_spearman']:.4f}"
    )
    print(
        f"  Spearman rho, p99 target: analytic "
        f"{stats_p99['analytic_spearman']:.4f}, learned "
        f"{stats_p99['learned_spearman']:.4f}"
    )
    print("=" * 78)
    print("RESULT: campaign completed. See the stability block above, not the "
          "single pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
