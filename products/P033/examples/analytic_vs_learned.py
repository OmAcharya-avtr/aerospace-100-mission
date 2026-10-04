#!/usr/bin/env python3
"""Example 3 --- the analytic baseline next to the learned predictor.

Runs the comparison the product is built around: a reproducible population of
small ONNX graphs is measured on this host, the analytic roofline baseline is
calibrated on a training split, a random forest is fitted on the same split,
and both are plotted against the held-out truth.

The figure shows the result whichever way it falls, for the median and for the
99th percentile separately, and the error bars on the learned model are its
ensemble-disagreement uncertainty.

Saves ``../screenshots/analytic_vs_learned.png``.

The full campaign, including a five-pass stability study, is
``validation/validate_predictor.py``. This example is the short version, sized
to run in well under a minute on one CPU core.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from edgeinfer.analytic import analytic_estimate, node_cost_arrays
from edgeinfer.backends import OnnxRuntimeBackend
from edgeinfer.dataset import generate_population
from edgeinfer.environment import environment_record
from edgeinfer.features import population_features
from edgeinfer.harness import benchmark
from edgeinfer.predictor import LatencyPredictor, compare_predictors, split_indices
from edgeinfer.roofline import calibrate_device

matplotlib.use("Agg")  # no display in a container; show() is never called

OUT = Path(__file__).resolve().parent.parent / "screenshots" / "analytic_vs_learned.png"
N_MODELS = 100
SEED = 20260401
REPEATS = 50
WARMUP = 8


def fit_and_predict(graphs, features, measured, train, test):
    node_flops, node_bytes, graph_index = node_cost_arrays(graphs)
    mask = np.isin(graph_index, train)
    remap = {g: i for i, g in enumerate(train)}
    fitted = calibrate_device(
        node_flops[mask],
        node_bytes[mask],
        np.asarray([remap[g] for g in graph_index[mask]]),
        measured[train],
        "container-fit",
        grid=16,
    )
    analytic = np.asarray([analytic_estimate(graphs[i], fitted).latency_s for i in test])
    forest = LatencyPredictor(n_estimators=120, seed=SEED).fit(
        features[train], measured[train]
    )
    predictions = forest.predict_with_uncertainty(features[test])
    learned = np.asarray([p.value for p in predictions])
    learned_u = np.asarray([p.uncertainty for p in predictions])
    return fitted, analytic, learned, learned_u


def main() -> None:
    models, _summaries = generate_population(N_MODELS, seed=SEED)
    graphs = [m.graph for m in models]
    features = population_features(graphs)

    p50 = np.empty(N_MODELS)
    p99 = np.empty(N_MODELS)
    method = ""
    for i, model in enumerate(models):
        backend = OnnxRuntimeBackend(model.model_bytes, model.input_feed(0))
        backend.prepare()
        try:
            profile = benchmark(
                backend.infer, label=model.graph.name, repeats=REPEATS, warmup=WARMUP,
                timer_bias_samples=200,
            )
        finally:
            backend.close()
        p50[i] = profile.p50_s
        p99[i] = profile.p99_s
        method = profile.method_line()
    train, test = split_indices(N_MODELS, 0.3, seed=SEED)
    print(f"population {N_MODELS} models, train {len(train)}, held-out {len(test)}")
    print(f"measurement method: {method}")

    env = environment_record(shared_host=True, note="")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))
    verdicts = []

    for ax, measured, target_label in (
        (axes[0], p50, "median latency (p50)"),
        (axes[1], p99, "worst-case latency (p99)"),
    ):
        fitted, analytic, learned, learned_u = fit_and_predict(
            graphs, features, measured, train, test
        )
        metrics_a, metrics_l, verdict = compare_predictors(
            analytic, learned, measured[test]
        )
        verdicts.append((target_label, verdict))
        print(f"\n{target_label}")
        print("-" * 70)
        print(f"  fitted device: {fitted.peak_flops / 1e9:.3f} GFLOP/s, "
              f"{fitted.peak_bandwidth_bytes_s / 1e9:.3f} GB/s, "
              f"{fitted.overhead_per_node_s * 1e6:.3f} us/node, "
              f"{fitted.fixed_overhead_s * 1e6:.3f} us fixed ({fitted.source})")
        for line in metrics_a.summary_lines():
            print(f"  {line}")
        for line in metrics_l.summary_lines():
            print(f"  {line}")
        print(f"  verdict: {verdict}")

        truth_us = measured[test] * 1e6
        lo = min(truth_us.min(), analytic.min() * 1e6, learned.min() * 1e6) * 0.6
        hi = max(truth_us.max(), analytic.max() * 1e6, learned.max() * 1e6) * 1.6
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--",
                label="exact")
        ax.errorbar(
            truth_us, learned * 1e6, yerr=learned_u * 1e6, fmt="o", markersize=5,
            color="#4c72b0", ecolor="#4c72b0", elinewidth=0.8, capsize=2, alpha=0.8,
            label=(
                f"learned forest, median |err| "
                f"{metrics_l.median_abs_rel_error * 100:.1f} %"
            ),
        )
        ax.plot(
            truth_us, analytic * 1e6, "s", markersize=5, color="#b5493f", alpha=0.8,
            label=(
                f"analytic roofline (4-param fit), median |err| "
                f"{metrics_a.median_abs_rel_error * 100:.1f} %"
            ),
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"measured {target_label} [us]")
        ax.set_ylabel(f"predicted {target_label} [us]")
        ax.set_title(
            f"{target_label}\n"
            f"Spearman: analytic {metrics_a.spearman:.3f}, "
            f"learned {metrics_l.spearman:.3f}"
        )
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=8)

    raw_footer = [
        f"Produced by examples/analytic_vs_learned.py on: {env.one_line()}",
        f"Population: {N_MODELS} synthetic ONNX graphs, seed {SEED}; 70/30 split, "
        f"seed {SEED}; {method}",
        "Error bars on the learned points are the random forest's "
        "ensemble-disagreement spread, which is NOT a calibrated interval -- see "
        "validation/validate_predictor.py for its measured coverage.",
        *(f"{label}: {verdict}" for label, verdict in verdicts),
        "WORKSTATION NUMBERS from a shared single-core container. Research-grade, "
        "not flight-qualified. The single-pass result above moves between runs; "
        "the five-pass study in validation/ is the figure to believe.",
    ]
    wrapped: list[str] = []
    for line in raw_footer:
        wrapped.extend(textwrap.wrap(line, width=168) or [""])
    fig.text(
        0.01, 0.012, "\n".join(wrapped), fontsize=6.6, va="bottom", family="monospace"
    )
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
