#!/usr/bin/env python3
"""Example 2 --- the roofline plot, and why the median is not the deadline.

Left panel: the roofline of Williams, Waterman & Patterson 2009 for a declared
device, with every node of a small CNN placed on it by its arithmetic
intensity, so a reader can see which nodes are memory bound and which are
compute bound.

Right panel: the measured latency distribution of one model, with p50, p99 and
the maximum marked. The distance between them is the whole argument for
reporting worst case separately: the median is not the number a control loop
has to survive.

Saves ``../screenshots/roofline_and_tail.png``.

Runtime: about 15 s on one CPU core.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from edgeinfer.analytic import analytic_estimate
from edgeinfer.backends import OnnxRuntimeBackend
from edgeinfer.dataset import random_cnn
from edgeinfer.environment import environment_record
from edgeinfer.harness import benchmark
from edgeinfer.roofline import DeviceModel

matplotlib.use("Agg")  # no display in a container; show() is never called

OUT = Path(__file__).resolve().parent.parent / "screenshots" / "roofline_and_tail.png"
REPEATS = 1500
WARMUP = 20

DEVICE = DeviceModel(
    "declared target",
    peak_flops=4.0e9,
    peak_bandwidth_bytes_s=10.0e9,
    source="declared; not measured and not a datasheet figure",
)


def main() -> None:
    rng = np.random.default_rng(20260401)
    model = random_cnn(
        rng, "roofline_demo", spatial=32, channels=(8, 16, 32), kernel=3, n_out=10
    )
    estimate = analytic_estimate(model.graph, DEVICE)

    print("analytic estimate")
    print("-" * 70)
    for line in estimate.summary_lines():
        print(f"  {line}")
    print(f"\n  ridge point: {DEVICE.ridge_point_flops_per_byte:.4f} FLOP/B")
    print(
        f"\n  {'node':<12}{'op':<18}{'I [FLOP/B]':>12}{'attainable':>14}"
        f"{'t [us]':>10}  bound by"
    )
    for node in estimate.nodes:
        intensity = node.cost.arithmetic_intensity
        attainable = DEVICE.attainable_flops(intensity)
        print(
            f"  {node.name:<12}{node.op_type:<18}{intensity:>12.4g}"
            f"{attainable / 1e9:>12.4f} G{node.roofline_s * 1e6:>10.4f}  "
            f"{'memory' if node.memory_bound else 'compute'}"
        )

    backend = OnnxRuntimeBackend(model.model_bytes, model.input_feed(0))
    backend.prepare()
    try:
        profile = benchmark(
            backend.infer, label=model.graph.name, repeats=REPEATS, warmup=WARMUP,
            timer_bias_samples=1000,
        )
    finally:
        backend.close()
    print("\nmeasured profile")
    print("-" * 70)
    for line in profile.summary_lines():
        print(f"  {line}")
    tail = profile.uncertainty("p99", n_resamples=400, seed=0)
    median = profile.uncertainty("p50", n_resamples=400, seed=0)
    print(f"\n  u_c(p50) = {median.combined_s * 1e6:.4f} us")
    print(f"  u_c(p99) = {tail.combined_s * 1e6:.4f} us")

    env = environment_record(shared_host=True, note="")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    ax = axes[0]
    intensities = np.logspace(-3, 3, 400)
    attainable = np.minimum(
        DEVICE.peak_flops, DEVICE.peak_bandwidth_bytes_s * intensities
    )
    ax.loglog(intensities, attainable / 1e9, color="black", linewidth=2, label="roofline")
    ax.axvline(
        DEVICE.ridge_point_flops_per_byte, color="grey", linestyle=":", linewidth=1.2
    )
    ax.text(
        DEVICE.ridge_point_flops_per_byte * 1.15,
        DEVICE.peak_flops / 1e9 * 0.012,
        f"ridge {DEVICE.ridge_point_flops_per_byte:.2f} FLOP/B",
        fontsize=8,
        rotation=90,
        va="bottom",
    )
    # Several nodes share an arithmetic intensity (every Relu in this graph has
    # the same one), so labels are stacked rather than written on top of each
    # other.
    seen: dict[float, int] = {}
    for node in estimate.nodes:
        intensity = max(node.cost.arithmetic_intensity, 1e-3)
        achieved = DEVICE.attainable_flops(intensity) / 1e9
        colour = "#b5493f" if node.memory_bound else "#4c72b0"
        ax.plot(intensity, achieved, "o", color=colour, markersize=8)
        bucket = round(float(np.log10(intensity)), 1)
        level = seen.get(bucket, 0)
        seen[bucket] = level + 1
        ax.annotate(
            node.name,
            (intensity, achieved),
            textcoords="offset points",
            xytext=(7, -11 - 10 * level),
            fontsize=7.5,
        )
    ax.plot([], [], "o", color="#b5493f", label="memory bound")
    ax.plot([], [], "o", color="#4c72b0", label="compute bound")
    ax.set_xlabel(
        "arithmetic intensity [FLOP per byte of compulsory traffic]\n"
        "node names: conv* Conv, crelu* Relu, pool* MaxPool, flat Reshape, fc Gemm"
    )
    ax.set_ylabel("attainable rate [GFLOP/s]")
    ax.set_title(
        "Roofline, Williams, Waterman & Patterson 2009\n"
        f"declared {DEVICE.peak_flops / 1e9:.0f} GFLOP/s and "
        f"{DEVICE.peak_bandwidth_bytes_s / 1e9:.0f} GB/s -- declared, not measured"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    samples_us = profile.samples_s * 1e6
    # Log-spaced bins on a log axis: the distribution spans two orders of
    # magnitude on a shared host, and linear bins would render the bulk as a
    # single spike and the tail as invisible singletons.
    bins = np.logspace(np.log10(samples_us.min() * 0.95),
                       np.log10(samples_us.max() * 1.05), 70)
    ax.hist(samples_us, bins=bins, color="#8d8d8d", edgecolor="none")
    ax.set_xscale("log")
    for value, colour, label in (
        (profile.p50_s * 1e6, "#4c72b0", f"p50 {profile.p50_s * 1e6:.2f} us"),
        (profile.p99_s * 1e6, "#b5493f", f"p99 {profile.p99_s * 1e6:.2f} us"),
        (profile.max_s * 1e6, "#000000", f"max {profile.max_s * 1e6:.2f} us"),
    ):
        ax.axvline(value, color=colour, linewidth=1.6, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("per-call latency [us], log scale, log-spaced bins")
    ax.set_ylabel("count (log scale)")
    ax.set_title(
        f"Measured latency distribution, n={REPEATS} repeats\n"
        f"p99/p50 = {profile.tail_ratio:.2f}: the tail is {profile.tail_ratio:.1f}x "
        "the median"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, which="both")

    footer = (
        f"Produced by examples/roofline_and_tail.py on: {env.one_line()}\n"
        f"Left: analytic only, nothing executed; device peaks are DECLARED. "
        f"Model {model.graph.name}: {estimate.total_flops} flops, "
        f"{estimate.total_traffic_bytes} B compulsory traffic, "
        f"{estimate.arithmetic_intensity:.3g} FLOP/B overall.\n"
        f"Right: perf_counter per-call bracket, n={REPEATS} repeats after {WARMUP} "
        f"warm-up; u_c(p50) = {median.combined_s * 1e6:.3f} us, "
        f"u_c(p99) = {tail.combined_s * 1e6:.3f} us (GUM + bootstrap).\n"
        "WORKSTATION NUMBERS from a shared single-core container -- not a measurement "
        "of any edge target. Research-grade, not flight-qualified."
    )
    fig.text(0.01, 0.015, footer, fontsize=7.2, va="bottom", family="monospace")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
