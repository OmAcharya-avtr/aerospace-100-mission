#!/usr/bin/env python3
"""Example 1 --- declare a budget, check a candidate, plot the verdict.

Declares a latency/memory/power/duty-cycle envelope, characterises three
candidate models analytically and by measurement, and plots utilisation
against each limit with the worst case and the median shown separately.

Saves ``../screenshots/budget_check.png``.

Every number in the figure comes from this script's own run on the host it is
executed on. That host is stated in the figure's footer, because a latency
without its environment is not a number anyone can use.

Runtime: about 15 s on one CPU core.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from edgeinfer.analytic import analytic_estimate
from edgeinfer.backends import OnnxRuntimeBackend
from edgeinfer.budget import Budget, build_report
from edgeinfer.dataset import random_cnn, random_mlp
from edgeinfer.environment import environment_record
from edgeinfer.harness import benchmark
from edgeinfer.roofline import DeviceModel

# No display in a container. matplotlib.use() after importing pyplot is
# supported as long as no figure exists yet, and keeping it here lets every
# import stay at the top of the file. show() is never called.
matplotlib.use("Agg")

OUT = Path(__file__).resolve().parent.parent / "screenshots" / "budget_check.png"
REPEATS = 200
WARMUP = 10

BUDGET = Budget(
    name="200 Hz attitude loop, 20 % duty",
    latency_s=1.0e-3,
    peak_memory_bytes=2 * 1024 * 1024,
    median_latency_s=400e-6,
    power_w=7.0,
    duty_cycle=0.20,
    period_s=5.0e-3,
    worst_case_quantile=0.99,
)

DEVICE = DeviceModel(
    "declared target",
    peak_flops=4.0e9,
    peak_bandwidth_bytes_s=10.0e9,
    source="declared; not measured and not a datasheet figure",
)


def candidates():
    rng = np.random.default_rng(20260401)
    return [
        ("small MLP", random_mlp(rng, "small_mlp", n_in=64, widths=(64,), n_out=8)),
        (
            "medium CNN",
            random_cnn(rng, "medium_cnn", spatial=24, channels=(8, 16), kernel=3, n_out=8),
        ),
        (
            "large MLP",
            random_mlp(rng, "large_mlp", n_in=512, widths=(1024, 512), n_out=64),
        ),
    ]


def main() -> None:
    print("budget declaration")
    print("-" * 70)
    for line in BUDGET.summary_lines():
        print(f"  {line}")
    feasibility = BUDGET.feasibility()
    print(f"  feasibility             : {'self-consistent' if feasibility else 'INFEASIBLE'}")
    if not feasibility:
        for reason in feasibility.reasons:
            print(f"    - {reason}")
        return

    env = environment_record(shared_host=True, note="")
    rows = []
    for label, model in candidates():
        estimate = analytic_estimate(model.graph, DEVICE)
        backend = OnnxRuntimeBackend(model.model_bytes, model.input_feed(0))
        backend.prepare()
        try:
            profile = benchmark(
                backend.infer, label=label, repeats=REPEATS, warmup=WARMUP,
                timer_bias_samples=500,
            )
        finally:
            backend.close()
        tail = profile.uncertainty("p99", n_resamples=400, seed=0)
        median = profile.uncertainty("p50", n_resamples=400, seed=0)
        report = build_report(
            BUDGET,
            candidate=label,
            environment=env.one_line(),
            worst_case_latency_s=tail.value,
            worst_case_uncertainty_s=tail.combined_s,
            median_latency_s=median.value,
            median_uncertainty_s=median.combined_s,
            peak_memory_bytes=float(estimate.peak_memory_bytes),
            peak_memory_uncertainty_bytes=0.0,
            declared_power_w=BUDGET.power_w,
            latency_method=profile.method_line(),
            memory_method="analytic liveness analysis (Aho et al. 2006 section 8.4)",
        )
        rows.append((label, estimate, profile, tail, median, report))
        print(f"\n{label}")
        print("-" * 70)
        for line in report.summary_lines():
            print(f"  {line}")
        print(f"  analytic latency estimate: {estimate.latency_s * 1e6:.3f} us")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    labels = [row[0] for row in rows]
    x = np.arange(len(labels))
    width = 0.34
    tail_util = [row[5].rows[0].utilisation * 100 for row in rows]
    tail_err = [
        row[3].combined_s / BUDGET.effective_latency_s * 100 for row in rows
    ]
    median_util = [row[5].rows[1].utilisation * 100 for row in rows]
    median_err = [row[4].combined_s / BUDGET.median_latency_s * 100 for row in rows]

    ax = axes[0]
    ax.bar(
        x - width / 2, tail_util, width, yerr=tail_err, capsize=4,
        color="#b5493f", label=f"worst case (p{BUDGET.worst_case_quantile * 100:g})",
    )
    ax.bar(
        x + width / 2, median_util, width, yerr=median_err, capsize=4,
        color="#4c72b0", label="median (p50)",
    )
    ax.axhline(100.0, color="black", linestyle="--", linewidth=1.2)
    ax.text(
        len(labels) - 0.45, 104.0, "declared limit", ha="right", va="bottom", fontsize=9
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("latency used, % of its own declared limit (log scale)")
    ax.set_title(
        "Latency against the budget\n"
        f"worst case vs {BUDGET.effective_latency_s * 1e6:.0f} us, "
        f"median vs {BUDGET.median_latency_s * 1e6:.0f} us"
    )
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_ylim(top=max(t + e for t, e in zip(tail_util, tail_err, strict=True)) * 4.0)
    for xi, (util, verdict) in enumerate(
        zip(tail_util, [row[5].rows[0].verdict.value for row in rows], strict=True)
    ):
        ax.annotate(
            verdict, (xi - width / 2, util), textcoords="offset points",
            xytext=(0, 12), ha="center", fontsize=8,
        )

    ax = axes[1]
    memory_util = [row[5].rows[2].utilisation * 100 for row in rows]
    analytic_latency_us = [row[1].latency_s * 1e6 for row in rows]
    measured_tail_us = [row[3].value * 1e6 for row in rows]
    ax.bar(x, memory_util, 0.5, color="#55a868")
    ax.axhline(100.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("analytic peak memory, % of the 2 MiB limit (log scale)")
    ax.set_title(
        "Peak memory against the budget\n"
        "analytic liveness analysis; a lower bound, no runtime arena"
    )
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_ylim(top=max(memory_util) * 4.0)
    for xi, (util, verdict) in enumerate(
        zip(memory_util, [row[5].rows[2].verdict.value for row in rows], strict=True)
    ):
        ax.annotate(
            verdict, (xi, util), textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=8,
        )

    footer = (
        f"Every number measured by examples/budget_check.py on: {env.one_line()}\n"
        f"Latency: perf_counter per-call bracket, n={REPEATS} repeats after "
        f"{WARMUP} warm-up calls; error bars are the combined standard uncertainty "
        "(GUM; bootstrap for a quantile).\n"
        "WORKSTATION NUMBERS from a shared single-core container -- not a "
        "measurement of any edge target. Power is declared, never measured. "
        "Research-grade.\n"
        "Analytic latency estimates for the same three models: "
        + ", ".join(
            f"{label} {value:.1f} us" for label, value in
            zip(labels, analytic_latency_us, strict=True)
        )
        + "; measured p99: "
        + ", ".join(
            f"{label} {value:.1f} us" for label, value in
            zip(labels, measured_tail_us, strict=True)
        )
    )
    fig.text(0.01, 0.015, footer, fontsize=7.2, va="bottom", family="monospace")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
