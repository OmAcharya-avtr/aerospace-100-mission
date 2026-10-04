#!/usr/bin/env python3
"""Example 4 --- the four failure modes, plotted.

Left panel: a declared thermal-throttle sweep. The analytic latency of one
candidate is recomputed for a range of declared clock factors and plotted
against the budget ceiling, showing the factor at which the candidate stops
fitting. Every factor is DECLARED; there is no thermal model and no device
here.

Right panel: a feasibility map over the two declarations that most often
contradict each other --- a worst-case latency ceiling and a duty cycle times
a control period. The shaded region is infeasible by construction and no model
can satisfy it, which this package detects before anything is measured.

The other two failure modes --- a model over the memory budget and an
unsupported operator --- are text, because they are categorical rather than
continuous; both are printed by this script and exercised in
``tests/test_failure_modes.py`` and ``validation/validate_budget.py``.

Saves ``../screenshots/failure_modes.png``.

Runtime: about 10 s on one CPU core.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from edgeinfer.analytic import analytic_estimate
from edgeinfer.budget import Budget, Verdict, build_report
from edgeinfer.dataset import random_cnn, random_mlp
from edgeinfer.environment import environment_record
from edgeinfer.graph import ModelGraph, Node, TensorSpec
from edgeinfer.ops import UnsupportedOperatorError
from edgeinfer.roofline import DeviceModel
from edgeinfer.thermal import ThrottleState, throttled_device

matplotlib.use("Agg")  # no display in a container; show() is never called

OUT = Path(__file__).resolve().parent.parent / "screenshots" / "failure_modes.png"

DEVICE = DeviceModel(
    "declared target",
    peak_flops=4.0e9,
    peak_bandwidth_bytes_s=10.0e9,
    source="declared; not measured and not a datasheet figure",
)


def memory_overrun_text() -> list[str]:
    rng = np.random.default_rng(20260401)
    model = random_mlp(rng, "oversized", n_in=256, widths=(1024, 1024), n_out=64)
    estimate = analytic_estimate(model.graph, DEVICE)
    budget = Budget("1 MiB ceiling", latency_s=10.0, peak_memory_bytes=1024 * 1024)
    report = build_report(
        budget,
        candidate=model.graph.name,
        environment="analytic only; no measurement taken",
        worst_case_latency_s=1e-5,
        worst_case_uncertainty_s=1e-9,
        peak_memory_bytes=float(estimate.peak_memory_bytes),
        peak_memory_uncertainty_bytes=0.0,
        memory_method="analytic liveness analysis",
        latency_method="placeholder",
    )
    memory_row = next(r for r in report.rows if r.quantity == "peak memory")
    return [
        "FAILURE MODE 1 -- model over the memory budget",
        f"  candidate              : {model.graph.name} (256-1024-1024-64 MLP, float32)",
        f"  analytic peak memory   : {estimate.peak_memory_bytes} B "
        f"({estimate.weight_bytes} B weights + "
        f"{estimate.peak_activation_bytes} B activations)",
        f"  declared ceiling       : {budget.peak_memory_bytes} B",
        f"  utilisation            : {memory_row.utilisation * 100:.1f} %",
        f"  verdict                : {memory_row.verdict.value}",
        "  decided with no measurement at all: integer arithmetic over the graph.",
    ]


def unsupported_operator_text() -> list[str]:
    lines = ["", "FAILURE MODE 2 -- unsupported operator"]
    for op in ("GRU", "Einsum", "Transpose"):
        graph = ModelGraph(
            name=f"has_{op}",
            inputs=(TensorSpec("X", (1, 8)),),
            outputs=("Y",),
            nodes=(Node("n", op, ("X",), ("Y",)),),
        )
        try:
            analytic_estimate(graph, DEVICE)
            lines.append(f"  {op:<10} NO ERROR -- the cost model charged it something")
        except UnsupportedOperatorError as exc:
            lines.append(f"  {op:<10} refused: {str(exc).split('. ')[0]}")
    lines.append(
        "  An unknown operator charged zero cost turns an overrun into a pass."
    )
    return lines


def main() -> None:
    rng = np.random.default_rng(7)
    model = random_cnn(
        rng, "candidate", spatial=24, channels=(8, 16), kernel=3, n_out=10
    )
    nominal = analytic_estimate(model.graph, DEVICE).latency_s
    budget = Budget(
        "worst case <= 2x nominal",
        latency_s=nominal * 2.0,
        peak_memory_bytes=8 * 1024 * 1024,
    )

    text_lines = memory_overrun_text() + unsupported_operator_text()
    for line in text_lines:
        print(line)

    print("\nFAILURE MODE 3 -- declared thermal throttle")
    factors = np.linspace(1.0, 0.1, 46)
    latencies = np.empty_like(factors)
    verdicts = []
    for i, factor in enumerate(factors):
        device = throttled_device(
            DEVICE,
            ThrottleState(
                f"declared {factor * 100:.0f} % clock",
                compute_factor=float(factor),
                bandwidth_factor=1.0,
                basis="declared for this example; no device measurement exists",
            ),
        )
        latencies[i] = analytic_estimate(model.graph, device).latency_s
        report = build_report(
            budget,
            candidate=model.graph.name,
            environment="analytic only; declared throttle, NOT a device measurement",
            worst_case_latency_s=latencies[i],
            worst_case_uncertainty_s=0.0,
            peak_memory_bytes=1.0,
            peak_memory_uncertainty_bytes=0.0,
            latency_method="analytic roofline on a declared-throttled device",
            memory_method="analytic liveness analysis",
        )
        verdicts.append(report.overall)
    failing = [f for f, v in zip(factors, verdicts, strict=True) if v is Verdict.FAIL]
    breaking_point = max(failing) if failing else None
    print(f"  nominal analytic latency : {nominal * 1e6:.3f} us")
    print(f"  declared ceiling         : {budget.latency_s * 1e6:.3f} us")
    if breaking_point is not None:
        print(
            f"  highest declared clock factor that still FAILS: "
            f"{breaking_point * 100:.0f} %"
        )
    else:
        print("  the candidate fits at every declared clock factor in the sweep")

    print("\nFAILURE MODE 4 -- budget infeasible by construction")
    demo = Budget(
        "contradictory", latency_s=2e-3, peak_memory_bytes=1 << 20,
        duty_cycle=0.10, period_s=10e-3,
    )
    for reason in demo.feasibility().reasons:
        print(f"  {reason}")

    env = environment_record(shared_host=True, note="")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    ax = axes[0]
    ax.plot(factors * 100, latencies * 1e6, color="#b5493f", linewidth=2)
    ax.axhline(
        budget.latency_s * 1e6, color="black", linestyle="--", linewidth=1.2,
        label=f"declared ceiling {budget.latency_s * 1e6:.1f} us",
    )
    if breaking_point is not None:
        ax.axvline(
            breaking_point * 100, color="grey", linestyle=":", linewidth=1.2,
            label=f"fails at or below {breaking_point * 100:.0f} % clock",
        )
    ax.set_xlabel("declared compute-clock factor [% of nominal]")
    ax.set_ylabel("analytic worst-case latency [us]")
    ax.set_title(
        "Failure mode 3: declared thermal throttle\n"
        "every factor is DECLARED -- no thermal model, no device measurement"
    )
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    latency_ms = np.linspace(0.1, 5.0, 220)
    duty_times_period_ms = np.linspace(0.1, 5.0, 220)
    grid_latency, grid_allowed = np.meshgrid(latency_ms, duty_times_period_ms)
    infeasible = (grid_allowed < grid_latency).astype(float)
    ax.contourf(
        grid_latency, grid_allowed, infeasible, levels=[0.5, 1.5], colors=["#e4b7b2"]
    )
    ax.contour(
        grid_latency, grid_allowed, infeasible, levels=[0.5], colors=["#b5493f"],
        linewidths=1.5,
    )
    ax.plot([], [], color="#b5493f", linewidth=6, alpha=0.5,
            label="infeasible by construction")
    ax.plot(
        [2.0], [1.0], "o", color="black", markersize=8,
        label="the example above: 2 ms ceiling, 10 % of 10 ms = 1 ms",
    )
    ax.set_xlabel("declared worst-case latency ceiling [ms]")
    ax.set_ylabel("duty cycle x control period [ms]")
    ax.set_title(
        "Failure mode 4: a budget that contradicts itself\n"
        "detected before any model is loaded or measured"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    footer_lines = [
        f"Produced by examples/failure_modes.py on: {env.one_line()}",
        "Both panels are ANALYTIC: nothing was executed and nothing was measured. "
        "The device peaks and every throttle factor are declared values.",
        text_lines[1].strip() + "  |  " + text_lines[3].strip(),
        "Failure modes 1 and 2 are categorical and printed as text by this script; "
        "all four are exercised in tests/test_failure_modes.py and "
        "validation/validate_budget.py.",
        "Research-grade. Not flight-qualified, not certified, not approved for "
        "operational aerospace use.",
    ]
    wrapped: list[str] = []
    for line in footer_lines:
        wrapped.extend(textwrap.wrap(line, width=168) or [""])
    fig.text(
        0.01, 0.012, "\n".join(wrapped), fontsize=6.8, va="bottom", family="monospace"
    )
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
