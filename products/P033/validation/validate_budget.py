#!/usr/bin/env python3
"""Validation 8 --- the budget decision path and the four failure modes.

Evidence that the pass/fail machinery decides the right way in the cases that
matter, including the ones where it must refuse to decide. The four failure
modes the batch specification names are each exercised end to end:

1. a model exceeding the memory budget
2. an unsupported operator
3. a thermally throttled device
4. a budget that is infeasible by construction

and two further cases that are the point of carrying uncertainty at all:

5. a pass whose margin is smaller than the measurement uncertainty, which is
   reported as MARGINAL PASS rather than PASS
6. a candidate that passes on the median and fails on the tail, which is
   reported as a failure --- a control loop is sized by its worst case

Runtime: about 20 s on one CPU core.
"""

from __future__ import annotations

import sys

import numpy as np

from edgeinfer.analytic import analytic_estimate
from edgeinfer.backends import OnnxRuntimeBackend
from edgeinfer.budget import Budget, Verdict, build_report
from edgeinfer.dataset import random_mlp
from edgeinfer.environment import describe_environment
from edgeinfer.graph import ModelGraph, Node, TensorSpec
from edgeinfer.harness import benchmark
from edgeinfer.ops import UnsupportedOperatorError
from edgeinfer.roofline import DeviceModel
from edgeinfer.thermal import ThrottleState, throttled_device

DEVICE = DeviceModel(
    "declared target", 4e9, 10e9, source="declared; not measured, not a datasheet figure"
)
REPEATS = 200
WARMUP = 10


def _expect(label: str, got: Verdict, want: Verdict) -> bool:
    ok = got is want
    print(f"  {label:<52} got {got.value:<14} want {want.value:<14} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def case_memory_overrun() -> list[bool]:
    print("\n(1) a model exceeding the memory budget")
    print("-" * 78)
    rng = np.random.default_rng(20260401)
    model = random_mlp(rng, "oversized", n_in=256, widths=(1024, 1024), n_out=64)
    estimate = analytic_estimate(model.graph, DEVICE)
    budget = Budget("1 MB ceiling", latency_s=10.0, peak_memory_bytes=1024 * 1024)
    print(
        f"  analytic peak memory : {estimate.peak_memory_bytes} B "
        f"({estimate.weight_bytes} B weights + {estimate.peak_activation_bytes} B "
        "activations)"
    )
    print(f"  declared ceiling     : {budget.peak_memory_bytes} B")
    print(
        f"  utilisation          : "
        f"{estimate.peak_memory_bytes / budget.peak_memory_bytes * 100:.1f} %"
    )
    report = build_report(
        budget,
        candidate=model.graph.name,
        environment="analytic only; no measurement needed for this decision",
        worst_case_latency_s=1e-5,
        worst_case_uncertainty_s=1e-9,
        peak_memory_bytes=float(estimate.peak_memory_bytes),
        peak_memory_uncertainty_bytes=0.0,
        memory_method="analytic liveness analysis (Aho et al. 2006 section 8.4)",
        latency_method="placeholder for this case",
    )
    results = [
        _expect("memory row", next(r for r in report.rows if r.quantity == "peak memory").verdict,
                Verdict.FAIL),
        _expect("latency row (unaffected)", report.rows[0].verdict, Verdict.PASS),
        _expect("overall", report.overall, Verdict.FAIL),
    ]
    print(
        "  note: the decision needed no measurement at all. The memory overrun is\n"
        "  integer arithmetic over the graph, available before the model is loaded."
    )
    return results


def case_unsupported_operator() -> list[bool]:
    print("\n(2) an unsupported operator")
    print("-" * 78)
    results: list[bool] = []
    for op in ("GRU", "Einsum", "Loop", "Transpose"):
        graph = ModelGraph(
            name=f"has_{op}",
            inputs=(TensorSpec("X", (1, 8)),),
            outputs=("Y",),
            nodes=(Node("n", op, ("X",), ("Y",)),),
        )
        try:
            analytic_estimate(graph, DEVICE)
            print(f"  {op:<20} NO ERROR RAISED -- the cost model charged it something")
            results.append(False)
        except UnsupportedOperatorError as exc:
            first_line = str(exc).split(". ")[0]
            print(f"  {op:<20} refused: {first_line}")
            results.append(True)
    print(
        "  The refusal is the correct behaviour: an unknown operator charged zero\n"
        "  cost turns a budget overrun into a budget pass, which is the single\n"
        "  failure this product exists to prevent."
    )
    return results


def case_thermal_throttle() -> list[bool]:
    print("\n(3) a thermally throttled device")
    print("-" * 78)
    rng = np.random.default_rng(7)
    model = random_mlp(rng, "candidate", n_in=128, widths=(256, 256), n_out=16)
    nominal = analytic_estimate(model.graph, DEVICE).latency_s
    budget = Budget(
        "2x nominal headroom", latency_s=nominal * 2.0, peak_memory_bytes=8 << 20
    )
    print(
        f"  nominal analytic latency : {nominal * 1e6:.3f} us on {DEVICE.name} "
        f"({DEVICE.source})"
    )
    print(f"  declared ceiling         : {budget.latency_s * 1e6:.3f} us")
    results: list[bool] = []
    print(
        f"\n  {'declared throttle':<34}{'compute':>9}{'bandwidth':>11}"
        f"{'latency [us]':>14}{'verdict':>16}"
    )
    for label, compute, bandwidth, want in (
        ("nominal", 1.0, 1.0, Verdict.PASS),
        ("mild, declared 70 % clock", 0.70, 1.0, Verdict.PASS),
        ("severe, declared 30 % clock", 0.30, 1.0, Verdict.FAIL),
        ("severe + memory derate", 0.30, 0.50, Verdict.FAIL),
    ):
        device = throttled_device(
            DEVICE,
            ThrottleState(
                label, compute, bandwidth,
                basis="declared for this validation; no device measurement exists",
            ),
        )
        latency = analytic_estimate(model.graph, device).latency_s
        report = build_report(
            budget,
            candidate=model.graph.name,
            environment="analytic only, declared throttle; NOT a device measurement",
            worst_case_latency_s=latency,
            worst_case_uncertainty_s=0.0,
            peak_memory_bytes=1.0,
            peak_memory_uncertainty_bytes=0.0,
            latency_method=(
                "analytic roofline on a declared-throttled device; no measurement"
            ),
            memory_method="analytic liveness analysis",
        )
        ok = report.overall is want
        results.append(ok)
        print(
            f"  {label:<34}{compute:>9.2f}{bandwidth:>11.2f}{latency * 1e6:>14.3f}"
            f"{report.overall.value:>16}  {'PASS' if ok else 'FAIL'}"
        )
    print(
        "\n  Every throttle factor above is DECLARED, not measured. There is no\n"
        "  thermal model here and no device: a real throttle curve for a Jetson\n"
        "  Orin Nano would come from its DVFS frequency table and thermal zones\n"
        "  logged under load on the device itself. The bandwidth factor defaults to\n"
        "  1.0 because memory clocks throttle on their own schedule and this package\n"
        "  has no basis to derate a quantity it cannot see."
    )
    return results


def case_infeasible_budget() -> list[bool]:
    print("\n(4) a budget that is infeasible by construction")
    print("-" * 78)
    results: list[bool] = []
    cases = (
        (
            "duty cycle tighter than the latency ceiling",
            Budget("a", latency_s=2e-3, peak_memory_bytes=1 << 20,
                   duty_cycle=0.10, period_s=10e-3),
        ),
        (
            "median ceiling above the worst-case ceiling",
            Budget("b", latency_s=1e-3, peak_memory_bytes=1 << 20,
                   median_latency_s=2e-3),
        ),
        (
            "median above the duty-cycle-implied ceiling",
            Budget("c", latency_s=4e-4, peak_memory_bytes=1 << 20,
                   median_latency_s=3e-4, duty_cycle=0.02, period_s=10e-3),
        ),
    )
    for label, budget in cases:
        feasibility = budget.feasibility()
        ok = not feasibility.feasible
        results.append(ok)
        print(f"  {label}")
        for reason in feasibility.reasons:
            print(f"    - {reason}")
        print(f"    feasible = {feasibility.feasible}  {'PASS' if ok else 'FAIL'}")

    print("\n  and the same budget against a candidate with zero latency:")
    budget = cases[0][1]
    report = build_report(
        budget,
        candidate="instantaneous, zero-memory candidate",
        environment="none; no measurement was taken",
        worst_case_latency_s=1e-12,
        worst_case_uncertainty_s=0.0,
        peak_memory_bytes=1.0,
        peak_memory_uncertainty_bytes=0.0,
    )
    results.append(_expect("overall for an impossible candidate", report.overall,
                           Verdict.FAIL))
    print(
        "  A contradictory declaration fails whatever is measured against it, and\n"
        "  the contradiction is detectable before any model is loaded. A\n"
        "  self-consistent budget is a different question from an achievable one:\n"
        "  a 1 ns ceiling is self-consistent and unachievable, and the two are\n"
        "  reported separately."
    )
    feasible = Budget("sane", latency_s=1e-3, peak_memory_bytes=1 << 20,
                      median_latency_s=5e-4, duty_cycle=0.5, period_s=10e-3)
    ok = feasible.feasibility().feasible
    results.append(ok)
    print(f"  a self-consistent budget is reported feasible  {'PASS' if ok else 'FAIL'}")
    return results


def case_marginal_and_tail() -> list[bool]:
    print("\n(5) and (6) a marginal pass, and a median pass with a tail failure")
    print("-" * 78)
    rng = np.random.default_rng(3)
    model = random_mlp(rng, "measured", n_in=64, widths=(128,), n_out=8)
    backend = OnnxRuntimeBackend(model.model_bytes, model.input_feed(0))
    backend.prepare()
    try:
        profile = benchmark(
            backend.infer, label=model.graph.name, repeats=REPEATS, warmup=WARMUP,
            timer_bias_samples=500,
        )
    finally:
        backend.close()
    tail = profile.uncertainty("p99", n_resamples=400, seed=0)
    median = profile.uncertainty("p50", n_resamples=400, seed=0)
    print(f"  measurement method : {profile.method_line()}")
    print(
        f"  measured p50 = {median.value * 1e6:.3f} us "
        f"+/- {median.combined_s * 1e6:.3f} us (u_c)"
    )
    print(
        f"  measured p99 = {tail.value * 1e6:.3f} us "
        f"+/- {tail.combined_s * 1e6:.3f} us (u_c)"
    )
    results: list[bool] = []

    # (5) a ceiling one standard uncertainty above the measured p99.
    marginal_budget = Budget(
        "ceiling 1 u above p99",
        latency_s=tail.value + tail.combined_s,
        peak_memory_bytes=8 << 20,
    )
    report = build_report(
        marginal_budget,
        candidate=model.graph.name,
        environment=profile.environment.one_line(),
        worst_case_latency_s=tail.value,
        worst_case_uncertainty_s=tail.combined_s,
        peak_memory_bytes=1.0,
        peak_memory_uncertainty_bytes=0.0,
        latency_method=profile.method_line(),
        memory_method="analytic liveness analysis",
    )
    results.append(_expect("ceiling 1 u above p99", report.overall, Verdict.MARGINAL_PASS))

    # A ceiling far above: an unambiguous pass.
    wide_budget = Budget(
        "ceiling 100x p99", latency_s=tail.value * 100.0, peak_memory_bytes=8 << 20
    )
    report = build_report(
        wide_budget,
        candidate=model.graph.name,
        environment=profile.environment.one_line(),
        worst_case_latency_s=tail.value,
        worst_case_uncertainty_s=tail.combined_s,
        peak_memory_bytes=1.0,
        peak_memory_uncertainty_bytes=0.0,
        latency_method=profile.method_line(),
        memory_method="analytic liveness analysis",
    )
    results.append(_expect("ceiling 100x p99", report.overall, Verdict.PASS))

    # (6) a ceiling between the median and the tail.
    tail_ceiling = (median.value + tail.value) / 2.0
    split_budget = Budget(
        "ceiling between p50 and p99",
        latency_s=tail_ceiling,
        # Kept below the worst-case ceiling so the budget stays self-consistent
        # and the failure below is attributable to the tail alone.
        median_latency_s=min(median.value * 4.0, tail_ceiling * 0.9),
        peak_memory_bytes=8 << 20,
    )
    assert split_budget.feasibility().feasible, "this case needs a feasible budget"
    report = build_report(
        split_budget,
        candidate=model.graph.name,
        environment=profile.environment.one_line(),
        worst_case_latency_s=tail.value,
        worst_case_uncertainty_s=tail.combined_s,
        median_latency_s=median.value,
        median_uncertainty_s=median.combined_s,
        peak_memory_bytes=1.0,
        peak_memory_uncertainty_bytes=0.0,
        latency_method=profile.method_line(),
        memory_method="analytic liveness analysis",
    )
    verdicts = {r.quantity: r.verdict for r in report.rows}
    results.append(
        _expect("median row", verdicts["median latency (p50)"], Verdict.PASS)
    )
    tail_verdict = verdicts["worst-case latency (p99)"]
    tail_failed = tail_verdict in (Verdict.FAIL, Verdict.MARGINAL_FAIL)
    results.append(tail_failed)
    print(
        f"  {'tail row (p99)':<52} got {tail_verdict.value:<14} "
        f"want FAIL or MARGINAL FAIL  {'PASS' if tail_failed else 'FAIL'}"
    )
    overall_failed = report.overall in (Verdict.FAIL, Verdict.MARGINAL_FAIL)
    results.append(overall_failed)
    print(
        f"  {'overall':<52} got {report.overall.value:<14} "
        f"want FAIL or MARGINAL FAIL  {'PASS' if overall_failed else 'FAIL'}"
    )
    print(
        "\n  This is the case the whole separation exists for: the candidate is\n"
        "  comfortably inside the median ceiling and outside the worst-case one, and\n"
        "  the overall verdict follows the worst case. A tool that reported only the\n"
        "  median would have passed this candidate."
    )
    print("\n  full report for case (6):")
    for line in report.summary_lines():
        print(f"    {line}")
    return results


def main() -> int:
    print("=" * 78)
    print("P033 edgeinfer -- Validation 8: the budget decision path")
    print("=" * 78)
    print(f"environment: {describe_environment(note='')}")
    print(
        "WORKSTATION NUMBERS where anything is measured at all. Cases (1) to (4) are\n"
        "analytic and host-independent; cases (5) and (6) use a measured profile from\n"
        "this shared single-CPU-core cloud container. Nothing here is a measurement\n"
        "of any edge target, and every throttle factor is declared, not measured."
    )
    results = case_memory_overrun()
    results += case_unsupported_operator()
    results += case_thermal_throttle()
    results += case_infeasible_budget()
    results += case_marginal_and_tail()
    print("\n" + "=" * 78)
    passed = sum(results)
    print(f"RESULT: {passed}/{len(results)} checks passed")
    print("=" * 78)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
