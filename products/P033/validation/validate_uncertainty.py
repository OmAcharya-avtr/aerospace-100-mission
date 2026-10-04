#!/usr/bin/env python3
"""Validation 4 --- the uncertainty budget over timing measurements.

Four things are checked:

(a) **The clock quantisation term** against its closed form, and the clock's
    actual resolution and the cost of reading it, both measured here.

(b) **The Type A term's coverage.** The standard uncertainty of the mean is
    ``s/sqrt(n)`` (JCGM 100:2008, the GUM, section 4.2.3). Whether that is
    *right* for real timing data is checked empirically: the same callable is
    timed in many independent batches, and the spread of the batch means is
    compared against the uncertainty each batch predicted for itself. On a
    shared host with a heavy-tailed latency distribution the predicted
    uncertainty is expected to understate the realised spread, and the measured
    ratio is reported whichever way it falls.

(c) **The bootstrap quantile uncertainty's coverage** (Efron & Tibshirani
    1993, ch. 6), by the same method applied to p99.

(d) **Where the budget is silent.** The combined standard uncertainty is the
    uncertainty of a *statistic*, not the width of the latency distribution.
    The two are printed side by side so they cannot be confused.

Runtime: about 40 s on one CPU core.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from edgeinfer.backends import PipelineStage, SimulatedBackend
from edgeinfer.environment import clock_resolution_s, describe_environment
from edgeinfer.harness import benchmark
from edgeinfer.uncertainty import (
    clock_quantisation_uncertainty_s,
    combine_standard_uncertainties,
    mean_uncertainty,
    timer_overhead_s,
)

N_BATCHES = 24
BATCH_REPEATS = 150


def clock_checks() -> list[bool]:
    print("\n(a) the clock and its quantisation term")
    print("-" * 78)
    results: list[bool] = []
    resolution = clock_resolution_s("perf_counter")
    info = time.get_clock_info("perf_counter")
    print(f"  perf_counter resolution      : {resolution:.3e} s (time.get_clock_info)")
    print(f"  perf_counter monotonic       : {info.monotonic}")
    print(f"  perf_counter implementation  : {info.implementation}")

    computed = clock_quantisation_uncertainty_s(resolution, reads=2)
    closed_form = resolution * np.sqrt(2.0 / 12.0)
    ok = abs(computed - closed_form) < 1e-24
    results.append(ok)
    print(
        f"  u_B for a start/stop pair    : {computed:.6e} s  "
        f"closed form d*sqrt(2/12) = {closed_form:.6e} s  {'PASS' if ok else 'FAIL'}"
    )
    print(
        "  source: GUM section 4.3.7, rectangular distribution of full width d has\n"
        "  standard uncertainty d/sqrt(12); two independent reads add in quadrature."
    )

    bias = timer_overhead_s(5000)
    print(
        f"  measured timer-pair cost     : {bias * 1e9:.1f} ns "
        "(median over 5000 back-to-back pairs)"
    )
    ok = 0.0 <= bias < 1e-5
    results.append(ok)
    print(
        f"  the timer-pair cost is {bias / max(resolution, 1e-18):.3g}x the clock "
        f"resolution, so on this host the dominant\n"
        f"  instrumental effect on a short interval is the call cost, not the\n"
        f"  quantisation.  {'PASS' if ok else 'FAIL'}"
    )
    return results


def _batches(label: str, fn, repeats: int, batches: int):
    """Time ``fn`` in independent batches, returning per-batch profiles."""
    profiles = []
    for _ in range(batches):
        profiles.append(
            benchmark(
                fn, label=label, repeats=repeats, warmup=5, timer_bias_samples=200
            )
        )
    return profiles


def coverage_checks() -> list[bool]:
    print("\n(b) and (c) does the predicted uncertainty match the realised spread?")
    print("-" * 78)
    results: list[bool] = []

    stages = (PipelineStage("stage", 120e-6, 24e-6, "lognormal"),)
    backend = SimulatedBackend(stages, seed=4242, consume_time=True)
    backend.prepare()
    profiles = _batches("uncertainty coverage", backend.infer, BATCH_REPEATS, N_BATCHES)

    means = np.asarray([p.mean_s for p in profiles])
    predicted_mean_u = np.asarray([mean_uncertainty(p.samples_s) for p in profiles])
    realised_mean_spread = float(np.std(means, ddof=1))
    median_predicted = float(np.median(predicted_mean_u))
    ratio_mean = realised_mean_spread / median_predicted
    print(
        f"  mean of a batch   : {N_BATCHES} batches of {BATCH_REPEATS} repeats"
    )
    print(
        f"    realised spread of batch means         : "
        f"{realised_mean_spread * 1e6:.4f} us"
    )
    print(
        f"    median predicted u_A = s/sqrt(n)       : {median_predicted * 1e6:.4f} us"
    )
    print(
        f"    realised / predicted                   : {ratio_mean:.3f}"
    )
    # A ratio near 1 means the GUM Type A term describes this data. A ratio
    # above 1 means the batches are not independent draws from one
    # distribution, which on a shared host they are not: host load drifts
    # between batches. Either outcome is reported; the check is only that the
    # figure is finite and the direction is stated.
    mean_ok = np.isfinite(ratio_mean) and ratio_mean > 0
    results.append(mean_ok)
    if 0.5 <= ratio_mean <= 2.0:
        verdict = "the GUM Type A term describes this data on this run"
    elif ratio_mean > 2.0:
        verdict = (
            "the realised spread EXCEEDS the within-batch prediction: host load "
            "drifts between batches, so the batches are not draws from one "
            "stationary distribution and u_A understates how much a repeated "
            "measurement would move"
        )
    else:
        verdict = (
            "the realised spread is SMALLER than the within-batch prediction: a few "
            "preempted repeats inflated the within-batch standard deviation while "
            "the batch means stayed close, so u_A overstates the spread of the mean "
            "on this run"
        )
    print(f"    interpretation: {verdict}")
    print(
        "    Neither direction is a general property. The ratio moves with host load,"
        "\n    and that is the point: a figure quoted from a shared machine carries "
        "the\n    machine's load with it."
    )

    tails = np.asarray([p.p99_s for p in profiles])
    predicted_tail_u = np.asarray(
        [p.uncertainty("p99", n_resamples=400, seed=0).type_a_s for p in profiles]
    )
    realised_tail_spread = float(np.std(tails, ddof=1))
    median_predicted_tail = float(np.median(predicted_tail_u))
    ratio_tail = realised_tail_spread / median_predicted_tail
    print("\n  p99 of a batch")
    print(
        f"    realised spread of batch p99s          : "
        f"{realised_tail_spread * 1e6:.4f} us"
    )
    print(
        f"    median bootstrap u_A (400 resamples)   : "
        f"{median_predicted_tail * 1e6:.4f} us"
    )
    print(f"    realised / predicted                   : {ratio_tail:.3f}")
    tail_ok = np.isfinite(ratio_tail) and ratio_tail > 0
    results.append(tail_ok)
    print(
        "    a bootstrap resamples the batch it was given, so it cannot see\n"
        "    variation BETWEEN batches; and with n=150 repeats the p99 is "
        "interpolated\n    between the top two order statistics, so the bootstrap "
        "spread is itself\n    driven by a handful of points. Both effects are "
        "present and they push in\n    opposite directions."
    )
    if ratio_tail > 1.5:
        print(
            "    on this run the realised spread EXCEEDED the bootstrap figure by "
            f"{ratio_tail:.2f}x:\n    host load moved the tail between batches, so a "
            "quoted p99 uncertainty is a\n    lower bound on how much that p99 "
            "would move if the measurement were repeated."
        )
    elif ratio_tail < 0.67:
        print(
            "    on this run the realised spread was SMALLER than the bootstrap "
            f"figure\n    ({ratio_tail:.2f}x): with n=150 the bootstrap's few "
            "extreme order statistics make\n    it pessimistic, and host load "
            "happened to be steady across the batches. On a\n    busier host the "
            "ratio goes the other way, so neither direction should be\n    quoted as "
            "a property of the method."
        )
    else:
        print(
            f"    on this run the two agreed to within {abs(ratio_tail - 1) * 100:.0f} "
            "%, which is not a general\n    property: the ratio moves with host load "
            "and with the repeat count."
        )
    return results


def scope_disclosure() -> None:
    print("\n(d) what the combined uncertainty is not")
    print("-" * 78)
    stages = (
        PipelineStage("preprocess", 60e-6, 12e-6, "lognormal"),
        PipelineStage("inference", 180e-6, 40e-6, "lognormal"),
    )
    backend = SimulatedBackend(stages, seed=7, consume_time=True)
    backend.prepare()
    profile = benchmark(
        backend.infer, label="scope", repeats=1500, warmup=20, timer_bias_samples=2000
    )
    mean_budget = profile.uncertainty("mean")
    print(f"  measurement method : {profile.method_line()}")
    print(f"  mean               : {profile.mean_s * 1e6:>9.3f} us")
    print(f"  u_c(mean)          : {mean_budget.combined_s * 1e6:>9.3f} us")
    print(f"  sd of the samples  : {profile.std_s * 1e6:>9.3f} us")
    print(f"  p50                : {profile.p50_s * 1e6:>9.3f} us")
    print(f"  p99                : {profile.p99_s * 1e6:>9.3f} us")
    print(f"  max                : {profile.max_s * 1e6:>9.3f} us")
    spread_ratio = profile.std_s / max(mean_budget.combined_s, 1e-18)
    print(
        f"\n  the sample spread is {spread_ratio:.0f}x the uncertainty of the mean."
    )
    print(
        "  A budget written against u_c would be written against the wrong number by\n"
        "  that factor. u_c says how well the *statistic* is known; the distribution's\n"
        "  width is what a deadline has to survive, and that is what p99 and max are\n"
        "  for."
    )
    rss = combine_standard_uncertainties(mean_budget.type_a_s, mean_budget.clock_s)
    print(
        f"  components of u_c(mean): u_A = {mean_budget.type_a_s * 1e6:.4f} us, "
        f"u_B(clock) = {mean_budget.clock_s * 1e9:.4f} ns, "
        f"RSS = {rss * 1e6:.4f} us"
    )


def main() -> int:
    print("=" * 78)
    print("P033 edgeinfer -- Validation 4: uncertainty budget over timing")
    print("=" * 78)
    print(f"environment: {describe_environment(note='')}")
    print(
        "WORKSTATION NUMBERS. Shared single-CPU-core cloud container with other "
        "build\njobs running concurrently. The coverage ratios in (b) and (c) are "
        "properties of\nthis host and will differ on another."
    )
    results = clock_checks()
    results += coverage_checks()
    scope_disclosure()
    print("\n" + "=" * 78)
    passed = sum(results)
    print(f"RESULT: {passed}/{len(results)} checks passed")
    print("=" * 78)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
