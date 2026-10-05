"""Mitigation evaluation: what each scheme buys and what it costs.

Writes ``../screenshots/mitigation_tradeoff.png``.

Three mitigations, each with its guarantee and its measured cost:

* parameter range clamping - no extra memory, a derived bound on the logit
  deviation, one clamp pass per inference. Measured here.
* selective triplication   - 3x memory on the protected region, exact recovery
  of every single upset, silent failure on same-bit double upsets.
* periodic reload          - one golden copy, expected live upsets
  ``lambda T_s / 2``, and a reload duty cycle of lost inference time.

Latency figures are ``time.perf_counter`` medians on the host that ran this
script, on a single contended core. They are numpy figures for a 147-parameter
model, not flight figures for any target, and they do not generalise.

Runtime: about 20 s on one core.
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bitflipsim.bitlayout import float_layout  # noqa: E402
from bitflipsim.campaign import parameter_campaign  # noqa: E402
from bitflipsim.criticality import (  # noqa: E402
    evaluate_protection,
    exponent_bit_baseline_scores,
    magnitude_baseline_scores,
    sweep_bit_criticality,
)
from bitflipsim.datasets import make_problem, reference_parameters  # noqa: E402
from bitflipsim.mitigation import (  # noqa: E402
    bit_majority_vote,
    clamp_logit_bound,
    clamp_parameters,
    clamping_cost,
    expected_live_upsets,
    measure_clamp_latency,
    reload_cost,
    triplication_cost,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "screenshots", "mitigation_tradeoff.png")

layout = float_layout("float32")
problem = make_problem()
params = reference_parameters(problem)
x = problem.evaluation.x
limit = float(np.abs(params.values).max())
max_input = float(np.abs(x).max())
bound = clamp_logit_bound(limit, params.layout.n_in, max_input)


def median_time(callable_, repeats: int = 40) -> float:
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        start = time.perf_counter()
        callable_()
        samples[index] = time.perf_counter() - start
    return float(np.median(samples))


inference_s = median_time(lambda: params.probabilities(x))
clamp_latency = measure_clamp_latency(params.layout.size, repeats=200)
triple = [params.values.copy() for _ in range(3)]
vote_s = median_time(lambda: bit_majority_vote(*triple), repeats=200)
clamped_inference_s = median_time(lambda: clamp_parameters(params, limit).probabilities(x))

print("Host measurement (single contended core, numpy; not a flight figure)")
print(f"inference, {x.shape[0]} samples   {inference_s * 1e6:.3f} us median of 40")
print(f"clamp pass, {params.layout.size} parameters  "
      f"{clamp_latency['median_s'] * 1e6:.3f} us median of 200 "
      f"({clamp_latency['per_parameter_ns']:.3f} ns per parameter)")
print(f"bitwise 3-way vote        {vote_s * 1e6:.3f} us median of 200")
print(f"clamp + inference         {clamped_inference_s * 1e6:.3f} us median of 40")
print()

print("Parameter range clamping: the derived bound")
print(f"clamp limit C = max |w|   {limit:.9f}")
print(f"Xmax = max |x|            {max_input:.9f}")
print(f"B = 2C max(C(n_in Xmax + 1), 1)  {bound:.6f} on max_k |d logits_k|")
print("(verified against all 4704 exhaustive single-upset cases in")
print(" validation/validate_clamp_bound.py, which also reports the tightness)")
print()

costs = [
    clamping_cost(params.layout.total_bytes, clamp_latency, inference_s),
    triplication_cost(params.layout.total_bytes, inference_s, vote_s),
    reload_cost(params.layout.total_bytes, 60.0, 0.01, 1.0e-3)[0],
]
print("Mitigation costs on the whole 588-byte parameter block")
print(f"{'scheme':<26} {'extra bytes':>12} {'mem factor':>11} {'latency fraction':>17}")
for cost in costs:
    print(f"{cost.scheme:<26} {cost.extra_memory_bytes:>12d} {cost.memory_factor:>11.3f} "
          f"{cost.latency_overhead_fraction:>17.6f}")
print()
for cost in costs:
    print(f"{cost.scheme}: {cost.notes}")
print()

print("Clamping effectiveness, measured by campaign (600 trials per point)")
print(f"{'mu':>6} {'unmitigated TV':>15} {'se':>9} {'clamped TV':>12} {'se':>9} "
      f"{'reduction':>11}")
clamp_rows = []
for mu in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
    plain = parameter_campaign(
        params, x, problem.evaluation.y, mu, 600, np.random.default_rng(int(mu * 13))
    )
    clamped = parameter_campaign(
        params, x, problem.evaluation.y, mu, 600, np.random.default_rng(int(mu * 13)),
        clamp_limit=limit,
    )
    reduction = (
        1.0 - clamped.mean_degradation / plain.mean_degradation
        if plain.mean_degradation > 0
        else float("nan")
    )
    clamp_rows.append((mu, plain, clamped, reduction))
    print(f"{mu:>6.1f} {plain.mean_degradation:>15.6f} "
          f"{plain.degradation_standard_error:>9.6f} {clamped.mean_degradation:>12.6f} "
          f"{clamped.degradation_standard_error:>9.6f} {reduction:>11.4f}")
print()

print("Periodic reload: expected live upsets and the reload duty cycle")
print("(lambda from the illustrative flux and cross-section applied to a 50 MB")
print(" model, so that the scrub interval has something to trade against)")
large_model_bits = 50 * 1024 * 1024 * 8
lam = 1.0e3 * 1.0e-14 * large_model_bits
print(f"lambda for 50 MB          {lam:.6e} upsets s^-1")
print(f"{'T_s (s)':>10} {'E[live upsets]':>16} {'duty cycle':>12} "
      f"{'reload time (s)':>16}")
scrub_rows = []
for interval in (1.0, 10.0, 60.0, 600.0, 3600.0, 86400.0):
    reload_time = 0.05
    cost, live = reload_cost(large_model_bits // 8, interval, reload_time, lam)
    scrub_rows.append((interval, live, cost.latency_overhead_fraction))
    print(f"{interval:>10.1f} {live:>16.6e} {cost.latency_overhead_fraction:>12.6e} "
          f"{reload_time:>16.3f}")
print()
print(f"check: expected_live_upsets(lam, 60) = {expected_live_upsets(lam, 60.0):.6e}")
print()

print("Protection targeting: avoided degradation per byte at a 5 % budget")
sweep = sweep_bit_criticality(params, x, problem.evaluation.y)
budget = params.layout.total_bytes * 0.05
for cost_model in ("bit", "word"):
    for name, scores in (
        ("magnitude baseline", magnitude_baseline_scores(params)),
        ("exponent heuristic", exponent_bit_baseline_scores(params.layout.size, layout)),
    ):
        result = evaluate_protection(
            scores, sweep.degradation, budget, params.layout.itemsize_bytes, cost_model, name
        )
        print(f"  {cost_model:<5} {name:<20} avoided {result.avoided:>10.6f} "
              f"per byte {result.avoided_per_byte:>10.6f} "
              f"fraction of total {result.avoided_fraction:>9.6f}")

figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

left = axes[0]
mus = np.array([row[0] for row in clamp_rows])
plain_tv = np.array([row[1].mean_degradation for row in clamp_rows])
plain_se = np.array([row[1].degradation_standard_error for row in clamp_rows])
clamp_tv = np.array([row[2].mean_degradation for row in clamp_rows])
clamp_se = np.array([row[2].degradation_standard_error for row in clamp_rows])
left.errorbar(mus, plain_tv, yerr=plain_se, marker="o", capsize=3, lw=1.7,
              color="#b2182b", label="no mitigation")
left.errorbar(mus, clamp_tv, yerr=clamp_se, marker="s", capsize=3, lw=1.7,
              color="#2166ac", label=f"clamped to |w| <= {limit:.3f}")
left.set_xscale("log", base=2)
left.set_xlabel("expected upsets (Poisson mean)")
left.set_ylabel("mean total-variation degradation")
left.set_title("Clamping: zero extra memory", fontsize=10)
left.grid(alpha=0.3)
left.legend(fontsize=8)

middle = axes[1]
scheme_labels = {
    "parameter_range_clamping": "range clamping",
    "selective_triplication": "selective triplication",
    "periodic_reload": "periodic reload",
}
scheme_colours = {
    "parameter_range_clamping": "#2166ac",
    "selective_triplication": "#b2182b",
    "periodic_reload": "#1b7837",
}
for cost in costs:
    middle.scatter(
        cost.memory_factor,
        cost.latency_overhead_fraction,
        s=140,
        color=scheme_colours[cost.scheme],
        zorder=3,
        label=scheme_labels[cost.scheme],
    )
    middle.annotate(
        f"{scheme_labels[cost.scheme]}\n"
        f"{cost.memory_factor:.1f}x memory, "
        f"{100.0 * cost.latency_overhead_fraction:.3f} % latency",
        xy=(cost.memory_factor, cost.latency_overhead_fraction),
        xytext=(0, -26),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )
middle.set_xscale("linear")
middle.set_yscale("log")
middle.set_xlim(0.4, 3.9)
latencies = [cost.latency_overhead_fraction for cost in costs]
middle.set_ylim(min(latencies) * 0.25, max(latencies) * 4.0)
middle.set_xlabel("memory factor on the protected region (x golden size)")
middle.set_ylabel("latency overhead (fraction of inference time)")
middle.set_title("Cost of each mitigation, measured on this host", fontsize=10)
middle.grid(alpha=0.3, which="both")

right = axes[2]
intervals = np.array([row[0] for row in scrub_rows])
live = np.array([row[1] for row in scrub_rows])
duty = np.array([row[2] for row in scrub_rows])
live_line, = right.loglog(intervals, live, marker="o", color="#1b7837", lw=1.7,
                          label="E[live upsets] = $\\lambda T_s / 2$")
right_twin = right.twinx()
duty_line, = right_twin.loglog(intervals, duty, marker="s", color="#b2182b", lw=1.7, ls="--",
                               label="reload duty cycle, 0.05 s reload")
right.set_xlabel("scrub interval $T_s$ (s)")
right.set_ylabel("expected live upsets, 50 MB model")
right_twin.set_ylabel("fraction of time spent reloading")
right.set_title("Periodic reload: the interval trade", fontsize=10)
right.grid(alpha=0.3, which="both")
# The legend goes on the twin axes, which is drawn last, so the dashed duty
# line cannot be painted over the top of it.
right_twin.legend([live_line, duty_line],
                  [live_line.get_label(), duty_line.get_label()],
                  fontsize=8, loc="center left", framealpha=0.95)

figure.suptitle(
    "Mitigation evaluation: effectiveness against measured memory and latency cost",
    fontsize=11,
)
figure.tight_layout()
figure.savefig(OUT, dpi=130)
print()
print(f"wrote screenshots/{os.path.basename(OUT)}")
