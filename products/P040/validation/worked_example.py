"""The worked example printed in README.md, kept here so that its output is
reproducible and committed (``worked_example_output.txt``) rather than typed.

Runtime: about 6 s on one core.
"""

import numpy as np

from bitflipsim import (
    BitUpset,
    apply_upsets,
    clamp_logit_bound,
    clamp_parameters,
    evaluate_protection,
    exponent_bit_baseline_scores,
    float_layout,
    make_problem,
    parameter_campaign,
    predict_flip,
    reference_parameters,
    sweep_bit_criticality,
    upset_rate,
    word_majority_vote,
)

problem = make_problem()
params = reference_parameters(problem)
layout = float_layout("float32")

# 1. What does one upset in the exponent MSB of a weight actually do?
weight = params.values[5]
prediction = predict_flip(weight, layout.exponent_msb, "float32")
print(f"w[5] = {weight!r}  bit {layout.exponent_msb} ({layout.role(layout.exponent_msb)})")
print(f"  -> {prediction.predicted_value!r}  regime {prediction.regime}")

# 2. How often should that happen? Rate = flux * cross-section * bits.
rate = upset_rate(flux_per_cm2_s=1.0e3, cross_section_cm2_per_bit=1.0e-14,
                  bit_count=params.layout.size * 32)
print(f"  lambda = {rate.rate_per_s:.4e} upsets/s = {rate.rate_fit:.4e} FIT "
      f"over {rate.bit_count} bits")

# 3. What does it cost in output terms, with and without a clamp?
limit = float(np.abs(params.values).max())
for label, clamp in (("unmitigated", None), ("clamped", limit)):
    result = parameter_campaign(params, problem.evaluation.x, problem.evaluation.y,
                                expected_upsets=8.0, trials=400,
                                rng=np.random.default_rng(0), clamp_limit=clamp)
    print(f"  mu=8 {label:<12} D = {result.mean_degradation:.4f} "
          f"+- {result.degradation_standard_error:.4f}, "
          f"accuracy {result.mean_accuracy:.4f}")

# 4. And the clamp comes with a bound, not just an improvement.
bound = clamp_logit_bound(limit, params.layout.n_in,
                          float(np.abs(problem.evaluation.x).max()))
print(f"  clamp C = {limit:.4f} bounds any single-upset logit deviation by {bound:.3f}")

# 5. Which bits are worth protecting, and what does the free heuristic get?
sweep = sweep_bit_criticality(params, problem.evaluation.x, problem.evaluation.y)
heuristic = exponent_bit_baseline_scores(params.layout.size, layout)
protection = evaluate_protection(heuristic, sweep.degradation,
                                 budget_bytes=params.layout.total_bytes * 0.05,
                                 itemsize_bytes=4, cost_model="bit",
                                 method="exponent heuristic")
print(f"  protecting {protection.protected_units} bits ({protection.cost_bytes:.1f} bytes) "
      f"avoids {protection.avoided_fraction:.1%} of all degradation, "
      f"{protection.avoided_per_byte:.4f} per byte")

# 6. Triplication recovers a single upset exactly.
trio = [params.values.copy() for _ in range(3)]
trio[1] = apply_upsets(trio[1], [BitUpset(5, layout.exponent_msb)])
voted, uncorrectable = word_majority_vote(*trio)
print(f"  TMR after one upset: recovered = {np.array_equal(voted, params.values)}, "
      f"flagged = {int(uncorrectable.sum())}")
_ = clamp_parameters(params, limit)
