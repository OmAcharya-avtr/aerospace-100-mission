"""Degradation avoided per protected byte: baselines versus the learned predictor.

Writes ``../screenshots/criticality_methods.png``.

Both baselines are evaluated first and the learned predictor is given both of
them as features, so it can only win by adding information they do not carry.
Two cost models are plotted side by side because the answer changes between
them: at bit granularity the exponent-bit heuristic is nearly as good as perfect
knowledge for zero effort; at word granularity it is exactly uninformative and
loses to a plain magnitude ranking.

Runtime: about 15 s on one core.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bitflipsim.bitlayout import float_layout  # noqa: E402
from bitflipsim.criticality import (  # noqa: E402
    evaluate_protection,
    exponent_bit_baseline_scores,
    magnitude_baseline_scores,
    oracle_scores,
    random_scores,
    sweep_bit_criticality,
)
from bitflipsim.datasets import make_problem, reference_parameters  # noqa: E402
from bitflipsim.predictor import (  # noqa: E402
    CriticalityPredictor,
    build_features,
    split_parameters,
    uncertainty_calibration,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "screenshots", "criticality_methods.png")
SEED = 7

layout = float_layout("float32")
problem = make_problem()
params = reference_parameters(problem)

sweep_calibration = sweep_bit_criticality(params, problem.calibration.x, problem.calibration.y)
sweep_evaluation = sweep_bit_criticality(params, problem.evaluation.x, problem.evaluation.y)
features = build_features(params, problem.calibration.x, layout)
split = split_parameters(params.layout.size, layout.total_bits, 0.6, seed=SEED)
train_mask, test_mask = split.site_mask("train"), split.site_mask("test")
predictor = CriticalityPredictor(n_estimators=160, max_depth=10, seed=SEED).fit(
    features[train_mask], sweep_calibration.flat_degradation()[train_mask]
)
prediction = predictor.predict(features[test_mask])

n_test = split.test_parameters.size
degradation = sweep_evaluation.degradation[split.test_parameters]
region_bytes = n_test * params.layout.itemsize_bytes
methods = {
    "magnitude baseline": magnitude_baseline_scores(params)[split.test_parameters],
    "exponent heuristic": exponent_bit_baseline_scores(params.layout.size, layout)[
        split.test_parameters
    ],
    "learned predictor": prediction.expected.reshape(n_test, layout.total_bits),
    "oracle (unachievable)": oracle_scores(degradation),
    "random": random_scores(degradation.shape, np.random.default_rng(SEED)),
}
STYLES = {
    "magnitude baseline": ("#7f7f7f", "o", "-"),
    "exponent heuristic": ("#2166ac", "s", "-"),
    "learned predictor": ("#b2182b", "^", "-"),
    "oracle (unachievable)": ("#1b7837", "", "--"),
    "random": ("#bdbdbd", "", ":"),
}

FRACTIONS = np.array([0.02, 0.05, 0.10, 0.15, 0.25, 0.40, 0.50, 0.75, 1.00])

print(f"test region               {n_test} of {params.layout.size} parameters "
      f"= {region_bytes} bytes (grouped split, seed {SEED})")
print(f"total degradation         {degradation.sum():.6f} over {degradation.size} bit sites")
print(f"training sites            {int(train_mask.sum())} "
      f"({split.train_parameters.size} parameters), measured on the calibration split")
print("scoring                   evaluation split, disjoint from calibration")
print()

curves: dict[str, dict[str, np.ndarray]] = {}
for cost_model in ("bit", "word"):
    print(f"cost model: {cost_model}  (degradation avoided per protected byte)")
    header = f"{'budget':>8} {'bytes':>8} {'units':>7} "
    header += " ".join(f"{name:>22}" for name in methods)
    print(header)
    per_byte = {name: [] for name in methods}
    for fraction in FRACTIONS:
        budget = region_bytes * fraction
        row = []
        units = 0
        for name, scores in methods.items():
            result = evaluate_protection(
                scores, degradation, budget, params.layout.itemsize_bytes, cost_model, name
            )
            per_byte[name].append(result.avoided_per_byte)
            units = result.protected_units
            row.append(f"{result.avoided_per_byte:>22.6f}")
        print(f"{fraction:>8.2f} {budget:>8.1f} {units:>7d} " + " ".join(row))
    curves[cost_model] = {name: np.array(values) for name, values in per_byte.items()}
    print()

report = uncertainty_calibration(prediction, sweep_evaluation.flat_degradation()[test_mask])
print("Learned predictor uncertainty (ensemble standard deviation across 160 trees)")
print(f"coverage at k = 2         {report['coverage']:.6f} "
      f"(Gaussian reference {report['gaussian_reference']:.6f})")
print(f"mean abs error (log1p)    {report['mean_abs_error_log1p']:.6e}")
print(f"median sigma (log1p)      {report['median_sigma_log1p']:.6e}")
print()
print("Top five features by importance:")
for name, importance in predictor.importance_table()[:5]:
    print(f"  {name:<34} {importance:.6f}")
print()

for cost_model in ("bit", "word"):
    data = curves[cost_model]
    print(f"Winner by budget, cost model {cost_model} "
          f"(oracle and random excluded from the comparison):")
    for index, fraction in enumerate(FRACTIONS):
        ranked = sorted(
            ((data[name][index], name) for name in
             ("magnitude baseline", "exponent heuristic", "learned predictor")),
            reverse=True,
        )
        best, runner = ranked[0], ranked[1]
        margin = best[0] - runner[0]
        relative = margin / runner[0] if runner[0] > 0 else float("inf")
        print(f"  budget {fraction:>5.2f}: {best[1]:<20} {best[0]:>10.6f} "
              f"beats {runner[1]:<20} {runner[0]:>10.6f} by {margin:>+10.6f} "
              f"({relative:+.1%})")
    print()

figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
for axis, cost_model, subtitle in (
    (axes[0], "bit", "bit-granular cost (idealised: protected bits / 8 bytes)"),
    (axes[1], "word", "word-granular cost (what ECC or word TMR charges)"),
):
    for name in methods:
        colour, marker, style = STYLES[name]
        axis.plot(
            FRACTIONS * 100.0,
            curves[cost_model][name],
            color=colour,
            marker=marker or None,
            ls=style,
            lw=1.8,
            ms=5,
            label=name,
        )
    axis.set_xscale("log")
    axis.set_xlabel("protection budget, per cent of the test region's bytes")
    axis.set_ylabel("degradation avoided per protected byte")
    axis.set_title(subtitle, fontsize=10)
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)

flat = curves["word"]["exponent heuristic"][0]
axes[1].annotate(
    f"exponent heuristic is flat at {flat:.4f}:\n"
    "every word gets the same score, so this\n"
    "is exactly the random-selection expectation",
    xy=(FRACTIONS[4] * 100.0, flat),
    xytext=(8.0, flat * 0.45),
    arrowprops={"arrowstyle": "->", "lw": 1.0},
    fontsize=8,
)

figure.suptitle(
    "Degradation avoided per protected byte; baselines first, "
    "learned predictor given both as features",
    fontsize=11,
)
figure.tight_layout()
figure.savefig(OUT, dpi=130)
print(f"wrote {os.path.normpath(OUT)}")
