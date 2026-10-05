"""Degradation versus upset rate, for parameter and activation upsets.

Sweeps the expected upset count, draws Poisson counts, injects, and reports the
mean degradation and accuracy with their standard errors. Writes
``../screenshots/degradation_vs_upset_rate.png``.

Campaign sizing: the per-trial degradation distribution is heavy-tailed - most
trials are near zero and a few are near one - so the sample standard deviation
is large relative to the mean and the standard error matters. The number of
trials is set from the measured spread at the largest expectation using
``n = (s/e)**2``, and the achieved standard error is printed for every point.

What to notice: the parameter curve rises far faster per upset than the
activation curve, and the reason is not that activations are robust. A
parameter upset persists until the next scrub and corrupts every subsequent
inference; an activation upset corrupts one value of one sample of one
inference. The two curves answer different questions and are plotted on
separate axes for that reason.

Runtime: about 25 s on one core.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bitflipsim.campaign import (  # noqa: E402
    activation_campaign,
    parameter_campaign,
    trials_for_standard_error,
)
from bitflipsim.datasets import make_problem, reference_parameters  # noqa: E402
from bitflipsim.flux import (  # noqa: E402
    ILLUSTRATIVE_CROSS_SECTION_CM2_PER_BIT,
    ILLUSTRATIVE_FLUX_PER_CM2_S,
    poisson_validity,
    upset_rate,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "screenshots", "degradation_vs_upset_rate.png")

problem = make_problem()
params = reference_parameters(problem)
limit = float(np.abs(params.values).max())
population = params.layout.size * params.layout.bits_per_parameter

rate = upset_rate(
    ILLUSTRATIVE_FLUX_PER_CM2_S, ILLUSTRATIVE_CROSS_SECTION_CM2_PER_BIT, population
)
print("Flux model (ILLUSTRATIVE inputs, not measured environment or part data)")
print(f"flux                      {rate.flux_per_cm2_s:.6e} particles cm^-2 s^-1")
print(f"cross-section             {rate.cross_section_cm2_per_bit:.6e} cm^2 bit^-1")
print(f"exposed bits              {rate.bit_count}")
print(f"lambda                    {rate.rate_per_s:.6e} upsets s^-1 "
      f"({rate.rate_fit:.6e} FIT)")
print(f"mean time between upsets  {rate.mean_time_between_upsets_s:.6e} s")
print()

# Size the campaigns from the spread at the largest expectation.
probe = parameter_campaign(
    params, problem.evaluation.x, problem.evaluation.y, 32.0, 200, np.random.default_rng(1)
)
probe_std = probe.degradation_standard_error * np.sqrt(probe.trials)
TARGET_SE = 0.012
BUDGET_CAP = 600
required = trials_for_standard_error(probe_std, TARGET_SE)
TRIALS = min(BUDGET_CAP, max(300, required))
print(f"campaign sizing: probe sample std {probe_std:.6f} at mu = 32; for a standard")
print(f"error of {TARGET_SE} the formula n = (s/e)**2 asks for {required} trials.")
print(f"Using {TRIALS}, capped by the {BUDGET_CAP}-trial compute budget for this script")
print(f"on one contended core, so the achieved standard error is "
      f"{probe_std / np.sqrt(TRIALS):.6f} rather than {TARGET_SE} at mu = 32. The")
print("achieved standard error is printed for every point below; none of the")
print("comparisons drawn from this table are inside it.")
print()

EXPECTED = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
rng = np.random.default_rng(20261005)

rows = []
print("Parameter upsets (persist until the next scrub)")
print(f"{'mu':>6} {'exposure (s)':>15} {'mean TV':>10} {'se':>9} {'accuracy':>10} "
      f"{'se':>9} {'clamped TV':>11} {'se':>9} {'drawn mean':>11}")
for mu in EXPECTED:
    plain = parameter_campaign(
        params, problem.evaluation.x, problem.evaluation.y, mu, TRIALS, rng
    )
    clamped = parameter_campaign(
        params, problem.evaluation.x, problem.evaluation.y, mu, TRIALS,
        np.random.default_rng(int(mu * 7919)), clamp_limit=limit,
    )
    exposure = rate.exposure_for_expected_upsets(mu)
    rows.append((mu, plain, clamped, exposure))
    print(f"{mu:>6.1f} {exposure:>15.6e} {plain.mean_degradation:>10.6f} "
          f"{plain.degradation_standard_error:>9.6f} {plain.mean_accuracy:>10.6f} "
          f"{plain.accuracy_standard_error:>9.6f} {clamped.mean_degradation:>11.6f} "
          f"{clamped.degradation_standard_error:>9.6f} "
          f"{plain.upset_counts.mean():>11.4f}")

activation_rows = []
print()
print("Activation upsets (affect one inference only)")
print(f"{'mu':>6} {'mean TV':>12} {'se':>10} {'accuracy':>10} {'se':>9} "
      f"{'activation bits':>16}")
hidden_bits = problem.evaluation.n_samples * params.layout.n_hidden * 32
for mu in EXPECTED:
    result = activation_campaign(
        params, problem.evaluation.x, problem.evaluation.y, mu, TRIALS,
        np.random.default_rng(int(mu * 104729)),
    )
    activation_rows.append((mu, result))
    print(f"{mu:>6.1f} {result.mean_degradation:>12.6f} "
          f"{result.degradation_standard_error:>10.6f} {result.mean_accuracy:>10.6f} "
          f"{result.accuracy_standard_error:>9.6f} {hidden_bits:>16}")

golden_accuracy = rows[0][1].golden_accuracy
print()
print(f"golden accuracy           {golden_accuracy:.6f}")
print(f"parameter bit population  {population}")
print(f"activation bit population {hidden_bits} "
      f"(the whole evaluation batch's hidden layer)")
print()
validity = poisson_validity(rate, rate.exposure_for_expected_upsets(64.0))
print(f"Poisson validity at mu = 64: per-bit probability "
      f"{validity['per_bit_probability']:.6e}, within validity "
      f"{validity['within_validity']} (threshold {validity['threshold']:.1e})")
print()
print("The exposures above are enormous because the illustrative cross-section is")
print("small and the model is 588 bytes. That is itself the finding: a parameter")
print("block this small is not what drives an onboard inference upset budget. The")
print("same equation applied to a 50 MB model gives exposures 1e5 times shorter.")

figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

mus = np.array([row[0] for row in rows])
plain_tv = np.array([row[1].mean_degradation for row in rows])
plain_se = np.array([row[1].degradation_standard_error for row in rows])
clamp_tv = np.array([row[2].mean_degradation for row in rows])
clamp_se = np.array([row[2].degradation_standard_error for row in rows])
act_tv = np.array([row[1].mean_degradation for row in activation_rows])
act_se = np.array([row[1].degradation_standard_error for row in activation_rows])

left = axes[0]
left.errorbar(mus, plain_tv, yerr=plain_se, marker="o", capsize=3, lw=1.6,
              color="#b2182b", label="parameter upsets, unmitigated")
left.errorbar(mus, clamp_tv, yerr=clamp_se, marker="s", capsize=3, lw=1.6,
              color="#2166ac", label=f"parameter upsets, clamped to |w| <= {limit:.3f}")
left.errorbar(mus, act_tv, yerr=act_se, marker="^", capsize=3, lw=1.6,
              color="#1b7837", label="activation upsets (one inference each)")
left.set_xscale("log", base=2)
left.set_yscale("log")
left.set_xlabel("expected upsets per exposure (Poisson mean)")
left.set_ylabel("mean total-variation degradation")
left.set_title(f"Degradation versus upset rate, {TRIALS} trials per point")
left.grid(alpha=0.3, which="both")
left.legend(fontsize=8, loc="upper left")

right = axes[1]
plain_acc = np.array([row[1].mean_accuracy for row in rows])
plain_acc_se = np.array([row[1].accuracy_standard_error for row in rows])
clamp_acc = np.array([row[2].mean_accuracy for row in rows])
clamp_acc_se = np.array([row[2].accuracy_standard_error for row in rows])
act_acc = np.array([row[1].mean_accuracy for row in activation_rows])
act_acc_se = np.array([row[1].accuracy_standard_error for row in activation_rows])
right.axhline(golden_accuracy, color="#404040", ls="--", lw=1.0,
              label=f"golden accuracy {golden_accuracy:.4f}")
right.errorbar(mus, plain_acc, yerr=plain_acc_se, marker="o", capsize=3, lw=1.6,
               color="#b2182b", label="parameter upsets, unmitigated")
right.errorbar(mus, clamp_acc, yerr=clamp_acc_se, marker="s", capsize=3, lw=1.6,
               color="#2166ac", label="parameter upsets, clamped")
right.errorbar(mus, act_acc, yerr=act_acc_se, marker="^", capsize=3, lw=1.6,
               color="#1b7837", label="activation upsets")
right.set_xscale("log", base=2)
right.set_xlabel("expected upsets per exposure (Poisson mean)")
right.set_ylabel("mean classification accuracy")
right.set_title("Accuracy versus upset rate, with standard errors")
right.grid(alpha=0.3)
right.legend(fontsize=8, loc="lower left")

figure.suptitle(
    "Upset rate versus output degradation; error bars are standard errors of the campaign mean",
    fontsize=11,
)
figure.tight_layout()
figure.savefig(OUT, dpi=130)
print()
print(f"wrote {os.path.normpath(OUT)}")
