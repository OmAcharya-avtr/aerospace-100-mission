"""Validation: does the Markov channel-state model fit? Where exactly does it fail?

Checks
------
1. **Positive controls.** A sequence simulated from a known two-state Markov
   chain must be recovered (transition matrix to within sampling error) and
   must NOT be rejected by either goodness-of-fit test. A sequence with an
   explicit second-order dependence must be rejected by the order test. Without
   both controls a rejection of the real channel below would be worthless.
2. **The measured channel, two states.** Transition matrix, stationary
   occupancy against measured occupancy, model mean dwell against measured
   mean dwell, dwell-time chi-square per state, and the Markov-order
   conditional-independence test with its effect size.
3. **Where the Markov assumption fails, quantified.** The geometric dwell law
   implies a coefficient of variation ``sqrt(1-p) <= 1``. The measured
   coefficient of variation is reported next to it for every state. This is the
   failure, stated without reference to any p-value, because at two million
   samples every test rejects and the p-value carries no information.
4. **More states do not rescue it.** The same tests for three and five states.
   Finer quantisation of a continuous Gauss-Markov process should move the
   quantised chain towards Markov; the effect size shows by how much.
5. **The semi-Markov alternative.** Empirical dwell laws reproduce the dwell
   distribution by construction; the cost is the parameter count, and a
   simulated sequence from the fitted semi-Markov model is compared against the
   original on occupancy, mean dwell and the dwell tail.

Runtime: about 40 s on one core.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np

from linkoutage.channel import lognormal_amplitude_series
from linkoutage.markov import (
    dwell_lengths,
    dwell_time_goodness_of_fit,
    fit_markov,
    fit_semi_markov,
    markov_order_test,
    state_sequence,
)

FS = 1.0e6
TAU = 2.0e-4
SI = 0.6
N = 2_000_000

print("=" * 78)
print("validate_markov_gof.py")
print("=" * 78)


def simulate_markov(p: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cum = np.cumsum(p, axis=1)
    out = np.empty(n, dtype=np.int64)
    s = 0
    draws = rng.random(n)
    for i in range(n):
        out[i] = s
        s = int(np.searchsorted(cum[s], draws[i]))
    return out


print()
print("[1] Positive controls")
p_true = np.array([[0.92, 0.08], [0.015, 0.985]])
control = simulate_markov(p_true, 1_000_000, seed=3101)
fit_control = fit_markov(control)
print("  (a) a true first-order chain")
print(f"      true P      : {p_true.tolist()!r}")
print(f"      fitted P    : {fit_control.transition_matrix.tolist()!r}")
print(
    f"      max |error| : "
    f"{float(np.max(np.abs(fit_control.transition_matrix - p_true)))!r}"
)
gof_control = dwell_time_goodness_of_fit(control, fit_control)
for g in gof_control:
    print(f"      dwell GOF {g.line()}")
order_control = markov_order_test(control)
print(f"      order test  : chi2={order_control.statistic!r} dof={order_control.dof} "
      f"p={order_control.p_value!r} V={order_control.cramers_v!r}")
ok_a = (
    float(np.max(np.abs(fit_control.transition_matrix - p_true))) < 0.005
    and all(g.p_value > 1e-3 for g in gof_control)
    and order_control.p_value > 1e-3
    and order_control.cramers_v < 0.02
)
print(f"      PASS (recovered and not rejected) = {ok_a}")

print("  (b) an explicitly second-order sequence")
rng = np.random.default_rng(3102)
n2 = 1_000_000
second = np.zeros(n2, dtype=np.int64)
draws = rng.random(n2)
for i in range(2, n2):
    stay = 0.97 if second[i - 1] == second[i - 2] else 0.45
    second[i] = second[i - 1] if draws[i] < stay else 1 - second[i - 1]
order_second = markov_order_test(second)
print(f"      order test  : chi2={order_second.statistic!r} dof={order_second.dof} "
      f"p={order_second.p_value!r} V={order_second.cramers_v!r}")
print(f"      PASS (rejected, large effect) = "
      f"{order_second.p_value < 1e-12 and order_second.cramers_v > 0.1}")

print()
print("[2] The measured channel, two states (threshold 0.6 = state 0)")
series = lognormal_amplitude_series(N, fs_hz=FS, tau_s=TAU, si=SI, seed=41)
states2 = state_sequence(series.amplitude, [0.6])
fit2 = fit_markov(states2)
print(fit2.report())
print()
print("  dwell-time goodness of fit (geometric implied by the fitted P[i,i]):")
for g in dwell_time_goodness_of_fit(states2, fit2):
    print(f"    {g.line()}")
print()
order2 = markov_order_test(states2)
print(order2.report())

print()
print("[3] Where the Markov assumption fails, without reference to a p-value")
header = (
    f"  {'state':>6s} {'runs':>7s} {'mean dwell [samp]':>18s} {'geometric mean':>15s} "
    f"{'cv measured':>12s} {'cv geometric':>13s} {'ratio':>7s}"
)
print(header)
print("  " + "-" * (len(header) - 2))
for g in dwell_time_goodness_of_fit(states2, fit2):
    print(
        f"  {g.state:>6d} {g.n_runs:>7d} {g.mean_observed:>18.4f} "
        f"{g.mean_geometric:>15.4f} {g.cv_observed:>12.4f} {g.cv_geometric:>13.4f} "
        f"{g.cv_observed / g.cv_geometric:>7.2f}"
    )
print("  The fitted chain reproduces the MEAN dwell almost exactly -- it is")
print("  fitted to do so -- and gets the DISPERSION wrong by a factor of about")
print("  2.7. The quantised state of a continuous Gauss-Markov process is not")
print("  Markov, because 'below the threshold' discards how far below. A fade")
print("  one sample old sits at the threshold and usually recovers at once; a")
print("  fade one correlation length old has gone deeper and lasts longer.")

print()
print("[4] More states")
header = (
    f"  {'K':>3s} {'thresholds':<26s} {'order chi2':>14s} {'dof':>5s} "
    f"{'p':>8s} {'Cramer V':>9s} {'max cv ratio':>12s}"
)
print(header)
print("  " + "-" * (len(header) - 2))
for thresholds in ([0.6], [0.5, 0.8], [0.4, 0.55, 0.7, 0.9]):
    st = state_sequence(series.amplitude, thresholds)
    f = fit_markov(st)
    o = markov_order_test(st)
    gofs = dwell_time_goodness_of_fit(st, f)
    ratios = [
        g.cv_observed / g.cv_geometric
        for g in gofs
        if np.isfinite(g.cv_observed) and g.cv_geometric > 0
    ]
    print(
        f"  {f.n_states:>3d} {str(thresholds):<26s} {o.statistic:>14.1f} {o.dof:>5d} "
        f"{o.p_value:>8.2g} {o.cramers_v:>9.4f} {max(ratios):>12.2f}"
    )
print("  Finer quantisation reduces the departure from Markov, as it must: in")
print("  the limit of infinitely many states the chain IS the Gauss-Markov")
print("  process and is exactly first-order. It does not reach that limit at")
print("  any state count a link budget would use.")

print()
print("[5] The semi-Markov alternative")
semi = fit_semi_markov(states2)
print(semi.report())
print()
sim = semi.simulate(N, np.random.default_rng(3103))
per_state_orig, _ = dwell_lengths(states2, n_states=2)
per_state_sim, _ = dwell_lengths(sim, n_states=2)
header = (
    f"  {'state':>6s} {'occupancy orig':>15s} {'occupancy sim':>14s} "
    f"{'mean dwell orig':>16s} {'mean dwell sim':>15s} {'P(dwell>100) orig':>18s} "
    f"{'sim':>10s}"
)
print(header)
print("  " + "-" * (len(header) - 2))
for i in range(2):
    print(
        f"  {i:>6d} {float(np.mean(states2 == i)):>15.6f} "
        f"{float(np.mean(sim == i)):>14.6f} "
        f"{float(per_state_orig[i].mean()):>16.4f} "
        f"{float(per_state_sim[i].mean()):>15.4f} "
        f"{float(np.mean(per_state_orig[i] > 100)):>18.6f} "
        f"{float(np.mean(per_state_sim[i] > 100)):>10.6f}"
    )
print(f"  Markov free parameters      : {fit2.n_parameters}")
print(f"  semi-Markov free parameters : {semi.n_parameters}")
print("  The semi-Markov model matches the dwell tail because it stores it.")
print("  That is a reproduction, not a prediction: it has no power to say what")
print("  the dwell law would be at a threshold or turbulence level it has not")
print("  seen, where the two-parameter channel fit does.")
print()
print("done")
