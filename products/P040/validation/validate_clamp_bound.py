"""Validation: a clamped parameter range provably bounds the output deviation.

The bound
---------
With every parameter read back through a clamp into ``[-C, C]``, a single-bit
upset in one parameter changes it by at most ``2C`` whatever bit pattern the
upset produces, because both the golden and the faulty value are clipped into
that interval. Propagating that through the two-layer network (full derivation
in the ``bitflipsim.mitigation`` module docstring, which also states the
per-tensor cases) gives

    B = 2 C * max( C * (n_in * Xmax + 1), 1 )

on ``max_k |d logits_k|``, where ``Xmax = max_i |x_i|`` over the batch.

Checks
------
1. Exhaustive: every one of the ``n_parameters * 32`` single-upset sites, with
   clamping applied, produces a logit deviation that does not exceed ``B``.
   This is not a sample - it is every single-upset case for this model.
2. The bound also holds at two tighter clamps, ``C = 0.5 max|w|`` and
   ``C = 0.25 max|w|``, where clamping itself perturbs the golden model. The
   golden perturbation is reported separately so that the two effects are not
   conflated.
3. Without clamping the deviation is unbounded in practice: the worst
   single-upset deviation is reported for comparison, and it exceeds the
   clamped bound by many orders of magnitude.
4. Tightness is reported, not hidden: the ratio ``B / worst measured``.

Runtime: about 18 s on one core (four exhaustive 4704-site sweeps over a
200-sample batch).
"""

from __future__ import annotations

import sys

import numpy as np

from bitflipsim.campaign import single_upset_sites
from bitflipsim.datasets import make_problem, reference_parameters
from bitflipsim.injection import apply_upsets
from bitflipsim.mitigation import clamp_logit_bound, clamp_parameters

failures: list[str] = []


def report(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


N_BATCH = 200
problem = make_problem()
params = reference_parameters(problem)
# The first N_BATCH evaluation samples. The bound depends on the batch only
# through Xmax, and four exhaustive 4704-site sweeps over the full 300-sample
# split cost about 27 s on one contended core; 200 samples keeps this script
# inside the 60 s per-script budget stated in the README. Xmax for this subset
# is printed below and is the Xmax the bound is claimed for.
x = problem.evaluation.x[:N_BATCH]
max_abs_weight = float(np.abs(params.values).max())
max_abs_input = float(np.abs(x).max())
sites = single_upset_sites(params)

print("=" * 78)
print("Model and batch under test")
print("=" * 78)
print(f"parameters                {params.layout.size} float32 "
      f"({params.layout.total_bytes} bytes)")
print(f"architecture              {params.layout.n_in} -> {params.layout.n_hidden} "
      f"(ReLU) -> {params.layout.n_out}")
print(f"evaluation batch          {x.shape[0]} of "
      f"{problem.evaluation.n_samples} evaluation samples")
print(f"max |w|                   {max_abs_weight:.9f}")
print(f"Xmax = max |x|            {max_abs_input:.9f}")
print(f"single-upset sites        {len(sites)} (every parameter x every bit)")

print()
print("=" * 78)
print("Check 1 to 3 - exhaustive single-upset deviation against the bound")
print("=" * 78)
print(f"{'clamp C':>14} {'bound B':>18} {'worst measured':>18} {'B / worst':>12} "
      f"{'violations':>11} {'golden shift':>14}")

results: list[tuple[float, float, float, int, float]] = []
for scale in (1.0, 0.5, 0.25):
    limit = max_abs_weight * scale
    bound = clamp_logit_bound(limit, params.layout.n_in, max_abs_input)
    clamped_golden = clamp_parameters(params, limit)
    golden_logits = clamped_golden.logits(x)
    unclamped_logits = params.logits(x)
    golden_shift = float(np.abs(golden_logits - unclamped_logits).max())
    worst = 0.0
    violations = 0
    for site in sites:
        faulty = clamp_parameters(
            params.with_values(apply_upsets(params.values, [site])), limit
        )
        deviation = float(np.abs(faulty.logits(x) - golden_logits).max())
        if not np.isfinite(deviation):
            violations += 1
            continue
        worst = max(worst, deviation)
        if deviation > bound:
            violations += 1
    ratio = bound / worst if worst > 0.0 else float("inf")
    results.append((limit, bound, worst, violations, golden_shift))
    print(f"{limit:>14.9f} {bound:>18.6f} {worst:>18.9f} {ratio:>12.3f} "
          f"{violations:>11d} {golden_shift:>14.9f}")

for limit, bound, worst, violations, golden_shift in results:
    report(
        f"clamp bound at C = {limit:.6f}",
        violations == 0 and worst <= bound,
        f"{len(sites)} exhaustive single-upset cases, worst deviation "
        f"{worst:.9f} <= bound {bound:.6f}, {violations} violations "
        f"(golden model shifted by {golden_shift:.9f} at this clamp)",
    )

print()
print("=" * 78)
print("Check 4 - the same sweep with no clamping, for comparison")
print("=" * 78)
golden_logits = params.logits(x)
unclamped_worst = 0.0
non_finite = 0
for site in sites:
    faulty = params.with_values(apply_upsets(params.values, [site]))
    with np.errstate(over="ignore", invalid="ignore"):
        deviation = float(np.abs(faulty.logits(x) - golden_logits).max())
    if not np.isfinite(deviation):
        non_finite += 1
        continue
    unclamped_worst = max(unclamped_worst, deviation)
print(f"worst finite deviation    {unclamped_worst:.6e}")
print(f"non-finite outcomes       {non_finite} of {len(sites)} sites "
      f"({100.0 * non_finite / len(sites):.2f} %)")
clamped_worst = results[0][2]
print(f"ratio to clamped worst    {unclamped_worst / clamped_worst:.6e}")
print()
print("The unclamped sweep produces non-finite logits, so no finite bound of any")
print("kind exists for it. That is the whole value of the clamp: the bound is not")
print("merely tighter, it is the difference between a bound and none.")
report("clamping is what makes a bound possible",
       non_finite > 0 and unclamped_worst > clamped_worst,
       f"{non_finite} of {len(sites)} unclamped single upsets give non-finite logits; "
       f"worst finite unclamped deviation {unclamped_worst:.6e} vs clamped "
       f"{clamped_worst:.9f}")

print()
print("=" * 78)
print("Bound tightness, stated plainly")
print("=" * 78)
limit, bound, worst, _, _ = results[0]
print(f"At C = max|w| = {limit:.6f} the bound is {bound:.6f} and the worst")
print(f"measured deviation over all {len(sites)} single-upset sites is {worst:.9f},")
print(f"so the bound is loose by a factor of {bound / worst:.3f}. It is loose because")
print("it assumes the worst input, the worst weight magnitude in the row sum and")
print("the worst alignment of signs simultaneously. A bound that is never")
print("exceeded and is known to be loose is more useful than a tight estimate")
print("that is sometimes exceeded.")

print()
print("=" * 78)
if failures:
    print(f"RESULT: {len(failures)} FAILED check(s): {failures}")
    sys.exit(1)
print("RESULT: all checks PASSED")
