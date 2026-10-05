"""Validation 5: same-budget benchmark of the three search strategies.

Claim under test: severe faults found per campaign budget, with confidence
intervals over seeds, for the learned prioritiser against the two classical
baselines -- and the verdict rule applied without exception.  Decision rule,
from the specification: if the uniform-random interval contains the learned
strategy's mean score, the result is ``no measurable advantage``.

This script reports; it does not steer.  It fails only on structural defects
(a strategy that exceeded its budget, executed a case twice, or returned the
wrong number of runs).  Whatever the comparison says, it says.

Run from products/P034/:  python validation/validate_benchmark.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from faultinject.benchmark import (  # noqa: E402
    DEFAULT_BUDGET,
    DEFAULT_N_POOLS,
    DEFAULT_N_SEEDS,
    DEFAULT_REPLICATES,
    DEFAULT_WARMUP,
    bootstrap_ci,
    run_benchmark,
)
from faultinject.campaign import build_pool, evaluate_pool  # noqa: E402
from faultinject.search import (  # noqa: E402
    STRATEGIES,
    coverage_greedy,
    kind_mean,
    learned,
    uniform_random,
)
from faultinject.severity import SEVERE_THRESHOLD  # noqa: E402

failures: list[str] = []

print("=" * 78)
print("VALIDATION 5 -- campaign search: two baselines, one learned model")
print("=" * 78)
print()
print("Protocol")
print(f"  pools                 {DEFAULT_N_POOLS}, each covering all 248 cells "
      f"{DEFAULT_REPLICATES}x -> 496 cases")
print(f"  strategy seeds        {DEFAULT_N_SEEDS} per pool -> "
      f"{DEFAULT_N_POOLS * DEFAULT_N_SEEDS} runs per strategy")
print(f"  budget                {DEFAULT_BUDGET} executions "
      f"({100.0 * DEFAULT_BUDGET / 496:.1f} % of the pool)")
print(f"  learned warm-up       {DEFAULT_WARMUP} random executions, counted "
      "against the budget")
print(f"  severe threshold      severity >= {SEVERE_THRESHOLD}")
print("  intervals             percentile bootstrap, 5000 resamples")
print()

t0 = time.time()
rep = run_benchmark()
elapsed = time.time() - t0

print(f"Pool severity ground truth: {rep.severe_in_pool * 100:.2f} % of pool cases "
      "are severe.")
print(f"A budget of {rep.budget} spent uniformly at random therefore has expectation "
      f"{rep.budget * rep.severe_in_pool:.2f} severe cases.")
print()

# ------------------------------------------------------------------ 5a structure
print("5a. Structural checks on every run")
pool = build_pool(pool_seed=1, replicates=DEFAULT_REPLICATES)
sev = evaluate_pool(pool)


def oracle(i: int) -> float:
    return sev[i]


checked = 0
for name in STRATEGIES:
    for s in range(3):
        rng = np.random.default_rng(7000 + s)
        if name == "learned":
            res, _ = learned(pool, oracle, DEFAULT_BUDGET, rng, warmup=DEFAULT_WARMUP)
        elif name == "kind_mean":
            res = kind_mean(pool, oracle, DEFAULT_BUDGET, rng, warmup=DEFAULT_WARMUP)
        elif name == "coverage_greedy":
            res = coverage_greedy(pool, oracle, DEFAULT_BUDGET, rng)
        else:
            res = uniform_random(pool, oracle, DEFAULT_BUDGET, rng)
        checked += 1
        if len(res.order) != DEFAULT_BUDGET:
            failures.append(f"5a: {name} executed {len(res.order)} cases, budget "
                            f"{DEFAULT_BUDGET}")
        if len(set(res.order)) != len(res.order):
            failures.append(f"5a: {name} executed a case more than once")
        if not 0 <= res.n_severe <= DEFAULT_BUDGET:
            failures.append(f"5a: {name} severe count {res.n_severe} out of range")
print(f"    runs checked  {checked}")
for name in STRATEGIES:
    n = len(rep.stats[name].scores)
    if n != DEFAULT_N_POOLS * DEFAULT_N_SEEDS:
        failures.append(f"5a: {name} has {n} runs, expected "
                        f"{DEFAULT_N_POOLS * DEFAULT_N_SEEDS}")
print(f"    run counts    {[len(rep.stats[n].scores) for n in STRATEGIES]}")
print("    PASS" if not failures else "    FAILED")
print()

# ------------------------------------------------------------------ 5b results
print("5b. Severe cases found per budget, with 95 % bootstrap intervals")
print()
hdr = (f"    {'strategy':<17} {'mean':>7} {'95 % interval':>20} {'mean sev':>9} "
       f"{'coverage':>9}")
print(hdr)
print("    " + "-" * (len(hdr) - 4))
for name in STRATEGIES:
    st = rep.stats[name]
    print(f"    {name:<17} {st.mean:>7.3f} "
          f"[{st.lo:>7.3f}, {st.hi:>7.3f}] {st.mean_severity:>9.4f} "
          f"{st.coverage:>9.4f}")
print()
print("    'mean sev' is the mean severity of the cases the strategy chose to run;")
print("    'coverage' is the fraction of the 248 cells reached inside the budget.")
print()

# ------------------------------------------------------------------ 5c verdicts
print("5c. Verdicts under the specification's overlap rule")
print()


def verdict(a_name: str, b_name: str) -> str:
    a, b = rep.stats[a_name], rep.stats[b_name]
    if b.lo <= a.mean <= b.hi:
        return "no measurable advantage"
    if a.mean > b.hi:
        return f"{a_name} ahead"
    return f"{b_name} ahead"


pairs = (
    ("learned", "uniform_random"),
    ("learned", "coverage_greedy"),
    ("learned", "kind_mean"),
    ("coverage_greedy", "uniform_random"),
    ("kind_mean", "uniform_random"),
)
for a_name, b_name in pairs:
    print(f"    {a_name:<16} vs {b_name:<16} : {verdict(a_name, b_name)}")
print()
print(f"    Required headline verdict (learned vs uniform_random): "
      f"{rep.verdict()}")
print()
print("5d. Paired bootstrap on the per-run difference (same pool, same seed)")
print()
for key, (mean, lo, hi) in rep.paired.items():
    contains_zero = lo <= 0.0 <= hi
    note = "interval contains zero" if contains_zero else "interval excludes zero"
    print(f"    {key:<34} {mean:>8.3f}  [{lo:>7.3f}, {hi:>7.3f}]  {note}")
print()

# ------------------------------------------------------------------ 5e model
print("5e. The learned model's uncertainty output")
print()
print(f"    empirical coverage of the nominal 95 % interval : "
      f"{rep.calibration[0]:.6f}")
print(f"    mean interval width (severity is in [0, 1])     : "
      f"{rep.calibration[1]:.6f}")
if not np.isfinite(rep.calibration[0]):
    failures.append("5e: no calibration measurement was produced")
elif rep.calibration[1] > 0.5:
    print("    NOTE  the interval is wider than half the severity range, so the")
    print("          nominal coverage is reached by being uninformative rather than")
    print("          by being calibrated. Stated as such in MODEL_CARD.md.")
print()
print("    top feature importances (impurity-based, last fitted forest)")
for name, value in rep.importances.items():
    print(f"      {name:<34} {value:.6f}")
print()

# ------------------------------------------------------------------ 5f sanity
print("5f. Coverage-greedy must reach at least as many cells as uniform random")
cg = rep.stats["coverage_greedy"].coverage
ur = rep.stats["uniform_random"].coverage
print(f"    coverage_greedy {cg:.6f}  uniform_random {ur:.6f}")
if cg < ur:
    failures.append(f"5f: coverage_greedy coverage {cg:.6f} below uniform_random {ur:.6f}")
else:
    print("    PASS")
print()

print(f"Wall clock for the benchmark: {elapsed:.1f} s on the build container "
      "(1 core, shared).")
print()
bc = bootstrap_ci(rep.stats["learned"].scores)
print(f"Re-derived learned interval from its raw scores: mean {bc[0]:.3f}, "
      f"[{bc[1]:.3f}, {bc[2]:.3f}]")
print()

print("=" * 78)
if failures:
    print(f"RESULT: FAILED ({len(failures)} structural check(s))")
    for f in failures:
        print(f"  FAILED  {f}")
    sys.exit(1)
print("RESULT: PASS -- validation 5 structural checks passed; "
      "comparison reported above")
print("=" * 78)
