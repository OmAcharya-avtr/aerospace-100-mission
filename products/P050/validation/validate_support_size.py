"""Validation: how many MODCODs the optimum actually needs, and therefore
whether the integer variables in the MILP do any work.

Two structural results are claimed in ``coderateopt.problem``:

* in ``"per_interval"`` mode the optimum is a single MODCOD;
* in ``"long_run"`` mode with no minimum dwell it needs at most two, because
  the linear programme has two active constraints.

Both are checked here by brute force. The interesting question is the third
case: with a minimum dwell fraction ``d > 0`` the implication
``x_m > 0 => x_m >= d`` is not a linear constraint, the basic-solution
argument no longer applies, and a support of three or more *could* be optimal.
Whether it ever is decides whether ``scipy.optimize.milp`` is doing anything
that enumeration over pairs could not.

Three searches:

1. a wide random search over tables (3 to 7 MODCODs), fade models, targets,
   K in 1..4 and dwell in {0, 0.05, 0.1, 0.2, 0.3}, solved by the MILP alone
   (its agreement with enumeration is established in
   ``validate_milp_vs_exhaustive.py``);
2. a targeted search, solved with the canonical path so ties cannot inflate a
   support, built to produce a three-entry optimum if one exists --
   one very-high-availability low-rate entry whose only use is to relax the
   availability constraint, plus a spread of higher-rate entries, with K and
   the dwell set so that three entries are permitted and two are not forced;
3. exhaustive verification on the targeted instances that no two-entry
   subset matches the best three-entry subset, when such a case is found.

Whatever the searches find is reported as found. A negative result is a
result: it says the MILP is a cross-check rather than a necessity on this
formulation, and the README says so.

Runtime: about 80 s on one heavily contended core.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from coderateopt import (  # noqa: E402
    EmpiricalFade,
    InfeasibleProblem,
    LognormalFade,
    ModcodSet,
    RateProblem,
    restricted_lp,
    select_rate,
    solve_milp,
)

SEED = 20261006
failures: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


print("1. wide random search")
rng = np.random.default_rng(SEED)
N = 1200
counts: dict[tuple[str, int], int] = {}
per_interval_violations = 0
long_run_no_dwell_violations = 0
largest = 0
largest_detail = ""
feasible = 0
for _ in range(N):
    n = int(rng.integers(3, 8))
    rates = rng.uniform(0.1, 8.0, size=n)
    thresholds = rng.choice(np.arange(-5.0, 25.01, 0.25), size=n, replace=False)
    table = ModcodSet.from_rows(
        [
            (f"m{i}", float(r), float(t))
            for i, (r, t) in enumerate(zip(rates, thresholds, strict=True))
        ]
    )
    if rng.random() < 0.5:
        fade = LognormalFade(float(rng.uniform(0.01, 1.5)))
    else:
        fade = EmpiricalFade(rng.uniform(-20.0, 5.0, size=int(rng.integers(5, 41))))
    mode = "per_interval" if rng.random() < 0.4 else "long_run"
    dwell = float(rng.choice([0.0, 0.05, 0.1, 0.2, 0.3]))
    problem = RateProblem(
        modcods=table,
        fade=fade,
        margin_db=float(rng.uniform(0.0, 25.0)),
        availability_target=float(rng.uniform(0.05, 0.995)),
        max_entries=int(rng.integers(1, 5)),
        mode=mode,
        min_dwell_fraction=dwell,
    )
    try:
        solution = solve_milp(problem)
    except InfeasibleProblem:
        continue
    feasible += 1
    size = len(solution.support)
    counts[(mode, size)] = counts.get((mode, size), 0) + 1
    if mode == "per_interval":
        # The structural claim is about the optimal *value*, not about which of
        # several tied optima the solver happens to return: a mix of two equally
        # good entries is also optimal. So compare the MILP optimum with the
        # best single allowed entry, computed directly.
        allowed = problem.allowed_mask()
        best_single = (
            float(np.max(problem.goodputs()[allowed])) if allowed.any() else float("-inf")
        )
        if solution.expected_goodput > best_single + 1e-9:
            per_interval_violations += 1
    if mode == "long_run" and dwell == 0.0 and size > 2:
        long_run_no_dwell_violations += 1
    if size > largest:
        largest = size
        largest_detail = (
            f"M={n} K={problem.max_entries} mode={mode} dwell={dwell} "
            f"target={problem.availability_target:.6f}"
        )

print(f"  instances generated                 {N}")
print(f"  feasible                            {feasible}")
print("  MILP support size by mode (a tie can inflate a support by one):")
for key in sorted(counts):
    print(f"    {key[0]:<13s} size {key[1]}: {counts[key]}")
print(f"  largest optimal support found       {largest}  ({largest_detail})")
check(
    "per_interval optimum never exceeds the best single allowed entry",
    per_interval_violations == 0,
)
check(
    "long_run optima with no dwell never exceed two entries",
    long_run_no_dwell_violations == 0,
)

print()
print("2. targeted search for a three-or-more entry optimum under a minimum dwell")
rng = np.random.default_rng(SEED + 1)
found: list[tuple[int, RateProblem, object]] = []
targeted = 0
for _ in range(2000):
    # One low-rate entry with a very low threshold (high availability) whose
    # only possible value is to relax the availability constraint, plus three
    # higher-rate entries spread above it.
    thresholds = [float(rng.uniform(-8.0, -2.0))]
    for _step in range(3):
        thresholds.append(thresholds[-1] + float(rng.uniform(0.5, 6.0)))
    rates = [float(rng.uniform(0.05, 0.5))]
    for _step in range(3):
        rates.append(rates[-1] + float(rng.uniform(0.2, 3.0)))
    table = ModcodSet.from_rows(
        [
            (f"m{i}", r, t)
            for i, (r, t) in enumerate(zip(rates, thresholds, strict=True))
        ]
    )
    dwell = float(rng.choice([0.05, 0.1, 0.15, 0.2, 0.3]))
    problem = RateProblem(
        modcods=table,
        fade=LognormalFade(float(rng.uniform(0.05, 1.2))),
        margin_db=float(rng.uniform(0.0, 12.0)),
        availability_target=float(rng.uniform(0.5, 0.995)),
        max_entries=int(rng.integers(3, 5)),
        mode="long_run",
        min_dwell_fraction=dwell,
    )
    try:
        solution = select_rate(problem)
    except InfeasibleProblem:
        continue
    targeted += 1
    if len(solution.support) >= 3:
        found.append((len(solution.support), problem, solution))

print(f"  targeted instances feasible         {targeted} of 2000")
print(f"  with a three-or-more entry optimum  {len(found)}")

if found:
    size, problem, solution = max(found, key=lambda item: item[0])
    print(f"  largest support {size}:")
    print(f"    rates          {np.array2string(problem.modcods.rates, precision=4)}")
    print(f"    thresholds dB  {np.array2string(problem.modcods.thresholds_db, precision=3)}")
    print(f"    availabilities {np.array2string(problem.availabilities(), precision=6)}")
    print(f"    target {problem.availability_target:.6f}  K {problem.max_entries}  "
          f"dwell {problem.min_dwell_fraction}")
    print(
        "    mix "
        + ", ".join(
            f"{problem.modcods.names[i]} x={solution.time_fractions[i]:.6f}"
            for i in solution.support
        )
    )
    print(f"    goodput {solution.expected_goodput:.9f}")
    print()
    print("3. is any pair as good as that support?")
    best_pair = None
    for pair in combinations(range(problem.n_modcods), 2):
        got = restricted_lp(problem, pair)
        if got is None:
            continue
        if best_pair is None or got[0] > best_pair[0]:
            best_pair = (got[0], pair)
    best_single = None
    for single in range(problem.n_modcods):
        got = restricted_lp(problem, (single,))
        if got is None:
            continue
        if best_single is None or got[0] > best_single[0]:
            best_single = (got[0], single)
    if best_pair is not None:
        print(
            f"    best pair  {tuple(problem.modcods.names[i] for i in best_pair[1])} "
            f"at {best_pair[0]:.9f}"
        )
    if best_single is not None:
        print(f"    best single {problem.modcods.names[best_single[1]]!r} at {best_single[0]:.9f}")
    margin_over_pair = (
        solution.expected_goodput - best_pair[0] if best_pair is not None else float("nan")
    )
    print(f"    the three-entry optimum beats the best pair by {margin_over_pair:.3e}")
    check("a three-entry optimum strictly beats every pair", margin_over_pair > 1e-9)
else:
    print()
    print("3. no three-entry optimum was found, in either search")
    print("   Total instances searched: "
          f"{N} wide + 2000 targeted, {feasible + targeted} of them feasible.")
    print("   Reading: on this formulation the binary selection variables never changed")
    print("   the answer. Enumerating singletons and pairs -- O(M**2) linear programmes,")
    print("   or the solver-free closed form for K <= 2 -- reproduced the MILP optimum on")
    print("   every instance tested. The MILP is kept as an independent implementation and")
    print("   as the path that survives additional constraints, not because this")
    print("   formulation needs it. The README states this plainly.")

print()
print(f"checks run: {len(failures)} failed")
if failures:
    for item in failures:
        print(f"  FAILED: {item}")
    sys.exit(1)
print("support-size structure is as the formulation documents it")
