"""Validation: the MILP encoding against two independent enumerations.

This is the check that decides whether the MILP in ``coderateopt.milp`` says
what the problem statement in ``coderateopt.problem`` says. Three solvers are
compared on the same randomly generated instances:

``milp``
    ``scipy.optimize.milp`` (HiGHS branch and bound) on the full encoding,
    with binary selection variables, linking rows, a cardinality row and --
    where configured -- minimum-dwell rows.
``exhaustive``
    Every MODCOD subset of size at most K enumerated by
    ``itertools.combinations``, each restricted problem solved as a plain
    linear programme by ``scipy.optimize.linprog``. No integer variables, no
    linking rows, no cardinality row.
``closed_form_k2``
    For an effective cardinality limit of at most 2, the vertices of the
    feasible segment written down and evaluated in NumPy. **No solver call at
    all.**

An agreement between ``milp`` and ``exhaustive`` rules out an error in the
integer encoding. An agreement with ``closed_form_k2`` additionally rules out
a shared error inside HiGHS, because that path never calls it.

Both disagreements this check has found so far are recorded in the source: the
default ``mip_rel_gap`` (``coderateopt.milp.MILP_OPTIONS``) and the raw HiGHS
allocation's primal-feasibility slack on a tight availability row
(``coderateopt.milp.solve_milp``). Neither was visible without a second
implementation to compare against, which is the argument for keeping one.

Instance generator: 2 to 7 MODCODs with rates in [0.1, 8] bits/symbol and
distinct thresholds in [-5, 25] dB; fade model lognormal (scintillation index
in [0.01, 1.5]) or empirical (4 to 40 dB samples in [-20, 5]); margin in
[0, 25] dB; availability target in [0.05, 0.995]; K in 1..4; mode either form;
minimum dwell drawn from {0, 0, 0.05, 0.2, 0.4}. Seeded, so the instance set
is reproducible.

Runtime: about 60 s on one contended core.
"""

from __future__ import annotations

import sys
import time
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
    count_subsets,
    solve_closed_form_k2,
    solve_exhaustive,
    solve_milp,
)

SEED = 20261006
N_INSTANCES = 3000
REL_TOL = 1e-7
ABS_TOL = 1e-9


def random_problem(rng: np.random.Generator) -> RateProblem:
    n = int(rng.integers(2, 8))
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
        count = int(rng.integers(4, 41))
        fade = EmpiricalFade(rng.uniform(-20.0, 5.0, size=count))
    return RateProblem(
        modcods=table,
        fade=fade,
        margin_db=float(rng.uniform(0.0, 25.0)),
        availability_target=float(rng.uniform(0.05, 0.995)),
        max_entries=int(rng.integers(1, 5)),
        mode="per_interval" if rng.random() < 0.5 else "long_run",
        min_dwell_fraction=float(rng.choice([0.0, 0.0, 0.05, 0.2, 0.4])),
    )


rng = np.random.default_rng(SEED)
compared = 0
infeasible = 0
disagree_both_ways = 0
mismatch_milp_exhaustive = 0
mismatch_closed_form = 0
closed_form_applicable = 0
worst_rel = 0.0
worst_rel_detail = ""
support_sizes: dict[int, int] = {}
milp_seconds = 0.0
exhaustive_seconds = 0.0

print(f"instances requested: {N_INSTANCES}  seed: {SEED}")
print(f"agreement tolerance: rel {REL_TOL:g}, abs {ABS_TOL:g}")
print()

for _ in range(N_INSTANCES):
    problem = random_problem(rng)
    t0 = time.perf_counter()
    try:
        by_milp = solve_milp(problem)
    except InfeasibleProblem:
        milp_seconds += time.perf_counter() - t0
        t1 = time.perf_counter()
        try:
            solve_exhaustive(problem)
        except InfeasibleProblem:
            infeasible += 1
        else:
            disagree_both_ways += 1
            print("  DISAGREEMENT on feasibility: milp infeasible, enumeration feasible")
        exhaustive_seconds += time.perf_counter() - t1
        continue
    milp_seconds += time.perf_counter() - t0

    t1 = time.perf_counter()
    by_enumeration = solve_exhaustive(problem)
    exhaustive_seconds += time.perf_counter() - t1

    compared += 1
    a, b = by_milp.expected_goodput, by_enumeration.expected_goodput
    rel = abs(a - b) / max(abs(b), 1e-12)
    if rel > worst_rel:
        worst_rel = rel
        worst_rel_detail = (
            f"M={problem.n_modcods} K={problem.max_entries} mode={problem.mode} "
            f"dwell={problem.min_dwell_fraction} target={problem.availability_target:.6f}"
        )
    if abs(a - b) > max(ABS_TOL, REL_TOL * abs(b)):
        mismatch_milp_exhaustive += 1
        print(f"  MISMATCH milp={a!r} exhaustive={b!r} rel={rel:.3e}")

    size = len(by_enumeration.support)
    support_sizes[size] = support_sizes.get(size, 0) + 1

    if problem.effective_k <= 2:
        closed_form_applicable += 1
        by_closed = solve_closed_form_k2(problem)
        if abs(by_closed.expected_goodput - b) > max(ABS_TOL, REL_TOL * abs(b)):
            mismatch_closed_form += 1
            print(
                f"  MISMATCH closed_form={by_closed.expected_goodput!r} exhaustive={b!r}"
            )

print("results")
print(f"  instances generated                     {N_INSTANCES}")
print(f"  infeasible in both solvers (agreed)     {infeasible}")
print(f"  feasibility disagreements               {disagree_both_ways}")
print(f"  feasible instances compared             {compared}")
print(f"  milp vs exhaustive mismatches           {mismatch_milp_exhaustive}")
print(f"  closed-form applicable (effective K<=2) {closed_form_applicable}")
print(f"  closed-form vs exhaustive mismatches    {mismatch_closed_form}")
print(f"  worst relative goodput difference       {worst_rel:.3e}")
print(f"    on instance                           {worst_rel_detail}")
print()
print("optimal support sizes over the compared instances")
for size in sorted(support_sizes):
    print(f"  {size} entr{'y' if size == 1 else 'ies'}: {support_sizes[size]}")
print()
print("solve cost, wall clock on a contended core (indicative, not a benchmark)")
print(f"  milp total        {milp_seconds:.3f} s  mean {milp_seconds / N_INSTANCES * 1e3:.3f} ms")
print(
    f"  exhaustive total  {exhaustive_seconds:.3f} s  "
    f"mean {exhaustive_seconds / N_INSTANCES * 1e3:.3f} ms"
)
print()
print("enumeration cost as the table grows: sum_{k=1..K} C(M, k) linear programmes")
for m_size, k in ((9, 2), (9, 3), (16, 3), (30, 3), (30, 5), (64, 4), (100, 5)):
    print(f"  M = {m_size:>3d}, K = {k}: {count_subsets(m_size, k):>12,d} subsets")
print()
if mismatch_milp_exhaustive or mismatch_closed_form or disagree_both_ways:
    print("FAILED: at least one disagreement")
    sys.exit(1)
print("all three solvers agree on every compared instance")
