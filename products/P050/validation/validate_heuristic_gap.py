"""Validation: how much the rules of thumb actually cost.

Two rules are measured against the constrained optimum over a grid of
margins, scintillation indices and availability targets on the illustrative
MODCOD table.

``highest_feasible_rate``
    "Take the fastest MODCOD that meets the availability target." Respects the
    constraint, so its loss is a genuine goodput loss.
``highest_rate_within_reserve``
    "Reserve R dB for fade and take the fastest MODCOD that still closes."
    Never looks at the fade distribution, so it can select a MODCOD that
    misses the target. Those cases are counted separately as constraint
    violations, not as goodput wins -- a rule that beats the constrained
    optimum has simply solved a different problem.

The honest headline is the fraction of the grid where each rule matches the
optimum exactly, alongside the worst case. A rule that is right most of the
time is worth keeping, and the numbers say which.

Runtime: about 40 s on one contended core.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from coderateopt import (  # noqa: E402
    InfeasibleProblem,
    LognormalFade,
    RateProblem,
    compare_to_optimum,
    illustrative_modcod_table,
)

TABLE = illustrative_modcod_table()
MARGINS = np.arange(6.0, 24.01, 1.0)
SCINTILLATIONS = (0.05, 0.1, 0.2, 0.4, 0.8)
TARGETS = (0.90, 0.95, 0.99, 0.995, 0.999)
RESERVE_DB = 3.0

print(f"table                 {len(TABLE)} MODCODs, thresholds "
      f"{TABLE.thresholds_db[0]:.1f} to {TABLE.thresholds_db[-1]:.1f} dB")
print(f"margins               {MARGINS[0]:.1f} to {MARGINS[-1]:.1f} dB in 1.0 dB steps")
print(f"scintillation indices {SCINTILLATIONS}")
print(f"availability targets  {TARGETS}")
print(f"fade reserve for the fixed-margin rule {RESERVE_DB:.1f} dB")
print()

rows: dict[str, dict[str, list[float]]] = {}
grid = 0
infeasible = 0
for margin in MARGINS:
    for scintillation in SCINTILLATIONS:
        for target in TARGETS:
            problem = RateProblem(
                modcods=TABLE,
                fade=LognormalFade(scintillation),
                margin_db=float(margin),
                availability_target=target,
                max_entries=1,
                mode="per_interval",
            )
            try:
                comparisons = compare_to_optimum(problem, reserve_db=RESERVE_DB)
            except InfeasibleProblem:
                infeasible += 1
                continue
            grid += 1
            for row in comparisons:
                bucket = rows.setdefault(
                    row.name,
                    {"loss": [], "violations": [], "matches": [], "refusals": []},
                )
                if np.isnan(row.heuristic_goodput):
                    bucket["refusals"].append(1.0)
                    continue
                if row.heuristic_meets_target:
                    bucket["loss"].append(row.relative_loss)
                    bucket["matches"].append(1.0 if row.matches_optimum else 0.0)
                else:
                    bucket["violations"].append(row.relative_loss)

print(f"grid points feasible  {grid}")
print(f"grid points infeasible for every MODCOD {infeasible}")
print()

for name, bucket in rows.items():
    loss = np.asarray(bucket["loss"])
    violations = np.asarray(bucket["violations"])
    matches = np.asarray(bucket["matches"])
    print(f"{name}")
    print(f"  constraint-respecting choices      {loss.size}")
    if loss.size:
        print(f"  matched the optimum exactly        {int(matches.sum())} "
              f"({matches.mean():.4f} of them)")
        print(f"  relative goodput loss: mean {loss.mean():.6f}  median "
              f"{np.median(loss):.6f}  p90 {np.quantile(loss, 0.9):.6f}  max {loss.max():.6f}")
        nonzero = loss[loss > 1e-12]
        if nonzero.size:
            print(f"  where it lost at all: n = {nonzero.size}, mean loss "
                  f"{nonzero.mean():.6f}, max {nonzero.max():.6f}")
        else:
            print("  it never lost on this grid")
    print(f"  choices that MISS the availability target {violations.size}")
    if violations.size:
        print(f"    those are constraint violations, not gains; their apparent 'loss' ranges "
              f"{violations.min():.6f} to {violations.max():.6f}")
    print(f"  refused to choose                  {len(bucket['refusals'])}")
    print()

print("worst single case for the constraint-respecting rule, reproduced")
worst = None
for margin in MARGINS:
    for scintillation in SCINTILLATIONS:
        for target in TARGETS:
            problem = RateProblem(
                modcods=TABLE,
                fade=LognormalFade(scintillation),
                margin_db=float(margin),
                availability_target=target,
            )
            try:
                comparisons = compare_to_optimum(problem, reserve_db=RESERVE_DB)
            except InfeasibleProblem:
                continue
            row = next(c for c in comparisons if c.name == "highest_feasible_rate")
            if row.heuristic_meets_target and (worst is None or row.relative_loss > worst[0]):
                worst = (row.relative_loss, float(margin), scintillation, target, row)
if worst is not None:
    loss, margin, scintillation, target, row = worst
    problem = RateProblem(
        modcods=TABLE,
        fade=LognormalFade(scintillation),
        margin_db=margin,
        availability_target=target,
    )
    print(f"  margin {margin:.1f} dB, scintillation index {scintillation}, target {target}")
    print(f"  availabilities {np.array2string(problem.availabilities(), precision=6)}")
    print(f"  goodputs       {np.array2string(problem.goodputs(), precision=6)}")
    print(f"  greedy goodput {row.heuristic_goodput:.6f}, optimum {row.optimal_goodput:.6f}, "
          f"relative loss {loss:.6f}")
print()
print("measured, not asserted: the numbers above are the whole claim")
