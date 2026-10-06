"""Validation: degenerate and adversarial instances behave as documented.

Each block below is a case that a rate optimiser can get wrong quietly. The
script states the expected behaviour, exercises it on all three solver paths
where they apply, and prints what happened.

1. Infeasible: no MODCOD meets the availability target. Required behaviour:
   raise ``InfeasibleProblem`` carrying the best achievable availability, the
   MODCOD that achieves it, and the margin that would fix it. Forbidden
   behaviour: returning the least-bad MODCOD.
2. Ties: two or more MODCODs attain the same optimal goodput. Required
   behaviour: a deterministic canonical choice plus an explicit list of the
   tied alternatives.
3. Only the lowest rate qualifies.
4. Non-monotone goodput over the rate set, where the greedy "fastest thing
   that closes" rule loses.
5. A MODCOD with zero availability and an enormous rate, which must never be
   selected despite having the largest rate in the table.
6. A minimum dwell fraction that makes every mix impossible.

Runtime: under 5 s.
"""

from __future__ import annotations

import sys
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
    compare_to_optimum,
    illustrative_modcod_table,
    select_rate,
    solve_closed_form_k2,
    solve_exhaustive,
    solve_milp,
)

QUARTER = EmpiricalFade(np.array([-6.0, -4.0, -2.0, 0.0]))
failures: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


print("1. infeasible instance")
table = ModcodSet.from_rows([("m2", 2.0, 6.0), ("m3", 4.0, 8.0), ("m4", 8.0, 10.0)])
infeasible_problem = RateProblem(
    modcods=table,
    fade=QUARTER,
    margin_db=10.0,
    availability_target=0.9,
    max_entries=2,
    mode="long_run",
)
print(f"  availabilities {np.array2string(infeasible_problem.availabilities(), precision=4)}")
print(f"  target {infeasible_problem.availability_target}")
for name, solver in (
    ("solve_milp", solve_milp),
    ("solve_exhaustive", solve_exhaustive),
    ("solve_closed_form_k2", solve_closed_form_k2),
    ("select_rate", select_rate),
):
    try:
        solver(infeasible_problem)
    except InfeasibleProblem as exc:
        print(f"  {name:<22s} raised InfeasibleProblem")
        print(f"    best_availability {exc.best_availability}  best_modcod {exc.best_modcod!r}")
        check(f"{name} raises on an infeasible instance", True)
    else:
        print(f"  {name:<22s} RETURNED A SOLUTION")
        check(f"{name} raises on an infeasible instance", False)

full = RateProblem(
    modcods=illustrative_modcod_table(),
    fade=LognormalFade(0.4),
    margin_db=3.0,
    availability_target=0.999,
)
try:
    select_rate(full)
except InfeasibleProblem as exc:
    print("  diagnostic message on the illustrative table at 3 dB margin, target 0.999:")
    for line in str(exc).split(". "):
        print(f"    {line.strip()}")
    needed = float(full.modcods.thresholds_db[0] - full.fade.quantile_db(0.999))
    print(f"  margin the message names as sufficient: {needed:.4f} dB")
    fixed = select_rate(full.with_margin(needed + 1e-6))
    print(
        f"  at that margin the instance solves: "
        f"{fixed.support_names(full.modcods)} at {fixed.expected_goodput:.6f} bits/symbol"
    )
    check("the margin named in the error message makes the instance feasible", True)

print()
print("2. exact ties")
tie_table = ModcodSet.from_rows([("a", 1.5, 4.0), ("b", 2.0, 6.0), ("c", 3.0, 8.0)])
tie_problem = RateProblem(
    modcods=tie_table, fade=QUARTER, margin_db=10.0, availability_target=0.4
)
print(f"  availabilities {np.array2string(tie_problem.availabilities(), precision=4)}")
print(f"  goodputs       {np.array2string(tie_problem.goodputs(), precision=6)}")
solution = select_rate(tie_problem)
print(f"  canonical support {solution.support_names(tie_table)}")
print(
    f"  tied alternatives "
    f"{[tuple(tie_table.names[i] for i in s) for s in solution.tied_supports]}"
)
print(f"  is_unique() {solution.is_unique()}")
check("three-way tie resolves to the lexicographically first support", solution.support == (0,))
check("both alternatives are reported", len(solution.tied_supports) == 2)
repeats = {select_rate(tie_problem).support for _ in range(20)}
print(f"  supports over 20 repeated solves: {repeats}")
check("the canonical choice is stable across repeated solves", len(repeats) == 1)

print()
print("3. constraint satisfied only by the lowest rate")
monotone = ModcodSet.from_rows(
    [("m1", 1.0, 4.0), ("m2", 2.0, 6.0), ("m3", 4.0, 8.0), ("m4", 8.0, 10.0)]
)
lowest = RateProblem(
    modcods=monotone, fade=QUARTER, margin_db=10.0, availability_target=0.9, max_entries=4
)
solution = select_rate(lowest)
print(f"  availabilities {np.array2string(lowest.availabilities(), precision=4)}")
print(
    f"  chosen {solution.support_names(monotone)} at {solution.expected_goodput:.6f} "
    f"bits/symbol, the lowest rate in the table"
)
check("only the lowest rate is selected", solution.support == (0,))
check("goodput is the lowest rate's own", abs(solution.expected_goodput - 1.0) < 1e-12)

print()
print("4. non-monotone goodput over the rate set")
zigzag_fade = EmpiricalFade(np.array([-9.0, -7.0, -5.0, -3.0, -1.0]))
zigzag = ModcodSet.from_rows(
    [("a", 1.0, 1.0), ("b", 1.0, 3.0), ("c", 6.0, 5.0), ("d", 2.0, 7.0), ("e", 8.0, 9.0)]
)
zig_problem = RateProblem(
    modcods=zigzag, fade=zigzag_fade, margin_db=10.0, availability_target=0.2
)
goodputs = zig_problem.goodputs()
print(f"  rates          {np.array2string(zigzag.rates, precision=3)}")
print(f"  availabilities {np.array2string(zig_problem.availabilities(), precision=4)}")
print(f"  goodputs       {np.array2string(goodputs, precision=6)}")
print(f"  goodput is monotone in rate: {bool(np.all(np.diff(goodputs) >= 0.0))}")
solution = select_rate(zig_problem)
print(f"  optimum {solution.support_names(zigzag)} at {solution.expected_goodput:.6f}")
greedy = next(c for c in compare_to_optimum(zig_problem) if c.name == "highest_feasible_rate")
print(
    f"  greedy 'fastest that closes' at {greedy.heuristic_goodput:.6f}, "
    f"relative loss {greedy.relative_loss:.6f}"
)
check(
    "the optimum is the interior peak, not the top rate",
    solution.support_names(zigzag) == ("c",),
)
check("the greedy rule loses here", greedy.relative_loss > 0.5)
for name, solver in (("milp", solve_milp), ("exhaustive", solve_exhaustive)):
    value = solver(zig_problem).expected_goodput
    print(f"  {name:<12s} {value:.9f}")
    check(f"{name} finds the global optimum on a non-monotone set", abs(value - 3.6) < 1e-9)

print()
print("5. a zero-availability MODCOD with the largest rate in the table")
trap = ModcodSet.from_rows([("useful", 1.0, 4.0), ("hopeless", 1000.0, 20.0)])
trap_problem = RateProblem(
    modcods=trap, fade=QUARTER, margin_db=10.0, availability_target=0.3, max_entries=2
)
print(f"  availabilities {np.array2string(trap_problem.availabilities(), precision=4)}")
print(f"  goodputs       {np.array2string(trap_problem.goodputs(), precision=4)}")
solution = select_rate(trap_problem)
print(f"  chosen {solution.support_names(trap)}")
check("the zero-availability MODCOD is never selected", solution.support_names(trap) == ("useful",))

print()
print("6. a minimum dwell fraction that forbids every mix")
dwell_problem = RateProblem(
    modcods=illustrative_modcod_table(),
    fade=LognormalFade(0.2),
    margin_db=12.0,
    availability_target=0.99,
    max_entries=4,
    mode="long_run",
    min_dwell_fraction=0.75,
)
print(f"  max_entries 4, min_dwell 0.75 -> effective_k {dwell_problem.effective_k}")
solution = select_rate(dwell_problem)
print(
    f"  chosen {solution.support_names(dwell_problem.modcods)} at "
    f"{solution.expected_goodput:.6f} bits/symbol, worst-interval availability "
    f"{solution.worst_interval_availability:.6f}"
)
check("a dwell above 0.5 collapses the cardinality limit to 1", dwell_problem.effective_k == 1)
check(
    "the single-entry answer still meets the target",
    solution.worst_interval_availability >= 0.99,
)

print()
print(f"checks run: {len(failures)} failed")
if failures:
    for item in failures:
        print(f"  FAILED: {item}")
    sys.exit(1)
print("every degenerate case behaved as documented")
