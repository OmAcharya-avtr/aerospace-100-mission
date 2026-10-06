"""Validation: the default HiGHS MIP gap changes the answer, measurably.

``scipy.optimize.milp`` does not set ``mip_rel_gap``, so HiGHS uses its own
default, which stops branch and bound as soon as the incumbent is within a
relative tolerance of the bound. On this problem the goodput error that buys
is negligible and the *decision* error is not: the optimiser returns a
different set of MODCODs, and every sensitivity boundary in
``coderateopt.sensitivity`` is found by watching that set change.

This script sweeps the scintillation index on the illustrative table and
counts, over the sweep:

* instances where the default-gap objective is below the enumerated optimum
  by more than 1e-9;
* instances where the default-gap *support* differs from the enumerated one;
* the largest relative objective shortfall, with the instance that produced
  it.

``coderateopt`` sets ``mip_rel_gap = 0.0`` for every call. The same sweep is
re-run with that setting to show the count fall to zero.

This measures scipy and HiGHS behaviour as installed, not a property of this
package, and the versions are printed so the result is attributable.

Runtime: about 90 s on one contended core.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: E402

from coderateopt import (  # noqa: E402
    InfeasibleProblem,
    LognormalFade,
    RateProblem,
    build_milp,
    illustrative_modcod_table,
    solve_exhaustive,
)
from coderateopt.milp import MILP_OPTIONS  # noqa: E402

TABLE = illustrative_modcod_table()
BASE = RateProblem(
    modcods=TABLE,
    fade=LognormalFade(0.2),
    margin_db=12.0,
    availability_target=0.99,
    max_entries=2,
    mode="long_run",
)
SWEEP = np.linspace(0.05, 1.2, 221)

print(f"scipy version                 {scipy.__version__}")
print(f"numpy version                 {np.__version__}")
print(f"package MILP options          {MILP_OPTIONS}")
print(f"sweep                         scintillation index {SWEEP[0]:.4f} to {SWEEP[-1]:.4f}, "
      f"{SWEEP.size} points")
print("instance                      illustrative table, margin 12.00 dB, target 0.990000, "
      "K = 2, long_run")
print()


def raw_milp(problem: RateProblem, options: dict[str, float] | None):
    enc = build_milp(problem)
    kwargs = {
        "c": enc.c,
        "constraints": LinearConstraint(enc.constraint_matrix, enc.lower, enc.upper),
        "integrality": enc.integrality,
        "bounds": Bounds(enc.variable_lower, enc.variable_upper),
    }
    if options is not None:
        kwargs["options"] = dict(options)
    result = milp(**kwargs)
    x = np.where(np.abs(result.x[: problem.n_modcods]) < 1e-10, 0.0, result.x[: problem.n_modcods])
    support = tuple(int(i) for i in np.flatnonzero(x > 0.0))
    return -float(result.fun), support


REFERENCES: list[tuple[float, object]] = []
for _value in SWEEP:
    _problem = BASE.with_fade(LognormalFade(float(_value)))
    try:
        REFERENCES.append((float(_value), solve_exhaustive(_problem)))
    except InfeasibleProblem:
        continue
print(f"feasible sweep points         {len(REFERENCES)} of {SWEEP.size}")
print()


def run(options: dict[str, float] | None, label: str) -> dict[str, float | int | str]:
    objective_shortfalls = 0
    support_differences = 0
    worst_rel = 0.0
    worst_detail = ""
    first_detail = ""
    for value, reference in REFERENCES:
        problem = BASE.with_fade(LognormalFade(value))
        objective, support = raw_milp(problem, options)
        shortfall = reference.expected_goodput - objective
        rel = shortfall / max(abs(reference.expected_goodput), 1e-12)
        if shortfall > 1e-9:
            objective_shortfalls += 1
        if support != reference.support:
            support_differences += 1
            detail = (
                f"scintillation {value!r}: solver support "
                f"{tuple(TABLE.names[i] for i in support)} at {objective:.12f}, "
                f"enumerated {tuple(TABLE.names[i] for i in reference.support)} at "
                f"{reference.expected_goodput:.12f}, relative shortfall {rel:.4e}"
            )
            if not first_detail:
                first_detail = detail
        if rel > worst_rel:
            worst_rel = rel
            worst_detail = (
                f"scintillation {value!r}, relative shortfall {rel:.4e}, "
                f"absolute {shortfall:.3e} bits/symbol"
            )
    print(f"{label}")
    print(f"  objective shortfalls > 1e-9           {objective_shortfalls}")
    print(f"  support differences from enumeration  {support_differences}")
    print(f"  worst relative shortfall              {worst_rel:.4e}")
    if worst_detail:
        print(f"    at                                  {worst_detail}")
    if first_detail:
        print(f"  first support difference              {first_detail}")
    print()
    return {
        "shortfalls": objective_shortfalls,
        "support_differences": support_differences,
        "worst_rel": worst_rel,
    }


default_arm = run(None, "A. scipy.optimize.milp with no options (HiGHS default gap)")
tight_arm = run(
    MILP_OPTIONS, "B. scipy.optimize.milp with mip_rel_gap = 0.0 (what this package does)"
)

print("C. random instances, default gap against enumeration")
print("   (the uniform sweep above walks a coarse grid and can step over the narrow")
print("    parameter bands where the two-entry mix is only marginally better, so the")
print("    frequency of the defect is measured by random search instead)")
rng = np.random.default_rng(20261006)
random_feasible = 0
random_support_differences = 0
random_shortfalls = 0
random_worst_rel = 0.0
random_worst_detail = ""
for _ in range(500):
    scintillation = float(rng.uniform(0.05, 1.2))
    margin = float(rng.uniform(8.0, 18.0))
    target = float(rng.uniform(0.80, 0.995))
    probe = RateProblem(
        modcods=TABLE,
        fade=LognormalFade(scintillation),
        margin_db=margin,
        availability_target=target,
        max_entries=2,
        mode="long_run",
    )
    try:
        reference = solve_exhaustive(probe)
    except InfeasibleProblem:
        continue
    random_feasible += 1
    objective, support = raw_milp(probe, None)
    shortfall = reference.expected_goodput - objective
    rel = shortfall / max(abs(reference.expected_goodput), 1e-12)
    if shortfall > 1e-9:
        random_shortfalls += 1
    if support != reference.support:
        random_support_differences += 1
    if rel > random_worst_rel:
        random_worst_rel = rel
        random_worst_detail = (
            f"scintillation {scintillation!r}, margin {margin!r} dB, target {target!r}: "
            f"default-gap support {tuple(TABLE.names[i] for i in support)} at "
            f"{objective:.12f}, enumerated "
            f"{tuple(TABLE.names[i] for i in reference.support)} at "
            f"{reference.expected_goodput:.12f}"
        )
print(f"  feasible random instances             {random_feasible} of 500")
print(f"  objective shortfalls > 1e-9           {random_shortfalls}")
print(f"  support differences from enumeration  {random_support_differences}")
if random_feasible:
    print(
        f"  support-difference rate               "
        f"{random_support_differences / random_feasible:.4f}"
    )
print(f"  worst relative shortfall              {random_worst_rel:.4e}")
if random_worst_detail:
    print(f"    at                                  {random_worst_detail}")
print()

print("the specific instance named in coderateopt.milp.MILP_OPTIONS")
named = BASE.with_fade(LognormalFade(0.5281008782676444))
reference = solve_exhaustive(named)
obj_default, sup_default = raw_milp(named, None)
obj_tight, sup_tight = raw_milp(named, MILP_OPTIONS)
print(f"  enumerated optimum    {reference.expected_goodput:.12f}  "
      f"support {tuple(TABLE.names[i] for i in reference.support)}")
print(
    f"  default gap           {obj_default:.12f}  "
    f"support {tuple(TABLE.names[i] for i in sup_default)}"
)
print(
    f"  mip_rel_gap = 0.0     {obj_tight:.12f}  "
    f"support {tuple(TABLE.names[i] for i in sup_tight)}"
)
print(f"  relative difference   "
      f"{(reference.expected_goodput - obj_default) / reference.expected_goodput:.4e}")
print()

print("verdict")
print(f"  random search          {random_support_differences} support differences in "
      f"{random_feasible} feasible instances under the default gap")
print(f"  default gap            {default_arm['support_differences']} support differences, "
      f"{default_arm['shortfalls']} objective shortfalls")
print(f"  mip_rel_gap = 0.0      {tight_arm['support_differences']} support differences, "
      f"{tight_arm['shortfalls']} objective shortfalls")
if tight_arm["support_differences"] or tight_arm["shortfalls"]:
    print("FAILED: the package's own options do not reproduce the enumerated optimum")
    sys.exit(1)
print("  the package's options reproduce the enumerated optimum at every feasible sweep point")
