"""Validation: the worked example printed in the README, reproduced here.

Everything the README shows under "A worked example" is produced by this
script. Running it is how the README's numbers are kept honest: if the API
changes, this script's output changes and the README is wrong until it is
updated from here.

Runtime: under 10 s.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from coderateopt import (  # noqa: E402
    LognormalFade,
    RateProblem,
    compare_to_optimum,
    illustrative_modcod_table,
    scintillation_sensitivity,
    select_rate,
)

table = illustrative_modcod_table()
fade = LognormalFade(scintillation_index=0.2)

problem = RateProblem(
    modcods=table,
    fade=fade,
    margin_db=14.0,
    availability_target=0.999,
    max_entries=1,
    mode="per_interval",
)

solution = select_rate(problem)
report = scintillation_sensitivity(problem)

print("MODCOD table with availabilities at a 14.00 dB margin")
print(f"{'name':<12s} {'rate':>7s} {'thr dB':>7s} {'A':>12s} {'R*A':>9s}")
availability = problem.availabilities()
for index, entry in enumerate(table):
    print(
        f"{entry.name:<12s} {entry.net_rate:>7.3f} {entry.threshold_db:>7.2f} "
        f"{availability[index]:>12.8f} {problem.goodputs()[index]:>9.6f}"
    )

print()
print(f"availability target        {problem.availability_target}")
print(f"chosen MODCOD              {solution.support_names(table)[0]}")
print(f"expected goodput           {solution.expected_goodput:.6f} bits/symbol")
print(f"achieved availability      {solution.achieved_availability:.8f}")
print(f"unique optimum             {solution.is_unique()}")

print()
print("sensitivity in the scintillation index")
print(f"nominal                    {report.nominal:.6f}")
print(f"stable over                [{report.lower:.6f}, {report.upper:.6f}]")
print(f"relative width             {report.relative_width:.6f}")
print(f"headroom down / up         {report.downward_headroom:.6f} / {report.upward_headroom:.6f}")
print(f"knife-edge at +/-10%       {report.is_knife_edge}")
below = report.boundary_signature_below
above = report.boundary_signature_above
print(
    "answer just below / above  "
    f"{table.names[below[0]] if below else 'infeasible'} / "
    f"{table.names[above[0]] if above else 'infeasible'}"
)

print()
print("the same instance with time-sharing allowed and the long-run reading")
mixed = RateProblem(
    modcods=table,
    fade=fade,
    margin_db=14.0,
    availability_target=0.999,
    max_entries=2,
    mode="long_run",
)
mixed_solution = select_rate(mixed)
for index in mixed_solution.support:
    print(
        f"  {table.names[index]:<12s} x = {mixed_solution.time_fractions[index]:.6f}  "
        f"A = {availability[index]:.8f}"
    )
print(f"  expected goodput          {mixed_solution.expected_goodput:.6f} bits/symbol")
print(f"  long-run availability     {mixed_solution.achieved_availability:.8f}")
print(f"  worst-interval availability {mixed_solution.worst_interval_availability:.8f}")
gain = mixed_solution.expected_goodput / solution.expected_goodput - 1.0
print(f"  goodput gain over per-interval {gain:.6f}")
print(
    "  paid for with a worst-interval availability of "
    f"{mixed_solution.worst_interval_availability:.6f} against the "
    f"{problem.availability_target} target"
)

print()
print("the rules of thumb on the same instance")
print(f"{'rule':<30s} {'goodput':>10s} {'rel loss':>10s} {'meets target':>13s}")
for row in compare_to_optimum(problem, reserve_db=3.0):
    loss = "nan" if np.isnan(row.relative_loss) else f"{row.relative_loss:.6f}"
    goodput = "nan" if np.isnan(row.heuristic_goodput) else f"{row.heuristic_goodput:.6f}"
    print(f"{row.name:<30s} {goodput:>10s} {loss:>10s} {str(row.heuristic_meets_target):>13s}")
