"""The worked example quoted in README.md, run so the output cannot drift.

Run from the repository root:

    python validation/worked_example.py

The code between the two markers below is exactly what README.md shows, and
the committed output beside this script is exactly what it printed.
"""

from __future__ import annotations

from _support import Emitter

# --- README example begins ---------------------------------------------
from budgetroll import load, rollup, run_checks, sensitivity

budget = load("examples/cases/cubesat_power.yaml")
roll = rollup(budget)
root = roll.root

lines = [
    f"{budget.name}  [{budget.unit}]",
    f"  nominal              {root.nominal:8.4f}",
    f"  contingency          {root.allowance:8.4f}",
    f"  predicted            {root.predicted:8.4f}",
    f"  allocation           {root.allocation:8.4f}",
    f"  margin               {root.margin_absolute:8.4f}"
    f"   ({100 * root.margin_fraction:.2f} %)",
    f"  1 sigma, declared    {root.uncertainty:8.4f}   (rule: {root.combination})",
    f"  1 sigma, exact       {root.exact_uncertainty:8.4f}   (declared correlations)",
    f"  margin at {budget.k_sigma:g} sigma    {root.margin_at_k:8.4f}",
    "",
    "worst three leaves by effect on the top-level margin:",
]
for row in sensitivity(budget, step=0.10)[:3]:
    lines.append(
        f"  {row.name:<30s}{row.margin_delta:+8.4f} {budget.unit} at +10 %"
        f"   variance share {100 * row.variance_share:5.2f} %"
    )
lines.append("")
lines.append("blocking findings:")
for finding in run_checks(budget):
    if finding.severity == "error":
        lines.append(f"  {finding.code} {finding.subject}")
# --- README example ends -----------------------------------------------


def main() -> int:
    """Print the example output and record it."""
    out = Emitter(__file__)
    out("budgetroll validation: the worked example quoted in README.md")
    out("=" * 72)
    out()
    for line in lines:
        out(line)
    out()
    out.check("the example produced output", bool(lines))
    out.check(
        "the root margin is 0.4975 W",
        out.close(root.margin_absolute, 0.4975, 1.0e-9),
        f"{root.margin_absolute:.10f}",
    )
    return out.finish()


if __name__ == "__main__":
    raise SystemExit(main())
