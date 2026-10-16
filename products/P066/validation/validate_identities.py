"""Exhaustive and randomised checks of the algebraic identities.

The identities are exact in exact arithmetic; what is measured here is the
size of the floating-point disagreement over a large sample, so the
tolerances in the test suite and in README.md are stated from measurement
rather than from habit.

Identities checked
------------------
1. RSS <= linear for non-negative uncertainties (L2 norm never exceeds L1).
2. Associativity of independent aggregation: grouping before RSS does not
   change the result.
3. R = I reduces the quadratic form to RSS.
4. R = J (all ones) reduces the quadratic form to the linear sum.
5. The variance shares of the quadratic form sum to 1.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from _support import Emitter

from budgetroll import (
    Budget,
    Correlation,
    Item,
    MarginPolicy,
    Node,
    correlation_matrix,
    linear_rollup_uncertainty,
    quadratic_form,
    rollup,
    rss_rollup_uncertainty,
    variance_shares,
)

POLICY = MarginPolicy(
    name="v",
    description="validation policy",
    source="declared in validation/validate_identities.py",
    levels={"mature": 0.0},
)
SEED = 20261016
RANDOM_CASES = 5000
GRID = (0.0, 0.25, 1.0, 3.5, 12.0, 100.0)


def flat(us) -> Budget:
    children = tuple(
        Item(id=f"i{n}", name=f"i{n}", value=1.0, uncertainty=float(u), maturity="mature")
        for n, u in enumerate(us)
    )
    return Budget(
        id="b",
        name="B",
        unit="kg",
        quantity="mass",
        policy=POLICY,
        root=Node(id="root", name="root", combination="rss", children=children),
    )


def grouped(us, cut: int) -> Budget:
    def node(ident, members):
        return Node(
            id=ident,
            name=ident,
            combination="rss",
            children=tuple(
                Item(
                    id=f"{ident}{n}",
                    name=f"{ident}{n}",
                    value=1.0,
                    uncertainty=float(u),
                    maturity="mature",
                )
                for n, u in enumerate(members)
            ),
        )

    root = Node(
        id="root",
        name="root",
        combination="rss",
        children=(node("L", us[:cut]), node("R", us[cut:])),
    )
    return Budget(id="b", name="B", unit="kg", quantity="mass", policy=POLICY, root=root)


def main() -> int:
    """Run the sweeps and report the worst disagreements found."""
    out = Emitter(__file__)
    out("budgetroll validation: algebraic identities")
    out("=" * 72)
    out("Educational-grade (validation level 1). Not flight-qualified, not certified,")
    out("not approved for operational aerospace use.")
    out(f"grid values {GRID}; random cases {RANDOM_CASES}; seed {SEED}")
    out()

    exhaustive = [list(combo) for n in (1, 2, 3) for combo in itertools.product(GRID, repeat=n)]
    rng = np.random.default_rng(SEED)
    randomised = [
        list(rng.uniform(0.0, 1000.0, size=int(rng.integers(2, 13))))
        for _ in range(RANDOM_CASES)
    ]
    cases = exhaustive + randomised
    out(f"1. RSS <= linear over {len(cases)} cases "
        f"({len(exhaustive)} exhaustive, {len(randomised)} random)")
    violations = 0
    worst_slack = math.inf
    for us in cases:
        budget = flat(us)
        rss = rss_rollup_uncertainty(budget)
        linear = linear_rollup_uncertainty(budget)
        slack = linear - rss
        worst_slack = min(worst_slack, slack)
        if slack < -1.0e-9 * max(1.0, linear):
            violations += 1
    out(f"   smallest (linear - RSS) seen     {worst_slack:.3e}")
    out.check("no case violated RSS <= linear", violations == 0, f"{violations} violations")
    out()

    out(f"2. Associativity of independent aggregation over {len(cases)} cases")
    worst_rel = 0.0
    worst_example = None
    mismatches = 0
    for us in cases:
        if len(us) < 2:
            continue
        for cut in range(1, len(us)):
            a = rollup(grouped(us, cut)).root.uncertainty
            b = rss_rollup_uncertainty(flat(us))
            if b == 0.0:
                continue
            rel = abs(a - b) / b
            if rel > worst_rel:
                worst_rel = rel
                worst_example = (
                    [float(u) for u in us],
                    cut,
                    float(a),
                    float(b),
                )
            if rel > 1.0e-12:
                mismatches += 1
    out(f"   worst relative disagreement      {worst_rel:.3e}")
    if worst_example is not None:
        us_w, cut_w, a_w, b_w = worst_example
        out(f"   worst case, {len(us_w)} leaves split after {cut_w}")
        out("     u = [" + ", ".join(f"{u:.12g}" for u in us_w) + "]")
        out(f"     grouped {a_w:.15g}  flat {b_w:.15g}")
    out.check(
        "every grouping agrees with the flat RSS to 1e-12 relative",
        mismatches == 0,
        f"{mismatches} cases above 1e-12",
    )
    out()

    out(f"3. R = I reduces the quadratic form to RSS, over {len(cases)} cases")
    worst_rel = 0.0
    for us in cases:
        matrix = np.eye(len(us))
        a = quadratic_form(us, matrix)
        b = math.sqrt(math.fsum(u * u for u in us))
        if b > 0.0:
            worst_rel = max(worst_rel, abs(a - b) / b)
    out(f"   worst relative disagreement      {worst_rel:.3e}")
    out.check("R = I agrees with RSS to 1e-12 relative", worst_rel <= 1.0e-12, f"{worst_rel:.3e}")
    out()

    out(f"4. R = J reduces the quadratic form to the linear sum, over {len(cases)} cases")
    worst_rel = 0.0
    for us in cases:
        matrix = np.ones((len(us), len(us)))
        a = quadratic_form(us, matrix)
        b = math.fsum(us)
        if b > 0.0:
            worst_rel = max(worst_rel, abs(a - b) / b)
    out(f"   worst relative disagreement      {worst_rel:.3e}")
    out.check("R = J agrees with the linear sum to 1e-12 relative", worst_rel <= 1.0e-12,
              f"{worst_rel:.3e}")
    out()

    out(f"5. Variance shares sum to 1, over {len(cases)} cases with a declared rho")
    worst_abs = 0.0
    for us in cases:
        if len(us) < 2:
            continue
        rho = float(rng.uniform(-0.9, 0.9))
        budget = Budget(
            id="b",
            name="B",
            unit="kg",
            quantity="mass",
            policy=POLICY,
            root=flat(us).root,
            correlations=(Correlation("i0", "i1", rho, "swept by the validation script"),),
        )
        shares = variance_shares(budget)
        values = [v for v in shares.values() if v is not None]
        if not values:
            continue
        worst_abs = max(worst_abs, abs(math.fsum(values) - 1.0))
    out(f"   worst absolute drift from 1      {worst_abs:.3e}")
    out.check("the shares sum to 1 to 1e-9 absolute", worst_abs <= 1.0e-9, f"{worst_abs:.3e}")
    out()

    out("6. Two-sided reference points for the correlated form")
    for rho, expected, note in (
        (1.0, 7.0, "rho = 1 gives the linear sum 3 + 4"),
        (0.5, math.sqrt(37.0), "rho = 0.5 gives sqrt(9 + 16 + 12) = sqrt(37)"),
        (0.0, 5.0, "rho = 0 gives the RSS sqrt(9 + 16) = 5"),
        (-1.0, 1.0, "rho = -1 gives the absolute difference |3 - 4|"),
    ):
        matrix = correlation_matrix(["a", "b"], (Correlation("a", "b", rho, "r"),))
        value = quadratic_form([3.0, 4.0], matrix)
        out(f"   rho = {rho:+.1f}  computed {value:.12f}  expected {expected:.12f}")
        out.check(note, out.close(value, expected, 1.0e-9), f"{value:.12f}")
    return out.finish()


if __name__ == "__main__":
    raise SystemExit(main())
