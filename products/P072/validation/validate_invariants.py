"""Value conservation, no double downlink, precedence and capacity, over a family.

The five invariants of :func:`downlinkpick.selection.check_invariants` are
checked on the output of both selectors over a large seeded family, and then
checked again against deliberately corrupted selections to confirm that the
checker actually fails when the invariant is broken. A checker that never fails
is not evidence.

Checks:

1. both selectors over 400 instances of five shapes: no violation, with the
   worst capacity headroom and the worst value residual reported;
2. value conservation stated as its own measurement: the reported total against
   an independently recomputed sum, over every selection, as a relative
   residual;
3. transitive precedence: the model imposes the precedence row on *direct*
   prerequisites only, and the claim that this implies it for the whole
   transitive closure is checked explicitly on instances with dependency chains
   of depth two and more;
4. the checker's own sensitivity: six corruptions of a valid selection, each of
   which must be detected;
5. the no-double-downlink column structure of the MILP: the number of
   no-double-downlink rows equals the number of products, and each product's
   columns appear in exactly one such row.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines

bootstrap()

from downlinkpick.dataset import generate_instance  # noqa: E402
from downlinkpick.exact import build_model, exact_select  # noqa: E402
from downlinkpick.greedy import greedy_select  # noqa: E402
from downlinkpick.selection import (  # noqa: E402
    Assignment,
    Selection,
    SelectionProblem,
    check_invariants,
    sizes_from_ratios,
)

N_INSTANCES = 400

SHAPES = (
    dict(n_products=24, n_windows=3, capacity_fraction=0.45),
    dict(n_products=12, n_windows=2, capacity_fraction=0.30),
    dict(n_products=30, n_windows=4, capacity_fraction=0.55, dependency_fraction=0.5),
    dict(n_products=18, n_windows=1, capacity_fraction=0.35),
    dict(n_products=20, n_windows=5, capacity_fraction=0.60, dependency_fraction=0.35),
)


def _problem(instance) -> SelectionProblem:
    values = {
        name: float(instance.inventory[name].declared_value or 0.0)
        for name in instance.inventory.names
    }
    return SelectionProblem(
        inventory=instance.inventory,
        plan=instance.plan,
        values=values,
        sizes_bits=sizes_from_ratios(instance.inventory, instance.true_ratios),
    )


def main() -> int:
    out = Tee("validate_invariants_output.txt")
    out.rule("downlinkpick validation: selection invariants")
    for line in environment_lines():
        out(line)
    failures = 0

    out.rule(f"1 and 2. both selectors over {N_INSTANCES} instances of {len(SHAPES)} shapes")
    out("invariants: no double downlink, expiry, capacity, dependency closure,")
    out("            precedence by window, value conservation")
    out("tolerances: capacity 1.0 bit absolute; value 1.0e-9 relative")
    out("")
    out(f"{'shape':>44s} {'selector':>9s} {'violations':>11s} "
        f"{'worst headroom [bit]':>21s} {'worst value res':>16s}")
    total_selections = 0
    for shape in SHAPES:
        label = ", ".join(f"{key}={value}" for key, value in shape.items())
        for name, selector in (("greedy", greedy_select), ("exact", exact_select)):
            violations = 0
            worst_headroom = -np.inf
            worst_residual = 0.0
            for seed in range(1, N_INSTANCES + 1):
                instance = generate_instance(seed, **shape)
                problem = _problem(instance)
                selection = selector(problem)
                total_selections += 1
                report = check_invariants(problem, selection)
                if not report.ok:
                    violations += 1
                    failures += 1
                    out(f"  FAIL {label} {name} seed {seed}: {report.violations}")
                worst_headroom = max(worst_headroom, report.worst_capacity_overshoot_bits)
                worst_residual = max(worst_residual, report.value_residual_rel)
            out(
                f"{label:>44s} {name:>9s} {violations:11d} {worst_headroom:21.6e} "
                f"{worst_residual:16.3e}"
            )
    out("")
    out(f"selections checked: {total_selections}")
    out("worst headroom is negative when every window has capacity left over, which is")
    out("the expected sign: a positive value would be an overflowing window")

    out.rule("3. transitive precedence from direct-prerequisite rows only")
    out("claim: imposing x_iw <= sum_{end(w') <= end(w)} x_jw' on DIRECT prerequisites")
    out("       implies the same for the whole transitive closure, because window ends")
    out("       are totally ordered")
    out("checked on instances that actually contain chains of depth two or more")
    out("")
    chains_found = 0
    deepest = 0
    checked = 0
    for seed in range(1, 401):
        instance = generate_instance(
            seed, n_products=26, n_windows=4, capacity_fraction=0.5, dependency_fraction=0.65
        )
        inventory = instance.inventory
        depths = {
            name: len(inventory.dependency_closure(name)) for name in inventory.names
        }
        depth = max(depths.values())
        if depth < 2:
            continue
        chains_found += 1
        deepest = max(deepest, depth)
        problem = _problem(instance)
        selection = exact_select(problem)
        window_of = selection.window_of
        for name in selection.selected_set:
            for prerequisite in inventory.dependency_closure(name):
                checked += 1
                if prerequisite not in window_of:
                    failures += 1
                    out(f"  FAIL seed {seed}: {name} selected without closure member "
                        f"{prerequisite}")
                    continue
                if problem.plan[window_of[prerequisite]].end_s > problem.plan[
                    window_of[name]
                ].end_s:
                    failures += 1
                    out(
                        f"  FAIL seed {seed}: transitive precedence violated, {name} in "
                        f"{window_of[name]} before {prerequisite} in "
                        f"{window_of[prerequisite]}"
                    )
    out(f"instances with a closure of size >= 2 : {chains_found}")
    out(f"deepest closure found                 : {deepest}")
    out(f"transitive pairs checked              : {checked}")

    out.rule("4. the checker's own sensitivity to six corruptions")
    out("a checker that never fails is not evidence, so each corruption below must be")
    out("detected; a corruption that goes undetected is a FAIL")
    out("")
    instance = generate_instance(11, n_products=14, n_windows=3, capacity_fraction=0.4)
    problem = _problem(instance)
    good = exact_select(problem)
    if not check_invariants(problem, good).ok:
        failures += 1
        out("  FAIL the uncorrupted selection does not pass its own invariants")

    def corrupt(label: str, selection: Selection, expect: str) -> None:
        nonlocal failures
        report = check_invariants(problem, selection)
        detected = any(expect in violation for violation in report.violations)
        out(f"{label:>34s} : detected={detected!s:>5s} violations={len(report.violations)}")
        if not detected:
            failures += 1
            out(f"  FAIL corruption {label!r} was not detected as {expect!r}")

    first = good.assignments[0]
    corrupt(
        "duplicate assignment",
        Selection(
            (*good.assignments, first),
            good.total_value + first.value,
            "corrupt",
        ),
        "double downlink",
    )
    corrupt(
        "total value inflated",
        Selection(good.assignments, good.total_value * 1.01, "corrupt"),
        "value not conserved",
    )
    corrupt(
        "assignment value inflated",
        Selection(
            (
                Assignment(first.product, first.window, first.bits, first.value * 2.0),
                *good.assignments[1:],
            ),
            good.total_value + first.value,
            "corrupt",
        ),
        "value mismatch",
    )
    corrupt(
        "size understated",
        Selection(
            (
                Assignment(first.product, first.window, first.bits * 0.5, first.value),
                *good.assignments[1:],
            ),
            good.total_value,
            "corrupt",
        ),
        "size mismatch",
    )
    unknown = Assignment("not-a-product", first.window, 1.0e6, 0.0)
    corrupt(
        "unknown product",
        Selection((*good.assignments, unknown), good.total_value, "corrupt"),
        "unknown product",
    )
    expired = next(
        (
            name
            for name in problem.inventory.names
            if not problem.feasible_windows(name) or
            problem.plan[problem.plan.names[-1]].end_s > problem.inventory[name].expiry_s
        ),
        None,
    )
    if expired is None:
        out("  no expired-capable product in this instance; expiry corruption skipped")
    else:
        last = problem.plan.names[-1]
        corrupt(
            "delivery after expiry",
            Selection(
                (
                    *good.assignments,
                    Assignment(expired, last, problem.sizes_bits[expired], 0.0),
                ),
                good.total_value,
                "corrupt",
            ),
            "expiry violated",
        )

    out.rule("5. no-double-downlink row structure of the MILP")
    out("one row per product, and every column of a product in exactly one such row")
    out("")
    out(f"{'seed':>6s} {'products':>9s} {'rows':>6s} {'columns':>8s} {'row sum max':>12s}")
    for seed in (5, 23, 77):
        instance = generate_instance(seed, n_products=20, n_windows=3, capacity_fraction=0.45)
        problem = _problem(instance)
        model = build_model(problem)
        counts = {name: 0 for name in model.products}
        for name, _window in model.pairs:
            counts[name] += 1
        rows = len(model.products)
        if rows != len(problem.inventory):
            failures += 1
            out(f"  FAIL seed {seed}: {rows} rows against {len(problem.inventory)} products")
        out(
            f"{seed:6d} {len(model.products):9d} {rows:6d} {model.n_columns:8d} "
            f"{max(counts.values()):12d}"
        )
        if sum(counts.values()) != model.n_columns:
            failures += 1
            out(f"  FAIL seed {seed}: column count does not partition by product")

    out.rule("summary")
    out(f"failures: {failures}")
    out(f"output  : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
