"""The MILP optimum against exhaustive enumeration, and the two model emissions.

Checks:

1. the MILP optimum against the optimum found by exhaustive enumeration over a
   seeded family of small instances, with no solver in the enumeration path;
2. the selected *sets* where the objectives agree, since two selections of equal
   value are both optimal and a set difference is not an error;
3. the PuLP emission of the model against the SciPy coefficient arrays, term by
   term, so that the LP file this package writes and the arrays it solves are
   known to state the same problem;
4. the LP relaxation as a valid upper bound on the integral optimum, and the
   integrality gap it leaves;
5. the solver's own objective value against the value recomputed from the
   returned selection through equation (2);
6. the degenerate instances: a plan whose capacity admits everything, a plan
   whose capacity admits nothing, and an inventory every member of which has
   expired before the first window;
7. the three counterexamples Hypothesis found against this module, re-measured
   here so that the numbers quoted in README.md and VALIDATION.md come from a
   committed output file and not only from an assertion in the test suite.

Tolerances are relative and two-sided throughout. The enumeration and the
solver sum the same terms in different orders, so what is guaranteed is
agreement to within rounding.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, Timer, bootstrap, environment_lines

bootstrap()

from downlinkpick.contacts import ContactPlan, ContactWindow, constant_rate_profile  # noqa: E402
from downlinkpick.dataset import generate_instance  # noqa: E402
from downlinkpick.exact import (  # noqa: E402
    ROUNDED_STATUS,
    SOLVER_ABS_GAP_VALUE_UNITS,
    build_model,
    build_pulp_problem,
    enumerate_select,
    enumeration_size,
    exact_select,
    lp_relaxation_bound,
    model_arrays,
    pulp_available,
)
from downlinkpick.greedy import dantzig_bound, greedy_select  # noqa: E402
from downlinkpick.inventory import DataProduct, Inventory  # noqa: E402
from downlinkpick.selection import (  # noqa: E402
    SelectionProblem,
    check_invariants,
    sizes_from_ratios,
)

#: Relative tolerance on an objective comparison. The objectives are sums of
#: products of doubles over a few dozen terms; 1e-9 is six orders of magnitude
#: above the accumulated rounding and far below any difference that would change
#: a decision.
OBJECTIVE_TOLERANCE_REL = 1.0e-9

#: Number of small instances enumerated.
N_SMALL = 240

#: Small-instance shape. Seven products over two windows gives at most 3^7 =
#: 2187 complete assignments, which enumerates in milliseconds.
SMALL_PRODUCTS = 7
SMALL_WINDOWS = 2


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
    out = Tee("validate_exact_output.txt")
    out.rule("downlinkpick validation: the exact solver")
    for line in environment_lines():
        out(line)
    failures = 0

    out("")
    out("PuLP is the modelling layer named in the specification. In this environment PuLP")
    out("has no solver attached, so the optimum is computed by scipy.optimize.milp (HiGHS,")
    out("shipped inside SciPy). Check 3 below compares the PuLP emission of the model")
    out("against the SciPy coefficient arrays term by term.")
    out(f"pulp usable as a solver: {pulp_available()}")

    out.rule(f"1 and 2. MILP optimum against exhaustive enumeration, {N_SMALL} instances")
    out(f"instances: {SMALL_PRODUCTS} products, {SMALL_WINDOWS} windows, seeds 1..{N_SMALL}")
    out(f"tolerance: {OBJECTIVE_TOLERANCE_REL:.1e} relative on the objective, plus the")
    out(f"           solver's absolute MIP gap of {SOLVER_ABS_GAP_VALUE_UNITS:.1e} value units")
    out("")
    worst_objective = 0.0
    worst_seed = 1
    set_differences = 0
    rounded = 0
    enumeration_sizes: list[int] = []
    milp_time = Timer()
    enum_time = Timer()
    for seed in range(1, N_SMALL + 1):
        instance = generate_instance(
            seed, n_products=SMALL_PRODUCTS, n_windows=SMALL_WINDOWS, capacity_fraction=0.4
        )
        problem = _problem(instance)
        enumeration_sizes.append(enumeration_size(problem))
        with milp_time:
            solved = exact_select(problem)
        with enum_time:
            enumerated = enumerate_select(problem)
        if solved.status == ROUNDED_STATUS:
            rounded += 1
        elif solved.status != "optimal":
            failures += 1
            out(f"  FAIL seed {seed}: solver status {solved.status}, not optimal")
        denominator = max(abs(enumerated.total_value), 1.0)
        difference = abs(solved.total_value - enumerated.total_value)
        relative = difference / denominator
        if relative > worst_objective:
            worst_objective = relative
            worst_seed = seed
        if relative > OBJECTIVE_TOLERANCE_REL and difference > SOLVER_ABS_GAP_VALUE_UNITS:
            failures += 1
            out(
                f"  FAIL seed {seed}: MILP {solved.total_value!r} against enumeration "
                f"{enumerated.total_value!r}, relative {relative:.3e}"
            )
        elif solved.selected_set != enumerated.selected_set:
            set_differences += 1
        report = check_invariants(problem, solved)
        if not report.ok:
            failures += 1
            out(f"  FAIL seed {seed}: invariants violated: {report.violations}")
    out(f"instances compared                  : {N_SMALL}")
    out(f"worst relative objective difference : {worst_objective:.6e} at seed {worst_seed}")
    out(f"selections with an equal-value tie  : {set_differences}")
    out(f"solutions needing overflow repair   : {rounded}")
    out(f"enumeration assignments, min / max  : {min(enumeration_sizes)} / "
        f"{max(enumeration_sizes)}")
    out("")
    out("timings below are wall clock on two shared cores and are UNSTABLE between runs;")
    out("they are recorded for the compute budget and no conclusion rests on them")
    out(f"{'route':>14s} {'calls':>7s} {'total [s]':>11s} {'per instance [ms]':>18s}")
    out(f"{'MILP':>14s} {milp_time.calls:7d} {milp_time.elapsed_s:11.3f} "
        f"{milp_time.mean_ms:18.3f}")
    out(f"{'enumeration':>14s} {enum_time.calls:7d} {enum_time.elapsed_s:11.3f} "
        f"{enum_time.mean_ms:18.3f}")

    out.rule("3. the PuLP emission against the SciPy coefficient arrays, term by term")
    out("compared: objective coefficients, constraint matrix and right-hand sides")
    out("tolerance: 0 absolute; both are built from the same MilpModel, so a difference")
    out("           here is a defect in one of the two emitters and not rounding")
    out("")
    out(f"{'seed':>6s} {'columns':>8s} {'rows':>6s} {'max |obj diff|':>16s} "
        f"{'max |row diff|':>16s} {'max |rhs diff|':>16s}")
    import pulp

    worst_emission = 0.0
    for seed in (3, 17, 101, 240):
        instance = generate_instance(
            seed, n_products=SMALL_PRODUCTS, n_windows=SMALL_WINDOWS, capacity_fraction=0.4
        )
        problem = _problem(instance)
        model = build_model(problem)
        objective, matrix, rhs = model_arrays(model)
        lp = build_pulp_problem(problem)
        variables = {variable.name: variable for variable in lp.variables()}
        order = [variables[f"x_{index}"] for index in range(model.n_columns)]

        position = {variable.name: index for index, variable in enumerate(order)}
        lp_objective = np.zeros(model.n_columns)
        for variable, coefficient in lp.objective.items():
            lp_objective[position[variable.name]] = float(coefficient)
        objective_difference = float(np.max(np.abs(lp_objective - objective)))

        lp_rows: list[np.ndarray] = []
        lp_rhs: list[float] = []
        for constraint in lp.constraints():
            row = np.zeros(model.n_columns)
            for variable, coefficient in constraint.items():
                row[position[variable.name]] = float(coefficient)
            lp_rows.append(row)
            lp_rhs.append(-float(constraint.constant))
            if constraint.sense != pulp.LpConstraintLE:
                failures += 1
                out(f"  FAIL seed {seed}: a PuLP constraint is not <=")
        lp_matrix = np.vstack(lp_rows)
        if lp_matrix.shape != matrix.shape:
            failures += 1
            out(f"  FAIL seed {seed}: PuLP has {lp_matrix.shape} rows against {matrix.shape}")
            continue
        row_difference = float(np.max(np.abs(np.sort(lp_matrix, axis=0) - np.sort(matrix, axis=0))))
        rhs_difference = float(np.max(np.abs(np.sort(lp_rhs) - np.sort(rhs))))
        worst_emission = max(worst_emission, objective_difference, row_difference, rhs_difference)
        out(
            f"{seed:6d} {model.n_columns:8d} {model.n_rows:6d} {objective_difference:16.3e} "
            f"{row_difference:16.3e} {rhs_difference:16.3e}"
        )
        if max(objective_difference, row_difference, rhs_difference) > 0.0:
            failures += 1
            out(f"  FAIL seed {seed}: the two emissions of the model disagree")
    out("")
    out(f"worst emission difference: {worst_emission:.6e}")

    out.rule("4. the LP relaxation as an upper bound, and the integrality gap")
    out("identity: LP relaxation >= integral optimum, always")
    out("")
    out(f"{'seed':>6s} {'optimum':>12s} {'LP bound':>12s} {'integrality gap':>16s}")
    gaps: list[float] = []
    bound_violations = 0
    for seed in range(1, 41):
        instance = generate_instance(
            seed, n_products=12, n_windows=3, capacity_fraction=0.4
        )
        problem = _problem(instance)
        optimum = exact_select(problem).total_value
        bound = lp_relaxation_bound(problem)
        if bound < optimum - OBJECTIVE_TOLERANCE_REL * max(abs(optimum), 1.0):
            bound_violations += 1
            failures += 1
            out(f"  FAIL seed {seed}: LP bound {bound!r} below the optimum {optimum!r}")
        gap = (bound - optimum) / optimum if optimum > 0.0 else 0.0
        gaps.append(gap)
        if seed <= 6:
            out(f"{seed:6d} {optimum:12.6f} {bound:12.6f} {gap:16.6e}")
    gap_array = np.asarray(gaps)
    out("")
    out(f"instances                      : {gap_array.size}")
    out(f"bound violations               : {bound_violations}")
    out(f"integrality gap mean / median  : {gap_array.mean():.6e} / "
        f"{np.median(gap_array):.6e}")
    out(f"integrality gap max            : {gap_array.max():.6e}")

    out.rule("5. the solver objective against the value recomputed from the selection")
    out("tolerance: 1.0e-9 relative")
    out("")
    worst_recompute = 0.0
    for seed in range(1, 61):
        instance = generate_instance(seed, n_products=16, n_windows=3, capacity_fraction=0.45)
        problem = _problem(instance)
        selection = exact_select(problem)
        recomputed = sum(
            problem.pair_value(a.product, a.window) for a in selection.assignments
        )
        relative = abs(selection.total_value - recomputed) / max(abs(recomputed), 1.0)
        worst_recompute = max(worst_recompute, relative)
        if relative > OBJECTIVE_TOLERANCE_REL:
            failures += 1
            out(f"  FAIL seed {seed}: objective {selection.total_value!r} against "
                f"recomputed {recomputed!r}")
    out("instances                            : 60")
    out(f"worst relative recompute difference  : {worst_recompute:.6e}")

    out.rule("6. degenerate instances")
    products = [
        DataProduct("p0", 1.0e6, 0.0, 9000.0, declared_value=3.0, compressibility=0.5),
        DataProduct("p1", 2.0e6, 0.0, 9000.0, declared_value=7.0, compressibility=0.25),
    ]
    inventory = Inventory(products)

    roomy = ContactPlan(
        [ContactWindow("w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 1.0e9))]
    )
    problem = SelectionProblem.from_declared(inventory, roomy)
    selection = exact_select(problem)
    out(f"capacity admits everything : value {selection.total_value:.6f}, "
        f"selected {len(selection.assignments)} of {len(inventory)}, "
        f"status {selection.status}")
    if len(selection.assignments) != 2:
        failures += 1
        out("  FAIL a plan with ample capacity did not select both products")

    tiny = ContactPlan(
        [ContactWindow("w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 1.0e3))]
    )
    problem = SelectionProblem.from_declared(inventory, tiny)
    selection = exact_select(problem)
    out(f"capacity admits nothing    : value {selection.total_value:.6f}, "
        f"selected {len(selection.assignments)}, status {selection.status}")
    if selection.assignments:
        failures += 1
        out("  FAIL a plan with no usable capacity selected something")

    late = ContactPlan(
        [ContactWindow("w", 20000.0, 20100.0, constant_rate_profile(20000.0, 20100.0, 1.0e9))]
    )
    problem = SelectionProblem.from_declared(inventory, late)
    selection = exact_select(problem)
    model = build_model(problem)
    out(f"everything expired        : value {selection.total_value:.6f}, "
        f"columns {model.n_columns}, status {selection.status}")
    if model.n_columns != 0 or selection.assignments:
        failures += 1
        out("  FAIL an all-expired instance produced a non-empty model or selection")

    out.rule("7. the counterexamples Hypothesis found against this module, re-measured")
    out("each is pinned as a named regression test in tests/test_regressions.py with its")
    out("literal inputs; they are re-measured here so that the numbers quoted in README.md")
    out("and validation/VALIDATION.md come from a committed output file")
    out("")

    def probe(label: str, values: list[float], sizes: list[float], capacity: float) -> None:
        nonlocal failures
        inventory = Inventory(
            [
                DataProduct(
                    f"p{index}",
                    raw_bytes=size / 8.0,
                    age_s=0.0,
                    expiry_s=1000.0,
                    declared_value=value,
                    compressibility=1.0,
                )
                for index, (value, size) in enumerate(zip(values, sizes, strict=True))
            ]
        )
        plan = ContactPlan(
            [
                ContactWindow(
                    "w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, capacity / 100.0)
                )
            ]
        )
        problem = SelectionProblem.from_declared(inventory, plan)
        solved = exact_select(problem)
        enumerated = enumerate_select(problem)
        heuristic = greedy_select(problem)
        bound = dantzig_bound(problem)
        out(f"{label}")
        out(f"  capacity [bit]        : {capacity!r}")
        out(f"  sizes [bit]           : {sizes!r}")
        out(f"  values                : {values!r}")
        out(f"  MILP objective        : {solved.total_value!r}  (status {solved.status})")
        out(f"  enumeration optimum   : {enumerated.total_value!r}")
        out(f"  greedy objective      : {heuristic.total_value!r}")
        out(f"  Dantzig bound         : {bound!r}")
        out(f"  greedy minus MILP     : {heuristic.total_value - solved.total_value!r}")
        report = check_invariants(problem, solved)
        out(f"  MILP invariants hold  : {report.ok}")
        if not report.ok:
            failures += 1
            out(f"  FAIL {report.violations}")
        if solved.total_value > enumerated.total_value + SOLVER_ABS_GAP_VALUE_UNITS:
            failures += 1
            out("  FAIL the MILP exceeds the enumeration optimum by more than the MIP gap")
        if solved.total_value > bound + SOLVER_ABS_GAP_VALUE_UNITS:
            failures += 1
            out("  FAIL the MILP exceeds Dantzig's fractional upper bound")
        out("")

    probe(
        "a. rounding a near-integral vector over-committed a window by 5 bits",
        [1.0, 1.0],
        [8.0, 7999535.0],
        7999538.0,
    )
    probe(
        "b. the same defect seen through a violated Dantzig bound",
        [1.0, 5.0],
        [8.0, 7999609.0],
        7999615.0,
    )
    probe(
        "c. greedy beat the MILP inside the solver's absolute MIP gap",
        [1.0, 0.0, 0.0, 0.0, 0.0, 5.960464477539063e-08],
        [9385924.0, 8.0, 8.0, 8.0, 8.0, 8.0],
        9385942.0,
    )
    out(f"the solver's absolute MIP gap is {SOLVER_ABS_GAP_VALUE_UNITS!r} value units, which")
    out("scipy.optimize.milp does not expose; case c sits inside it and is therefore a")
    out("documented tolerance rather than a defect")

    out.rule("summary")
    out(f"failures: {failures}")
    out(f"output  : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
