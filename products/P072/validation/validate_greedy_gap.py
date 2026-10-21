"""The greedy optimality gap over a seeded instance family, as a distribution.

The gap of an instance is ``(optimum - greedy) / optimum``, dimensionless and
zero when greedy is optimal. A single mean gap is not a usable answer: a family
in which greedy is optimal nine times out of ten and catastrophic on the tenth
has the same mean as a family in which it is uniformly mediocre, and the two
call for different operational advice. The whole distribution is therefore
reported -- deciles, the 95th and 99th percentiles, the maximum, the count of
exact matches and a histogram -- together with the seed of the worst instance so
that it can be reproduced on its own.

Three families are measured, because the gap is a property of the regime and not
of the algorithm alone:

* **tight** -- capacity 45 per cent of the queue, 24 products, 3 windows. The
  regime this product exists for.
* **very tight** -- capacity 20 per cent of the queue. Fewer products fit, so
  the choice matters more.
* **loose** -- capacity 85 per cent of the queue. Most things fit and greedy
  should be near-optimal.

A fourth family varies the dependency density, because the bundle rollback in
:func:`downlinkpick.greedy.greedy_select` is the part of the baseline most
likely to lose value and the measurement should say by how much.

Every comparison uses a relative tolerance on both sides. The two objectives
are independent floating-point sums over the same terms in different orders, so
"greedy matched the optimum" means agreement to within rounding.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, Timer, bootstrap, environment_lines

bootstrap()

from downlinkpick.benchmark import gap_statistics  # noqa: E402
from downlinkpick.dataset import generate_instance  # noqa: E402
from downlinkpick.exact import (  # noqa: E402
    ROUNDED_STATUS,
    SOLVER_ABS_GAP_VALUE_UNITS,
    exact_select,
)
from downlinkpick.greedy import dantzig_bound, greedy_select  # noqa: E402
from downlinkpick.selection import (  # noqa: E402
    SelectionProblem,
    check_invariants,
    sizes_from_ratios,
)

#: Relative tolerance within which greedy counts as having matched the optimum.
OPTIMAL_TOLERANCE_REL = 1.0e-9

#: Instances per family.
N_INSTANCES = 300

FAMILIES = (
    ("tight (capacity 45% of queue)", dict(n_products=24, n_windows=3, capacity_fraction=0.45)),
    ("very tight (capacity 20%)", dict(n_products=24, n_windows=3, capacity_fraction=0.20)),
    ("loose (capacity 85%)", dict(n_products=24, n_windows=3, capacity_fraction=0.85)),
    (
        "tight, no dependencies",
        dict(n_products=24, n_windows=3, capacity_fraction=0.45, dependency_fraction=0.0),
    ),
    (
        "tight, dense dependencies",
        dict(n_products=24, n_windows=3, capacity_fraction=0.45, dependency_fraction=0.6),
    ),
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
    out = Tee("validate_greedy_gap_output.txt")
    out.rule("downlinkpick validation: the greedy optimality gap, as a distribution")
    for line in environment_lines():
        out(line)
    failures = 0
    out("")
    out(f"instances per family: {N_INSTANCES}, seeds 1..{N_INSTANCES}")
    out("gap definition      : (optimum - greedy) / optimum, relative")
    out(f"matched-optimum tol : {OPTIMAL_TOLERANCE_REL:.1e} relative")

    greedy_timer = Timer()
    exact_timer = Timer()
    summary: list[tuple[str, object]] = []
    rounded_total = 0

    for label, kwargs in FAMILIES:
        gaps: list[float] = []
        seeds: list[int] = []
        negative = 0
        for seed in range(1, N_INSTANCES + 1):
            instance = generate_instance(seed, **kwargs)
            problem = _problem(instance)
            with greedy_timer:
                greedy = greedy_select(problem)
            with exact_timer:
                exact = exact_select(problem)
            if exact.status == ROUNDED_STATUS:
                rounded_total += 1
            elif exact.status != "optimal":
                failures += 1
                out(f"  FAIL {label} seed {seed}: solver status {exact.status}")
                continue
            for selection in (greedy, exact):
                report = check_invariants(problem, selection)
                if not report.ok:
                    failures += 1
                    out(
                        f"  FAIL {label} seed {seed}: {selection.method} violated "
                        f"{report.violations}"
                    )
            if exact.total_value <= 0.0:
                gaps.append(0.0)
                seeds.append(seed)
                continue
            gap = (exact.total_value - greedy.total_value) / exact.total_value
            excess = greedy.total_value - exact.total_value
            if gap < -OPTIMAL_TOLERANCE_REL and excess > SOLVER_ABS_GAP_VALUE_UNITS:
                negative += 1
                failures += 1
                out(
                    f"  FAIL {label} seed {seed}: greedy {greedy.total_value!r} exceeds the "
                    f"proved optimum {exact.total_value!r}, gap {gap:.3e}"
                )
            gaps.append(max(gap, 0.0))
            seeds.append(seed)

        statistics = gap_statistics(
            gaps, seeds, optimal_tolerance=OPTIMAL_TOLERANCE_REL
        )
        summary.append((label, statistics))
        out.rule(f"family: {label}")
        out(f"instances                    : {statistics.n_instances}")
        out(f"greedy matched the optimum   : {statistics.n_optimal} "
            f"({100.0 * statistics.n_optimal / statistics.n_instances:.2f} per cent)")
        out(f"greedy above the optimum     : {negative} (must be 0)")
        out(f"mean gap                     : {statistics.mean:.6e}")
        out(f"median gap                   : {statistics.median:.6e}")
        for level, value in statistics.quantiles:
            out(f"gap quantile q{level:<5.2f}          : {value:.6e}")
        out(f"maximum gap                  : {statistics.maximum:.6e} "
            f"at seed {statistics.worst_seed}")
        out("")
        out("histogram of the gap")
        out(f"{'lower':>12s} {'upper':>12s} {'count':>7s} {'fraction':>9s}")
        for index, count in enumerate(statistics.histogram_counts):
            lower = statistics.histogram_edges[index]
            upper = statistics.histogram_edges[index + 1]
            out(
                f"{lower:12.4e} {upper:12.4e} {count:7d} "
                f"{count / statistics.n_instances:9.4f}"
            )

    out.rule("families side by side")
    out(f"{'family':>32s} {'matched':>8s} {'mean':>11s} {'median':>11s} "
        f"{'q0.95':>11s} {'max':>11s}")
    for label, statistics in summary:
        quantiles = dict(statistics.quantiles)  # type: ignore[attr-defined]
        out(
            f"{label:>32s} {statistics.n_optimal:8d} {statistics.mean:11.4e} "  # type: ignore[attr-defined]
            f"{statistics.median:11.4e} {quantiles[0.95]:11.4e} "  # type: ignore[attr-defined]
            f"{statistics.maximum:11.4e}"  # type: ignore[attr-defined]
        )

    out.rule("solver status over every family")
    out(f"solutions that needed overflow repair: {rounded_total} of "
        f"{len(FAMILIES) * N_INSTANCES}")
    out("a rounded solution is one where HiGHS returned a vector integral only to within")
    out("its own tolerance and rounding it up over-committed a window; the repair drops")
    out("the least valuable products until it fits, and the status records it. On")
    out("realistic instances, whose product sizes differ by millions of bits, this does")
    out("not arise; it was found by a property test at a 5-bit margin.")

    out.rule("the Dantzig fractional bound against the proved optimum")
    out("Dantzig's bound ignores the per-window capacity split and the precedence rows,")
    out("so it is a valid upper bound and a loose one. Measured looseness below.")
    out("")
    out(f"{'seed':>6s} {'optimum':>12s} {'Dantzig':>12s} {'excess':>12s}")
    excesses: list[float] = []
    violations = 0
    for seed in range(1, 61):
        instance = generate_instance(seed, n_products=24, n_windows=3, capacity_fraction=0.45)
        problem = _problem(instance)
        optimum = exact_select(problem).total_value
        bound = dantzig_bound(problem)
        if (
            bound < optimum - OPTIMAL_TOLERANCE_REL * max(abs(optimum), 1.0)
            and optimum - bound > SOLVER_ABS_GAP_VALUE_UNITS
        ):
            violations += 1
            failures += 1
            out(f"  FAIL seed {seed}: Dantzig bound {bound!r} below the optimum {optimum!r}")
        excess = (bound - optimum) / optimum if optimum > 0.0 else 0.0
        excesses.append(excess)
        if seed <= 6:
            out(f"{seed:6d} {optimum:12.6f} {bound:12.6f} {excess:12.6e}")
    excess_array = np.asarray(excesses)
    out("")
    out(f"instances            : {excess_array.size}")
    out(f"bound violations     : {violations}")
    out(f"excess mean / median : {excess_array.mean():.6e} / {np.median(excess_array):.6e}")
    out(f"excess min / max     : {excess_array.min():.6e} / {excess_array.max():.6e}")

    out.rule("compute budget")
    out("timings are wall clock on two shared cores and are UNSTABLE between runs;")
    out("they are recorded for the compute budget and no conclusion rests on them")
    out(f"{'route':>14s} {'calls':>7s} {'total [s]':>11s} {'mean [ms]':>11s}")
    out(f"{'greedy':>14s} {greedy_timer.calls:7d} {greedy_timer.elapsed_s:11.3f} "
        f"{greedy_timer.mean_ms:11.3f}")
    out(f"{'exact':>14s} {exact_timer.calls:7d} {exact_timer.elapsed_s:11.3f} "
        f"{exact_timer.mean_ms:11.3f}")

    out.rule("summary")
    out(f"failures: {failures}")
    out(f"output  : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
