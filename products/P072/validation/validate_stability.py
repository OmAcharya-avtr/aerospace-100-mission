"""Selection stability under perturbed value estimates.

A selector that flips on noise is unusable regardless of its optimality, so this
is measured and not asserted. For each relative noise level ``sigma`` the value
table is perturbed multiplicatively by ``exp(N(0, sigma))``, the selector is
re-run, and two quantities are recorded against the unperturbed selection: the
Jaccard similarity of the selected set, and the fraction of the unperturbed
value that the perturbed selection actually retains when scored against the
unperturbed values.

Reporting both is the point. A selection that swaps two products of equal value
loses Jaccard and retains all its value, so Jaccard alone overstates the harm;
a selection that keeps most of its membership but drops the single most valuable
product retains little value at a high Jaccard, so value alone hides the churn
an operator has to review.

Checks:

1. both selectors over a family of instances, at seven noise levels, with the
   mean, the 5th percentile and the minimum of each quantity;
2. the degenerate level ``sigma = 0``, at which both quantities must be exactly
   1.0 and the flip count exactly 0, because the selectors are deterministic;
3. the two selectors compared at the same noise level, because the exact solver
   is the one with more to lose: it sits at a vertex chosen by the value table
   and greedy only sorts by it;
4. the relationship between the gap greedy leaves and the value lost to noise.
   If one per cent of value noise costs more than the greedy gap, the optimality
   is not worth having on that instance family, and that comparison is the
   practical conclusion of this product.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, Timer, bootstrap, environment_lines

bootstrap()

from downlinkpick.dataset import generate_instance  # noqa: E402
from downlinkpick.exact import exact_select  # noqa: E402
from downlinkpick.greedy import greedy_select  # noqa: E402
from downlinkpick.selection import SelectionProblem, sizes_from_ratios  # noqa: E402
from downlinkpick.stability import jaccard, stability_curve  # noqa: E402

SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4)
N_INSTANCES = 24
N_TRIALS = 24
SEED = 20261021


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
    out = Tee("validate_stability_output.txt")
    out.rule("downlinkpick validation: selection stability under perturbed values")
    for line in environment_lines():
        out(line)
    failures = 0
    out("")
    out(f"instances   : {N_INSTANCES} (seeds 1..{N_INSTANCES}), 20 products, 3 windows,")
    out("              capacity 45 per cent of the queue")
    out(f"noise levels: {', '.join(f'{sigma:g}' for sigma in SIGMAS)}")
    out(f"trials      : {N_TRIALS} per instance per level, seed {SEED}")
    out("perturbation: v -> v * exp(N(0, sigma)), multiplicative and non-negative")

    timer = Timer()
    pooled: dict[str, dict[float, dict[str, list[float]]]] = {}
    for label, selector in (("greedy", greedy_select), ("exact", exact_select)):
        pooled[label] = {sigma: {"j": [], "v": [], "f": []} for sigma in SIGMAS}
        for seed in range(1, N_INSTANCES + 1):
            instance = generate_instance(
                seed, n_products=20, n_windows=3, capacity_fraction=0.45
            )
            problem = _problem(instance)
            with timer:
                curve = stability_curve(
                    problem, SIGMAS, selector=selector, n_trials=N_TRIALS, seed=SEED + seed
                )
            for point in curve:
                pooled[label][point.sigma]["j"].append(point.mean_jaccard)
                pooled[label][point.sigma]["v"].append(point.mean_value_retained)
                pooled[label][point.sigma]["f"].append(point.mean_flips)
                if point.sigma == 0.0:
                    if point.min_jaccard != 1.0 or point.mean_flips != 0.0:
                        failures += 1
                        out(
                            f"  FAIL {label} seed {seed}: sigma=0 gave Jaccard "
                            f"{point.min_jaccard!r} and {point.mean_flips!r} flips; the "
                            "selectors must be deterministic"
                        )
                    if abs(point.mean_value_retained - 1.0) > 1.0e-12:
                        failures += 1
                        out(
                            f"  FAIL {label} seed {seed}: sigma=0 retained "
                            f"{point.mean_value_retained!r} of the value"
                        )

    for label in ("greedy", "exact"):
        out.rule(f"1 and 2. {label} selector, pooled over {N_INSTANCES} instances")
        out(
            f"{'sigma':>7s} {'mean J':>9s} {'p05 J':>9s} {'min J':>9s} "
            f"{'mean kept':>10s} {'p05 kept':>9s} {'min kept':>9s} {'mean flips':>11s}"
        )
        for sigma in SIGMAS:
            similarity = np.asarray(pooled[label][sigma]["j"])
            retained = np.asarray(pooled[label][sigma]["v"])
            flips = np.asarray(pooled[label][sigma]["f"])
            out(
                f"{sigma:7.3f} {similarity.mean():9.5f} "
                f"{np.quantile(similarity, 0.05):9.5f} {similarity.min():9.5f} "
                f"{retained.mean():10.6f} {np.quantile(retained, 0.05):9.6f} "
                f"{retained.min():9.6f} {flips.mean():11.4f}"
            )

    out.rule("3. the two selectors at the same noise level")
    out("the exact solver sits at a vertex chosen by the value table; greedy only sorts")
    out("by it, so greedy is expected to be the more stable of the two in membership")
    out("")
    out(f"{'sigma':>7s} {'greedy J':>10s} {'exact J':>10s} {'greedy kept':>12s} "
        f"{'exact kept':>11s}")
    for sigma in SIGMAS:
        out(
            f"{sigma:7.3f} "
            f"{np.mean(pooled['greedy'][sigma]['j']):10.5f} "
            f"{np.mean(pooled['exact'][sigma]['j']):10.5f} "
            f"{np.mean(pooled['greedy'][sigma]['v']):12.6f} "
            f"{np.mean(pooled['exact'][sigma]['v']):11.6f}"
        )
    out("")
    out("greedy's retained fraction above 1.0 is not an error. Greedy's nominal selection")
    out("is not optimal, so a perturbation that reshuffles its density ordering can land")
    out("on a better selection than the unperturbed ordering did, and the fraction is")
    out("measured against greedy's own unperturbed value. The exact solver has no such")
    out("headroom: its unperturbed selection is the optimum, so its retained fraction")
    out("cannot exceed 1.0 and the amount by which it falls below is the cost of noise.")

    out.rule("4. the greedy gap against the value lost to value noise")
    out("if a given level of value noise costs the exact solver more than the gap greedy")
    out("leaves, the optimality is not worth having at that level of noise")
    out("")
    gaps: list[float] = []
    for seed in range(1, N_INSTANCES + 1):
        instance = generate_instance(seed, n_products=20, n_windows=3, capacity_fraction=0.45)
        problem = _problem(instance)
        optimum = exact_select(problem).total_value
        greedy = greedy_select(problem).total_value
        gaps.append((optimum - greedy) / optimum if optimum > 0.0 else 0.0)
    gap_mean = float(np.mean(gaps))
    out(f"mean greedy gap on this family : {gap_mean:.6e}")
    out("")
    out(f"{'sigma':>7s} {'exact value lost':>17s} {'greedy gap':>12s} {'verdict':>28s}")
    crossover: float | None = None
    for sigma in SIGMAS:
        lost = 1.0 - float(np.mean(pooled["exact"][sigma]["v"]))
        verdict = "optimality still pays" if lost < gap_mean else "noise costs more than the gap"
        if lost >= gap_mean and crossover is None:
            crossover = sigma
        out(f"{sigma:7.3f} {lost:17.6e} {gap_mean:12.6e} {verdict:>28s}")
    out("")
    if crossover is None:
        out("no crossover within the levels measured: the exact solver keeps its advantage")
        out(f"over greedy at every noise level up to sigma = {SIGMAS[-1]:g}")
    else:
        out(f"crossover between sigma = {crossover:g} and the level below it: at and above")
        out("that level the value the exact solver loses to value noise exceeds the mean")
        out("gap greedy leaves, so the optimality is not worth having")

    out.rule("5. the Jaccard identity on trivial arguments")
    out("jaccard(empty, empty) is defined as 1.0: nothing selected either time is")
    out("agreement, not disagreement")
    cases = (
        (frozenset(), frozenset(), 1.0),
        (frozenset({"a"}), frozenset({"a"}), 1.0),
        (frozenset({"a"}), frozenset(), 0.0),
        (frozenset({"a", "b"}), frozenset({"b", "c"}), 1.0 / 3.0),
    )
    for left, right, expected in cases:
        measured = jaccard(left, right)
        ok = abs(measured - expected) <= 1.0e-15
        out(f"  jaccard({sorted(left)}, {sorted(right)}) = {measured:.6f}, "
            f"expected {expected:.6f}, ok={ok}")
        if not ok:
            failures += 1
            out("  FAIL jaccard identity")

    out.rule("compute budget")
    out("timings are wall clock on two shared cores and are UNSTABLE between runs")
    out(f"{'quantity':>26s} {'value':>12s}")
    out(f"{'stability_curve calls':>26s} {timer.calls:12d}")
    out(f"{'selector invocations':>26s} "
        f"{timer.calls * len(SIGMAS) * N_TRIALS + timer.calls:12d}")
    out(f"{'total wall clock [s]':>26s} {timer.elapsed_s:12.3f}")

    out.rule("summary")
    out(f"failures: {failures}")
    out(f"output  : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
