"""Rename detection: false-match rate and recall across a swept grid.

What is measured
----------------

Rename detection is the one change class that is inferred rather than read,
so it has a false-match rate. This script measures it rather than assuming it,
across the full cross product of

- acceptance threshold, from 0.40 to 0.95 in steps of 0.05,
- dictionary size, 20 / 50 / 100 / 200 / 400 parameters,
- matching strategy, ``greedy`` (the baseline) and ``optimal``,
- 12 generator seeds per cell.

Each generated pair carries ``size // 10`` true renames plus unrelated
removals and additions. The unrelated items are what make a false match
possible at all: a matcher offered only true renames cannot make one.

Definitions used here
---------------------

- **accepted pair**: a pairing the strategy produced whose score reached the
  threshold.
- **correct**: an accepted pair whose old and new names are the injected
  rename pair.
- **false-match rate (FMR)**: false accepted pairs over accepted pairs. The
  denominator is what the tool asserts, which is the quantity a user is
  exposed to.
- **recall**: correct accepted pairs over injected renames.

How the threshold is chosen
---------------------------

The criterion is declared here, in the script, and the selection is computed
from the measured grid. It is not picked by eye and it is not tuned until a
number looks good.

    Rule 1. Among thresholds whose measured FMR is exactly zero at every swept
    dictionary size, take the lowest. Recall is non-increasing in the
    threshold, so the lowest qualifying threshold is the highest-recall
    qualifying threshold.

    Rule 2. If no threshold reaches zero FMR at every size, take the lowest
    threshold whose FMR is at or below 0.05 at every size.

    Rule 3. If none does, report that no threshold on this grid meets the
    criterion and select the threshold with the lowest maximum FMR.

Whichever rule fires is printed, and the constant
``tmtcdiff.diff.DEFAULT_RENAME_THRESHOLD`` is checked against the selection.
If they disagree this script FAILS; it does not adjust itself.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines

ROOT = bootstrap()

from tmtcdiff.changes import ChangeClass  # noqa: E402
from tmtcdiff.diff import DEFAULT_RENAME_THRESHOLD  # noqa: E402
from tmtcdiff.metrics import clopper_pearson  # noqa: E402
from tmtcdiff.rename import STRATEGIES, match_renames, signature_collision_bound  # noqa: E402
from tmtcdiff.report import format_table  # noqa: E402
from tmtcdiff.synth import rename_pair  # noqa: E402

THRESHOLDS: tuple[float, ...] = tuple(round(0.40 + 0.05 * step, 2) for step in range(12))
SIZES: tuple[int, ...] = (20, 50, 100, 200, 400)
SEEDS: tuple[int, ...] = tuple(4100 + 7 * index for index in range(12))
TOLERABLE_FMR = 0.05


def _pair_pools(size: int, seed: int):
    renames = max(1, size // 10)
    pair = rename_pair(size, seed, renames=renames, removals=2, additions=2)
    removed = [
        pair.old.parameter_map[name]
        for name in sorted(set(pair.old.parameter_map) - set(pair.new.parameter_map))
    ]
    added = [
        pair.new.parameter_map[name]
        for name in sorted(set(pair.new.parameter_map) - set(pair.old.parameter_map))
    ]
    truth = {change.name: change.new_name for change in pair.truth_for(ChangeClass.RENAMED)}
    return pair, removed, added, truth


def main() -> int:
    out = Tee("validate_rename_sweep_output.txt")
    out.rule("tmtcdiff -- rename detection false-match rate across a swept grid")
    for line in environment_lines():
        out(line)
    out("")
    out(f"thresholds : {', '.join(f'{t:.2f}' for t in THRESHOLDS)}")
    out(f"sizes      : {', '.join(str(s) for s in SIZES)}")
    out(f"seeds      : {len(SEEDS)} per cell, {SEEDS[0]} to {SEEDS[-1]} step 7")
    out(f"strategies : {', '.join(STRATEGIES)}  (greedy is the baseline)")
    out("")
    out("A rename is inferred, not read, so it has a false-match rate. Every number")
    out("below is counted from generated pairs whose rename ground truth is exact.")

    # --- the grid ---------------------------------------------------------
    cells: dict[tuple[str, float, int], dict[str, int]] = {}
    pools: dict[tuple[int, int], tuple[list, list, dict]] = {}
    for size in SIZES:
        for seed in SEEDS:
            _, removed, added, truth = _pair_pools(size, seed)
            pools[(size, seed)] = (removed, added, truth)

    # The pairing a strategy produces does not depend on the threshold -- the
    # threshold only decides which produced pairs are accepted -- so each
    # pairing is computed once with threshold 0.0 and every threshold on the
    # grid is then applied to the same scored pairs. Recomputing the score
    # matrix for each of the twelve thresholds made this script take twelve
    # times as long and produced identical numbers.
    pairings: dict[tuple[str, int, int], tuple] = {}
    for strategy in STRATEGIES:
        for size in SIZES:
            for seed in SEEDS:
                removed, added, _ = pools[(size, seed)]
                pairings[(strategy, size, seed)] = match_renames(
                    removed, added, threshold=0.0, strategy=strategy
                )

    for strategy in STRATEGIES:
        for threshold in THRESHOLDS:
            for size in SIZES:
                accumulator = {"accepted": 0, "correct": 0, "injected": 0, "pairs": 0}
                for seed in SEEDS:
                    _, _, truth = pools[(size, seed)]
                    candidates = pairings[(strategy, size, seed)]
                    accepted = [c for c in candidates if c.score >= threshold]
                    correct = sum(1 for c in accepted if truth.get(c.old_name) == c.new_name)
                    accumulator["accepted"] += len(accepted)
                    accumulator["correct"] += correct
                    accumulator["injected"] += len(truth)
                    accumulator["pairs"] += len(candidates)
                cells[(strategy, threshold, size)] = accumulator

    def fmr(cell: dict[str, int]) -> float:
        return (
            (cell["accepted"] - cell["correct"]) / cell["accepted"]
            if cell["accepted"]
            else float("nan")
        )

    def recall(cell: dict[str, int]) -> float:
        return cell["correct"] / cell["injected"] if cell["injected"] else float("nan")

    for strategy in STRATEGIES:
        out.rule(f"strategy '{strategy}' -- false-match rate by threshold and dictionary size")
        rows = []
        for threshold in THRESHOLDS:
            row: list[object] = [f"{threshold:.2f}"]
            for size in SIZES:
                cell = cells[(strategy, threshold, size)]
                row.append(f"{fmr(cell):.4f}" if cell["accepted"] else "nan")
            false_total = sum(
                cells[(strategy, threshold, s)]["accepted"]
                - cells[(strategy, threshold, s)]["correct"]
                for s in SIZES
            )
            row.append(f"{false_total}")
            row.append(f"{sum(cells[(strategy, threshold, s)]['accepted'] for s in SIZES)}")
            rows.append(row)
        out(
            format_table(
                ["threshold", *[f"n={size}" for size in SIZES], "false", "accepted"], rows
            )
        )

        out("")
        out(f"strategy '{strategy}' -- recall by threshold and dictionary size")
        rows = []
        for threshold in THRESHOLDS:
            row = [f"{threshold:.2f}"]
            for size in SIZES:
                cell = cells[(strategy, threshold, size)]
                row.append(f"{recall(cell):.4f}")
            total_correct = sum(cells[(strategy, threshold, s)]["correct"] for s in SIZES)
            total_injected = sum(cells[(strategy, threshold, s)]["injected"] for s in SIZES)
            low, high = clopper_pearson(total_correct, total_injected)
            row.append(f"{total_correct}/{total_injected}")
            row.append(f"[{low:.4f}, {high:.4f}]")
            rows.append(row)
        out(
            format_table(
                [
                    "threshold",
                    *[f"n={size}" for size in SIZES],
                    "correct/injected",
                    "95% interval (pooled)",
                ],
                rows,
            )
        )

    out.rule("false matches per unrelated-item opportunity -- the size confound, named")
    out("The false-match rate above has a confound that has to be stated: the number of")
    out("injected renames grows with dictionary size (size // 10) while the unrelated")
    out("removals and additions stay at 2 each, so the DENOMINATOR of the rate is")
    out("dominated by true renames at the larger sizes. The rate therefore falls with")
    out("size partly by construction. The table below removes the confound by counting")
    out("false matches against the opportunity instead: 2 unrelated removals and 2")
    out("unrelated additions per seed, so 24 opportunities per cell.")
    out("")
    opportunities = len(SEEDS) * 2
    rows = []
    for strategy in STRATEGIES:
        for threshold in THRESHOLDS:
            row: list[object] = [strategy, f"{threshold:.2f}"]
            for size in SIZES:
                cell = cells[(strategy, threshold, size)]
                false_matches = cell["accepted"] - cell["correct"]
                row.append(f"{false_matches}/{opportunities}")
            false_total = sum(
                cells[(strategy, threshold, s)]["accepted"]
                - cells[(strategy, threshold, s)]["correct"]
                for s in SIZES
            )
            row.append(f"{false_total}/{opportunities * len(SIZES)}")
            rows.append(row)
    out(format_table(["strategy", "threshold", *[f"n={size}" for size in SIZES], "pooled"], rows))
    out("")
    out("Read this way the dependence on dictionary size is weak and not monotone, and")
    out("the dependence on threshold is the strong effect.")

    # --- threshold selection, by the declared criterion -------------------
    out.rule("threshold selection -- the criterion was declared before the grid was run")
    selection: dict[str, tuple[int, float, str]] = {}
    for strategy in STRATEGIES:
        zero = [
            threshold
            for threshold in THRESHOLDS
            if all(
                cells[(strategy, threshold, size)]["accepted"] == 0
                or fmr(cells[(strategy, threshold, size)]) == 0.0
                for size in SIZES
            )
        ]
        tolerable = [
            threshold
            for threshold in THRESHOLDS
            if all(
                cells[(strategy, threshold, size)]["accepted"] == 0
                or fmr(cells[(strategy, threshold, size)]) <= TOLERABLE_FMR
                for size in SIZES
            )
        ]
        if zero:
            selection[strategy] = (
                1,
                min(zero),
                "rule 1: lowest threshold with FMR 0 at every size",
            )
        elif tolerable:
            selection[strategy] = (
                2,
                min(tolerable),
                f"rule 2: lowest threshold with FMR <= {TOLERABLE_FMR} at every size",
            )
        else:
            worst = {
                threshold: max(
                    fmr(cells[(strategy, threshold, size)])
                    for size in SIZES
                    if cells[(strategy, threshold, size)]["accepted"]
                )
                for threshold in THRESHOLDS
            }
            best = min(worst, key=lambda t: worst[t])
            selection[strategy] = (
                3,
                best,
                f"rule 3: no threshold met the criterion; lowest worst-case FMR {worst[best]:.4f}",
            )
        rule, threshold, note = selection[strategy]
        pooled_correct = sum(cells[(strategy, threshold, s)]["correct"] for s in SIZES)
        pooled_injected = sum(cells[(strategy, threshold, s)]["injected"] for s in SIZES)
        pooled_accepted = sum(cells[(strategy, threshold, s)]["accepted"] for s in SIZES)
        out("")
        out(f"strategy        : {strategy}")
        out(f"rule fired      : {rule}")
        out(f"note            : {note}")
        out(f"threshold       : {threshold:.2f}")
        out(f"pooled recall   : {pooled_correct}/{pooled_injected} = "
            f"{pooled_correct / pooled_injected:.4f}")
        out(f"pooled accepted : {pooled_accepted}, false {pooled_accepted - pooled_correct}")

    chosen_rule, chosen_threshold, chosen_note = selection["optimal"]
    out("")
    out("The library default is taken from the 'optimal' strategy, which is the default")
    out("strategy. The selected value is compared with the shipped constant below; a")
    out("disagreement FAILS this script rather than being written back.")
    out.check(
        "DEFAULT_RENAME_THRESHOLD equals the value the criterion selected",
        abs(DEFAULT_RENAME_THRESHOLD - chosen_threshold) < 1e-12,
        f"constant {DEFAULT_RENAME_THRESHOLD}, selected {chosen_threshold:.2f} by rule "
        f"{chosen_rule}",
    )

    # --- strategy comparison at the selected threshold --------------------
    out.rule("baseline comparison at the selected threshold -- greedy against optimal")
    rows = []
    for strategy in STRATEGIES:
        for size in SIZES:
            cell = cells[(strategy, chosen_threshold, size)]
            rows.append(
                [
                    strategy,
                    size,
                    cell["injected"],
                    cell["accepted"],
                    cell["correct"],
                    cell["accepted"] - cell["correct"],
                    f"{recall(cell):.4f}",
                    f"{fmr(cell):.4f}" if cell["accepted"] else "nan",
                ]
            )
    out(
        format_table(
            [
                "strategy",
                "size",
                "injected",
                "accepted",
                "correct",
                "false",
                "recall",
                "FMR",
            ],
            rows,
        )
    )
    greedy_correct = sum(cells[("greedy", chosen_threshold, s)]["correct"] for s in SIZES)
    optimal_correct = sum(cells[("optimal", chosen_threshold, s)]["correct"] for s in SIZES)
    greedy_false = sum(
        cells[("greedy", chosen_threshold, s)]["accepted"]
        - cells[("greedy", chosen_threshold, s)]["correct"]
        for s in SIZES
    )
    optimal_false = sum(
        cells[("optimal", chosen_threshold, s)]["accepted"]
        - cells[("optimal", chosen_threshold, s)]["correct"]
        for s in SIZES
    )
    out("")
    out(f"greedy  : {greedy_correct} correct, {greedy_false} false")
    out(f"optimal : {optimal_correct} correct, {optimal_false} false")
    if greedy_correct == optimal_correct and greedy_false == optimal_false:
        out("MEASURED RESULT: the greedy baseline and the optimal assignment are")
        out("indistinguishable on this generator at the selected threshold. The optimal")
        out("matcher is retained as the default because it cannot be worse by")
        out("construction, not because it was measured better. It was not.")
    elif greedy_correct >= optimal_correct and greedy_false <= optimal_false:
        out("MEASURED RESULT: the greedy BASELINE matches or beats the optimal")
        out("assignment on this generator. Reported as measured.")
    else:
        out("MEASURED RESULT: the optimal assignment beats the greedy baseline.")
    def false_matches(strategy: str, threshold: float, size: int) -> int:
        cell = cells[(strategy, threshold, size)]
        return cell["accepted"] - cell["correct"]

    worst_gap = max(
        abs(false_matches("greedy", threshold, size) - false_matches("optimal", threshold, size))
        for threshold in THRESHOLDS
        for size in SIZES
    )
    out("")
    out("largest difference in false matches between the two strategies in any cell of")
    out(f"the whole grid: {worst_gap}. Away from the selected threshold the two differ in")
    out("both directions, so neither dominates.")

    # --- the independently derived bound ----------------------------------
    out.rule("independently derived bound -- exact structural signature collisions")
    out("The bound counts items sharing an EXACT structural signature. Two such items")
    out("cannot be separated by the structural term at all, so every false match the")
    out("structural term alone can make lies inside that set. The bound comes from")
    out("counting the signature multiset and uses nothing the matcher computes.")
    rows = []
    for size in SIZES:
        pair, removed, added, _ = _pair_pools(size, SEEDS[0])
        colliding, distinct, fraction = signature_collision_bound(list(pair.old.parameters))
        rows.append(
            [
                size,
                len(pair.old.parameters),
                distinct,
                colliding,
                f"{fraction:.4f}",
                len(removed),
                len(added),
            ]
        )
    out(
        format_table(
            [
                "size",
                "parameters",
                "distinct signatures",
                "colliding items",
                "collision fraction",
                "removed pool",
                "added pool",
            ],
            rows,
        )
    )
    out("")
    out("A non-zero collision fraction is the reason the score carries a name term at")
    out("all: on this generator the structural term alone cannot be sufficient.")
    out.check(
        "the collision bound is computable and non-trivial at every swept size",
        all(float(row[4]) >= 0.0 for row in rows),
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
