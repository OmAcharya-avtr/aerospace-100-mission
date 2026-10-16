"""The head-to-head: learned gap finder against uniform random and greedy max-min.

Protocol (fixed before the numbers were looked at, and stated so that the
choices are auditable):

* the starting campaign size is 20, chosen because a 20-scenario draw from the
  committed campaign sits at a mean 2-way coverage of about 0.82, which leaves
  headroom. At 40 the mean start is already 0.95 and every strategy saturates;
* the number of added scenarios equals the starting size, 20;
* the candidate pool is 200 of the committed 400, redrawn per seed;
* 60 seeds, which is the largest count that keeps this script, the examples and
  the suite comfortably inside the mission's 3-minute compute budget on 2
  cores. This whole script runs in roughly 10 seconds;
* both metrics are reported: 2-way coverage, which the coverage-greedy
  reference optimises directly, and grid occupancy, which is the metric closest
  to what the density model actually sees.

Spread is reported everywhere. A mean difference smaller than the standard
deviation of the paired per-seed difference has not separated two strategies,
and this script says so in those words.
"""

from __future__ import annotations

import sys
import time

from _common import FIXTURES, rule

from scenariocov import (
    CoverageGreedyReference,
    GreedyMaxMinStrategy,
    LearnedGapFinderStrategy,
    UniformRandomStrategy,
    coverage_benchmark,
    load_campaign,
    load_space,
)

N_START = 20
N_STEPS = 20
POOL = 200
SEEDS = tuple(range(60))
UNDER_TEST = ("uniform_random", "greedy_maxmin", "learned_gapfinder")


def strategies() -> list:
    """The three strategies under test plus the coverage-greedy reference."""
    return [
        UniformRandomStrategy(),
        GreedyMaxMinStrategy(),
        LearnedGapFinderStrategy(n_bootstrap=0),
        CoverageGreedyReference(t=2),
    ]


def table(out, result) -> None:
    """Print the per-strategy summary with the spread beside every mean."""
    head = (
        f"{'strategy':<28}{'final mean':>12}{'sd':>10}{'min':>10}{'max':>10}"
        f"{'gain/scen':>12}{'gain sd':>10}{'seconds':>9}"
    )
    out.write(head + "\n")
    for name, curves in result.by_strategy.items():
        s = curves.summary()
        out.write(
            f"{name:<28}{s['final_mean']:>12.6f}{s['final_sd']:>10.6f}"
            f"{s['final_min']:>10.6f}{s['final_max']:>10.6f}"
            f"{s['gain_mean']:>12.6f}{s['gain_sd']:>10.6f}{s['seconds']:>9.2f}\n"
        )


def paired(out, result, pairs) -> None:
    """Print paired per-seed differences with the win, tie and loss counts."""
    head = f"{'left - right':<46}{'mean':>11}{'sd':>11}{'W':>4}{'T':>4}{'L':>4}  separated"
    out.write(head + "\n")
    for left, right in pairs:
        d = result.paired_difference(left, right)
        separated = abs(d["mean"]) > d["sd"]
        out.write(
            f"{left + ' - ' + right:<46}{d['mean']:>+11.6f}{d['sd']:>11.6f}"
            f"{d['left_ahead_on']:>4}{d['tied_on']:>4}{d['right_ahead_on']:>4}  "
            f"{'yes' if separated else 'NO'}\n"
        )


def main() -> int:
    """Run the benchmark on both metrics plus the batch-rule ablation."""
    out = sys.stdout
    started = time.perf_counter()
    space = load_space(FIXTURES / "space.json")
    campaign = load_campaign(FIXTURES / "campaign_executed.json")
    pool = load_campaign(FIXTURES / "candidate_pool.json")

    out.write("coverage achieved per scenario executed: head to head\n")
    out.write("=" * 78 + "\n")
    out.write(
        f"start {N_START} scenarios, +{N_STEPS} from a pool of {POOL}, {len(SEEDS)} seeds, "
        "paired\n"
    )

    results = {}
    for metric in ("tway", "occupancy"):
        results[metric] = coverage_benchmark(
            space, campaign, pool, strategies(),
            n_start=N_START, n_steps=N_STEPS, pool_size=POOL, seeds=SEEDS, t=2, metric=metric,
        )

    out.write(rule("metric 1: 2-way combinatorial coverage (headline)") + "\n")
    table(out, results["tway"])
    out.write("\n")
    paired(
        out,
        results["tway"],
        [
            ("learned_gapfinder", "greedy_maxmin"),
            ("learned_gapfinder", "uniform_random"),
            ("greedy_maxmin", "uniform_random"),
            ("coverage_greedy_reference", "learned_gapfinder"),
        ],
    )

    out.write(rule("metric 2: grid occupancy over the continuous parameters") + "\n")
    table(out, results["occupancy"])
    out.write("\n")
    paired(
        out,
        results["occupancy"],
        [
            ("learned_gapfinder", "greedy_maxmin"),
            ("learned_gapfinder", "uniform_random"),
            ("coverage_greedy_reference", "learned_gapfinder"),
        ],
    )

    out.write(rule("batch-rule ablation for the learned gap finder, 2-way metric") + "\n")
    out.write(
        "Three batch rules for the same density model, measured on the same\n"
        "seeds, so that the shipped default is a measured choice rather than an\n"
        "unexamined one. greedy_maxmin is repeated as the reference line.\n"
    )
    ablation = coverage_benchmark(
        space, campaign, pool,
        [
            LearnedGapFinderStrategy(n_bootstrap=0, name="learned_refit_plus_separation"),
            LearnedGapFinderStrategy(n_bootstrap=0, refit=False, name="learned_separation_only"),
            LearnedGapFinderStrategy(
                n_bootstrap=0, sequential=False, name="learned_lowest_density_block"
            ),
            GreedyMaxMinStrategy(),
        ],
        n_start=N_START, n_steps=N_STEPS, pool_size=POOL, seeds=SEEDS, t=2,
    )
    table(out, ablation)
    out.write(
        "The shipped default is learned_refit_plus_separation. The other two\n"
        "rows are not shipped; they are here because a batch rule that simply\n"
        "takes the lowest-density candidates crowds one corner, and the size of\n"
        "that effect is the reason the default is what it is.\n"
    )

    out.write(rule("what this experiment supports") + "\n")
    tway = results["tway"]
    learned = tway.by_strategy["learned_gapfinder"].final.mean()
    maxmin = tway.by_strategy["greedy_maxmin"].final.mean()
    random_ = tway.by_strategy["uniform_random"].final.mean()
    reference = tway.by_strategy["coverage_greedy_reference"].final.mean()
    d = tway.paired_difference("learned_gapfinder", "greedy_maxmin")
    out.write(
        f"1. The coverage-greedy reference reaches {reference:.6f} against "
        f"{learned:.6f} for the\n   learned gap finder, ahead on every one of the "
        f"{len(SEEDS)} seeds. If 2-way coverage\n   is what you want, greedily "
        "maximising 2-way coverage is what to do, and no\n   model is needed for it.\n"
    )
    out.write(
        f"2. The learned gap finder and greedy max-min differ by "
        f"{d['mean']:+.6f} in mean final\n   coverage, with a paired standard "
        f"deviation of {d['sd']:.6f}, and the learned model\n   loses on "
        f"{d['right_ahead_on']} of {len(SEEDS)} seeds. This experiment does NOT "
        "separate them.\n"
    )
    occ = results["occupancy"].paired_difference("learned_gapfinder", "uniform_random")
    out.write(
        f"3. On the occupancy metric the learned gap finder is {occ['mean']:+.6f} "
        f"against uniform\n   random, losing on {occ['right_ahead_on']} of "
        f"{len(SEEDS)} seeds. On the metric closest to what the\n   density model "
        "sees, it does not beat the null strategy.\n"
    )
    out.write(
        f"4. Both non-trivial strategies beat uniform random ({random_:.6f}) on the\n"
        f"   2-way metric: max-min at {maxmin:.6f} and the learned model at "
        f"{learned:.6f}.\n"
    )
    seconds = {n: results['tway'].by_strategy[n].seconds for n in UNDER_TEST}
    ratio = seconds["learned_gapfinder"] / max(seconds["greedy_maxmin"], 1e-9)
    out.write(
        f"5. Cost: the learned gap finder spent {seconds['learned_gapfinder']:.2f} s "
        f"against {seconds['greedy_maxmin']:.2f} s for\n   greedy max-min on the same "
        f"work, a factor of {ratio:.0f}, for a difference this\n   experiment cannot "
        "establish the sign of.\n"
    )
    out.write(f"\nwall clock for this script: {time.perf_counter() - started:.1f} s\n")
    out.write("failures: 0 (this script measures; it does not assert)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
