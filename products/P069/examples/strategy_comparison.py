"""Benchmark the learned gap finder against uniform random and greedy max-min.

    python examples/strategy_comparison.py

Writes ../screenshots/strategy_comparison.png. The result is the one in
README.md: the deterministic coverage-greedy reference wins, and the learned
gap finder is not separated from the greedy max-min baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scenariocov import (  # noqa: E402
    CoverageGreedyReference,
    GreedyMaxMinStrategy,
    LearnedGapFinderStrategy,
    UniformRandomStrategy,
    coverage_benchmark,
    load_campaign,
    load_space,
)
from scenariocov.plotting import plot_coverage_curves  # noqa: E402

SEEDS = tuple(range(30))


def main() -> int:
    """Run the benchmark, print the table with its spread and save the figure."""
    space = load_space(ROOT / "fixtures" / "space.json")
    campaign = load_campaign(ROOT / "fixtures" / "campaign_executed.json")
    pool = load_campaign(ROOT / "fixtures" / "candidate_pool.json")
    result = coverage_benchmark(
        space,
        campaign,
        pool,
        [
            UniformRandomStrategy(),
            GreedyMaxMinStrategy(),
            LearnedGapFinderStrategy(n_bootstrap=0),
            CoverageGreedyReference(t=2),
        ],
        n_start=20,
        n_steps=20,
        pool_size=200,
        seeds=SEEDS,
        t=2,
    )
    print(f"2-way coverage, {len(SEEDS)} seeds, start 20, +20 from a pool of 200")
    print(f"{'strategy':<28}{'final':>10}{'sd':>9}{'min':>9}{'max':>9}{'gain/scen':>11}")
    for name, curves in result.by_strategy.items():
        s = curves.summary()
        print(
            f"{name:<28}{s['final_mean']:>10.6f}{s['final_sd']:>9.6f}"
            f"{s['final_min']:>9.6f}{s['final_max']:>9.6f}{s['gain_mean']:>11.6f}"
        )
    diff = result.paired_difference("learned_gapfinder", "greedy_maxmin")
    print(
        f"\nlearned - greedy_maxmin, paired: mean {diff['mean']:+.6f}, sd {diff['sd']:.6f}, "
        f"W/T/L {diff['left_ahead_on']}/{diff['tied_on']}/{diff['right_ahead_on']}"
    )
    print("separated by this experiment:", "yes" if abs(diff["mean"]) > diff["sd"] else "no")
    out = plot_coverage_curves(result, ROOT / "screenshots" / "strategy_comparison.png")
    print(f"wrote screenshots/{out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
