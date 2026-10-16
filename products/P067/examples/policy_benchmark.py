"""Held-out head-to-head: fixed patterns, Bayesian-greedy and the learned policy.

    python examples/policy_benchmark.py

Writes ../screenshots/policy_benchmark.png relative to this file. This is a
reduced version of validation/validate_policies.py, sized to run in about
half a minute on two cores; the full numbers are in
validation/validate_policies_output.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from acqsearch import (  # noqa: E402
    LearnedScanPolicy,
    build_training_set,
    evaluate_policies,
    generate_scenarios,
    paired_difference,
    split_scenarios,
    summarise,
    winner,
)
from acqsearch.plotting import plot_policy_comparison  # noqa: E402

N_SCENARIOS = 150
SEED = 20261016


def main() -> int:
    """Run the reduced benchmark and draw the comparison."""
    scenarios = generate_scenarios(N_SCENARIOS, SEED)
    train, test = split_scenarios(scenarios, 0.65)
    features, target = build_training_set(train, episodes_per_scenario=2)
    learned = LearnedScanPolicy().fit(features, target)
    results = evaluate_policies(
        test, learned, targets_per_scenario=4, seed=20261017, include_all_patterns=True
    )
    scores = summarise(results)
    print(f"train {len(train)}  held out {len(test)}  rows {features.shape[0]}  "
          f"episodes per policy {len(results['learned'])}")
    print(f"{'policy':<22}{'mean_s':>10}{'s.e._s':>9}{'median_s':>10}{'acquired':>10}")
    for score in sorted(scores.values(), key=lambda s: s.mean_time_s):
        print(f"{score.name:<22}{score.mean_time_s:>10.4f}{score.standard_error_s:>9.4f}"
              f"{score.median_time_s:>10.4f}{score.acquisition_rate:>10.4f}")
    print(f"winner {winner(scores)}")
    for other in ("bayesian_greedy", "fixed_spiral"):
        difference = paired_difference("learned", results["learned"], other, results[other])
        print(f"paired learned - {other}: {difference.mean_difference_s:+.4f} s "
              f"+- {difference.standard_error_s:.4f} s  z = {difference.z_score:+.2f}")

    output = plot_policy_comparison(
        scores,
        ROOT / "screenshots" / "policy_benchmark.png",
        title=f"Held-out comparison, {len(test)} scenarios x 4 target draws",
    )
    print(f"wrote {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
