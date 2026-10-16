"""Compare the committed numbers at both ends of every declared dependency range.

A suite that passes at both ends of a dependency range while its validation
numbers move is the defect class recorded in the build guide on 2026-10-14.
The test verdict is not enough; the numbers have to be diffed.

Method used in the build session of 2026-10-16:

1. a virtual environment was created with the FLOOR of every declared range --
   numpy 2.3.4, scikit-learn 1.8.0, matplotlib 3.10.7, pytest 9.0.1,
   hypothesis 6.169.3, ruff 0.17.0 -- and the full suite plus this script were
   run in it;
2. the floor numbers this script printed were pasted into ``FLOOR_REFERENCE``
   below;
3. the script was then run again at the CEILING end actually installed in the
   build container -- numpy 2.5.3, scikit-learn 1.9.1, matplotlib 3.11.2,
   pytest 9.1.1, hypothesis 6.170.0, ruff 0.17.1 -- and its output, which is
   the committed ``validate_dependency_ends_output.txt``, shows both columns
   and their difference.

The declared ceilings exclude the next minor release of numpy, scikit-learn
and matplotlib. That is a real cost, recorded in README.md under Limitations:
this product cannot take a scikit-learn 1.10 release without re-running the
benchmark, because ``KernelDensity`` bandwidth selection and the value of
``score_samples`` are not contractually stable across minor versions.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import sys

import numpy as np
from _common import FIXTURES, rule

from scenariocov import (
    CoverageGreedyReference,
    GapFinder,
    GreedyMaxMinStrategy,
    LearnedGapFinderStrategy,
    UniformRandomStrategy,
    build_report,
    coverage_benchmark,
    grid_occupancy,
    load_campaign,
    load_class_document,
    load_space,
)

# Printed by this script when run under the FLOOR environment described above.
FLOOR_REFERENCE = {
    "bench_learned_final": 0.9601769911504424,
    "bench_learned_minus_maxmin": 0.007374631268436553,
    "bench_maxmin_final": 0.9528023598820058,
    "bench_reference_final": 0.9959439528023598,
    "bench_uniform_final": 0.9487463126843657,
    "boundary_nearest_handover": 3.5860000000000127,
    "class_covered": 14.0,
    "gapfinder_bandwidth": 0.7244152328943627,
    "gapfinder_top_log_density": -10.978379779526692,
    "gapfinder_top_sd": 0.0643967185168227,
    "n_uncovered": 3.0,
    "occupancy_fraction": 0.75,
    "occupancy_largest_empty_block": 11.0,
    "two_way_covered": 223.0,
    "two_way_fraction": 0.9867256637168141,
    "two_way_total": 226.0,
}
TOLERANCE = 1e-9
PACKAGES = ("numpy", "scikit-learn", "scipy", "matplotlib", "pytest", "hypothesis", "ruff")


def measure() -> dict[str, float]:
    """Compute every number that a dependency change could move."""
    space = load_space(FIXTURES / "space.json")
    campaign = load_campaign(FIXTURES / "campaign_executed.json")
    pool = load_campaign(FIXTURES / "candidate_pool.json")
    classes, boundaries = load_class_document(FIXTURES / "classes.json")

    report = build_report(space, campaign, 2, classes, boundaries)
    occ = grid_occupancy(space, space.encode_many(campaign))
    handover = next(h for h in report.boundaries if h.spec.name == "handover_200m")

    finder = GapFinder(n_bootstrap=16, random_state=0).fit(space.normalise(campaign))
    top = finder.propose(space.normalise(pool), 1)[0]

    bench = coverage_benchmark(
        space, campaign, pool,
        [
            UniformRandomStrategy(),
            GreedyMaxMinStrategy(),
            LearnedGapFinderStrategy(n_bootstrap=0),
            CoverageGreedyReference(t=2),
        ],
        n_start=20, n_steps=20, pool_size=200, seeds=tuple(range(12)), t=2,
    )
    paired = bench.paired_difference("learned_gapfinder", "greedy_maxmin")
    return {
        "two_way_covered": float(report.tway.covered),
        "two_way_total": float(report.tway.total),
        "two_way_fraction": float(report.tway.fraction),
        "n_uncovered": float(report.tway.n_uncovered),
        "class_covered": float(report.classes.covered),
        "boundary_nearest_handover": float(handover.nearest_distance),
        "occupancy_fraction": float(occ.fraction),
        "occupancy_largest_empty_block": float(occ.largest_empty_block()),
        "gapfinder_bandwidth": float(finder.fitted_bandwidth),
        "gapfinder_top_log_density": float(top.log_density),
        "gapfinder_top_sd": float(top.log_density_sd),
        "bench_uniform_final": float(bench.by_strategy["uniform_random"].final.mean()),
        "bench_maxmin_final": float(bench.by_strategy["greedy_maxmin"].final.mean()),
        "bench_learned_final": float(bench.by_strategy["learned_gapfinder"].final.mean()),
        "bench_reference_final": float(
            bench.by_strategy["coverage_greedy_reference"].final.mean()
        ),
        "bench_learned_minus_maxmin": float(paired["mean"]),
    }


def main() -> int:
    """Measure, compare against the floor reference and print both columns."""
    out = sys.stdout
    out.write("validation numbers at both ends of every declared dependency range\n")
    out.write("=" * 78 + "\n")
    out.write(rule("environment this run was made in") + "\n")
    for name in PACKAGES:
        try:
            out.write(f"  {name:<16}{md.version(name)}\n")
        except md.PackageNotFoundError:
            out.write(f"  {name:<16}ABSENT\n")
    out.write(f"  {'python':<16}{sys.version.split()[0]}\n")

    measured = measure()
    out.write(rule("floor run versus this run") + "\n")
    out.write(f"{'quantity':<34}{'floor':>18}{'this run':>18}{'difference':>14}\n")
    moved = 0
    for key, value in measured.items():
        reference = FLOOR_REFERENCE.get(key, float("nan"))
        delta = value - reference
        if not np.isfinite(delta) or abs(delta) > TOLERANCE:
            moved += 1
        out.write(f"{key:<34}{reference:>18.10g}{value:>18.10g}{delta:>+14.3g}\n")
    out.write(
        f"\nquantities that moved by more than {TOLERANCE:g}: {moved} of {len(measured)}\n"
    )
    out.write(rule("measured values, for pasting into FLOOR_REFERENCE") + "\n")
    out.write(json.dumps(measured, indent=4, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
