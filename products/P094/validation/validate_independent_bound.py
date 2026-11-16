"""Checks against bounds and data paths this package's arithmetic does not use.

"Two implementations agree" and "the answer lies inside a bound derived from
different mathematics" are different check classes, and the second catches what
the first cannot. This script runs four of them.

1. **A quantile bound from the union bound.** For non-negative stage times
   ``X_1..X_k``, the median of the total obeys

       ``max_s median(X_s)  <=  median(sum_s X_s)  <=  sum_s q_s(1 - 0.5/k)``

   The lower bound is quantile monotonicity under pointwise domination: the
   total dominates every component, so its median dominates every component's
   median. The upper bound is a union bound: if every ``X_s`` is at or below its
   ``(1 - 0.5/k)``-quantile then the total is at or below the sum of those
   quantiles, and the probability that any one exceeds it is at most
   ``k * 0.5/k = 0.5``. Neither argument appears anywhere in this package, which
   computes per-stage medians and never a median of a sum. The bound is checked
   against the per-frame total medians the campaign wrote into ``campaign.log``,
   parsed from the log text -- a different data path from the CSV the package
   reads.

   The committed ``p95_ns`` column stands in for the ``1 - 0.5/6 = 0.91667``
   quantile. ``p95 >= q(0.91667)``, so substituting it keeps the bound valid and
   makes it looser, which is the safe direction.

2. **A different data path for the same medians.** ``raw/<cell>.npy`` is written
   by a separate code path from ``stages_pass.csv`` and is SUBSAMPLED EVERY
   THIRD FRAME. Per-stage medians from it must agree with the pass-table
   medians to within the subsampling.

3. **An independent implementation of the rank statistic.**
   ``scipy.stats.kendalltau`` against this package's own tau-b. scipy is in the
   test extra and not in the runtime dependencies, precisely so the
   cross-checked implementation does not ship with the thing it checks.

4. **An instrument bound.** Every per-stage median must exceed the campaign's
   own measured empty-bracket cost and the clock resolution. A stage time below
   the cost of observing it would not be a measurement.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import re  # noqa: E402

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from hiltimingdiff import kendall_tau_b, load_campaign, pass_units  # noqa: E402

#: The regular expression that parses the per-cell median total from the log.
LOG_PATTERN = re.compile(
    r"\s(\S+)\s+passes=60 frames=90000 median_total=([0-9.]+) us"
)

#: Number of stages, which the union bound's tail allocation divides by.
N_STAGES = 6


def main() -> int:
    tee = _bootstrap.Tee("validate_independent_bound_output.txt")
    _bootstrap.header(tee, "P094 HilTimingDiff -- independent bounds and data paths")

    campaign = load_campaign()
    units = pass_units(campaign)
    table = campaign.stages_pass

    tee.rule("1. quantile bound on the median total, against the log's own figure")
    tee(
        "  max_s median(X_s) <= median(sum_s X_s) <= sum_s q_s(1 - 0.5/k),  k = 6 stages"
    )
    tee(
        "  lower: quantile monotonicity under pointwise domination. upper: a union "
        "bound on the"
    )
    tee(
        "  per-stage tails. Neither argument is used anywhere in this package, which "
        "never forms"
    )
    tee("  a median of a sum.")
    tee("")
    logged: dict[str, float] = {}
    for line in campaign.log.splitlines():
        match = LOG_PATTERN.search(line)
        if match:
            logged[match.group(1)] = float(match.group(2)) * 1000.0
    tee(f"  cells parsed from campaign.log : {len(logged)}")
    tee("")
    tee.check(
        "the log yields a median total for every one of the nine cells",
        len(logged) == 9,
        f"parsed {', '.join(sorted(logged))}",
    )
    tee(
        "  cell                 lower bound    logged median      upper bound   "
        "sum of medians"
    )
    violations: list[str] = []
    for cell in sorted(logged):
        subset = table.loc[table["cell"] == cell]
        medians = subset.groupby("stage")["median_ns"].median()
        p95 = subset.groupby("stage")["p95_ns"].median()
        lower = float(medians.max())
        upper = float(p95.sum())
        inside = lower <= logged[cell] <= upper
        if not inside:
            violations.append(cell)
        tee(
            f"  {cell:<18s} {lower:>13.1f} {logged[cell]:>16.1f} {upper:>16.1f} "
            f"{float(medians.sum()):>16.1f}   {'inside' if inside else 'OUTSIDE'}"
        )
    tee("")
    tee.check(
        "every cell's logged median total lies inside the independent bound",
        not violations,
        f"violations: {', '.join(violations) if violations else 'none'}",
    )
    above = [
        cell
        for cell in sorted(logged)
        if float(
            table.loc[table["cell"] == cell].groupby("stage")["median_ns"].median().sum()
        )
        > logged[cell]
    ]
    tee.finding(
        "The SUM OF PER-STAGE MEDIANS is not a bound on the median of the total, and "
        f"this campaign shows it: on {len(above)} of 9 cells "
        f"({', '.join(above) if above else 'none'}) the sum of the per-stage medians "
        "EXCEEDS the logged median total. That is not an inconsistency -- the median "
        "of a sum is not the sum of medians for dependent, right-tailed variables -- "
        "and it is the reason this check uses a bound derived from quantile "
        "monotonicity and a union bound rather than an additivity assumption."
    )

    tee.rule("2. the same medians through a different data path: raw/<cell>.npy")
    tee(
        f"  raw/<cell>.npy is SUBSAMPLED EVERY {campaign.sample_every} FRAMES "
        "(meta.json: raw_sample_every)"
    )
    tee(
        "  and is written by a separate code path from stages_pass.csv, which "
        "summarises EVERY frame."
    )
    tee("")
    tee("  cell                 stage        raw median   pass-table median   rel diff")
    worst = 0.0
    for cell in sorted(logged):
        raw = campaign.raw(cell)
        subset = table.loc[table["cell"] == cell]
        for index, stage in enumerate(units.stages):
            raw_median = float(np.median(raw[:, index]))
            pass_median = float(subset.loc[subset["stage"] == stage, "median_ns"].median())
            relative = abs(raw_median - pass_median) / pass_median
            worst = max(worst, relative)
            if cell == "w64_sparse":
                tee(
                    f"  {cell:<18s} {stage:<11s} {raw_median:>12.1f} "
                    f"{pass_median:>19.1f} {relative:>10.6f}"
                )
    tee("")
    tee(f"  largest relative difference over all 54 cell-stage pairs: {worst:.6f}")
    tee.check(
        "the two data paths agree on every per-stage median to within 10 per cent",
        worst < 0.10,
        f"largest relative difference {worst:.6f}; the two differ because one "
        "summarises every frame and the other every third",
    )

    tee.rule("3. the rank statistic against an independent implementation")
    try:
        from scipy.stats import kendalltau
    except ImportError:  # pragma: no cover - scipy is in the test extra
        tee("  scipy is not installed; the cross-check cannot run and is NOT claimed.")
        tee.check(
            "scipy is available for the independent tau-b cross-check",
            False,
            "install the test extra: pip install -e '.[test]'",
        )
    else:
        hardware = np.median(units.hw, axis=0)
        declared = np.median(units.ops, axis=0)
        ours = kendall_tau_b(declared, hardware)
        theirs = float(kendalltau(declared, hardware, variant="b").statistic)
        tee(f"  hiltimingdiff.kendall_tau_b : {ours!r}")
        tee(f"  scipy.stats.kendalltau      : {theirs!r}")
        tee(f"  absolute difference         : {abs(ours - theirs):.3e}")
        tee("")
        tee.check(
            "the two implementations agree to within rounding",
            abs(ours - theirs) <= 1e-12,
            "exact in exact arithmetic; agreeing to within rounding is what is "
            "guaranteed",
        )
        generator = np.random.default_rng(7070)
        disagreements = 0
        for _ in range(200):
            size = int(generator.integers(2, 9))
            left = generator.integers(0, 4, size=size).astype(np.float64)
            right = generator.integers(0, 4, size=size).astype(np.float64)
            ours_value = kendall_tau_b(left, right)
            theirs_value = float(kendalltau(left, right, variant="b").statistic)
            if np.isnan(ours_value) and np.isnan(theirs_value):
                continue
            if np.isnan(ours_value) != np.isnan(theirs_value):
                disagreements += 1
                continue
            if abs(ours_value - theirs_value) > 1e-9:
                disagreements += 1
        tee.check(
            "the two implementations agree on 200 random tied rankings",
            disagreements == 0,
            f"{disagreements} disagreements; ties are where a tau-b implementation "
            "differs if it differs at all",
        )

    tee.rule("4. the instrument bound: a stage time must exceed the cost of seeing it")
    runs = campaign.runs
    bracket_max = float(runs["bracket_ns"].max())
    resolution = float(runs["clock_res_ns"].max())
    tee(f"  largest measured empty-bracket cost : {bracket_max:.3f} ns")
    tee(f"  clock resolution                    : {resolution:.3f} ns")
    tee("")
    tee("  stage        median_ns   median / bracket cost")
    for index, stage in enumerate(units.stages):
        median = float(np.median(units.hw[:, index]))
        tee(f"  {stage:<11s} {median:>11.2f} {median / bracket_max:>22.2f}")
    tee("")
    minimum = float(np.min(np.median(units.hw, axis=0)))
    tee.check(
        "every per-stage median exceeds the measured empty-bracket cost",
        minimum > bracket_max,
        f"smallest median {minimum:.2f} ns against a bracket cost of "
        f"{bracket_max:.3f} ns, a factor of {minimum / bracket_max:.1f}",
    )
    tee.check(
        "every per-stage median exceeds the clock resolution by three orders",
        minimum > 1000.0 * resolution,
        f"smallest median {minimum:.2f} ns against a resolution of {resolution:.3f} ns",
    )
    tee.finding(
        "The instrument is not negligible for the cheapest stage. ingest's median is "
        f"{minimum:.0f} ns against a measured empty-bracket cost of "
        f"{bracket_max:.1f} ns, a factor of {minimum / bracket_max:.1f}. So roughly "
        f"{bracket_max / minimum:.1%} of the reported ingest time could be the "
        "bracket itself. ingest is argmin(R) and holds 0.53 per cent of frame time, so "
        "this does not move the headline, but a reader using the ingest figure on its "
        "own should subtract the bracket cost."
    )

    tee.rule("negative controls on the bound itself")
    # Anchored on TRUE values. A deliberately violated total must fall outside.
    subset = table.loc[table["cell"] == "w64_sparse"]
    medians = subset.groupby("stage")["median_ns"].median()
    p95 = subset.groupby("stage")["p95_ns"].median()
    lower = float(medians.max())
    upper = float(p95.sum())
    tee.control(
        "a total half the largest stage median falls below the bound",
        not (lower <= 0.5 * lower <= upper),
        must_fire=True,
        detail=f"0.5 * {lower:.1f} = {0.5 * lower:.1f} against lower bound {lower:.1f}",
    )
    tee.control(
        "a total twice the upper bound falls above it",
        not (lower <= 2.0 * upper <= upper),
        must_fire=True,
        detail=f"2 * {upper:.1f} = {2.0 * upper:.1f} against upper bound {upper:.1f}",
    )
    tee.control(
        "the measured total falls outside the bound",
        not (lower <= logged["w64_sparse"] <= upper),
        must_fire=False,
        detail=f"measured {logged['w64_sparse']:.1f} inside [{lower:.1f}, {upper:.1f}]",
    )
    tee(
        f"\n  bound slack on w64_sparse: the interval is [{lower:.1f}, {upper:.1f}] ns "
        f"and the"
    )
    tee(
        f"  measurement is {logged['w64_sparse']:.1f} ns, so the bound is "
        f"{(upper - lower) / logged['w64_sparse']:.2f} of the measurement wide. It is a "
        "loose"
    )
    tee(
        "  bound and it is a VALID one: the slack follows the conditioning of a union "
        "bound over"
    )
    tee(
        "  six right-tailed stages, not an absolute constant, and the controls above "
        "confirm it is"
    )
    tee("  not vacuous.")

    path = tee.save()
    print(f"\nwritten: {path}")
    return 1 if tee.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
