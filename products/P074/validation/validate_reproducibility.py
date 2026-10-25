"""Determinism: the same input must give the same numbers, bit for bit.

Three separate claims are checked, because they fail for different reasons.

1. **The committed fixtures are what the committed code produces.** The
   population is regenerated from its seed and compared byte for byte against
   ``fixtures/population.tle``.
2. **The divergence computation is deterministic.** A subset of the population
   is propagated twice in the same process and the two tables are compared bit
   for bit.
3. **The divergence of a subset reproduces the committed rows**, provided the
   integration step is pinned. It is derived from the shortest period in the
   file, so a subset has a different one unless the reference period is passed
   back in; without that the rows differ in the fourth significant figure,
   which is the truncation floor correctly visible.

One thing that is *not* bit-exact and the reason is recorded rather than
absorbed into a loose tolerance: the luni-solar ephemeris is tabulated over the
propagation span, so a one-day run evaluates the trigonometric series on a
shorter array than a seven-day run, and numpy's vectorised transcendental loops
are not required to give bit-identical results for arrays of different length.
The measured consequence is reported.

Every predictor in the benchmark is also re-fitted and compared, because a
gradient-boosted ensemble with a fixed random state must be reproducible and a
conformal half-width derived from a seeded split must be too.
"""

from __future__ import annotations

import hashlib
import sys

from _bootstrap import (
    POPULATION_PATH,
    Tee,
    Timer,
    bootstrap,
    environment_lines,
    is_fast,
    load_dataset,
)

ROOT = bootstrap()

import numpy as np  # noqa: E402

from orbitdrift import (  # noqa: E402
    build_divergence_table,
    run_benchmark,
    synthetic_population_tle,
)
from orbitdrift.benchmark import run_extrapolation_benchmark  # noqa: E402


def main() -> int:
    out = Tee("validate_reproducibility_output.txt")
    out.rule("orbitdrift -- determinism of the population, the dataset and the fits")
    for line in environment_lines():
        out(line)

    out.rule("claim 1: the committed population is what the seed produces")
    committed_text = (ROOT / POPULATION_PATH).read_text()
    regenerated = synthetic_population_tle()
    committed_digest = hashlib.sha256(committed_text.encode()).hexdigest()
    regenerated_digest = hashlib.sha256(regenerated.encode()).hexdigest()
    out(f"  committed sha256              : {committed_digest}")
    out(f"  regenerated sha256            : {regenerated_digest}")
    out(f"  bytes                         : {len(committed_text)}")
    out.check(
        "the committed population file is reproduced byte for byte from its seed",
        committed_text == regenerated,
        f"{committed_digest[:16]} against {regenerated_digest[:16]}",
    )
    out.check(
        "a different seed gives a different population",
        synthetic_population_tle(seed=7) != regenerated,
        "so the seed is actually in use",
    )

    out.rule("claim 2: the divergence computation is deterministic")
    lines = committed_text.splitlines()
    header = lines[:2]
    body: list[str] = []
    index = 2
    objects_wanted = 2 if is_fast() else 6
    while index < len(lines) and len(body) < 3 * objects_wanted:
        body.extend(lines[index : index + 3])
        index += 3
    subset_text = "\n".join(header + body) + "\n"
    span_days = 0.2 if is_fast() else 1.0
    table = load_dataset()
    reference_period = float(table.detail.split("reference period of ")[1].split(" s at")[0])
    out(f"  objects                       : {objects_wanted}")
    out(f"  span                          : {span_days} days")
    out(f"  reference period pinned at    : {reference_period:.6f} s")

    timer = Timer()
    with timer:
        first = build_divergence_table(
            subset_text, span_days=span_days, reference_period_s=reference_period
        )
    with timer:
        second = build_divergence_table(
            subset_text, span_days=span_days, reference_period_s=reference_period
        )
    identical = bool(np.array_equal(first.values, second.values))
    out(f"  rows                          : {first.n_rows}")
    out(f"  bit-identical between runs    : {identical}")
    out.check(
        "two runs of the same computation in the same process agree bit for bit",
        identical,
        f"worst difference "
        f"{float(np.max(np.abs(first.values - second.values))) if not identical else 0.0:.3e}",
    )

    out.rule("claim 3: a subset reproduces the committed rows with the step pinned")
    objects = table.column("object_index")
    worst_absolute = 0.0
    worst_relative = 0.0
    compared = 0
    for value in np.unique(first.column("object_index")):
        mine = first.values[first.column("object_index") == value]
        theirs = table.values[objects == value][: mine.shape[0]]
        difference = np.abs(mine - theirs)
        scale = np.maximum(np.abs(theirs), 1e-12)
        worst_absolute = max(worst_absolute, float(difference.max()))
        worst_relative = max(worst_relative, float((difference / scale).max()))
        compared += mine.shape[0]
    out(f"  rows compared                 : {compared}")
    out(f"  worst absolute difference     : {worst_absolute:.6e}")
    out(f"  worst relative difference     : {worst_relative:.6e}")
    out("")
    out("  This is NOT bit-exact, and the reason is the tabulated luni-solar")
    out("  ephemeris. A one-day run evaluates the trigonometric series on 1442")
    out("  epochs where the seven-day committed run used 10082, and numpy's")
    out("  vectorised transcendental loops are not required to give bit-identical")
    out("  results for arrays of different length -- the tail of a SIMD loop is")
    out("  handled differently from its body. A few thousand Runge-Kutta steps")
    out("  amplify that last-bit difference to the value above.")
    out.check(
        "a pinned-step subset reproduces the committed rows to 1e-6 relative",
        worst_relative < 1e-6,
        f"{worst_relative:.3e}",
    )

    out.rule("claim 3b: without pinning the step, the rows differ measurably")
    unpinned = build_divergence_table(subset_text, span_days=span_days)
    unpinned_period = float(
        unpinned.detail.split("reference period of ")[1].split(" s at")[0]
    )
    worst_unpinned = 0.0
    for value in np.unique(unpinned.column("object_index")):
        mine = unpinned.values[unpinned.column("object_index") == value]
        theirs = table.values[objects == value][: mine.shape[0]]
        scale = np.maximum(np.abs(theirs), 1e-12)
        worst_unpinned = max(worst_unpinned, float((np.abs(mine - theirs) / scale).max()))
    out(f"  subset's own reference period : {unpinned_period:.6f} s")
    out(f"  committed reference period    : {reference_period:.6f} s")
    out(f"  worst relative difference     : {worst_unpinned:.6e}")
    out("")
    out("  This is the integrator's truncation floor made visible by a change of")
    out("  step, and it is why the step is an input to build_divergence_table")
    out("  rather than an emergent property of which objects happen to share an")
    out("  epoch. The number is reported, not hidden.")
    out.check(
        "an unpinned step changes the rows by more than the pinned run does "
        "(so the pinning is doing something)",
        worst_unpinned > 10.0 * worst_relative,
        f"{worst_unpinned:.3e} against {worst_relative:.3e}",
    )

    if is_fast():
        out.rule("claims 4 and 5")
        out("  skipped in probe mode: four benchmark fits dominate this script's")
        out("  runtime and the probe only checks the exit status.")
        out("")
        out(f"wrote {out.save()}")
        return 1 if out.failures else 0

    out.rule("claim 4: every fit in the benchmark is reproducible")
    first_cold = run_benchmark(table)
    second_cold = run_benchmark(table)
    out("  predictor                             coverage match   score match")
    mismatches = 0
    for left, right in zip(first_cold.rows, second_cold.rows, strict=True):
        coverage_same = left.metrics.coverage == right.metrics.coverage
        score_same = left.metrics.interval_score == right.metrics.interval_score
        if not (coverage_same and score_same):
            mismatches += 1
        out(
            f"  {left.name:37s} {str(coverage_same):14s}   {str(score_same)}"
        )
    out.check(
        "the cold-prediction benchmark is bit-reproducible across runs",
        mismatches == 0,
        f"{mismatches} mismatched rows",
    )

    first_extrapolation = run_extrapolation_benchmark(table)
    second_extrapolation = run_extrapolation_benchmark(table)
    extrapolation_mismatches = sum(
        1
        for left, right in zip(
            first_extrapolation.rows, second_extrapolation.rows, strict=True
        )
        if left.metrics.interval_score != right.metrics.interval_score
    )
    out.check(
        "the extrapolation benchmark is bit-reproducible across runs",
        extrapolation_mismatches == 0,
        f"{extrapolation_mismatches} mismatched rows",
    )

    out.rule("claim 5: a different split seed changes the answer")
    out("  Otherwise the split is not in use and the held-out measurement is not")
    out("  held out.")
    other = run_benchmark(table, seed=12345)
    changed = sum(
        1
        for left, right in zip(first_cold.rows, other.rows, strict=True)
        if left.metrics.coverage != right.metrics.coverage
    )
    out(f"  rows whose coverage changed   : {changed} of {len(first_cold.rows)}")
    out.check(
        "a different split seed changes most predictors' measured coverage",
        changed >= len(first_cold.rows) - 1,
        f"{changed} of {len(first_cold.rows)}",
    )

    out.rule("timings (UNSTABLE: wall clock on a shared two-core container)")
    out(f"  subset propagations           : {timer.calls}, {timer.elapsed_s:.1f} s total")

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())
