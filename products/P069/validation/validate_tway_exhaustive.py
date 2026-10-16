"""Cross-check t-way coverage against exhaustive enumeration on small spaces.

Three independent checks:

1. the closed form for ``|T_t|`` (equation (1) in
   :mod:`scenariocov.combinatorial`) against a count of enumerated tuples;
2. the package's set-based coverage count against a deliberately slow
   reference that builds every t-way value tuple and asks whether any scenario
   realises it;
3. the exhaustive campaign, where coverage must be exactly complete -- an
   integer identity over a set, not a floating-point one.

The spaces are kept genuinely small, because the enumeration is exhaustive in
``|T_t|`` and that is what blows up. The largest space exercised here has
level counts (4, 3, 3, 2, 2) and the largest enumeration is |T_5| = 144. The
limit the package enforces is ``max_tuples`` (default 2,000,000); the cost
figures at the end of this output say where the method stops being usable.
"""

from __future__ import annotations

import sys
from itertools import combinations, product

import numpy as np
from _common import rule

from scenariocov import Parameter, ScenarioSpace, count_tway_tuples, tway_coverage

SPACES = (
    [2, 2],
    [2, 3],
    [3, 3, 2],
    [2, 2, 2, 2],
    [4, 3, 3, 2, 2],
)
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)


def space_from(counts: list[int]) -> ScenarioSpace:
    """A categorical space with the given level counts."""
    return ScenarioSpace(
        [Parameter.categorical(f"p{i}", [f"l{j}" for j in range(c)]) for i, c in enumerate(counts)]
    )


def reference_coverage(counts: list[int], rows: np.ndarray, t: int) -> tuple[int, int]:
    """Slow, obvious reference: enumerate T_t and test membership directly."""
    total = covered = 0
    listed = [tuple(int(v) for v in row) for row in rows]
    for subset in combinations(range(len(counts)), t):
        for values in product(*(range(counts[i]) for i in subset)):
            total += 1
            if any(all(r[i] == v for i, v in zip(subset, values, strict=True)) for r in listed):
                covered += 1
    return covered, total


def main() -> int:
    """Run the three checks and report pass or fail for each."""
    out = sys.stdout
    failures = 0
    out.write("t-way coverage against exhaustive enumeration\n")
    out.write("=" * 78 + "\n")

    out.write(rule("1. closed form |T_t| versus enumerated tuple count") + "\n")
    out.write(f"{'level counts':<22}{'t':>3}{'closed form':>14}{'enumerated':>12}  verdict\n")
    for counts in SPACES:
        space = space_from(counts)
        empty = np.zeros((0, len(counts)), dtype=np.int64)
        for t in range(1, len(counts) + 1):
            closed = count_tway_tuples(counts, t)
            enumerated = len(tway_coverage(space, empty, t=t).uncovered)
            ok = closed == enumerated
            failures += 0 if ok else 1
            out.write(
                f"{str(tuple(counts)):<22}{t:>3}{closed:>14}{enumerated:>12}  "
                f"{'PASS' if ok else 'FAIL'}\n"
            )

    out.write(rule("2. package count versus the slow reference, random campaigns") + "\n")
    out.write(f"{'level counts':<22}{'t':>3}{'n':>4}{'package':>10}{'reference':>11}  verdict\n")
    for counts in SPACES:
        space = space_from(counts)
        for t in range(1, len(counts) + 1):
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                n = int(rng.integers(0, 9))
                rows = np.column_stack(
                    [rng.integers(0, c, size=n) for c in counts]
                ).astype(np.int64) if n else np.zeros((0, len(counts)), dtype=np.int64)
                package = tway_coverage(space, rows, t=t)
                ref_cov, ref_total = reference_coverage(counts, rows, t)
                ok = (package.covered, package.total) == (ref_cov, ref_total)
                failures += 0 if ok else 1
                if seed == SEEDS[-1] or not ok:
                    out.write(
                        f"{str(tuple(counts)):<22}{t:>3}{n:>4}"
                        f"{package.covered:>10}{ref_cov:>11}  {'PASS' if ok else 'FAIL'}\n"
                    )
    out.write(
        f"({len(SPACES)} spaces x all t x {len(SEEDS)} seeds compared; one row printed per "
        "space and t, plus any failure)\n"
    )

    out.write(rule("3. exhaustive campaign gives complete coverage") + "\n")
    out.write(f"{'level counts':<22}{'t':>3}{'covered':>10}{'total':>8}  verdict\n")
    for counts in SPACES:
        space = space_from(counts)
        rows = np.asarray(list(product(*(range(c) for c in counts))), dtype=np.int64)
        for t in range(1, len(counts) + 1):
            result = tway_coverage(space, rows, t=t)
            ok = result.covered == result.total and result.uncovered == ()
            failures += 0 if ok else 1
            out.write(
                f"{str(tuple(counts)):<22}{t:>3}{result.covered:>10}{result.total:>8}  "
                f"{'PASS' if ok else 'FAIL'}\n"
            )

    out.write(rule("4. where the enumeration stops being usable") + "\n")
    out.write(f"{'parameters':>11}{'levels':>8}{'t':>3}{'|T_t|':>16}\n")
    for k, v, t in ((7, 3, 2), (7, 3, 3), (20, 4, 2), (20, 4, 3), (20, 4, 6), (40, 5, 4)):
        out.write(f"{k:>11}{v:>8}{t:>3}{count_tway_tuples([v] * k, t):>16}\n")
    out.write(
        "The default max_tuples is 2000000, so the (20, 4, 6) and (40, 5, 4) rows\n"
        "above are refused rather than attempted. Enumerating the uncovered set is\n"
        "what costs: a percentage alone would be cheaper and is not what this\n"
        "package is for.\n"
    )

    out.write(f"\nfailures: {failures}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
