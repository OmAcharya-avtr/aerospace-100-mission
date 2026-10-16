"""The gap finder's uncertainty output: does it mean anything.

A confidence number that does not move with the evidence is decoration. Three
checks, all measured on the committed fixtures:

1. the bootstrap standard deviation of the log density falls as the executed
   campaign grows, which is the minimum a resampling uncertainty must do;
2. ``sparsity_agreement`` separates candidates that lie in a genuinely empty
   grid cell from candidates that lie in a well-populated one;
3. the agreement number is informative rather than saturated: the fraction of
   the pool at agreement 1.0 and at 0.0 is reported, because a score that is
   1.0 for everything carries no information even if it correlates.

None of this makes the number a probability. It is the fraction of bootstrap
resamples that agree, and nothing in this repository calibrates it against an
observed frequency.
"""

from __future__ import annotations

import sys

import numpy as np
from _common import FIXTURES, rule

from scenariocov import GapFinder, grid_occupancy, load_campaign, load_space

SIZES = (20, 40, 80, 160, 240)
BOOTSTRAP = 24


def main() -> int:
    """Run the three checks."""
    out = sys.stdout
    failures = 0
    space = load_space(FIXTURES / "space.json")
    campaign = load_campaign(FIXTURES / "campaign_executed.json")
    pool = load_campaign(FIXTURES / "candidate_pool.json")
    pool_norm = space.normalise(pool)

    out.write("gap finder uncertainty output\n")
    out.write("=" * 78 + "\n")
    out.write(f"bootstrap replicates: {BOOTSTRAP}   candidate pool: {len(pool)}\n")

    out.write(rule("1. bootstrap spread against the size of the executed campaign") + "\n")
    out.write(f"{'n executed':>12}{'bandwidth':>12}{'mean sd':>12}{'median sd':>12}\n")
    rng = np.random.default_rng(20261016)
    spreads = []
    for n in SIZES:
        idx = rng.choice(len(campaign), size=n, replace=False)
        finder = GapFinder(n_bootstrap=BOOTSTRAP, random_state=7)
        finder.fit(space.normalise([campaign[int(i)] for i in idx]))
        scores = finder.score_candidates(pool_norm)
        spreads.append(float(np.mean(scores.log_density_sd)))
        out.write(
            f"{n:>12}{finder.fitted_bandwidth:>12.6f}"
            f"{np.mean(scores.log_density_sd):>12.6f}"
            f"{np.median(scores.log_density_sd):>12.6f}\n"
        )
    ok = spreads[-1] < spreads[0]
    failures += 0 if ok else 1
    out.write(
        f"  {'PASS' if ok else 'FAIL'}  mean spread falls from {spreads[0]:.6f} nats at "
        f"n={SIZES[0]} to {spreads[-1]:.6f} at n={SIZES[-1]}\n"
    )
    monotone = all(b <= a + 1e-9 for a, b in zip(spreads, spreads[1:], strict=False))
    out.write(
        f"  {'monotone' if monotone else 'NOT monotone'} across the five sizes; "
        "monotonicity is not claimed, only the end-to-end fall.\n"
    )

    out.write(rule("2. agreement against whether the candidate's cell is empty") + "\n")
    finder = GapFinder(n_bootstrap=BOOTSTRAP, random_state=7)
    finder.fit(space.normalise(campaign))
    scores = finder.score_candidates(pool_norm)
    occ = grid_occupancy(space, space.encode_many(campaign))
    grid_cols = [space.index_of(name) for name in occ.parameters]
    pool_encoded = space.encode_many(pool)
    empty_flags = np.array(
        [occ.counts[tuple(int(row[c]) for c in grid_cols)] == 0 for row in pool_encoded]
    )
    in_empty = scores.sparsity_agreement[empty_flags]
    in_full = scores.sparsity_agreement[~empty_flags]
    out.write(f"candidates in an EMPTY grid cell:    {int(empty_flags.sum())}\n")
    out.write(f"candidates in an OCCUPIED grid cell: {int((~empty_flags).sum())}\n")
    out.write(
        f"{'group':<36}{'mean agreement':>16}{'median':>10}{'n':>6}\n"
        f"{'candidate in an empty cell':<36}{in_empty.mean():>16.6f}"
        f"{np.median(in_empty):>10.4f}{len(in_empty):>6}\n"
        f"{'candidate in an occupied cell':<36}{in_full.mean():>16.6f}"
        f"{np.median(in_full):>10.4f}{len(in_full):>6}\n"
    )
    ok = in_empty.mean() > in_full.mean()
    failures += 0 if ok else 1
    out.write(
        f"  {'PASS' if ok else 'FAIL'}  mean agreement is higher for candidates in empty "
        f"cells, by {in_empty.mean() - in_full.mean():+.6f}\n"
    )
    out.write(
        "  The grid cell is a coarse ground truth: a candidate can sit in an\n"
        "  occupied cell and still be in a thin part of it, which is precisely\n"
        "  the case the density model exists to see and the grid cannot.\n"
    )

    out.write(rule("3. is the agreement number saturated") + "\n")
    at_one = float(np.mean(scores.sparsity_agreement >= 0.999))
    at_zero = float(np.mean(scores.sparsity_agreement <= 0.001))
    out.write(f"  fraction of the pool at agreement 1.0:  {at_one:.4f}\n")
    out.write(f"  fraction of the pool at agreement 0.0:  {at_zero:.4f}\n")
    out.write(f"  fraction strictly between:              {1.0 - at_one - at_zero:.4f}\n")
    out.write(
        f"  declared sparsity quantile: {finder.sparsity_quantile:.2f}, so about "
        f"{finder.sparsity_quantile:.0%} of the pool\n  is inside the sparse set in any one "
        "resample and a saturated 1.0 for most of\n  the pool would be the warning sign.\n"
    )
    ok = at_one < 0.5
    failures += 0 if ok else 1
    out.write(f"  {'PASS' if ok else 'FAIL'}  fewer than half the pool sits at agreement 1.0\n")

    out.write(rule("4. the top five proposals, with their uncertainty") + "\n")
    proposals = finder.propose(pool_norm, 5)
    out.write(f"{'#':>3}{'log_density':>14}{'sd':>9}{'confidence':>12}  scenario\n")
    for p in proposals:
        record = pool[p.candidate_index]
        rendered = " ".join(f"{k}={record[k]}" for k in space.names)
        out.write(
            f"{p.order:>3}{p.log_density:>14.4f}{p.log_density_sd:>9.4f}"
            f"{p.sparsity_agreement:>12.4f}  {rendered}\n"
        )

    out.write(f"\nfailures: {failures}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
