"""The exchangeability test: size on exchangeable data, power on a swept card.

Exchangeability is the load-bearing assumption of the bootstrap and conformal
bounds and the specification this package was built to requires it to be
STATED AND TESTED, not assumed. This script measures the test's SIZE -- how
often it rejects when the assumption holds, which must not exceed the declared
alpha -- and its POWER -- how often it rejects when the assumption fails,
which is what decides whether the test is worth running at a field sample size
of tens.

Both numbers are measured over many independent campaigns. A single campaign's
p-value is one draw and is not evidence either way: measured below, the
``swept`` campaign at n = 40 is rejected on 90 per cent of draws and NOT
rejected on the other 10, so a build that tested one draw would have had a one
in ten chance of concluding the test does not work.
"""

from __future__ import annotations

# Bytecode must not be written anywhere inside this tree. validation/
# validate_cli.py asserts that no __pycache__ directory and no .pyc file
# exists in the repository, so an interpreter that caches its own imports
# here makes that check report itself. It is set in each entry script,
# before _bootstrap is imported, and not inside _bootstrap, because CPython
# writes a module's cache BEFORE executing that module: _bootstrap cannot
# set this on its own behalf. The imports below therefore follow a statement
# and carry the same E402 waiver the post-bootstrap imports already carry.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

from _bootstrap import Tee, bootstrap, environment_lines  # noqa: E402

ROOT = bootstrap()

from transfergap import synthetic  # noqa: E402
from transfergap.exchangeability import permutation_test  # noqa: E402
from transfergap.report import format_table  # noqa: E402

ALPHA = 0.05
PERMUTATIONS = 399
CAMPAIGNS_PER_POINT = 300
SIZES = (15, 25, 40, 60, 120)


def rejection_rate(n: int, campaign: str, statistic: str, base_seed: int) -> tuple[float, float]:
    """Fraction of independent campaigns the test rejects, and its standard error."""
    rejected = 0
    for index in range(CAMPAIGNS_PER_POINT):
        gaps = synthetic.generate(n, campaign, seed=base_seed + index).gaps
        result = permutation_test(
            gaps, statistic, permutations=PERMUTATIONS, alpha=ALPHA, seed=index
        )
        rejected += int(result.rejected)
    rate = rejected / CAMPAIGNS_PER_POINT
    return rate, math.sqrt(max(rate * (1.0 - rate), 0.0) / CAMPAIGNS_PER_POINT)


def main() -> int:
    out = Tee("validate_exchangeability_output.txt")
    out.rule("transfergap -- the exchangeability test, size and power")
    for line in environment_lines():
        out(line)
    out("")
    out(f"declared alpha            : {ALPHA}")
    out(f"permutations per test     : {PERMUTATIONS} (smallest attainable p-value "
        f"{1.0 / (PERMUTATIONS + 1):.4f})")
    out(f"independent campaigns     : {CAMPAIGNS_PER_POINT} per table cell")
    out(f"binomial standard error at a rate of {ALPHA}: "
        f"{math.sqrt(ALPHA * (1 - ALPHA) / CAMPAIGNS_PER_POINT):.4f}")
    out("")
    out("The 'random' campaign draws every operating condition independently and is")
    out("exchangeable by construction. The 'swept' campaign sweeps Mach number")
    out("monotonically across the index, as a flight-test card is actually flown, so")
    out("the discrepancy distribution depends on where in the campaign a point sits")
    out("and the sample is NOT exchangeable in the index.")

    out.rule("1. SIZE on the exchangeable campaign -- must not exceed alpha")
    rows = []
    size_failures = 0
    for statistic in ("trend", "half_split"):
        for n in SIZES:
            rate, error = rejection_rate(n, "random", statistic, 500_000 + 7 * n)
            # Three standard errors of the DECLARED alpha, which is the
            # conditioning of this estimate, not an absolute constant.
            allowance = 3.0 * math.sqrt(ALPHA * (1 - ALPHA) / CAMPAIGNS_PER_POINT)
            ok = rate <= ALPHA + allowance
            if not ok:
                size_failures += 1
            rows.append([statistic, str(n), f"{rate:.4f}", f"{error:.4f}",
                         f"{ALPHA + allowance:.4f}", "ok" if ok else "TOO LARGE"])
    out(format_table(["statistic", "n", "rejection rate", "+/- se", "alpha + 3 se",
                      "verdict"], rows))
    out("")
    out.check(
        "the test's size does not exceed the declared alpha on exchangeable data",
        size_failures == 0,
        f"{len(rows)} cells, {size_failures} above alpha + 3 se",
    )

    out.rule("2. POWER on the deliberately non-exchangeable swept campaign")
    rows = []
    powers: dict[tuple[str, int], float] = {}
    for statistic in ("trend", "half_split"):
        for n in SIZES:
            rate, error = rejection_rate(n, "swept", statistic, 600_000 + 11 * n)
            powers[(statistic, n)] = rate
            rows.append([statistic, str(n), f"{rate:.4f}", f"{error:.4f}"])
    out(format_table(["statistic", "n", "rejection rate (power)", "+/- se"], rows))
    out("")
    out("REPORTED AS FOUND. The power is not 1.0 at the sample sizes this product is")
    out("about, and that is the honest and useful number: at n = 15 the test misses a")
    out("genuinely non-exchangeable campaign most of the time. A programme with")
    out("fifteen paired points cannot establish that its conformal bound is valid,")
    out("and the distribution-free bound of equation (3), which needs no such")
    out("assumption, is the only answer available to it.")
    out("")
    for statistic in ("trend", "half_split"):
        out.check(
            f"the {statistic} statistic has power above 0.80 at n = 60 or more",
            min(powers[(statistic, 60)], powers[(statistic, 120)]) > 0.80,
            f"n=60 {powers[(statistic, 60)]:.4f}, n=120 {powers[(statistic, 120)]:.4f}",
        )
        out.check(
            f"the {statistic} statistic has power well below 1.0 at n = 15",
            powers[(statistic, 15)] < 0.80,
            f"n=15 power {powers[(statistic, 15)]:.4f}; a test that claimed to detect "
            f"drift from fifteen points would be the defect here",
        )

    out.rule("3. a single campaign is not evidence")
    out("The campaign at n = 40, seed 0, of the 'swept' shape -- the one the command")
    out("line prints by default -- is shown here against the measured power, so that")
    out("a reader who runs that command and sees a non-rejection knows why.")
    out("")
    rows = []
    for statistic in ("trend", "half_split"):
        result = permutation_test(
            synthetic.generate(40, "swept", seed=0).gaps, statistic,
            permutations=PERMUTATIONS, alpha=ALPHA, seed=0,
        )
        rows.append([statistic, f"{result.observed:.4f}", f"{result.p_value:.4f}",
                     "REJECTED" if result.rejected else "not rejected",
                     f"{powers[(statistic, 40)]:.4f}"])
    out(format_table(["statistic", "observed", "p", "verdict on this one draw",
                      "measured power at n=40"], rows))
    out("")
    out("A hard case for comparison: 40 strictly increasing values, where the trend")
    out("statistic takes its maximum of 1.0 and no permutation can match it.")
    hard = permutation_test(np.arange(40.0), "trend", permutations=PERMUTATIONS, seed=1)
    out(f"  observed {hard.observed:.6f}, p = {hard.p_value:.6f} (the floor, "
        f"1/(B+1) = {1.0 / (PERMUTATIONS + 1):.6f})")
    out.check(
        "a strictly monotone sample gives the smallest attainable p-value",
        hard.p_value == 1.0 / (PERMUTATIONS + 1),
        f"p = {hard.p_value:.6f}",
    )
    out.check(
        "the p-value is never zero, as equation (18) requires",
        hard.p_value > 0.0,
        "the +1 in both numerator and denominator is what makes this a valid p-value",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
