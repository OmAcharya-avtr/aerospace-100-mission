"""Validation: what spread the S-random search actually reaches.

Claims under test
-----------------
1. Every permutation the search returns satisfies the S-random condition it was
   asked for. Verified with ``s_parameter``, which re-derives the condition from
   the permutation instead of trusting the search.
2. The bound ``minimum_spread >= s_parameter + 1`` holds, which is proved in
   ``src/interleavekit/metrics.py``.
3. The search is bounded, not optimal. This script measures the largest spread it
   reaches at each length and compares that to the ``sqrt(length / 2)`` rule of
   thumb, so a reader can see where the rule of thumb holds here and where this
   implementation falls short of it. **The shortfall is the honest result and it
   is reported, not hidden.**

Reference
---------
The ``sqrt(length / 2)`` figure is the rule of thumb quoted for the practical
ceiling of S-random search in the turbo-code literature. It is used here only as a
yardstick to report against. No claim is made that this implementation attains it,
and the measured shortfall below shows that at lengths above 128 it does not.

Note that the scan below does not stop at the first failure: a bounded randomised
search is not monotone in the requested spread, so every spread is attempted and
both the largest success and any gaps are reported.

Runtime: about 45 s on one core.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

from interleavekit import SRandomInterleaver  # noqa: E402
from interleavekit.metrics import minimum_spread, s_parameter  # noqa: E402

LENGTHS = [32, 64, 128, 256, 512, 1024]


def main() -> int:
    failures = 0

    print("1. Achieved S-parameter against the request, and the spread bound")
    print(f"{'length':>7} {'request':>8} {'achieved S':>11} {'min spread':>11} {'S+1 <= ms':>10}")
    for length, spread in [(64, 4), (128, 6), (256, 8), (512, 10), (1024, 12)]:
        il = SRandomInterleaver(length, spread, seed=0)
        pi = il.permutation()
        achieved = s_parameter(pi)
        ms = minimum_spread(pi)
        bound_ok = ms >= achieved + 1
        if achieved < spread:
            print(f"   CHECK FAILED: asked {spread}, achieved {achieved}")
            failures += 1
        if not bound_ok:
            print(f"   CHECK FAILED: minimum spread {ms} < S + 1 = {achieved + 1}")
            failures += 1
        print(
            f"{length:>7} {spread:>8} {achieved:>11} {ms:>11} {'yes' if bound_ok else 'NO':>10}"
        )

    print()
    print("2. Largest spread this bounded search reaches, against sqrt(length/2)")
    print(
        f"{'length':>7} {'sqrt(N/2)':>10} {'largest':>8} {'ratio':>7} "
        f"{'gaps below largest':>20}"
    )
    for length in LENGTHS:
        ceiling = float(np.sqrt(length / 2))
        reached = []
        failed = []
        for spread in range(1, int(ceiling) + 2):
            try:
                # max_attempts is cut from the default 20 to 8 here so that the whole
                # scan stays inside the 3-minute compute budget on a shared pair of
                # cores. A longer budget reaches slightly larger spreads; the point of
                # the table is the shortfall against the rule of thumb, not a record.
                il = SRandomInterleaver(length, spread, seed=0, max_attempts=8)
            except ValueError:
                failed.append(spread)
                continue
            achieved = s_parameter(il.permutation())
            if achieved < spread:
                print(f"   CHECK FAILED: length {length} spread {spread} achieved {achieved}")
                failures += 1
            reached.append(spread)
        largest = max(reached) if reached else 0
        gaps = [s for s in failed if s < largest]
        print(
            f"{length:>7} {ceiling:>10.2f} {largest:>8} {largest / ceiling:>7.2f} "
            f"{str(gaps):>20}"
        )

    print()
    print("3. Determinism: the same (length, spread, seed) gives the same permutation")
    same = 0
    different = 0
    for seed in range(6):
        a = SRandomInterleaver(128, 5, seed=seed).permutation()
        b = SRandomInterleaver(128, 5, seed=seed).permutation()
        if np.array_equal(a, b):
            same += 1
        else:
            different += 1
            failures += 1
    print(f"   seeds tested: {same + different}  reproduced: {same}  differed: {different}")

    print()
    print("4. Distinct seeds give distinct permutations")
    perms = [tuple(SRandomInterleaver(128, 5, seed=s).permutation().tolist()) for s in range(8)]
    distinct = len(set(perms))
    print(f"   seeds 0..7 gave {distinct} distinct permutations out of 8")
    if distinct != 8:
        failures += 1

    print()
    print("RESULT: PASS" if failures == 0 else f"RESULT: FAIL ({failures} checks failed)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
