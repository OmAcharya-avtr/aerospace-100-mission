"""Validation: the block interleaver minimum-spread closed form against brute force.

Claim under test
----------------
``BlockInterleaver.minimum_spread_closed_form()`` returns

    min(span + 1, depth + 1, depth + span - 2)   for depth, span >= 2
    2                                            if depth == 1 or span == 1
    0                                            for depth * span == 1

Reference
---------
The closed form is derived in ``src/interleavekit/block.py``. The reference it is
checked against is brute-force enumeration of all N(N-1)/2 index pairs, computing
``|i-j| + |pi[i]-pi[j]|`` directly. No external reference is involved: this check
is the derivation against the definition.

Every (depth, span) pair with depth * span <= 400 is tested.

Runtime: about 2 s on one core.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

from interleavekit import BlockInterleaver  # noqa: E402

MAX_N = 400


def brute_force_minimum_spread(pi: np.ndarray) -> int:
    """Minimum over all index pairs of |i-j| + |pi[i]-pi[j]|, by enumeration."""
    n = pi.size
    if n < 2:
        return 0
    i, j = np.triu_indices(n, k=1)
    return int((np.abs(j - i) + np.abs(pi[j] - pi[i])).min())


def main() -> int:
    cases = 0
    mismatches = 0
    worst = []
    for depth in range(1, MAX_N + 1):
        for span in range(1, MAX_N // depth + 1):
            if depth * span > MAX_N:
                break
            il = BlockInterleaver(depth, span)
            closed = il.minimum_spread_closed_form()
            brute = brute_force_minimum_spread(il.permutation())
            cases += 1
            if closed != brute:
                mismatches += 1
                if len(worst) < 10:
                    worst.append((depth, span, closed, brute))

    print("Block interleaver minimum spread: closed form vs brute-force enumeration")
    print(f"ceiling on depth * span: {MAX_N}")
    print(f"(depth, span) pairs tested: {cases}")
    print(f"mismatches: {mismatches}")
    if worst:
        print("first mismatches (depth, span, closed form, brute force):")
        for row in worst:
            print(f"  {row}")

    print()
    print("Sample of the closed form (depth, span, minimum spread):")
    for depth, span in [(1, 1), (1, 8), (8, 1), (2, 2), (4, 4), (2, 10), (10, 2), (16, 16)]:
        il = BlockInterleaver(depth, span)
        print(
            f"  depth={depth:>3} span={span:>3} N={il.length:>4} "
            f"minimum spread={il.minimum_spread_closed_form():>3}"
        )

    print()
    print("RESULT: PASS" if mismatches == 0 else "RESULT: FAIL")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
