"""Validation: de-interleave o interleave = identity, and permutations are bijections.

Claims under test
-----------------
1. For every construction and every parameter set below,
   ``deinterleave(interleave(x)) == x`` elementwise and exactly, with no tolerance.
2. Every block construction's permutation is a bijection on its index set.
3. The convolutional interleaver's transmitted-position map is injective, which is
   the corresponding statement for a construction that is not a permutation of a
   contiguous block.
4. For the block interleaver, the closed-form inverse
   ``position_of_input()`` equals ``numpy.argsort(permutation())``.

Reference
---------
The identity is the definition of a de-interleaver; there is no external number to
compare against. The point of the check is that it holds at the edges -- depth 1,
span 1, step 0, slope 0, length 1 -- where an off-by-one would otherwise hide.

Runtime: about 3 s on one core.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

from interleavekit import (  # noqa: E402
    BlockInterleaver,
    ConvolutionalInterleaver,
    HelicalInterleaver,
    SRandomInterleaver,
)
from interleavekit.metrics import is_bijection  # noqa: E402

RNG = np.random.default_rng(20261006)


def main() -> int:
    failures = 0
    checks = 0

    print("1. Block interleaver: round trip and bijection")
    block_cases = [(d, s) for d in range(1, 13) for s in range(1, 13)]
    bad = []
    for depth, span in block_cases:
        il = BlockInterleaver(depth, span)
        x = RNG.integers(-10**6, 10**6, size=il.length)
        ok = bool(np.array_equal(il.deinterleave(il.interleave(x)), x))
        ok &= is_bijection(il.permutation())
        ok &= bool(np.array_equal(il.position_of_input(), np.argsort(il.permutation())))
        checks += 1
        if not ok:
            failures += 1
            bad.append((depth, span))
    print(f"   parameter sets: {len(block_cases)}  failures: {len(bad)}")
    if bad:
        print(f"   failing: {bad[:10]}")

    print("2. Helical interleaver: round trip and bijection, step 0..12")
    hel_cases = [
        (r, c, st) for r in range(1, 10) for c in range(1, 10) for st in range(0, 13)
    ]
    bad = []
    for rows, cols, step in hel_cases:
        il = HelicalInterleaver(rows, cols, step)
        x = RNG.integers(-10**6, 10**6, size=il.length)
        ok = bool(np.array_equal(il.deinterleave(il.interleave(x)), x))
        ok &= is_bijection(il.permutation())
        checks += 1
        if not ok:
            failures += 1
            bad.append((rows, cols, step))
    print(f"   parameter sets: {len(hel_cases)}  failures: {len(bad)}")
    if bad:
        print(f"   failing: {bad[:10]}")

    print("3. S-random interleaver: round trip and bijection")
    sr_cases = [(1, 1), (16, 2), (64, 4), (128, 6), (256, 8), (512, 10)]
    bad = []
    for length, spread in sr_cases:
        il = SRandomInterleaver(length, spread, seed=0)
        x = RNG.integers(-10**6, 10**6, size=il.length)
        ok = bool(np.array_equal(il.deinterleave(il.interleave(x)), x))
        ok &= is_bijection(il.permutation())
        checks += 1
        if not ok:
            failures += 1
            bad.append((length, spread))
    print(f"   parameter sets: {len(sr_cases)}  failures: {len(bad)}")
    if bad:
        print(f"   failing: {bad[:10]}")

    print("4. Convolutional interleaver: round trip and injective positions")
    conv_cases = [
        (r, m) for r in range(1, 11) for m in range(0, 5)
    ]
    bad = []
    for registers, slope in conv_cases:
        ci = ConvolutionalInterleaver(registers, slope)
        n = ci.max_delay_symbols + 40
        x = np.arange(1, n + 1)
        ok = bool(np.array_equal(ci.deinterleave(ci.interleave(x)), x))
        pos = ci.transmitted_position(np.arange(n))
        ok &= bool(np.unique(pos).size == n)
        checks += 1
        if not ok:
            failures += 1
            bad.append((registers, slope))
    print(f"   parameter sets: {len(conv_cases)}  failures: {len(bad)}")
    if bad:
        print(f"   failing: {bad[:10]}")

    print()
    print(f"total parameter sets checked: {checks}")
    print(f"total failures: {failures}")
    print("RESULT: PASS" if failures == 0 else "RESULT: FAIL")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
