"""Validation: the O(N) burst-dispersion fast path against its O(N*L) definition.

Claim under test
----------------
``burst_dispersion`` computes the worst-case surviving run from the span profile
``m(R)``, using the identity

    R consecutive source symbols can be hit by a burst of L  <=>  m(R) <= L - 1

rather than by scanning burst windows. The definitional scan is kept as
``burst_dispersion_by_window_scan``. The two must agree on every case.

This check exists because the fast path is the one the library actually uses and
the one every other number in this directory depends on. Replacing the scan with
the reformulation took one design search from 106 s to under 5 s, which is the
only reason the cost-trade-off sweep fits the compute budget at all.

Reference
---------
``burst_dispersion_by_window_scan``, which implements the definition directly:
for every admissible window, take the longest run of consecutive source indices
among the symbols the window damages, and maximise over windows.

Runtime: about 20 s on one core.
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
from interleavekit.metrics import (  # noqa: E402
    burst_dispersion,
    burst_dispersion_by_window_scan,
)


def compare(label: str, pos: np.ndarray, lengths, window=None) -> tuple[int, int]:
    """Return (checks, mismatches) for one position map over several burst lengths."""
    checks = 0
    mismatches = 0
    for length in lengths:
        fast = burst_dispersion(pos, int(length), window)
        slow = burst_dispersion_by_window_scan(pos, int(length), window)
        checks += 1
        if fast != slow:
            mismatches += 1
            print(f"   MISMATCH {label} L={length}: fast={fast} scan={slow}")
    return checks, mismatches


def main() -> int:
    total = 0
    bad = 0

    print("1. Block interleaver, every (depth, span) in 1..8, every burst length 1..N")
    for depth in range(1, 9):
        for span in range(1, 9):
            il = BlockInterleaver(depth, span)
            c, m = compare(
                f"block {depth}x{span}",
                il.position_of_input(),
                range(1, il.length + 1),
            )
            total += c
            bad += m

    print("2. Helical interleaver, rows and columns in 1..7, step 0..5")
    for rows in range(1, 8):
        for cols in range(1, 8):
            for step in range(0, 6):
                il = HelicalInterleaver(rows, cols, step)
                c, m = compare(
                    f"helical {rows}x{cols}s{step}",
                    il.position_of_input(),
                    range(1, min(il.length, 16) + 1),
                )
                total += c
                bad += m

    print("3. S-random interleaver, several lengths and seeds")
    for length, spread in [(32, 3), (64, 4), (128, 6)]:
        for seed in range(3):
            il = SRandomInterleaver(length, spread, seed=seed)
            c, m = compare(
                f"srandom {length}/{spread}/{seed}",
                il.position_of_input(),
                range(1, 25),
            )
            total += c
            bad += m

    print("4. Convolutional interleaver, on its steady-state range")
    for registers in range(2, 8):
        for slope in range(1, 4):
            ci = ConvolutionalInterleaver(registers, slope)
            n = ci.max_delay_symbols + 60
            pos = ci.transmitted_position(np.arange(n))
            window = ci.steady_state_range(n)
            c, m = compare(
                f"conv R{registers}M{slope}",
                pos,
                range(1, min(window[1] - window[0], 30) + 1),
                window,
            )
            total += c
            bad += m

    print()
    print(f"(position map, burst length) comparisons: {total}")
    print(f"mismatches between fast path and definitional window scan: {bad}")
    print("RESULT: PASS" if bad == 0 else "RESULT: FAIL")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
