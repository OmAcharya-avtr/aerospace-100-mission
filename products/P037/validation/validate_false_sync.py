"""False-sync probability against the combinatorial expression, Eq. (3).

The expression, derived
-----------------------
The detector declares the marker at a window position when the Hamming
distance between the L-bit received window and the L-bit reference is at
most T. For an i.i.d. uniform bit stream each of the L comparisons
mismatches with probability 1/2, independently of the others and
independently of the reference pattern, so the window distance D is
Binomial(L, 1/2) and

    P_fa(T) = P[D <= T] = 2^-L * sum_{k=0}^{T} C(L, k)

This script checks that expression three ways:

1. Against an independent evaluation, ``scipy.stats.binom.cdf(T, L, 0.5)``,
   over every T from 0 to L. Exact-integer arithmetic against a continuous
   library path.
2. Against Monte Carlo over random bit windows, at the T values whose
   probability is large enough to resolve with the frame counts the compute
   budget allows. Overlapping windows are correlated, so the acceptance band
   is widened to 6 sigma and the reason is stated.
3. Against a brute-force count over the whole 2^16 space for a 16-bit
   marker, where the expression can be verified exhaustively rather than
   statistically.

Run: ``python3 validate_false_sync.py`` from this directory.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
from scipy.stats import binom

sys.path.insert(0, "../src")

from framesync.asm import (  # noqa: E402
    ASM_32_HEX,
    asm_autocorrelation,
    bits_from_int,
    expected_false_syncs,
    false_sync_probability,
    hamming_distances,
)

SEED = 20261005
MC_WINDOWS = 2_000_000
MC_TOLERANCES = (8, 10, 12, 14, 16)


def main() -> int:
    t0 = time.time()
    all_pass = True
    print("framesync validation 3 -- ASM false-sync probability")
    print("=" * 78)
    print(f"marker 0x{ASM_32_HEX:08X}, L = 32 bits, MSB first")
    print("P_fa(T) = 2^-L * sum_{k=0}^{T} C(L, k)   (per window position)")
    print()

    print("Check 1: exact expression against scipy.stats.binom.cdf(T, 32, 0.5)")
    print("-" * 78)
    worst = 0.0
    for t in range(33):
        a = false_sync_probability(t, 32)
        b = float(binom.cdf(t, 32, 0.5))
        worst = max(worst, abs(a - b) / max(b, 1e-300))
    ok1 = worst < 1e-12
    all_pass &= ok1
    print(f"worst relative difference over T = 0..32: {worst:.3e}  "
          f"(tolerance 1e-12) -> {'PASS' if ok1 else 'FAIL'}")
    print()
    print("Operational thresholds, exact values (below Monte Carlo resolution):")
    print(f"{'T':>3}  {'P_fa(T)':>14}  {'expected false syncs in 1e9 bits':>34}")
    for t in range(0, 7):
        print(f"{t:3d}  {false_sync_probability(t):14.6e}  "
              f"{expected_false_syncs(10**9, t):34.6e}")
    print()

    print(f"Check 2: Monte Carlo over {MC_WINDOWS} random window positions")
    print("-" * 78)
    print("overlapping windows are correlated, so the band is 6 sigma of the")
    print("binomial standard error, not 2; the reason is the correlation, not a")
    print("loosened tolerance")
    rng = np.random.default_rng(SEED)
    stream = rng.integers(0, 2, MC_WINDOWS + 31, dtype=np.uint8)
    dist = hamming_distances(stream)
    n = int(dist.size)
    header = (
        f"{'T':>3}  {'hits':>9}  {'measured':>12}  {'Eq. (3)':>12}  "
        f"{'sigma':>11}  {'dev/sigma':>9}  result"
    )
    print(header)
    print("-" * len(header))
    for t in MC_TOLERANCES:
        hits = int(np.count_nonzero(dist <= t))
        measured = hits / n
        expected = false_sync_probability(t, 32)
        sigma = math.sqrt(expected * (1.0 - expected) / n)
        dev = abs(measured - expected) / sigma
        ok = dev < 6.0
        all_pass &= ok
        print(
            f"{t:3d}  {hits:9d}  {measured:12.6e}  {expected:12.6e}  "
            f"{sigma:11.4e}  {dev:9.2f}  {'PASS' if ok else 'FAIL'}"
        )
    print()

    print("Check 3: exhaustive count over the whole 2^16 space, L = 16 marker")
    print("-" * 78)
    pat16 = bits_from_int(0x1ACF, 16)
    words = np.arange(1 << 16, dtype=np.uint32)
    bits = ((words[:, None] >> np.arange(15, -1, -1)[None, :]) & 1).astype(np.uint8)
    dists = np.count_nonzero(bits ^ pat16, axis=1)
    ok3 = True
    print(f"{'T':>3}  {'exhaustive count':>16}  {'2^16 * P_fa':>14}  result")
    for t in range(0, 17):
        count = int(np.count_nonzero(dists <= t))
        predicted = false_sync_probability(t, 16) * (1 << 16)
        hit = count == round(predicted)
        ok3 &= hit
        if t <= 4 or t == 8 or t == 16:
            print(f"{t:3d}  {count:16d}  {predicted:14.1f}  "
                  f"{'PASS' if hit else 'FAIL'}")
    all_pass &= ok3
    print(f"all 17 thresholds exact: {ok3}")
    print()

    print("Marker autocorrelation (reported, not a pass/fail criterion)")
    print("-" * 78)
    ac = asm_autocorrelation()
    print(f"minimum Hamming distance over the 31 non-zero cyclic shifts: "
          f"{int(ac[1:].min())} bits")
    print(f"mean over non-zero shifts: {float(ac[1:].mean()):.2f} bits "
          f"(a random 32-bit pattern would average 16)")
    print()
    print(f"ALL CHECKS PASS: {all_pass}")
    print(f"wall time: {time.time() - t0:.1f} s")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
