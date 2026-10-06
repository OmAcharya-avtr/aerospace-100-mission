"""Interleaver latency and memory: measured by impulse, not asserted from a formula.

The latency and memory numbers in the README are what a designer trades depth
against, so they are measured from the implementations rather than taken from
equations (24) and (25). The block interleaver's spreading property is measured by
marking one codeword and finding the spacing of its symbols in the channel stream;
the convolutional interleaver's delay is measured by pushing a ramp through and
finding the shift at which input and output agree.

Run: ``PYTHONPATH=../src python3 validate_interleaver_cost.py``
Runtime: under 5 s on one core.
"""

from __future__ import annotations

import numpy as np

from codedfade.interleave import (
    BlockInterleaver,
    ConvolutionalInterleaver,
    burst_dispersion,
)

SPAN = 31
RS_HZ = 1.0e6
BITS = 5
T = 5


def measured_spacing(depth: int, span: int) -> int:
    """Channel-time spacing between adjacent symbols of one codeword."""
    il = BlockInterleaver(depth, span)
    marks = np.zeros(depth * span, dtype=np.int64)
    marks[:span] = 1
    positions = np.nonzero(il.interleave(marks))[0]
    diffs = np.unique(np.diff(positions))
    return int(diffs[0]) if diffs.size == 1 else -1


def measured_delay(branches: int, increment: int) -> int:
    """End-to-end delay of the convolutional interleaver, found by search.

    The record is sized from the construction's own nominal delay so that the
    search range always covers it; returning -1 would mean the delay is larger
    than ``M*B*(B-1)``, which is the thing being checked.
    """
    ci = ConvolutionalInterleaver(branches, increment)
    nominal = max(ci.total_delay_symbols, 1)
    n = 4 * nominal + 400
    x = np.arange(1, n + 1)
    out = ci.deinterleave(ci.interleave(x))
    for d in range(2 * nominal + 1):
        if np.array_equal(out[d:], x[: n - d]):
            return d
    return -1


def main() -> None:
    print("=" * 78)
    print("INTERLEAVER COST VALIDATION")
    print("=" * 78)
    print(f"span (codeword length)   {SPAN} symbols")
    print(f"symbol rate              {RS_HZ:.6e} Hz")
    print(f"bits per symbol          {BITS}")
    print(f"code correction radius t {T} symbols")
    print()

    print("-- block interleaver: spacing, latency, memory -----------------------")
    print(f"{'depth':>7} {'spacing':>9} {'eq(24) lat sym':>15} {'latency ms':>12} "
          f"{'memory B':>12} {'max burst t*D':>14}")
    for depth in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        il = BlockInterleaver(depth, SPAN)
        cost = il.cost(RS_HZ, bits_per_symbol=BITS)
        print(
            f"{depth:7d} {measured_spacing(depth, SPAN):9d} {cost.latency_symbols:15d} "
            f"{cost.latency_ms:12.4f} {cost.memory_bytes:12.1f} {T * depth:14d}"
        )
    print()
    print("The measured spacing column must equal the depth column exactly: that is")
    print("the property equation (23) relies on. The last column is the longest burst")
    print("that still deposits at most t errors in one codeword.")
    print()

    print("-- burst dispersion, equation (22) -----------------------------------")
    print(f"{'burst L':>9} " + " ".join(f"{f'D={d}':>7}" for d in (1, 8, 64, 512)))
    for burst in (1, 8, 20, 64, 200, 512, 2000):
        row = " ".join(f"{burst_dispersion(d, burst):7d}" for d in (1, 8, 64, 512))
        print(f"{burst:9d} {row}")
    print()

    print("-- convolutional interleaver: measured delay vs M*B*(B-1) ------------")
    print(f"{'branches B':>11} {'increment M':>12} {'M*B*(B-1)':>11} {'measured':>10} "
          f"{'memory sym':>11} {'block equiv':>12}")
    for branches, inc in ((2, 1), (4, 1), (4, 2), (8, 1), (8, 4), (16, 1), (31, 1)):
        ci = ConvolutionalInterleaver(branches, inc)
        bi_mem = BlockInterleaver(branches * inc, SPAN).cost(RS_HZ, BITS).memory_symbols
        print(
            f"{branches:11d} {inc:12d} {ci.total_delay_symbols:11d} "
            f"{measured_delay(branches, inc):10d} {ci.memory_symbols:11d} {bi_mem:12d}"
        )
    print()
    print("The 'block equiv' column is the block interleaver memory for a comparable")
    print("depth, for the halving claim. The convolutional construction stores less")
    print("for the same end-to-end delay, at the cost of a transient at start-up.")
    print()
    print("=" * 78)


if __name__ == "__main__":
    main()
