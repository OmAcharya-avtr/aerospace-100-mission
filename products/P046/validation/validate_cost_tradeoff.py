"""Validation: latency and memory actually paid to disperse a given burst.

Claim under test
----------------
For a stated burst length that must be fully dispersed, the shift-register-bank
(convolutional) interleaver costs strictly less pair memory and pair latency than
any buffered block or helical permutation, by a factor measured here rather than
quoted.

Every candidate's burst dispersion is **measured** with
``max_burst_fully_dispersed``; nothing is predicted from a depth parameter. The
search is ``interleavekit.cost.cheapest_for_burst``, which minimises pair memory
in symbols with pair latency as the tie-break.

Cost models
-----------
Block and helical: full-block buffering, so each end holds N = depth * span
symbols and the pair adds 2N symbol times. Conventional, stated in
``src/interleavekit/base.py``.

Convolutional: shift-register bank, pair latency
``registers * slope * (registers - 1)`` and pair memory
``slope * registers * (registers - 1)``, derived in
``src/interleavekit/convolutional.py``. The latency constant agrees with the
total delay documented for the MathWorks Communications Toolbox Convolutional
Interleaver block, N x slope x (N - 1) for N rows of shift registers
(documentation read on 2026-10-06).

A note on the commonly quoted factor
------------------------------------
The folklore figure is that a convolutional interleaver needs half the memory of
a block interleaver for the same job. The measured ratio under these two cost
models is larger than two, and the reason is stated with the numbers below: the
cheapest block candidate at a target burst L is depth L + 1 at span 2, so
N = 2L + 2 and the pair pays 4L + 4, while the cheapest bank is two registers of
slope about L/2, paying about L + 2. Whether a reader should accept the larger
ratio depends on whether they accept full-block buffering as the block cost
model; an implementation that overlaps read-out with write-in pays less. The
ratio is reported, not asserted to be universal.

Runtime: about 40 s on one core.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from interleavekit.cost import cheapest_for_burst, format_cost_table  # noqa: E402

SYMBOL_RATE_HZ = 1.0e6
BITS_PER_SYMBOL = 8
TARGETS = [4, 8, 16, 32, 64]


def main() -> int:
    print("Cheapest parameter set per family that fully disperses a target burst")
    print(f"symbol rate {SYMBOL_RATE_HZ:.6g} sym/s, {BITS_PER_SYMBOL} bits per symbol")
    print("burst dispersion of every candidate measured, not derived from depth")
    print()

    failures = 0
    summary = []
    for target in TARGETS:
        best = cheapest_for_burst(
            target,
            symbol_rate_hz=SYMBOL_RATE_HZ,
            max_block_symbols=1024,
            max_registers=48,
            max_slope=48,
        )
        rows = [r for r in best.values() if r is not None]
        print(f"--- target burst {target} symbols ---")
        if rows:
            print(format_cost_table(rows, SYMBOL_RATE_HZ))
        for family, row in best.items():
            if row is None:
                print(f"   {family}: no candidate inside the search ceilings")
                failures += 1
        print()

        block = best["block"]
        conv = best["convolutional"]
        helical = best["helical"]
        if block is None or conv is None or helical is None:
            continue
        for row in (block, conv, helical):
            if row.max_burst_fully_dispersed < target:
                print(f"   CHECK FAILED: {row.name} disperses only "
                      f"{row.max_burst_fully_dispersed} < {target}")
                failures += 1
        summary.append(
            (
                target,
                block.pair_memory_symbols,
                helical.pair_memory_symbols,
                conv.pair_memory_symbols,
                block.pair_memory_symbols / conv.pair_memory_symbols,
                block.pair_latency_ms,
                conv.pair_latency_ms,
                block.pair_latency_symbols / conv.pair_latency_symbols,
            )
        )

    print("Summary: pair memory in symbols, and the block-to-bank ratios")
    print(
        f"{'burst':>6} {'block':>7} {'helical':>8} {'bank':>6} {'mem ratio':>10} "
        f"{'block ms':>9} {'bank ms':>8} {'lat ratio':>10}"
    )
    for row in summary:
        print(
            f"{row[0]:>6} {row[1]:>7} {row[2]:>8} {row[3]:>6} {row[4]:>10.3f} "
            f"{row[5]:>9.4f} {row[6]:>8.4f} {row[7]:>10.3f}"
        )

    if summary:
        ratios = [r[4] for r in summary]
        print()
        print(f"memory ratio range over targets {TARGETS}: "
              f"{min(ratios):.3f} to {max(ratios):.3f}")
        if min(ratios) <= 1.0:
            print("   CHECK FAILED: the bank did not beat the block on memory everywhere")
            failures += 1

    print()
    print("RESULT: PASS" if failures == 0 else f"RESULT: FAIL ({failures} checks failed)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
