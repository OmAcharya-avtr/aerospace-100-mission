"""Validation: measured burst dispersion of each construction.

Claims under test
-----------------
1. **The textbook rule "a block interleaver of depth D fully disperses a burst of
   D" is false at span 2.** At span 2 the source index in the last column is
   adjacent to the first column of the next row, and those two symbols sit only
   D - 1 transmitted positions apart, so the largest fully dispersed burst is
   D - 1. From span 3 upward the rule holds exactly. Measured, not assumed.
2. The helical interleaver's largest fully dispersed burst grows with its step:
   measured against the step at fixed array size.
3. The convolutional interleaver's largest fully dispersed burst, measured on its
   steady-state range so that start-up fill symbols cannot flatter it.

Reference
---------
The metric is ``max_burst_fully_dispersed``, which is ``m(2)``, the narrowest
transmitted window that can hold two adjacent source symbols. It is checked
against the definitional window scan in
``validate_burst_metric_equivalence.py``, and here it is additionally checked
against a direct computation of ``min |pos[j+1] - pos[j]|`` for every block case.

Runtime: about 5 s on one core.
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
)
from interleavekit.metrics import (  # noqa: E402
    burst_dispersion,
    max_burst_fully_dispersed,
)


def main() -> int:
    failures = 0

    print("1. Block interleaver: largest fully dispersed burst vs depth and span")
    print(f"   {'depth':>6} {'span':>5} {'N':>6} {'measured':>9} {'depth?':>7} {'direct':>7}")
    rule_holds_from_span = {}
    direct_mismatches = 0
    for depth in [4, 8, 16, 32]:
        for span in [1, 2, 3, 4, 8, 16]:
            il = BlockInterleaver(depth, span)
            pos = il.position_of_input()
            measured = max_burst_fully_dispersed(pos)
            direct = int(np.abs(np.diff(pos)).min()) if pos.size > 1 else 0
            if measured != direct:
                direct_mismatches += 1
            flag = "yes" if measured == depth else f"no ({measured - depth:+d})"
            print(
                f"   {depth:>6} {span:>5} {il.length:>6} {measured:>9} {flag:>7} {direct:>7}"
            )
            rule_holds_from_span.setdefault(depth, {})[span] = measured == depth
    print(f"   measured vs direct min adjacent gap: {direct_mismatches} mismatches")
    if direct_mismatches:
        failures += 1

    print()
    print("   The rule 'largest fully dispersed burst == depth':")
    for depth, by_span in rule_holds_from_span.items():
        holds = sorted(s for s, ok in by_span.items() if ok)
        fails = sorted(s for s, ok in by_span.items() if not ok)
        print(f"     depth {depth:>3}: holds at span {holds}, fails at span {fails}")

    print()
    print("2. Block interleaver at span 2: measured burst is depth - 1, not depth")
    span2_ok = True
    for depth in [4, 8, 16, 32, 64, 127]:
        il = BlockInterleaver(depth, 2)
        measured = max_burst_fully_dispersed(il.position_of_input())
        expected = depth - 1
        ok = measured == expected
        span2_ok &= ok
        verdict = "ok" if ok else "MISMATCH"
        print(
            f"   depth={depth:>4} measured={measured:>4} "
            f"depth-1={expected:>4} {verdict}"
        )
    if not span2_ok:
        failures += 1

    print()
    print("3. Helical interleaver, 16x16 array: burst dispersion vs step")
    print(f"   {'step':>5} {'N':>5} {'burst':>6} {'rows*step-1':>12}")
    for step in range(0, 10):
        il = HelicalInterleaver(16, 16, step)
        measured = max_burst_fully_dispersed(il.position_of_input())
        print(f"   {step:>5} {il.length:>5} {measured:>6} {16 * step - 1:>12}")

    print()
    print("4. Convolutional interleaver on its steady-state range")
    print(f"   {'R':>4} {'slope':>6} {'pair mem':>9} {'pair lat':>9} {'burst':>6} {'R*slope':>8}")
    for registers in [2, 4, 8, 16]:
        for slope in [1, 2, 4]:
            ci = ConvolutionalInterleaver(registers, slope)
            n = ci.max_delay_symbols + 4 * registers * slope + 200
            pos = ci.transmitted_position(np.arange(n))
            window = ci.steady_state_range(n)
            measured = max_burst_fully_dispersed(pos, window)
            cost = ci.cost()
            print(
                f"   {registers:>4} {slope:>6} {cost.pair_memory_symbols:>9} "
                f"{cost.pair_latency_symbols:>9} {measured:>6} {registers * slope:>8}"
            )

    print()
    print("5. No interleaving: a burst of L damages L consecutive source symbols")
    identity = np.arange(64)
    row = [burst_dispersion(identity, length) for length in (1, 2, 4, 8, 16)]
    print(f"   burst lengths (1, 2, 4, 8, 16) -> surviving runs {row}")
    if row != [1, 2, 4, 8, 16]:
        failures += 1

    print()
    print("RESULT: PASS" if failures == 0 else f"RESULT: FAIL ({failures} checks failed)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
