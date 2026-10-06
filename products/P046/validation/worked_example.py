"""The worked example reproduced in the README, with its printed output.

Runs the whole decision this library exists for: a link that must survive a burst
of 32 symbols, and the question of what each construction costs to deliver that.

Runtime: under 5 s on one core.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

from interleavekit import (  # noqa: E402
    BlockInterleaver,
    ConvolutionalInterleaver,
    cost_table,
    format_cost_table,
)
from interleavekit.metrics import (  # noqa: E402
    burst_dispersion,
    max_burst_fully_dispersed,
    minimum_spread,
)

SYMBOL_RATE_HZ = 2.5e6
TARGET_BURST = 32


def main() -> int:
    block = BlockInterleaver(depth=33, span=3)
    bank = ConvolutionalInterleaver(registers=2, slope=17)

    print(f"Target: fully disperse a channel burst of {TARGET_BURST} symbols")
    print(f"Symbol rate: {SYMBOL_RATE_HZ:.3g} symbols/s")
    print()

    pos = block.position_of_input()
    print(f"{block!r}")
    print(f"  block length                 {block.length} symbols")
    print(f"  minimum spread               {minimum_spread(block.permutation())}")
    print(f"  largest fully dispersed burst {max_burst_fully_dispersed(pos)} symbols")
    print(f"  surviving run at burst {TARGET_BURST}     "
          f"{burst_dispersion(pos, TARGET_BURST)} symbol(s)")
    print(f"  surviving run at burst {TARGET_BURST * 2}     "
          f"{burst_dispersion(pos, TARGET_BURST * 2)} symbols")
    cost = block.cost()
    print(f"  pair latency                 {cost.pair_latency_symbols} symbols "
          f"= {cost.latency_ms(SYMBOL_RATE_HZ):.4f} ms")
    print(f"  pair memory                  {cost.pair_memory_symbols} symbols "
          f"= {cost.memory_bytes(8):.0f} bytes at 8 bits/symbol")
    print()

    n = bank.max_delay_symbols + 400
    bank_pos = bank.transmitted_position(np.arange(n))
    window = bank.steady_state_range(n)
    print(f"{bank!r}")
    print(f"  register delays              {bank.register_delays().tolist()} symbols")
    print(f"  start-up transient           {bank.max_delay_symbols} symbols")
    print(f"  largest fully dispersed burst "
          f"{max_burst_fully_dispersed(bank_pos, window)} symbols")
    print(f"  surviving run at burst {TARGET_BURST}     "
          f"{burst_dispersion(bank_pos, TARGET_BURST, window)} symbol(s)")
    cost = bank.cost()
    print(f"  pair latency                 {cost.pair_latency_symbols} symbols "
          f"= {cost.latency_ms(SYMBOL_RATE_HZ):.4f} ms")
    print(f"  pair memory                  {cost.pair_memory_symbols} symbols "
          f"= {cost.memory_bytes(8):.0f} bytes at 8 bits/symbol")
    print()

    rows = cost_table(
        [block, bank], SYMBOL_RATE_HZ, burst_search_limit=80, conv_window=120
    )
    print(format_cost_table(rows, SYMBOL_RATE_HZ))
    print()
    ratio = rows[0].pair_memory_symbols / rows[1].pair_memory_symbols
    print(f"The bank delivers the same dispersed burst for {ratio:.2f}x less pair memory.")
    print("It gives no minimum-spread guarantee, so it is the wrong choice if the")
    print("permutation also has to feed a turbo decoder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
