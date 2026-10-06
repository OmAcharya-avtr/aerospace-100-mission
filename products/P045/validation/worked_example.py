"""The worked example printed in the README, so the README cannot drift from it.

Six questions, answered with the public API, in the order a practitioner would
ask them.

Runtime: about 25 s on one core.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
)

from arqlonghaul import (  # noqa: E402
    GilbertElliottChannel,
    HarqConfig,
    IndependentFrameChannel,
    crossover_rtt,
    gbn_throughput,
    harq_goodput_exact,
    preset,
    simulate_go_back_n,
    simulate_selective_repeat,
    size_window,
    sr_throughput,
)


def main() -> None:
    # 1. What does the link geometry force on me?
    link = preset("lunar")
    print(f"1. {link.name}")
    print(f"   N = {link.slots_per_cycle:.1f} slots, "
          f"bandwidth-delay = {link.bdp_bytes:.0f} bytes "
          f"= {link.bdp_frames:.1f} frames")
    print(f"   smallest window for continuous transmission: "
          f"{link.min_continuous_window} frames")

    # 2. What does the textbook window buy me at a 5 per cent frame error rate?
    sizing = size_window(link, 0.05)
    print(f"2. at the textbook window of {sizing.knee_frames} frames "
          f"({sizing.knee_bytes:.0f} bytes):")
    print(f"   selective repeat eta = {sizing.sr_goodput_at_knee:.4f} "
          f"-> {sizing.sr_bps_at_knee / 1e3:.1f} kbit/s")
    print(f"   go-back-N        eta = {sizing.gbn_goodput_at_knee:.4f} "
          f"-> {sizing.gbn_bps_at_knee / 1e3:.1f} kbit/s")

    # 3. Is the closed form telling me the truth? Simulate the state machine.
    n = 60
    rng = np.random.default_rng(0)
    iid = IndependentFrameChannel(0.05).errors(300_000, rng)
    sim = simulate_selective_repeat(iid, n, n)
    print(f"3. N = {n}, W = {n}, independent errors at 5 per cent:")
    print(f"   closed form says eta = {sr_throughput(0.05, n, n):.4f}")
    print(f"   the state machine measures eta = {sim.goodput:.4f} "
          f"+- {sim.goodput_stderr:.4f}")
    print("   the difference is head-of-line blocking, which the formula omits")

    # 4. Now make the errors bursty, holding the marginal rate fixed.
    burst = GilbertElliottChannel.from_mean_and_burst(0.05, 25.0)
    ge = burst.errors(300_000, rng)
    gbn_iid = simulate_go_back_n(iid, n, n)
    gbn_ge = simulate_go_back_n(ge, n, n)
    print(f"4. same 5 per cent marginal rate, bursts of "
          f"{burst.mean_burst_slots:.0f} frames:")
    print(f"   go-back-N closed form        eta = {gbn_throughput(0.05, n, n):.4f}")
    print(f"   go-back-N, independent errors eta = {gbn_iid.goodput:.4f}")
    print(f"   go-back-N, bursty errors      eta = {gbn_ge.goodput:.4f}")
    print(f"   the formula is wrong by "
          f"{100 * (gbn_throughput(0.05, n, n) / gbn_ge.goodput - 1):+.1f} per cent")

    # 5. HARQ: what does one incremental-redundancy configuration deliver?
    cfg = HarqConfig(
        k=200, n1=300, delta=40, max_rounds=4, esn0_db=1.0,
        rtt_symbols=500.0, alpha=0.5, scheme="type_ii",
    )
    res = harq_goodput_exact(cfg)
    print(f"5. type-II IR, k = {cfg.k}, first rate {cfg.first_rate:.3f}, "
          f"D = {cfg.rtt_symbols:.0f} symbols:")
    print(f"   goodput {res.goodput:.4f}, residual frame error "
          f"{res.residual_fer:.2e}, {res.mean_rounds:.3f} rounds per frame")

    # 6. And the crossover: retransmit, or pay for redundancy up front?
    cx = crossover_rtt(
        k=200, esn0_db=1.0, delta=40, retransmit_rate=0.90,
        upfront_rate=0.50, max_rounds=4, alpha=0.5,
    )
    print(f"6. crossover at D = {cx['crossover_d']:.0f} symbol times; this link's "
          f"D is {link.bdp_bits:.3g},")
    print(f"   a factor of {link.bdp_bits / cx['crossover_d']:.0f} past it, so for a "
          "single HARQ process")
    print("   with no pipelining the answer is always to pay up front")


if __name__ == "__main__":
    main()
