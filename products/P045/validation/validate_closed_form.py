"""Validation: the protocol simulator against the classical closed forms.

This is the check that licenses everything else.  The three state-machine
simulators in ``arqlonghaul.protocols`` are run on *independent* frame errors
with large windows, which is the one regime where the textbook throughput
expressions are exact, and the measured goodput is compared with them.  Only
after that agreement is established is the simulator used on a correlated
channel, where no closed form exists.

Checks
------
1. Stop-and-wait against ``eta = (1-p)/N``.
2. Go-back-N against ``eta = (1-p)/(1-p+Np)``, with ``W = ceil(N)`` so the
   sender transmits continuously, which the expression assumes.
3. Selective repeat against ``eta = 1-p`` with a window far above the
   head-of-line requirement, which is the infinite-buffer case the expression
   assumes.
4. The consistency identity ``sr_throughput(p, N, 1) == sw_throughput(p, N)``,
   and the window-limited expression at W = 1 where it is exact.
5. Where the closed forms *break*, measured and reported rather than hidden:
   the window-limited selective-repeat expression ``eta = (1-p) W/N`` assumes a
   window that never blocks, which is false for 1 < W < N, and the ideal
   expression ``eta = 1-p`` assumes an unbounded window.  Both shortfalls are
   tabulated.

All deviations are reported both as a relative error and as a z score against
the batch-means standard error of the simulation, because a relative error
without its Monte Carlo uncertainty is not evidence.

Runtime: about 100 s on one core.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
)

from arqlonghaul import closedform as cf  # noqa: E402
from arqlonghaul.channel import IndependentFrameChannel  # noqa: E402
from arqlonghaul.protocols import (  # noqa: E402
    simulate_go_back_n,
    simulate_selective_repeat,
    simulate_stop_and_wait,
)

SLOTS_SLIDING = 1_200_000
"""Slots per sliding-window run.  Sets the Monte Carlo standard error."""

SLOTS_PER_CYCLE_SW = 40_000
"""Stop-and-wait makes one attempt per N slots, so its run length scales with N."""

P_GRID = (0.001, 0.01, 0.05, 0.10, 0.20, 0.40)
N_GRID = (5, 20, 60, 200)
REL_SE_LIMIT = 0.005
"""Cases whose own relative standard error exceeds this are excluded from the
relative-deviation claim, because a relative deviation smaller than its own
uncertainty is not evidence of agreement or of disagreement."""


def _row(label: str, sim: float, ref: float, se: float) -> tuple[str, float, float]:
    """Print one comparison row and return (label, rel error, z)."""
    rel = sim / ref - 1.0 if ref > 0 else float("nan")
    z = (sim - ref) / se if se > 0 and math.isfinite(se) else float("nan")
    print(f"{label:<34}{sim:>12.6f}{ref:>12.6f}{100 * rel:>11.3f}{z:>9.2f}")
    return label, abs(rel), abs(z)


def main() -> None:
    rng = np.random.default_rng(45045)
    worst_z: dict[str, tuple[str, float, float]] = {}
    worst_rel: dict[str, tuple[str, float, float]] = {}

    def track(key: str, entry: tuple[str, float, float], rel_se: float) -> None:
        cur = worst_z.get(key)
        if cur is None or entry[2] > cur[2]:
            worst_z[key] = entry
        if rel_se < REL_SE_LIMIT:
            cur = worst_rel.get(key)
            if cur is None or entry[1] > cur[1]:
                worst_rel[key] = entry

    print(f"slots per sliding-window run: {SLOTS_SLIDING}")
    print(f"stop-and-wait attempts per run: {SLOTS_PER_CYCLE_SW}")
    print("channel: independent frame errors (the regime where the closed forms")
    print("are exact)")
    print()
    print("=" * 78)
    print("1-3. Each protocol against its closed form, in its exact regime")
    print("=" * 78)
    print(f"{'case':<34}{'simulated':>12}{'closed form':>12}{'rel %':>11}{'z':>9}")
    for n in N_GRID:
        for p in P_GRID:
            sw_slots = SLOTS_PER_CYCLE_SW * n
            sw_errors = IndependentFrameChannel(p).errors(sw_slots, rng)
            sw = simulate_stop_and_wait(sw_errors, n)
            ref = cf.sw_throughput(p, n)
            track(
                "stop_and_wait",
                _row(f"SW   N={n:<4} p={p:<6g}", sw.goodput, ref, sw.goodput_stderr),
                sw.goodput_stderr / ref if ref else 1.0,
            )
            del sw_errors
            errors = IndependentFrameChannel(p).errors(SLOTS_SLIDING, rng)
            w = math.ceil(n)
            gbn = simulate_go_back_n(errors, n, w)
            ref = cf.gbn_throughput(p, n, w)
            track(
                "go_back_n",
                _row(
                    f"GBN  N={n:<4} p={p:<6g} W={w}",
                    gbn.goodput,
                    ref,
                    gbn.goodput_stderr,
                ),
                gbn.goodput_stderr / ref if ref else 1.0,
            )
            w_big = max(40 * n, 400)
            sr = simulate_selective_repeat(errors, n, w_big)
            ref = cf.sr_throughput(p, n)
            track(
                "selective_repeat",
                _row(
                    f"SR   N={n:<4} p={p:<6g} W={w_big}",
                    sr.goodput,
                    ref,
                    sr.goodput_stderr,
                ),
                sr.goodput_stderr / ref if ref else 1.0,
            )
        print("-" * 78)

    print()
    print("=" * 78)
    print("4. Window-limited expression at W = 1, where it is exact, and the")
    print("   consistency identity sr_throughput(p, N, 1) == sw_throughput(p, N)")
    print("=" * 78)
    print(f"{'case':<34}{'simulated':>12}{'closed form':>12}{'rel %':>11}{'z':>9}")
    for n in (60, 200):
        for p in (0.01, 0.05, 0.20):
            errors = IndependentFrameChannel(p).errors(SLOTS_SLIDING, rng)
            sr = simulate_selective_repeat(errors, n, 1)
            ref = cf.sr_throughput(p, n, 1)
            track(
                "sr_window_1",
                _row(f"SRw  N={n:<4} p={p:<6g} W=1", sr.goodput, ref, sr.goodput_stderr),
                sr.goodput_stderr / ref if ref else 1.0,
            )
    max_gap = 0.0
    for n in N_GRID:
        for p in P_GRID:
            max_gap = max(
                max_gap, abs(cf.sr_throughput(p, n, 1) - cf.sw_throughput(p, n))
            )
    print(f"identity: largest absolute difference over "
          f"{len(N_GRID) * len(P_GRID)} grid points: {max_gap:.3e}")

    print()
    print("=" * 78)
    print("WORST-CASE AGREEMENT IN THE EXACT REGIME")
    print("=" * 78)
    print(f"{'protocol':<20}{'worst |z| case':<34}{'rel %':>9}{'|z|':>8}")
    for key, (label, rel, z) in worst_z.items():
        print(f"{key:<20}{label:<34}{100 * rel:>9.3f}{z:>8.2f}")
    print()
    print(f"{'protocol':<20}{'worst |rel| case (rel SE < 0.5%)':<34}{'rel %':>9}{'|z|':>8}")
    for key, (label, rel, z) in worst_rel.items():
        print(f"{key:<20}{label:<34}{100 * rel:>9.3f}{z:>8.2f}")
    overall_z = max(v[2] for v in worst_z.values())
    overall_rel = max(v[1] for v in worst_rel.values()) if worst_rel else float("nan")
    print()
    print(f"cases compared:                                     "
          f"{len(N_GRID) * len(P_GRID) * 3 + 6}")
    print(f"worst |z| against the batch-means standard error:   {overall_z:.2f}")
    print(f"worst |relative| where rel SE < 0.5 %:              "
          f"{100 * overall_rel:.3f} %")
    print()
    print("Tolerance adopted: |z| <= 4 on every case, and |relative| <= 1.0 % on")
    print("every case whose own Monte Carlo relative standard error is below")
    print("0.5 %. The z criterion is the real one: it is what distinguishes")
    print("Monte Carlo noise from a modelling error. The relative criterion is")
    print("the engineering statement, and it is only meaningful where the")
    print("simulation is precise enough to support it.")
    print(f"PASS (z <= 4):          {overall_z <= 4.0}")
    print(f"PASS (relative <= 1 %): {overall_rel <= 0.01}")

    print()
    print("=" * 78)
    print("5. WHERE THE CLOSED FORMS BREAK")
    print("=" * 78)
    print("5a. The window-limited expression eta = (1-p) W/N for 1 < W < N.")
    print("    It assumes the sender always holds W transmissions in flight.")
    print("    A real selective-repeat sender cannot advance its window base")
    print("    past an unacknowledged frame, so a frame under retransmission")
    print("    holds its sequence number and the window fills behind it.")
    print()
    print(f"{'N':>5}{'p':>8}{'W':>8}{'W/N':>7}{'simulated':>12}"
          f"{'(1-p)W/N':>12}{'misstated %':>13}")
    for n in (60, 200):
        for p in (0.01, 0.05, 0.20):
            errors = IndependentFrameChannel(p).errors(SLOTS_SLIDING, rng)
            for w in (1, max(2, n // 8), n // 2):
                sr = simulate_selective_repeat(errors, n, w)
                ref = cf.sr_throughput(p, n, w)
                print(
                    f"{n:>5}{p:>8g}{w:>8}{w / n:>7.3f}{sr.goodput:>12.6f}"
                    f"{ref:>12.6f}{100 * (ref / sr.goodput - 1):>13.2f}"
                )
    print()
    print("5b. The ideal expression eta = 1-p, which assumes an unbounded")
    print("    window. How much window selective repeat actually needs.")
    print()
    print(f"{'N':>5}{'p':>8}{'W':>8}{'W/N':>7}{'simulated':>12}{'ideal 1-p':>12}"
          f"{'shortfall %':>13}")
    for n in (60,):
        for p in (0.05, 0.20, 0.40):
            errors = IndependentFrameChannel(p).errors(SLOTS_SLIDING, rng)
            for w in (n, 2 * n, 3 * n, 4 * n, 6 * n, 10 * n, 40 * n):
                sr = simulate_selective_repeat(errors, n, w)
                ideal = cf.sr_throughput(p, n)
                print(
                    f"{n:>5}{p:>8g}{w:>8}{w / n:>7.3f}{sr.goodput:>12.6f}"
                    f"{ideal:>12.6f}{100 * (sr.goodput / ideal - 1):>13.3f}"
                )
    print()
    print("Reading: at W = N, which is one bandwidth-delay product plus one")
    print("frame and is the textbook recommendation, selective repeat is")
    print("already well short of its ideal throughput, and the shortfall grows")
    print("with the frame error rate. The window needed to come within 1 per")
    print("cent of 1-p is several times N, not N. That multiple is the number")
    print("the textbook rule hides, and it is the reason this repository")
    print("simulates the state machine instead of evaluating the formula.")


if __name__ == "__main__":
    main()
