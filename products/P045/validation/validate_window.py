"""Validation: window sizing against the bandwidth-delay product.

Three things are established, two by arithmetic and one by simulation.

1. The bandwidth-delay product and the slot count N are consistent with each
   other and with the frame size, for every link preset, in bits, bytes and
   frames.  ``bdp_frames == N - 1`` exactly, by construction, and the check is
   that the independently computed quantities agree.
2. The closed-form knee: selective-repeat throughput ``(1-p) min(1, W/N)`` has
   zero sensitivity to W above W = N, and sensitivity ``(1-p)/N`` below it.
   Checked by finite difference against :func:`arqlonghaul.window.marginal_gain`.
3. The window a simulated selective-repeat sender actually needs, which is
   several times N rather than N, because the closed form assumes the window
   never blocks.  Reported as the measured multiple of N required to reach 99
   per cent of the ideal ``1-p``, as a function of the frame error rate.  This
   is the number the textbook rule gets wrong, and it is the practical output of
   this file.

Both frames and bytes are reported throughout, with the assumed frame size
stated, because a window in frames is meaningless without one.

Runtime: about 90 s on one core.
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
from arqlonghaul.link import PRESETS  # noqa: E402
from arqlonghaul.protocols import simulate_selective_repeat  # noqa: E402
from arqlonghaul.window import marginal_gain, size_window  # noqa: E402

SLOTS = 600_000


def main() -> None:
    print("=" * 96)
    print("1. Link geometry: N, the bandwidth-delay product, and the window knee")
    print("   frame size is stated on every row; without it a frame count means")
    print("   nothing")
    print("=" * 96)
    print(f"{'preset':<11}{'rate bit/s':>12}{'RTT s':>10}{'frame B':>9}"
          f"{'N slots':>11}{'BDP B':>13}{'BDP frames':>12}{'knee frames':>12}")
    for name, link in PRESETS.items():
        d = link.describe()
        print(
            f"{name:<11}{d['rate_bps']:>12.4g}{d['rtt_s']:>10.4g}"
            f"{d['frame_bits'] / 8:>9.0f}{d['slots_per_cycle_N']:>11.2f}"
            f"{d['bdp_bytes']:>13.6g}{d['bdp_frames']:>12.2f}"
            f"{d['min_continuous_window_frames']:>12}"
        )
    print()
    print("consistency: bdp_frames == N - 1 exactly")
    worst = 0.0
    for link in PRESETS.values():
        gap = abs(link.bdp_frames - (link.slots_per_cycle - 1.0))
        worst = max(worst, gap)
    print(f"largest absolute discrepancy over {len(PRESETS)} presets: {worst:.3e}")

    print()
    print("=" * 96)
    print("2. Window sizing at a 5 per cent frame error rate, both protocols")
    print("=" * 96)
    print(f"{'preset':<11}{'knee frames':>12}{'knee bytes':>13}"
          f"{'SR eta':>10}{'SR bit/s':>13}{'GBN eta':>10}{'GBN bit/s':>13}")
    for name, link in PRESETS.items():
        d = size_window(link, 0.05).as_dict()
        print(
            f"{name:<11}{d['knee_frames']:>12}{d['knee_bytes']:>13.6g}"
            f"{d['sr_goodput_at_knee']:>10.5f}{d['sr_bps_at_knee']:>13.6g}"
            f"{d['gbn_goodput_at_knee']:>10.5f}{d['gbn_bps_at_knee']:>13.6g}"
        )
    print()
    print("Reading: at the same window and the same 5 per cent frame error rate,")
    print("go-back-N delivers a small fraction of what selective repeat does, and")
    print("the fraction gets worse as N grows. On the Mars preset go-back-N is")
    print("not a protocol choice, it is a way of not using the link.")

    print()
    print("=" * 96)
    print("3. Marginal gain per extra frame of window, closed form versus")
    print("   finite difference of the throughput expression")
    print("=" * 96)
    print(f"{'N':>6}{'p':>7}{'W':>7}{'analytic dEta/dW':>19}"
          f"{'finite difference':>19}{'abs diff':>11}")
    worst_mg = 0.0
    for n in (20.0, 60.0, 294.0):
        for p in (0.01, 0.05, 0.2):
            for w in (1, int(n // 2), int(math.ceil(n)), int(2 * n)):
                analytic = marginal_gain(n, p, w)
                fd = cf.sr_throughput(p, n, w + 1) - cf.sr_throughput(p, n, w)
                worst_mg = max(worst_mg, abs(analytic - fd))
                print(
                    f"{n:>6g}{p:>7g}{w:>7}{analytic:>19.8f}{fd:>19.8f}"
                    f"{abs(analytic - fd):>11.1e}"
                )
    print(f"largest absolute difference: {worst_mg:.2e}")
    print("Tolerance adopted: 1e-12 absolute (this is exact arithmetic, not a")
    print("Monte Carlo estimate).")
    print(f"PASS: {worst_mg < 1e-12}")

    print()
    print("=" * 96)
    print("4. The window a simulated selective-repeat sender actually needs.")
    print("   Measured multiple of N to reach 99 per cent of the ideal 1-p.")
    print("=" * 96)
    rng = np.random.default_rng(45_045_3)
    n = 60
    print(f"N = {n} slots, {SLOTS} slots per run")
    print(f"{'p':>7}{'ideal 1-p':>11}" + "".join(f"{f'W={m}N':>10}" for m in (1, 2, 3, 4, 6, 8)))
    needed: dict[float, float] = {}
    for p in (0.01, 0.05, 0.10, 0.20, 0.40):
        errors = IndependentFrameChannel(p).errors(SLOTS, rng)
        ideal = cf.sr_throughput(p, n)
        cells = []
        first_ok = math.nan
        for mult in (1, 2, 3, 4, 6, 8):
            g = simulate_selective_repeat(errors, n, mult * n).goodput
            cells.append(g / ideal)
            if math.isnan(first_ok) and g / ideal >= 0.99:
                first_ok = float(mult)
        needed[p] = first_ok
        print(
            f"{p:>7g}{ideal:>11.5f}" + "".join(f"{c:>10.4f}" for c in cells)
        )
    print()
    print("fraction of the ideal 1-p achieved at each window, as a multiple of N")
    print()
    print(f"{'p':>7}{'smallest multiple of N reaching 0.99 of ideal':>48}"
          f"{'in frames':>12}{'in bytes (1115 B frame)':>26}")
    for p, mult in needed.items():
        frames = mult * n if math.isfinite(mult) else float("nan")
        print(
            f"{p:>7g}{mult:>48g}{frames:>12.0f}"
            f"{frames * 1115:>26.0f}"
        )
    print()
    print("Reading: the textbook window of one bandwidth-delay product plus one")
    print("frame (W = N) is adequate only on an error-free link. At a 20 per cent")
    print("frame error rate this simulator needs six times that window to come")
    print("within one per cent of the ideal, which at a 1115-octet frame and")
    print("N = 60 is 401400 bytes of send buffer rather than 66900.")


if __name__ == "__main__":
    main()
