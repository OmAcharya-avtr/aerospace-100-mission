"""Is the depth sweep's decoding shortcut sound? Run both paths on one realisation.

``CodedLink`` skips the Berlekamp-Massey decoder for codewords whose pre-decoding
symbol error count is at most ``t``, recording success with no residual errors.
That is justified by the maximum-distance-separable property, which
``validate_rs_known_answers.py`` checks exhaustively for RS(15,11). This script
checks the consequence directly: the same configuration and seed run with
``exact_decode=False`` and ``exact_decode=True`` must give identical results, and
the wall-clock saving is reported so the reason for the shortcut is visible.

The timing figures here are wall-clock measurements of this package on a shared
cloud core. They are a note on the compute budget, not a hardware claim, and they
are not compared to any model output.

Run: ``PYTHONPATH=../src python3 validate_decode_shortcut.py``
Runtime: about 60 s on one core.
"""

from __future__ import annotations

import time

from codedfade.channel import ChannelConfig
from codedfade.link import CodedLink
from codedfade.reedsolomon import ReedSolomon

SI, TAU, FS, SNR_DB, SEED = 0.6, 2.0e-4, 1.0e6, 14.0, 9
CODEWORDS = 512


def main() -> None:
    code = ReedSolomon(31, 21, 5)
    cfg = ChannelConfig(SI, TAU, FS, "lognormal", "exp", seed=SEED)
    print("=" * 88)
    print("DECODING SHORTCUT VALIDATION")
    print("=" * 88)
    print(f"code                     RS({code.n},{code.k}) over GF(2^{code.m}), t={code.t}")
    print(f"channel                  lognormal, SI={SI}, tau={TAU:.1e} s, "
          f"fs={FS:.1e} Hz, seed={SEED}")
    print(f"mean SNR                 {SNR_DB} dB")
    print(f"codewords per point      {CODEWORDS}")
    print()
    header = (
        f"{'depth':>7} {'mode':>8} {'FER':>10} {'post BER':>11} {'fail':>6} {'mis':>5} "
        f"{'wall s':>8}"
    )
    print(header)
    print("-" * len(header))
    mismatches = 0
    for depth in (1, 8, 64, 512):
        rows = []
        for exact in (False, True):
            link = CodedLink(code, cfg, SNR_DB, exact_decode=exact)
            t0 = time.perf_counter()
            r = link.run(depth, CODEWORDS)
            wall = time.perf_counter() - t0
            rows.append((r, wall))
            print(
                f"{depth:7d} {'exact' if exact else 'shortcut':>8} "
                f"{r.frame_error_rate:10.6f} {r.post_bit_error_rate:11.4e} "
                f"{r.decoder_failures:6d} {r.miscorrections:5d} {wall:8.2f}"
            )
        fast, slow = rows[0][0], rows[1][0]
        same = (
            fast.frame_error_rate == slow.frame_error_rate
            and fast.post_bit_error_rate == slow.post_bit_error_rate
            and fast.decoder_failures == slow.decoder_failures
            and fast.miscorrections == slow.miscorrections
        )
        if not same:
            mismatches += 1
        print(
            f"{'':7} {'speedup':>8} x{rows[1][1] / max(rows[0][1], 1e-9):<9.2f} "
            f"identical results: {same}"
        )
    print()
    print(f"depths with a mismatch   {mismatches} (must be 0)")
    print()
    print("Timing is wall-clock on a shared cloud core, reported as a compute-budget")
    print("note only. It is not hardware evidence and is not compared to a model.")
    print("=" * 88)


if __name__ == "__main__":
    main()
