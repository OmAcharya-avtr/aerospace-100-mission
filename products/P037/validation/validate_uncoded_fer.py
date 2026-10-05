"""Uncoded frame error rate against the analytic 1-(1-BER)^n expression.

Checks
------
1. Measured FER over an Eb/N0 sweep against Eq. (11), with the binomial
   standard error on every point. Point sizes come from
   ``n_frames_for_target`` at a 5% relative standard error, capped by the
   compute budget.
2. The Frame Error Control Field detection gap: how many frames with a bit
   error the CRC-16 fails to reject, against the 2^-16 asymptote.
3. The CRC catalogue check value, as a known-answer anchor.

Run: ``python3 validate_uncoded_fer.py`` from this directory.
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "../src")

from framesync.channel import bpsk_ber  # noqa: E402
from framesync.crc import crc16  # noqa: E402
from framesync.fer import (  # noqa: E402
    measure_uncoded_fer,
    n_frames_for_target,
    uncoded_fer,
)
from framesync.frames import FrameGeometry  # noqa: E402

SEED = 20261005
GEOM = FrameGeometry(data_octets=1115, fecf=True, asm_bits=32)
EBN0 = [7.0, 8.0, 9.0, 10.0, 11.0]
MIN_FRAMES = 400
MAX_FRAMES = 12_000
TOL_SIGMA = 4.0


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("framesync validation 1 -- uncoded frame error rate vs Eq. (11)")
    print("=" * 78)
    print(f"frame: {GEOM.data_octets} octets + 2 octet FECF = {GEOM.frame_bits} bits")
    print("Eq. (11): FER = 1 - (1 - p)^n,  p = Q(sqrt(2 Eb/N0)) (Eq. (1), R = 1)")
    print(f"acceptance: |measured - analytic| < {TOL_SIGMA:g} binomial standard errors")
    print(f"seed {SEED}; frame counts sized for 5% relative standard error, "
          f"clamped to [{MIN_FRAMES}, {MAX_FRAMES}] frames")
    print()
    print("CRC-16 known answer: crc16(b'123456789') = "
          f"0x{crc16(b'123456789'):04X} (catalogue check value 0x29B1) -> "
          f"{'PASS' if crc16(b'123456789') == 0x29B1 else 'FAIL'}")
    print()

    header = (
        f"{'Eb/N0 dB':>8}  {'p_bit':>11}  {'N':>7}  {'errors':>7}  "
        f"{'FER meas':>11}  {'stderr':>10}  {'FER Eq.11':>11}  {'dev/sigma':>9}  result"
    )
    print(header)
    print("-" * len(header))

    all_pass = True
    undetected_total = 0
    errored_total = 0
    for ebn0 in EBN0:
        analytic = float(uncoded_fer(ebn0, GEOM.frame_bits)[0])
        n = int(np.clip(n_frames_for_target(max(analytic, 1e-4), 0.05), MIN_FRAMES, MAX_FRAMES))
        pt = measure_uncoded_fer(ebn0, GEOM, n, rng)
        se = pt.stderr if pt.stderr > 0 else 1.0 / n
        dev = abs(pt.fer - analytic) / se
        ok = dev < TOL_SIGMA
        all_pass &= ok
        undetected_total += pt.extra["n_undetected"]
        errored_total += pt.n_frame_errors
        print(
            f"{ebn0:8.2f}  {float(bpsk_ber(ebn0)[0]):11.4e}  {n:7d}  "
            f"{pt.n_frame_errors:7d}  {pt.fer:11.4e}  {se:10.3e}  {analytic:11.4e}  "
            f"{dev:9.2f}  {'PASS' if ok else 'FAIL'}"
        )

    print()
    print("Frame Error Control Field detection gap")
    print("-" * 78)
    frac = undetected_total / errored_total if errored_total else float("nan")
    print(f"frames with at least one bit error      : {errored_total}")
    print(f"of those, FECF failed to reject         : {undetected_total}")
    print(f"undetected fraction                     : {frac:.3e}")
    print("asymptotic expectation for a 16-bit CRC : 2^-16 = 1.526e-05")
    expected_undetected = errored_total * 2.0**-16
    print(f"expected undetected count at that rate  : {expected_undetected:.3f}")
    print("NOTE: most frames here carry a single bit error, which a CRC-16 always")
    print("      detects, so the measured gap is expected BELOW 2^-16 and this is")
    print("      reported, not used as a pass/fail criterion.")

    print()
    print(f"ALL CHECKS PASS: {all_pass}")
    print(f"wall time: {time.time() - t0:.1f} s")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
