"""Validation: the in-package CRC against two independent references.

Checks
------
1. Every CRC in ``arqlonghaul.crc.CATALOGUE`` reproduces its catalogue *check*
   value, the CRC of the ASCII byte string ``123456789``.  The expected values
   were read from the ``crcmod`` 1.7 source distribution file
   ``python3/crcmod/predefined.py`` on 2026-10-06; ``crcmod`` itself is not
   installed and is not imported here, because it does not install in this
   environment.
2. The table-driven implementation agrees with the bit-at-a-time shift-register
   definition on random payloads, for every CRC.
3. CRC-32 agrees with :func:`zlib.crc32` from the Python standard library on
   random payloads.  This is a wholly independent implementation and is the
   stronger of the two references.
4. The frame check sequence detects injected bit errors.  A CRC of width w has
   an undetected-error probability of about 2**-w for a random error pattern;
   the measured detection rate is reported against that, with its binomial
   standard error, rather than asserted.

Runtime: about 10 s on one core.
"""

from __future__ import annotations

import os
import sys
import zlib

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
)

from arqlonghaul.crc import CATALOGUE, append_fcs, check_fcs, crc, crc_bitwise  # noqa: E402

RNG = np.random.default_rng(20261006)


def random_payloads(n: int, max_len: int = 96) -> list[bytes]:
    """``n`` random byte strings of length 1..``max_len``."""
    return [
        bytes(RNG.integers(0, 256, size=int(RNG.integers(1, max_len + 1)), dtype=np.uint8))
        for _ in range(n)
    ]


def main() -> None:
    print("=" * 72)
    print("1. Catalogue check values (CRC of the ASCII string 123456789)")
    print("   reference: crcmod 1.7 python3/crcmod/predefined.py, read 2026-10-06")
    print("=" * 72)
    print(f"{'name':<18}{'width':>6}{'computed':>14}{'catalogue':>14}{'agree':>8}")
    all_ok = True
    for name, spec in CATALOGUE.items():
        value = crc(b"123456789", spec)
        ok = value == spec.check
        all_ok &= ok
        w = spec.width // 4
        shown = f"0x{value:0{w}x}"
        want = f"0x{spec.check:0{w}x}"
        print(f"{name:<18}{spec.width:>6}{shown:>14}{want:>14}{str(ok):>8}")
    print(f"all five agree: {all_ok}")

    print()
    print("=" * 72)
    print("2. Table-driven implementation versus the bit-at-a-time definition")
    print("=" * 72)
    payloads = random_payloads(400)
    print(f"{'name':<18}{'payloads':>10}{'mismatches':>12}")
    for name, spec in CATALOGUE.items():
        bad = sum(1 for d in payloads if crc(d, spec) != crc_bitwise(d, spec))
        print(f"{name:<18}{len(payloads):>10}{bad:>12}")

    print()
    print("=" * 72)
    print("3. CRC-32 versus zlib.crc32 (Python standard library)")
    print("=" * 72)
    spec32 = CATALOGUE["crc-32"]
    big = random_payloads(3000, max_len=512)
    bad = sum(1 for d in big if crc(d, spec32) != zlib.crc32(d))
    total_bytes = sum(len(d) for d in big)
    print(f"payloads compared      {len(big)}")
    print(f"bytes compared         {total_bytes}")
    print(f"mismatches             {bad}")

    print()
    print("=" * 72)
    print("4. Frame check sequence detection of injected bit errors")
    print("=" * 72)
    print(f"{'name':<18}{'trials':>8}{'detected':>10}{'rate':>10}{'stderr':>10}"
          f"{'2**-w':>12}")
    for name, spec in CATALOGUE.items():
        trials = 8000
        detected = 0
        body_len = 64
        for _ in range(trials):
            body = bytes(RNG.integers(0, 256, size=body_len, dtype=np.uint8))
            frame = bytearray(append_fcs(body, spec))
            n_flips = int(RNG.integers(1, 9))
            # Without replacement: a repeated position would flip a bit twice
            # and leave the frame unaltered, which would be counted as an
            # undetected error when in fact no error was injected.
            bit_positions = RNG.choice(len(frame) * 8, size=n_flips, replace=False)
            for pos in bit_positions:
                frame[int(pos) // 8] ^= 1 << (int(pos) % 8)
            if not check_fcs(bytes(frame), spec):
                detected += 1
        rate = detected / trials
        se = (rate * (1 - rate) / trials) ** 0.5
        print(
            f"{name:<18}{trials:>8}{detected:>10}{rate:>10.5f}{se:>10.5f}"
            f"{2.0 ** -spec.width:>12.3e}"
        )
    print()
    print("Notes. The error patterns are 1 to 8 distinct bit flips over a")
    print("544-bit frame (CRC-32 case).  A first pass of this check sampled flip")
    print("positions with replacement, which silently injected no-error frames")
    print("and produced 7 spurious misses in 20000 for CRC-32; the sampling is")
    print("now without replacement and that artefact is gone.  These rates are")
    print("for a pattern a CRC is good at.  They are not a residual-undetected-")
    print("error model for a real channel, and this package does not provide one.")


if __name__ == "__main__":
    main()
