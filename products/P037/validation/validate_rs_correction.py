"""RS(255,223) corrects up to 16 symbol errors and fails at 17.

Checks
------
1. For every error count E = 0..16, a seeded sweep of error patterns must
   decode back to the original message every time. The sweep covers random
   positions, the first E positions, the last E positions, and alternating
   positions, over several seeds, so the boundary is exercised from the
   structured directions as well as the random one.
2. For E = 17..20, no pattern may decode back to the original message.
   ``reedsolo`` either raises (declared failure) or, if the corrupted word
   lands inside another codeword's sphere, returns a wrong message; both
   outcomes count as "not corrected" and the two are counted separately.
3. The parameter arithmetic: d_min = n - k + 1 = 33, E = (d_min - 1)/2 = 16.

Run: ``python3 validate_rs_correction.py`` from this directory.
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "../src")

from framesync.rs import RS_E, RS_K, RS_N, ReedSolomonLink  # noqa: E402

SEEDS = (20261005, 777, 31337)
MESSAGE = bytes((13 * i + 29) % 256 for i in range(RS_K))


def patterns(n_err: int, seed: int) -> list[np.ndarray]:
    """Four structured families of error positions for a given error count."""
    if n_err == 0:
        return [np.array([], dtype=int)]
    rng = np.random.default_rng(seed)
    return [
        rng.choice(RS_N, size=n_err, replace=False),
        np.arange(n_err),
        np.arange(RS_N - n_err, RS_N),
        (np.arange(n_err) * (RS_N // max(n_err, 1))) % RS_N,
    ]


def main() -> int:
    t0 = time.time()
    link = ReedSolomonLink(interleave=1)
    clean = link.encode_codeword(MESSAGE)

    print("framesync validation 2 -- RS(255,223) correction boundary")
    print("=" * 78)
    print(f"n = {RS_N}, k = {RS_K}, n - k = {RS_N - RS_K}, "
          f"d_min = n - k + 1 = {RS_N - RS_K + 1}, E = (d_min - 1)/2 = {RS_E}")
    print(f"codec: reedsolo RSCodec({RS_N - RS_K}); a decoding failure raises "
          f"ReedSolomonError")
    print(f"patterns per error count: 4 families x {len(SEEDS)} seeds = "
          f"{4 * len(SEEDS)}")
    print("error values are drawn non-zero, so every chosen position is a real "
          "symbol error")
    print()

    header = (
        f"{'errors':>7}  {'patterns':>8}  {'corrected':>9}  {'declared fail':>13}  "
        f"{'wrong msg':>9}  expected  result"
    )
    print(header)
    print("-" * len(header))

    all_pass = True
    for n_err in range(0, 21):
        corrected = declared = wrong = 0
        total = 0
        for seed in SEEDS:
            for pos in patterns(n_err, seed):
                rng = np.random.default_rng(seed + 991 * n_err)
                cw = bytearray(clean)
                for p in pos:
                    cw[int(p)] ^= int(rng.integers(1, 256))
                res = link.decode_codeword(bytes(cw))
                total += 1
                if not res.corrected:
                    declared += 1
                elif res.message == MESSAGE:
                    corrected += 1
                else:
                    wrong += 1
        expect = "all corrected" if n_err <= RS_E else "none corrected"
        ok = corrected == total if n_err <= RS_E else corrected == 0
        all_pass &= ok
        print(
            f"{n_err:7d}  {total:8d}  {corrected:9d}  {declared:13d}  {wrong:9d}  "
            f"{expect:>13}  {'PASS' if ok else 'FAIL'}"
        )

    print()
    print("Interleaved codeblock: 16 errors in each of I codewords must all correct,")
    print("17 in any one codeword must fail that codeword and so lose the frame.")
    print("-" * 78)
    for interleave in (1, 3, 5):
        lk = ReedSolomonLink(interleave=interleave)
        rng = np.random.default_rng(4242 + interleave)
        data = rng.integers(0, 256, lk.frame_data_octets, dtype=np.uint8).tobytes()
        arr = np.frombuffer(lk.encode_frame(data), dtype=np.uint8).reshape(
            RS_N, interleave
        ).copy()
        for j in range(interleave):
            pos = rng.choice(RS_N, size=RS_E, replace=False)
            arr[pos, j] ^= np.uint8(0x5A)
        out, failed = lk.decode_frame(arr.tobytes())
        ok16 = out == data and failed == 0
        arr2 = np.frombuffer(lk.encode_frame(data), dtype=np.uint8).reshape(
            RS_N, interleave
        ).copy()
        pos = rng.choice(RS_N, size=RS_E + 1, replace=False)
        arr2[pos, 0] ^= np.uint8(0x5A)
        out2, failed2 = lk.decode_frame(arr2.tobytes())
        ok17 = out2 is None and failed2 >= 1
        all_pass &= ok16 and ok17
        print(
            f"  I = {interleave}: 16 per codeword -> frame recovered: {ok16}  "
            f"(PASS {ok16});  17 in codeword 0 -> frame lost: {ok17} "
            f"(failed codewords {failed2}) (PASS {ok17})"
        )

    print()
    print(f"ALL CHECKS PASS: {all_pass}")
    print(f"wall time: {time.time() - t0:.1f} s")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
