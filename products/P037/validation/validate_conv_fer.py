"""Convolutionally coded frame error rate, hard and soft decision.

The CCSDS (171, 133) rate-1/2 K=7 code has no tractable exact frame-error
expression, so this leg is Monte Carlo only and no analytic comparison is
claimed. What is checked is structural and ordinal:

1. The decoded frame error rate must fall monotonically with Eb/N0.
2. At every Eb/N0 the coded link must beat the uncoded link of the same
   information length at the same Eb/N0 per information bit, i.e. the gain
   must be positive once the code is out of its threshold region.
3. Soft decision must not be worse than hard decision at the same Eb/N0,
   and the measured separation is reported.
4. Noiseless decoding must be exact (a regression guard on the decoder).

Every point carries its binomial standard error, and zero-error points are
reported with the one-sided rule-of-three upper limit instead.

Compute budget: the Viterbi decoder is vectorised across frames, so the
cost is 2*(n_info + 6) NumPy steps on arrays of shape (batch, 64). The
sizes below were chosen to keep this script inside 60 s on one contended
core, which is why the frame counts are modest and the error bars are
correspondingly wide. Wider bars honestly reported are preferred to a
tighter number this machine cannot produce.

Run: ``python3 validate_conv_fer.py`` from this directory.
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "../src")

from framesync.conv import ConvCode, free_distance  # noqa: E402
from framesync.fer import measure_conv_fer, uncoded_fer  # noqa: E402

SEED = 20261005
INFO_BITS = 512
POINTS = ((2.0, 300), (3.0, 300), (4.0, 400), (5.0, 500))


def main() -> int:
    t0 = time.time()
    all_pass = True
    code = ConvCode()
    print("framesync validation 6 -- convolutionally coded frame error rate")
    print("=" * 78)
    print(f"CCSDS (171, 133) rate-1/2, K = {code.k}, {code.n_states} states, "
          f"G2 inverted = {code.invert_g2}")
    print(f"computed free distance: {free_distance(code)} (literature value 10)")
    print(f"frame: {INFO_BITS} information bits + {code.tail_bits()} tail bits "
          f"-> {2 * (INFO_BITS + code.tail_bits())} channel bits")
    print("Eb/N0 is per information bit; the rate-1/2 loss (3.0103 dB) is inside")
    print("the channel bit error rate Eq. (1).")
    print()

    print("Regression guard: noiseless decoding must be exact")
    print("-" * 78)
    rng = np.random.default_rng(SEED)
    info = rng.integers(0, 2, (16, INFO_BITS), dtype=np.uint8)
    exact = np.array_equal(code.decode_batch(code.encode_batch(info)), info)
    all_pass &= exact
    print(f"16 frames round-trip with no channel errors: {exact} -> "
          f"{'PASS' if exact else 'FAIL'}")
    print()

    header = (
        f"{'Eb/N0 dB':>8}  {'mode':>5}  {'p_chan':>11}  {'N':>5}  {'errors':>6}  "
        f"{'FER':>11}  {'uncertainty':>22}  {'dec BER':>11}  {'FER uncoded':>11}"
    )
    print("Measured frame error rate")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    results: dict[str, list[float]] = {"hard": [], "soft": []}
    for ebn0, n in POINTS:
        unc = float(uncoded_fer(ebn0, INFO_BITS)[0])
        for mode in ("hard", "soft"):
            rng = np.random.default_rng(SEED + (0 if mode == "hard" else 1))
            pt = measure_conv_fer(ebn0, INFO_BITS, n, rng, soft=(mode == "soft"))
            results[mode].append(pt.fer)
            if pt.n_frame_errors == 0:
                unc_txt = f"<{pt.rule_of_three_upper:.3e} (95% 1-sided)"
            else:
                unc_txt = f"+-{pt.stderr:.3e}"
            print(
                f"{ebn0:8.2f}  {mode:>5}  {pt.extra['p_bit']:11.4e}  {n:5d}  "
                f"{pt.n_frame_errors:6d}  {pt.fer:11.4e}  {unc_txt:>22}  "
                f"{pt.extra['decoded_ber']:11.4e}  {unc:11.4e}"
            )
    print()

    print("Check: frame error rate falls monotonically with Eb/N0")
    print("-" * 78)
    for mode in ("hard", "soft"):
        arr = np.array(results[mode])
        ok = bool(np.all(np.diff(arr) <= 0.0))
        all_pass &= ok
        print(f"  {mode:>5}: {' -> '.join(f'{v:.4e}' for v in arr)}  "
              f"-> {'PASS' if ok else 'FAIL'}")
    print()

    print("Check: soft decision is not worse than hard decision at any point")
    print("-" * 78)
    for i, (ebn0, _) in enumerate(POINTS):
        h, s = results["hard"][i], results["soft"][i]
        ok = s <= h + 1e-12
        all_pass &= ok
        ratio = (h / s) if s > 0 else float("inf")
        print(f"  {ebn0:5.2f} dB: hard {h:.4e}, soft {s:.4e}, ratio "
              f"{ratio:8.2f} -> {'PASS' if ok else 'FAIL'}")
    print("  (the soft/hard separation is reported, not asserted at a particular")
    print("   value; the frame counts here do not resolve a dB figure)")
    print()

    print("Check: the coded link beats the uncoded link of the same length")
    print("-" * 78)
    for i, (ebn0, _) in enumerate(POINTS):
        unc = float(uncoded_fer(ebn0, INFO_BITS)[0])
        ok = results["soft"][i] < unc
        all_pass &= ok
        print(f"  {ebn0:5.2f} dB: soft-decision coded {results['soft'][i]:.4e} "
              f"< uncoded {unc:.4e} -> {'PASS' if ok else 'FAIL'}")
    print()
    print(f"ALL CHECKS PASS: {all_pass}")
    print(f"wall time: {time.time() - t0:.1f} s")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
