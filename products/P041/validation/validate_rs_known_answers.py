"""Reed-Solomon: exhaustive correction at the radius and the behaviour just beyond it.

The code's guarantee is that bounded-distance decoding corrects **every** pattern
of at most ``t = (n-k)/2`` symbol errors, because Reed-Solomon codes are maximum
distance separable. That is a statement about all patterns, so for a small code it
is checked by enumerating all of them rather than by sampling.

At ``t+1`` errors the honest statement is not "it fails" but "it does not recover
the message": the decoder either declares the word uncorrectable or miscorrects to
a different codeword. Both counts are reported. A single recovery at ``t+1`` would
be a defect in the test, not a feature of the code.

Run: ``PYTHONPATH=../src python3 validate_rs_known_answers.py``
Runtime: about 40 s on one core.
"""

from __future__ import annotations

import itertools

import numpy as np

from codedfade.convolutional import ConvolutionalCode
from codedfade.reedsolomon import ReedSolomon


def classify(result, message: np.ndarray) -> str:
    """Four buckets, because 'message is right' is not the same as 'decoding worked'.

    ``recovered``             decoder reported success and the message is right
    ``miscorrected``          reported success, message wrong
    ``failed_intact``         reported failure, but every error fell in the parity
                              symbols so the message came back untouched
    ``failed_wrong``          reported failure, message wrong
    """
    right = bool(np.array_equal(result.message, message))
    if result.success:
        return "recovered" if right else "miscorrected"
    return "failed_intact" if right else "failed_wrong"


def exhaustive(code: ReedSolomon, message: np.ndarray, errors: int) -> dict[str, int]:
    """Bucket counts over all ``errors``-symbol error patterns."""
    cw = code.encode(message)
    values = range(1, 1 << code.m)
    counts = {
        "recovered": 0,
        "miscorrected": 0,
        "failed_intact": 0,
        "failed_wrong": 0,
    }
    for positions in itertools.combinations(range(code.n), errors):
        for vals in itertools.product(values, repeat=errors):
            received = cw.copy()
            for p, v in zip(positions, vals, strict=True):
                received[p] ^= v
            counts[classify(code.decode(received), message)] += 1
    return counts


def sampled(
    code: ReedSolomon, message: np.ndarray, errors: int, trials: int, seed: int
) -> dict[str, int]:
    cw = code.encode(message)
    rng = np.random.default_rng(seed)
    counts = {
        "recovered": 0,
        "miscorrected": 0,
        "failed_intact": 0,
        "failed_wrong": 0,
    }
    for _ in range(trials):
        received = cw.copy()
        for p in rng.choice(code.n, errors, replace=False):
            received[p] ^= int(rng.integers(1, 1 << code.m))
        counts[classify(code.decode(received), message)] += 1
    return counts


def main() -> None:
    print("=" * 78)
    print("REED-SOLOMON KNOWN-ANSWER VALIDATION")
    print("=" * 78)

    small = ReedSolomon(15, 11, 4)
    msg = np.arange(small.k, dtype=np.int64)
    print(f"code                     RS({small.n},{small.k}) over GF(2^{small.m})")
    print(f"t = (n-k)/2              {small.t}")
    print(f"minimum distance n-k+1   {small.n - small.k + 1} (MDS)")
    print(f"generator polynomial     {small.generator.tolist()} (ascending degree)")
    print(f"generator degree         {small.generator.size - 1} = 2t")
    print(f"message under test       {msg.tolist()}")
    print(f"codeword                 {small.encode(msg).tolist()}")
    print(f"syndromes of codeword    {small.syndromes(small.encode(msg)).tolist()}")
    print()

    print("-- exhaustive error patterns, RS(15,11) ------------------------------")
    print("buckets: recovered = success AND right; miscorrected = success, wrong;")
    print("failed_intact = failure but every error fell in parity so the message")
    print("came back untouched; failed_wrong = failure, message wrong.")
    print(f"{'errors':>7} {'patterns':>10} {'recovered':>10} {'miscorr':>9} "
          f"{'fail_intact':>12} {'fail_wrong':>11} {'verdict':>30}")
    for e in (1, 2):
        total = len(list(itertools.combinations(range(small.n), e))) * 15**e
        c = exhaustive(small, msg, e)
        rec = c["recovered"]
        verdict = "all corrected" if rec == total else f"DEFECT: {total - rec} not corrected"
        print(
            f"{e:7d} {total:10d} {rec:10d} {c['miscorrected']:9d} "
            f"{c['failed_intact']:12d} {c['failed_wrong']:11d} {verdict:>30}"
        )
    print()
    print("-- t+1 = 3 errors, 20000 random patterns, RS(15,11) ------------------")
    c = sampled(small, msg, 3, 20_000, seed=101)
    n = 20_000
    print(f"recovered (must be 0)    {c['recovered']}")
    print(f"miscorrected             {c['miscorrected']}  ({c['miscorrected'] / n:.6f})")
    print(f"failed, message intact   {c['failed_intact']}  ({c['failed_intact'] / n:.6f})"
          f"   expected C(4,3)/C(15,3) = {4 / 455:.6f}")
    print(f"failed, message wrong    {c['failed_wrong']}  ({c['failed_wrong'] / n:.6f})")
    print()

    work = ReedSolomon(31, 21, 5)
    msgw = np.arange(work.k, dtype=np.int64)
    print(f"-- working code RS({work.n},{work.k}) over GF(2^{work.m}), t = {work.t} --------------")
    print(f"rate                     {work.rate:.6f}")
    print(f"{'errors':>7} {'trials':>8} {'recovered':>10} {'miscorr':>9} "
          f"{'fail_intact':>12} {'fail_wrong':>11}")
    for e, trials in ((work.t, 4000), (work.t + 1, 4000), (work.t + 3, 2000)):
        c = sampled(work, msgw, e, trials, seed=200 + e)
        print(
            f"{e:7d} {trials:8d} {c['recovered']:10d} {c['miscorrected']:9d} "
            f"{c['failed_intact']:12d} {c['failed_wrong']:11d}"
        )
    print()
    print("The t row must recover every trial; the t+1 and t+3 rows must show")
    print("recovered = 0. A non-zero failed_intact column on those rows is not a")
    print("recovery: it is an error pattern confined to the parity symbols.")
    print()

    print("-- convolutional code hand trace -------------------------------------")
    cc = ConvolutionalCode()
    bits = np.array([1, 0, 1, 1], dtype=np.uint8)
    encoded = cc.encode(bits)
    print(f"code                     {cc!r}, {cc.n_states} states")
    print(f"message                  {bits.tolist()}")
    print(f"encoded                  {''.join(map(str, encoded.tolist()))}")
    print("hand trace expects       111000010111  (11 10 00 01 01 11)")
    print(f"match                    {''.join(map(str, encoded.tolist())) == '111000010111'}")
    dec = cc.decode(encoded, bits.size)
    print(f"viterbi on clean input   {dec.message.tolist()}  metric {dec.path_metric}")
    print(f"min terminated weight    {cc.minimum_terminated_weight(12)} bits over all "
          f"{2**12 - 1} non-zero L=12 messages")
    print()
    print("-- convolutional single-bit error correction -------------------------")
    rng = np.random.default_rng(7)
    msgc = rng.integers(0, 2, 60).astype(np.uint8)
    enc = cc.encode(msgc)
    ok = 0
    for pos in range(enc.size):
        received = enc.copy()
        received[pos] ^= 1
        if np.array_equal(cc.decode(received, msgc.size).message, msgc):
            ok += 1
    print(f"single-bit errors tested {enc.size}")
    print(f"recovered                {ok}  ({ok / enc.size:.6f})")
    print()
    print("-- convolutional burst: what the interleaver is for ------------------")
    print(f"{'burst bits':>11} {'recovered':>10}")
    for burst in (1, 2, 3, 4, 6, 8, 12):
        recovered = 0
        trials = 0
        for start in range(0, enc.size - burst, 11):
            received = enc.copy()
            received[start : start + burst] ^= 1
            trials += 1
            if np.array_equal(cc.decode(received, msgc.size).message, msgc):
                recovered += 1
        print(f"{burst:11d} {recovered}/{trials}")
    print()
    print("=" * 78)


if __name__ == "__main__":
    main()
