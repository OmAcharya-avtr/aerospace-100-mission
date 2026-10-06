"""Validation 3: what the max-log approximation and LLR clipping actually cost.

Two costs are reported for every approximation, because they are different
quantities and only the second one matters to a user:

* **LLR error** against the exact LLR on the same samples: RMS, maximum, and
  mean signed deviation.
* **Decoded performance**: bit error rate after sum-product decoding of the
  shipped LDPC code on the *same* channel realisations, plus the generalised
  mutual information, which is the calibration-sensitive scalar.

A demapper can have a large LLR error and lose almost nothing after decoding;
the clipping sweep below shows exactly that, and it is the main practical
result of this file.

Configuration, frozen
---------------------
LDPC (3, 6)-regular, n = 96, k = 50, rate 50/96, seed 20261006, 20 flooding
iterations, early stop on a satisfied syndrome. OOK over lognormal fading
with scintillation index 0.3, independently drawn per channel bit (ideal
interleaving). Eb/N0 is per information bit. ``BLOCKS`` codewords per point,
one seeded generator per point.

Runtime: about 60 s on one shared core.
"""

from __future__ import annotations

import numpy as np

from softdecode.channel import GammaGammaFading, LognormalFading
from softdecode.ldpc import make_regular_ldpc
from softdecode.llr import clip_llr
from softdecode.metrics import generalised_mutual_information, llr_error
from softdecode.simulate import decode_ber, demap_ook, simulate_ook

BLOCKS = 1200
SEED = 20261006
EBN0_DB = (2.0, 4.0, 6.0, 8.0, 10.0)
CLIP_LEVELS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0)


def banner(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


def setup():
    code = make_regular_ldpc()
    print(f"LDPC n={code.length} m={code.n_checks} k={code.dimension} "
          f"rate={code.rate:.6f} length-4 cycles={code.cycles4}")
    print(f"blocks per point {BLOCKS} ({BLOCKS * code.dimension} information bits, "
          f"{BLOCKS * code.length} channel bits), seed {SEED}")
    return code


def maxlog_cost(code, fading, label: str) -> None:
    banner(f"1. Max-log versus exact, no channel-state information, {label}")
    print("   The fading average is taken over the prior; max-log replaces the")
    print("   logsumexp over quadrature nodes by a max, so L_maxlog >= L_exact.")
    print(f"  {'Eb/N0':>7} {'LLR rmse':>11} {'max |e|':>11} {'mean signed':>12} "
          f"{'one-signed':>11} {'BER exact':>13} {'BER max-log':>13} {'BER ratio':>10} "
          f"{'GMI exact':>10} {'GMI max-log':>12}")
    for ebn0 in EBN0_DB:
        r = simulate_ook(code, ebn0, fading, None, BLOCKS, SEED + int(10 * ebn0))
        exact = demap_ook(r, "no_csi_exact", fading)
        maxlog = demap_ook(r, "no_csi_maxlog", fading)
        err = llr_error(maxlog, exact)
        one_signed = bool(np.all(maxlog >= exact - 1e-9))
        be = decode_ber(code, r, exact)
        bm = decode_ber(code, r, maxlog)
        ratio = bm.rate / be.rate if be.rate > 0 else float("inf")
        print(f"  {ebn0:7.2f} {err.rmse:11.5f} {err.max_abs:11.5f} {err.mean_signed:+12.5f} "
              f"{str(one_signed):>11} {be.rate:13.6e} {bm.rate:13.6e} {ratio:10.3f} "
              f"{generalised_mutual_information(exact, r.codeword):10.5f} "
              f"{generalised_mutual_information(maxlog, r.codeword):12.5f}")


def maxlog_known_csi(code, fading) -> None:
    banner("2. Max-log versus exact with an estimated channel state")
    print("   Here the fading average is over the posterior p(h | h_hat), which is")
    print("   the quantity a receiver with an imperfect estimate should average")
    print("   over. bias +2.00 dB, jitter 1.00 dB.")
    from softdecode.csi import MultiplicativeCsiError

    error = MultiplicativeCsiError(2.0, 1.0)
    print(f"  {'Eb/N0':>7} {'LLR rmse':>11} {'max |e|':>11} {'BER exact':>13} "
          f"{'BER max-log':>13} {'BER ratio':>10} {'GMI exact':>10} {'GMI max-log':>12}")
    for ebn0 in EBN0_DB:
        r = simulate_ook(code, ebn0, fading, error, BLOCKS, SEED + 7 + int(10 * ebn0))
        exact = demap_ook(r, "csi_aware_exact", fading, error)
        maxlog = demap_ook(r, "csi_aware_maxlog", fading, error)
        err = llr_error(maxlog, exact)
        be = decode_ber(code, r, exact)
        bm = decode_ber(code, r, maxlog)
        ratio = bm.rate / be.rate if be.rate > 0 else float("inf")
        print(f"  {ebn0:7.2f} {err.rmse:11.5f} {err.max_abs:11.5f} {be.rate:13.6e} "
              f"{bm.rate:13.6e} {ratio:10.3f} "
              f"{generalised_mutual_information(exact, r.codeword):10.5f} "
              f"{generalised_mutual_information(maxlog, r.codeword):12.5f}")


def clipping_cost(code, fading) -> None:
    banner("3. LLR clipping: LLR error against decoded cost")
    print("   Exact LLRs with no CSI, then saturated at L_max before decoding.")
    print("   The two costs diverge: the LLR error is already large at L_max = 5")
    print("   while the decoded BER has not moved.")
    for ebn0 in (4.0, 8.0):
        r = simulate_ook(code, ebn0, fading, None, BLOCKS, SEED + int(10 * ebn0))
        exact = demap_ook(r, "no_csi_exact", fading)
        base = decode_ber(code, r, exact)
        base_gmi = generalised_mutual_information(exact, r.codeword)
        print(f"\n  Eb/N0 = {ebn0:.2f} dB, unclipped BER {base}, GMI {base_gmi:.6f}")
        print(f"  {'L_max':>7} {'clipped frac':>13} {'LLR rmse':>11} {'rel rmse':>10} "
              f"{'BER':>13} {'BER / base':>11} {'GMI':>10} {'GMI loss':>10}")
        for limit in CLIP_LEVELS:
            clipped = clip_llr(exact, limit)
            frac = float(np.mean(np.abs(exact) > limit))
            err = llr_error(clipped, exact)
            ber = decode_ber(code, r, clipped)
            gmi = generalised_mutual_information(clipped, r.codeword)
            ratio = ber.rate / base.rate if base.rate > 0 else float("inf")
            print(f"  {limit:7.2f} {frac:13.6f} {err.rmse:11.5f} {err.relative_rmse:10.5f} "
                  f"{ber.rate:13.6e} {ratio:11.3f} {gmi:10.5f} {base_gmi - gmi:10.5f}")


def ppm_maxlog(fading) -> None:
    banner("4. 4-PPM max-log bit-LLR error (LLR units only, no decoder)")
    from softdecode.channel import amplitude_quadrature
    from softdecode.detection import DetectionModel
    from softdecode.llr import llr_ppm_marginal, llr_ppm_maxlog

    det = DetectionModel()
    hq, wq = amplitude_quadrature(fading)
    rng = np.random.default_rng(SEED + 99)
    print("   inner_max=False applies max-log only to the symbol sums, which is")
    print("   the classical Hagenauer-Hoeher demapper; inner_max=True also")
    print("   replaces the fading average by its dominant node.")
    print(f"  {'Eb/N0':>7} {'rmse symbol-only':>17} {'max |e| symbol-only':>20} "
          f"{'rmse both':>11} {'max |e| both':>13}")
    for ebn0 in EBN0_DB:
        a = det.ppm_amplitude(ebn0, 4)
        h = fading.sample(4000, rng)
        symbols = rng.integers(0, 4, 4000)
        y = a * h[:, None] * np.eye(4)[symbols] + rng.standard_normal((4000, 4))
        exact = llr_ppm_marginal(y, a, hq, wq, det.sigma)
        outer = llr_ppm_maxlog(y, a, hq, wq, det.sigma, inner_max=False)
        both = llr_ppm_maxlog(y, a, hq, wq, det.sigma, inner_max=True)
        e1 = llr_error(outer, exact)
        e2 = llr_error(both, exact)
        print(f"  {ebn0:7.2f} {e1.rmse:17.5f} {e1.max_abs:20.5f} "
              f"{e2.rmse:11.5f} {e2.max_abs:13.5f}")
    print()
    print("   Known answer for the symbol-only column: for 4-PPM each of the two")
    print("   sums in a bit LLR has M/2 = 2 terms, and")
    print("     0 <= logsumexp(a, b) - max(a, b) <= log 2,")
    print("   with equality at a = b. The bit LLR is a difference of two such")
    print(f"   reductions, so its max-log error lies in [-log 2, +log 2] = "
          f"[-{np.log(2.0):.6f}, +{np.log(2.0):.6f}].")
    print(f"   The measured maximum above is {e1.max_abs:.6f}, i.e. "
          f"{e1.max_abs / np.log(2.0):.6f} of the bound.")


def main() -> int:
    print("softdecode validation 3: max-log and clipping costs")
    print("raw output committed as validation/maxlog_clipping_output.txt")
    code = setup()
    lognormal = LognormalFading(0.3)
    maxlog_cost(code, lognormal, "lognormal sigma_I2 = 0.3")
    maxlog_cost(code, GammaGammaFading(4.0, 2.0), "gamma-gamma (4, 2)")
    maxlog_known_csi(code, lognormal)
    clipping_cost(code, lognormal)
    ppm_maxlog(lognormal)
    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
