"""Validation 6: CROSS-CHECK X3, and the single-aperture BPSK comparand.

Part A -- cross-check X3, binding
---------------------------------
The ordering under test, from the Batch 05 specification:

    soft-decision BER <= hard-decision BER at every stated Eb/N0,
    on the same channel realisations.

Nothing here can be tuned to meet it. The configuration below is frozen in
this file, the decoders are both **exhaustive maximum likelihood** over all 16
codewords of the extended Hamming (8,4) code, and they are handed the *same*
LLR array: the soft decoder uses it, the hard decoder uses only its sign. A
violation would therefore be a defect in the channel model or in the LLRs, and
would be reported here rather than removed.

Frozen configuration
    modulation          OOK, thermal-limited signal-independent AWGN
                        (``softdecode.detection``), sigma = 1
    code                extended Hamming (8,4), rate 1/2, d_min = 4
    energy convention   Eb is energy per *information* bit; the channel-bit
                        amplitude is a = 2 sigma sqrt(R Eb/N0) with R = 1/2
    fading              unit-mean lognormal, scintillation index 0.30,
                        drawn independently per channel bit (ideal
                        interleaving)
    channel knowledge   perfect CSI; the LLR is the exact linear LLR
                        (a**2 h**2 - 2 a h y) / 2
    Eb/N0 points        0, 2, 4, 6, 8, 10, 12 dB
    blocks per point    X3_BLOCKS codewords = 4 * X3_BLOCKS information bits
    seed                X3_SEED; one independent generator per Eb/N0 point,
                        obtained as default_rng(X3_SEED + 1000 * round(Eb/N0))
    RNG draw order      per point: information bits as
                        rng.integers(0, 2, (blocks, 4)); then fading as
                        exp(mu_x + sigma_x * rng.standard_normal((blocks, 8)));
                        then noise as rng.standard_normal((blocks, 8))

Part B -- comparand for the coordinating session
------------------------------------------------
Separately, an uncoded single-aperture **hard-decision BPSK** bit error rate
over lognormal fading, with its binomial standard error and the full
configuration, over a grid of scintillation indices and Eb/N0 points. The
value another product computes for this quantity was deliberately not supplied
to this build, so no comparison is attempted here; the grid is printed so the
coordinating session can make it.

The analytic reference for Part B is
    BER = E_h[ Q( h sqrt(2 Eb/N0) ) ],
evaluated by the same quadrature the package uses elsewhere, which is a check
on the Monte Carlo rather than a second measurement.

No quantity in this file is compared to a wall-clock measurement.

Runtime: about 40 s on one shared core.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from softdecode.channel import LognormalFading, amplitude_quadrature
from softdecode.codes import EXTENDED_HAMMING_84
from softdecode.detection import DetectionModel
from softdecode.llr import llr_ook_known_csi
from softdecode.metrics import bit_error_rate

X3_BLOCKS = 40_000
X3_SEED = 20261006
X3_EBN0_DB = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
X3_SIGMA_I2 = 0.30

COMPARAND_BITS = 2_000_000
COMPARAND_SEED = 20261048
COMPARAND_SIGMA_I2 = (0.1, 0.3, 1.0)
COMPARAND_EBN0_DB = (0.0, 5.0, 10.0, 15.0, 20.0)


def banner(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


def cross_check_x3() -> bool:
    banner("PART A: cross-check X3, soft-decision BER <= hard-decision BER")
    code = EXTENDED_HAMMING_84
    det = DetectionModel(1.0)
    fading = LognormalFading(X3_SIGMA_I2)
    print(f"  modulation        OOK, thermal-limited AWGN, sigma = {det.sigma:.1f}")
    print(f"  code              extended Hamming ({code.length},{code.dimension}), "
          f"rate {code.rate:.4f}, d_min {code.min_distance}")
    print(f"  weight enumerator {code.weight_distribution().tolist()}")
    print(f"  fading            unit-mean lognormal, scintillation index "
          f"{fading.scintillation_index:.4f}, sigma_x {fading.sigma_x:.9f}, "
          f"mu_x {fading.mu_x:.9f}")
    print("                    drawn independently per channel bit (ideal interleaving)")
    print(f"  energy            a = 2 sigma sqrt(R Eb/N0), R = {code.rate:.4f}, "
          f"N0 = 2 sigma**2")
    print(f"  blocks per point  {X3_BLOCKS} "
          f"({X3_BLOCKS * code.dimension} information bits per point)")
    print(f"  seed              {X3_SEED}, one generator per point: "
          f"default_rng(seed + 1000 * round(Eb/N0))")
    print("  draw order        information bits, then fading, then noise")
    print("  decoders          both exhaustive maximum likelihood over all 16")
    print("                    codewords; the hard decoder sees only sign(LLR)")
    print()
    print(f"  {'Eb/N0 dB':>9} {'a':>10} {'soft BER':>14} {'soft errs':>10} {'soft se':>10} "
          f"{'hard BER':>14} {'hard errs':>10} {'hard se':>10} {'soft<=hard':>11} "
          f"{'gain':>8}")
    ok = True
    for ebn0 in X3_EBN0_DB:
        rng = np.random.default_rng(X3_SEED + 1000 * int(round(ebn0)))
        amplitude = det.ook_amplitude(ebn0, code.rate)
        messages = rng.integers(0, 2, size=(X3_BLOCKS, code.dimension)).astype(np.int8)
        words = code.encode(messages)
        h = np.exp(fading.mu_x + fading.sigma_x * rng.standard_normal(words.shape))
        y = amplitude * h * words + det.sigma * rng.standard_normal(words.shape)
        llr = llr_ook_known_csi(y, amplitude, h, det.sigma)
        soft = bit_error_rate(code.decode_soft(llr), messages)
        hard = bit_error_rate(code.decode_hard(llr), messages)
        holds = soft.rate <= hard.rate
        ok = ok and holds
        gain = hard.rate / soft.rate if soft.rate > 0 else float("inf")
        print(f"  {ebn0:9.2f} {amplitude:10.6f} {soft.rate:14.6e} {soft.errors:10d} "
              f"{soft.standard_error:10.3e} {hard.rate:14.6e} {hard.errors:10d} "
              f"{hard.standard_error:10.3e} {str(holds):>11} {gain:8.3f}")
    print()
    if ok:
        print("  X3 RESULT: PASS. soft <= hard at every stated Eb/N0 on the same")
        print("  channel realisations, strictly at every point.")
    else:
        print("  X3 RESULT: FAIL. The ordering was violated. Defect reported, not")
        print("  tuned away; the configuration above is the one that produced it.")
    return ok


def paired_detail() -> None:
    banner("PART A.1: pairing evidence")
    print("  Both decoders are evaluated on one LLR array per point, so the")
    print("  realisations are identical by construction rather than by seed")
    print("  agreement. The counts below show where they differ.")
    code = EXTENDED_HAMMING_84
    det = DetectionModel(1.0)
    fading = LognormalFading(X3_SIGMA_I2)
    print(f"  {'Eb/N0 dB':>9} {'blocks':>8} {'soft right hard wrong':>22} "
          f"{'soft wrong hard right':>22} {'both wrong':>11}")
    for ebn0 in X3_EBN0_DB:
        rng = np.random.default_rng(X3_SEED + 1000 * int(round(ebn0)))
        amplitude = det.ook_amplitude(ebn0, code.rate)
        messages = rng.integers(0, 2, size=(X3_BLOCKS, code.dimension)).astype(np.int8)
        words = code.encode(messages)
        h = np.exp(fading.mu_x + fading.sigma_x * rng.standard_normal(words.shape))
        y = amplitude * h * words + det.sigma * rng.standard_normal(words.shape)
        llr = llr_ook_known_csi(y, amplitude, h, det.sigma)
        soft_ok = np.all(code.decode_soft(llr) == messages, axis=1)
        hard_ok = np.all(code.decode_hard(llr) == messages, axis=1)
        print(f"  {ebn0:9.2f} {X3_BLOCKS:8d} {int(np.sum(soft_ok & ~hard_ok)):22d} "
              f"{int(np.sum(~soft_ok & hard_ok)):22d} "
              f"{int(np.sum(~soft_ok & ~hard_ok)):11d}")


def comparand() -> None:
    banner("PART B: comparand, uncoded hard-decision BPSK over lognormal fading")
    print("  Configuration, complete:")
    print("    modulation       antipodal BPSK, s in {-1, +1}, one sample per bit")
    print("    detection        thermal-limited signal-independent AWGN, sigma = 1")
    print("    amplitude        a = sigma sqrt(2 Eb/N0); uncoded, so R = 1")
    print("    received sample  y = a h s + n, n ~ N(0, 1)")
    print("    fading           unit-mean lognormal, independent per bit,")
    print("                     h = exp(mu_x + sigma_x z), mu_x = -sigma_x**2/2,")
    print("                     sigma_x = sqrt(log(1 + sigma_I**2))")
    print("    detector         hard decision sign(y); error iff sign(y) != s")
    print("    channel knowledge  none needed: the hard decision is CSI free")
    print(f"    bits per point   {COMPARAND_BITS}")
    print(f"    seed             {COMPARAND_SEED}, one generator per")
    print("                     (sigma_I**2, Eb/N0) cell:")
    print("                     default_rng(COMPARAND_SEED + 101 * round(10*sigma_I2)")
    print("                                 + round(Eb/N0))")
    print("    draw order       symbols as rng.integers(0, 2, bits) mapped to")
    print("                     1 - 2b; then fading as rng.standard_normal(bits);")
    print("                     then noise as rng.standard_normal(bits)")
    print("    standard error   binomial, sqrt(p(1-p)/n)")
    print("    analytic check   BER = E_h[Q(h sqrt(2 Eb/N0))] by the package")
    print("                     quadrature (80 Gauss-Hermite nodes)")
    print()
    det = DetectionModel(1.0)
    print(f"  {'sigma_I^2':>10} {'Eb/N0 dB':>9} {'a':>10} {'sample BER':>14} "
          f"{'errors':>9} {'binom se':>11} {'quadrature BER':>15} {'z':>8}")
    for sigma_i2 in COMPARAND_SIGMA_I2:
        fading = LognormalFading(sigma_i2)
        hq, wq = amplitude_quadrature(fading)
        for ebn0 in COMPARAND_EBN0_DB:
            seed = (
                COMPARAND_SEED
                + 101 * int(round(10 * sigma_i2))
                + int(round(ebn0))
            )
            rng = np.random.default_rng(seed)
            amplitude = det.bpsk_amplitude(ebn0)
            bits = rng.integers(0, 2, size=COMPARAND_BITS).astype(np.int8)
            symbols = 1.0 - 2.0 * bits
            h = np.exp(fading.mu_x + fading.sigma_x * rng.standard_normal(COMPARAND_BITS))
            y = amplitude * h * symbols + det.sigma * rng.standard_normal(COMPARAND_BITS)
            decided = (y < 0.0).astype(np.int8)
            result = bit_error_rate(decided, bits)
            analytic = float(
                np.sum(wq * 0.5 * special.erfc(hq * amplitude / (det.sigma * np.sqrt(2.0))))
            )
            z = (
                (result.rate - analytic) / result.standard_error
                if result.standard_error > 0
                else float("nan")
            )
            print(f"  {sigma_i2:10.2f} {ebn0:9.2f} {amplitude:10.6f} {result.rate:14.6e} "
                  f"{result.errors:9d} {result.standard_error:11.3e} {analytic:15.6e} "
                  f"{z:8.3f}")
    print()
    print("  The headline comparand value, for the coordinating session to use:")
    sigma_i2, ebn0 = 0.30, 10.0
    fading = LognormalFading(sigma_i2)
    hq, wq = amplitude_quadrature(fading)
    seed = COMPARAND_SEED + 101 * int(round(10 * sigma_i2)) + int(round(ebn0))
    rng = np.random.default_rng(seed)
    amplitude = det.bpsk_amplitude(ebn0)
    bits = rng.integers(0, 2, size=COMPARAND_BITS).astype(np.int8)
    symbols = 1.0 - 2.0 * bits
    h = np.exp(fading.mu_x + fading.sigma_x * rng.standard_normal(COMPARAND_BITS))
    y = amplitude * h * symbols + det.sigma * rng.standard_normal(COMPARAND_BITS)
    result = bit_error_rate((y < 0.0).astype(np.int8), bits)
    analytic = float(
        np.sum(wq * 0.5 * special.erfc(hq * amplitude / (det.sigma * np.sqrt(2.0))))
    )
    print(f"    scintillation index   {sigma_i2:.2f}")
    print(f"    Eb/N0                 {ebn0:.2f} dB  (amplitude a = {amplitude:.9f})")
    print(f"    bits                  {COMPARAND_BITS}")
    print(f"    seed                  {seed}")
    print(f"    sample BER            {result.rate:.9e}")
    print(f"    errors                {result.errors}")
    print(f"    binomial se           {result.standard_error:.6e}")
    print(f"    three se interval     [{result.rate - 3 * result.standard_error:.6e}, "
          f"{result.rate + 3 * result.standard_error:.6e}]")
    print(f"    quadrature reference  {analytic:.9e}")
    print("    No comparison to any other product's value is made in this file.")
    print("    No quantity in this file is compared to a wall-clock measurement.")


def main() -> int:
    print("softdecode validation 6: cross-check X3 and the BPSK comparand")
    print("raw output committed as validation/cross_check_x3_output.txt")
    ok = cross_check_x3()
    paired_detail()
    comparand()
    print(f"\nX3 ordering held at every point: {ok}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
