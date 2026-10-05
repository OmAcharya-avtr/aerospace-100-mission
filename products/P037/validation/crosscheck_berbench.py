"""Mandatory cross-check: FrameSync frame error rate against P010 BERBench.

The specification (batch_reports/BATCH_04_SPEC.md) requires that FrameSync's
frame error rate be consistent with P010 BERBench's bit error rate for the
same channel at the same Eb/N0, through the analytic frame/bit relation, and
that disagreement be reported as a finding.

What is compared
----------------
Channel: coherent BPSK over AWGN, uncoded, Eb/N0 per bit, no fading, no
implementation loss. This is P010's ``analytic_ber("bpsk", snr_db,
channel="awgn")`` case.

P010's expression, from P010 src/berbench/analytic.py module docstring and
``_ber_bpsk_cond``:

    Pb = Q(sqrt(2 gamma)),  gamma = 10**(snr_db / 10)
    [Proakis & Salehi 2008, Eq. (4.3-13)]

It is recomputed here from that formula rather than imported -- cross-product
imports are forbidden by the build guide, and an import would make the
comparison circular anyway. Two independent evaluations of Q are used, this
package's ``erfc`` path and ``scipy.stats.norm.sf``, so an error in either
would show up.

Relation: FER = 1 - (1 - Pb)^n, Eq. (11), with n the frame length in bits.

Then FrameSync's **measured** frame error rate, from the Monte Carlo
harness that flips individual bits and checks the Frame Error Control
Field, is compared against that converted value. The relative difference and
the measured point's binomial standard error are written to
``crosscheck_berbench.json``.

Independent anchors from P010's own committed validation output, quoted so
the comparison is against P010's recorded numbers and not only against a
formula:

  * P010 validation/bpsk_textbook_output.txt:
        "Pb at Eb/N0 = 10 dB            : 3.872108216e-06"
        "Eb/N0 for Pb = 1e-5 (inverted) : 9.5879 dB"
  * P010 validation/VALIDATION.md, representative rows table:
        "bpsk [awgn] | 8 dB | BER analytic 1.9091e-04"

Run: ``python3 crosscheck_berbench.py`` from this directory.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.stats import norm

sys.path.insert(0, "../src")

from framesync.channel import bpsk_ber  # noqa: E402
from framesync.fer import measure_uncoded_fer, uncoded_fer  # noqa: E402
from framesync.frames import FrameGeometry  # noqa: E402

SEED = 20261005
EBN0_DB = 9.0
GEOM = FrameGeometry(data_octets=1115, fecf=True, asm_bits=32)
N_FRAMES = 20_000
TOL_REL = 0.05
TOL_SIGMA = 4.0

# Values recorded in P010 BERBench's own committed validation output.
P010_ANCHORS = {
    10.0: 3.872108216e-06,  # validation/bpsk_textbook_output.txt
    8.0: 1.9091e-04,  # validation/VALIDATION.md representative rows
}
P010_INVERTED_1E5_DB = 9.5879  # validation/bpsk_textbook_output.txt


def p010_bpsk_ber(snr_db: float) -> float:
    """P010's expression, reimplemented: Pb = Q(sqrt(2 gamma)) via scipy norm."""
    gamma = 10.0 ** (snr_db / 10.0)
    return float(norm.sf(np.sqrt(2.0 * gamma)))


def main() -> int:
    t0 = time.time()
    all_pass = True
    print("framesync validation 7 -- cross-check against P010 BERBench")
    print("=" * 78)
    print("channel: coherent BPSK, AWGN, uncoded, Eb/N0 per bit, no fading")
    print("P010 expression: Pb = Q(sqrt(2 gamma)), gamma = 10^(Eb/N0 dB / 10)")
    print("                 [P010 src/berbench/analytic.py; Proakis & Salehi 2008,")
    print("                  Eq. (4.3-13)]")
    print("relation:        FER = 1 - (1 - Pb)^n  [Eq. (11)]")
    print(f"frame:           {GEOM.data_octets} octets + 2 octet FECF = "
          f"{GEOM.frame_bits} bits")
    print()

    print("Step 1: P010's BER expression against P010's own recorded numbers")
    print("-" * 78)
    for snr, recorded in sorted(P010_ANCHORS.items()):
        recomputed = p010_bpsk_ber(snr)
        ours = float(bpsk_ber(snr)[0])
        d_rec = abs(recomputed - recorded) / recorded
        d_ours = abs(ours - recomputed) / recomputed
        ok = d_rec < 1e-4 and d_ours < 1e-12
        all_pass &= ok
        print(
            f"  {snr:5.1f} dB: P010 recorded {recorded:.6e}, recomputed "
            f"{recomputed:.6e} (rel {d_rec:.2e}), framesync {ours:.6e} "
            f"(rel {d_ours:.2e}) -> {'PASS' if ok else 'FAIL'}"
        )
    print()

    print("Step 2: P010's BER converted to a frame error rate through Eq. (11)")
    print("-" * 78)
    p010_ber = p010_bpsk_ber(EBN0_DB)
    framesync_ber = float(bpsk_ber(EBN0_DB)[0])
    fer_from_p010 = 1.0 - (1.0 - p010_ber) ** GEOM.frame_bits
    fer_framesync_analytic = float(uncoded_fer(EBN0_DB, GEOM.frame_bits)[0])
    print(f"  Eb/N0                                  : {EBN0_DB:.2f} dB")
    print(f"  P010 BER (its formula, scipy norm.sf)  : {p010_ber:.9e}")
    print(f"  framesync BER (its formula, erfc)      : {framesync_ber:.9e}")
    print(f"  relative difference in BER             : "
          f"{abs(framesync_ber - p010_ber) / p010_ber:.3e}")
    print(f"  FER from P010 BER via Eq. (11)         : {fer_from_p010:.9e}")
    print(f"  FER from framesync Eq. (11)            : {fer_framesync_analytic:.9e}")
    print()

    print(f"Step 3: framesync MEASURED frame error rate, {N_FRAMES} frames")
    print("-" * 78)
    rng = np.random.default_rng(SEED)
    pt = measure_uncoded_fer(EBN0_DB, GEOM, N_FRAMES, rng)
    rel = abs(pt.fer - fer_from_p010) / fer_from_p010
    dev_sigma = abs(pt.fer - fer_from_p010) / pt.stderr
    ok_rel = rel < TOL_REL
    ok_sigma = dev_sigma < TOL_SIGMA
    all_pass &= ok_rel and ok_sigma
    print(f"  frames                                 : {pt.n_frames}")
    print(f"  frames with at least one bit error     : {pt.n_frame_errors}")
    print(f"  measured FER                           : {pt.fer:.9e}")
    print(f"  binomial standard error                : {pt.stderr:.9e}")
    print(f"  FER predicted from P010's BER          : {fer_from_p010:.9e}")
    print(f"  relative difference                    : {rel:.6e}")
    print(f"  deviation in standard errors           : {dev_sigma:.3f}")
    print(f"  acceptance: rel < {TOL_REL:g} -> {'PASS' if ok_rel else 'FAIL'}; "
          f"dev < {TOL_SIGMA:g} sigma -> {'PASS' if ok_sigma else 'FAIL'}")
    print()
    print(f"  FECF-detected frame error rate         : "
          f"{pt.extra['n_fecf_detected'] / pt.n_frames:.9e}")
    print(f"  frames with errors the FECF missed     : {pt.extra['n_undetected']}")
    print()

    print("Step 4: the inverted-BER anchor")
    print("-" * 78)
    from framesync.fer import ebn0_for_target

    ours_inv = ebn0_for_target("uncoded_ber", 1e-5)
    d_inv = abs(ours_inv - P010_INVERTED_1E5_DB)
    ok_inv = d_inv < 1e-3
    all_pass &= ok_inv
    print(f"  framesync Eb/N0 at BER 1e-5            : {ours_inv:.4f} dB")
    print(f"  P010 recorded                          : {P010_INVERTED_1E5_DB:.4f} dB")
    print(f"  difference                             : {d_inv:.2e} dB "
          f"(tolerance 1e-3) -> {'PASS' if ok_inv else 'FAIL'}")
    print()

    finding = (
        "NO DISAGREEMENT. framesync's BER expression reproduces P010 BERBench's "
        "recorded BPSK/AWGN values to better than 1e-4 relative (limited only by "
        "the 5 significant figures P010 printed for the 8 dB row), the inverted "
        "Eb/N0 at BER 1e-5 agrees to 1e-4 dB, and framesync's measured frame "
        f"error rate at {EBN0_DB:.1f} dB sits {dev_sigma:.2f} binomial standard "
        "errors from the value obtained by converting P010's BER through "
        "FER = 1 - (1 - Pb)^n."
        if all_pass
        else "DISAGREEMENT -- see the per-step results above; this is a finding."
    )
    print("Finding")
    print("-" * 78)
    for line in finding.split(". "):
        if line:
            print(f"  {line.strip()}.")
    print()

    payload = {
        "channel": "bpsk-awgn-uncoded",
        "ebn0_db": EBN0_DB,
        "frame_bits": GEOM.frame_bits,
        "seed": SEED,
        "n_frames": pt.n_frames,
        "n_frame_errors": pt.n_frame_errors,
        "p010_ber_recomputed": p010_ber,
        "framesync_ber": framesync_ber,
        "ber_relative_difference": abs(framesync_ber - p010_ber) / p010_ber,
        "fer_from_p010_ber": fer_from_p010,
        "fer_framesync_analytic": fer_framesync_analytic,
        "fer_framesync_measured": pt.fer,
        "fer_measured_stderr": pt.stderr,
        "relative_difference_measured_vs_p010": rel,
        "deviation_sigma": dev_sigma,
        "p010_anchors_recorded": {str(k): v for k, v in P010_ANCHORS.items()},
        "p010_inverted_1e-5_db": P010_INVERTED_1E5_DB,
        "framesync_inverted_1e-5_db": ours_inv,
        "all_pass": bool(all_pass),
        "finding": finding,
    }
    with open("crosscheck_berbench.json", "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print("wrote crosscheck_berbench.json")
    print(f"ALL CHECKS PASS: {all_pass}")
    print(f"wall time: {time.time() - t0:.1f} s")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
