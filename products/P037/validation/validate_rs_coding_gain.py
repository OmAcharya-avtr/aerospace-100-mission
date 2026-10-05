"""RS(255,223) coding gain at 10^-5, and the state of the reference figure.

What this checks
----------------
1. The semi-analytic RS frame error rate of Eqs. (7) and (8) against Monte
   Carlo with the real ``reedsolo`` decoder, at Eb/N0 values where the rate
   is large enough to measure inside the compute budget. This is what makes
   the extrapolation to 10^-5 credible: the curve is validated where it can
   be measured, then inverted where it cannot.
2. The coding gain at 10^-5, reported for two metrics -- equal output bit
   error rate and equal frame error rate -- with the convention stated.
3. The gain against the asymptotic-coding-gain bound
   G_inf = 10 log10(R (E+1)) dB for a hard-decision bounded-distance
   decoder, which is a formula and can therefore be stated exactly:
   [standard result; Lin & Costello 2004, "Error Control Coding" 2nd ed.,
   coding-gain discussion; Sklar 2001, "Digital Communications:
   Fundamentals and Applications" 2nd ed., Ch. 8]. Real gain at a finite
   error rate must be strictly below G_inf.
4. The uncoded anchor: Eb/N0 for BER 1e-5, against the textbook-quoted
   9.6 dB for coherent BPSK.

On the reference range this product was asked to compare against
---------------------------------------------------------------
The build specification asks for the measured gain to fall "in the range the
standard references quote". The performance curves for the CCSDS Reed-Solomon
code are carried by CCSDS 130.1-G (TM Synchronization and Channel Coding --
Summary of Concept and Rationale). This build container has no network
access and that document was not available to be read, so **no numeric
range is quoted from it here**: quoting a figure and a page from memory
would be inventing a citation, which is the one thing this mission's build
guide forbids outright. What is checked instead is the asymptotic bound,
which is a formula rather than a remembered number, plus sign and ordering.
The reference-range comparison is therefore reported as NOT VERIFIED in this
environment, and that is recorded in VALIDATION.md and in the README
Limitations as a deviation from the Level 2 requirement as written.

Run: ``python3 validate_rs_coding_gain.py`` from this directory.
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, "../src")

from framesync.conv import ConvCode, free_distance  # noqa: E402
from framesync.fer import (  # noqa: E402
    coding_gain_db,
    ebn0_for_target,
    measure_rs_fer,
)
from framesync.rs import RS_E, RS_RATE  # noqa: E402

SEED = 20261005
MC_POINTS = ((5.25, 100), (5.5, 150), (5.75, 200), (6.0, 250))
INTERLEAVE = 5


def main() -> int:
    t0 = time.time()
    all_pass = True
    print("framesync validation 5 -- RS(255,223) coding gain at 10^-5")
    print("=" * 78)
    print(f"R = 223/255 = {RS_RATE:.9f}, E = {RS_E}, interleave I = {INTERLEAVE}")
    print("Eb/N0 is always per INFORMATION bit, so the 0.5821 dB rate loss of the")
    print("coded link is already inside the channel bit error rate Eq. (1).")
    print()

    print("Check 1: semi-analytic Eqs. (7)/(8) against Monte Carlo with reedsolo")
    print("-" * 78)
    rng = np.random.default_rng(SEED)
    header = (
        f"{'Eb/N0 dB':>8}  {'p_bit':>11}  {'N':>5}  {'errors':>6}  {'FER meas':>11}  "
        f"{'stderr':>10}  {'FER Eq.8':>11}  {'dev/sigma':>9}  result"
    )
    print(header)
    print("-" * len(header))
    for ebn0, n in MC_POINTS:
        pt = measure_rs_fer(ebn0, n, rng, interleave=INTERLEAVE)
        analytic = pt.extra["analytic_fer"]
        se = pt.stderr if pt.stderr > 0 else 1.0 / n
        dev = abs(pt.fer - analytic) / se
        ok = dev < 4.0
        all_pass &= ok
        print(
            f"{ebn0:8.2f}  {pt.extra['p_bit']:11.4e}  {n:5d}  {pt.n_frame_errors:6d}  "
            f"{pt.fer:11.4e}  {se:10.3e}  {analytic:11.4e}  {dev:9.2f}  "
            f"{'PASS' if ok else 'FAIL'}"
        )
    print("acceptance: within 4 binomial standard errors of the semi-analytic value")
    print()

    print("Check 2: the uncoded anchor")
    print("-" * 78)
    unc_ber = ebn0_for_target("uncoded_ber", 1e-5)
    anchor_ok = abs(unc_ber - 9.6) < 0.05
    all_pass &= anchor_ok
    print(f"Eb/N0 for uncoded BPSK BER = 1e-5: {unc_ber:.4f} dB")
    print("textbook-quoted value for coherent BPSK: ~9.6 dB "
          "[Proakis & Salehi 2008, Sec. 4.3; Sklar 2001, Sec. 4.7.1]")
    print("P010 BERBench records the same inversion at 9.5879 dB "
          "(validation/bpsk_textbook_output.txt)")
    print(f"within 0.05 dB of 9.6: {anchor_ok} -> {'PASS' if anchor_ok else 'FAIL'}")
    print()

    print("Check 3: coding gain at 10^-5, both metrics")
    print("-" * 78)
    gains = {}
    for metric in ("ber", "fer"):
        g = coding_gain_db(1e-5, metric=metric, interleave=INTERLEAVE)
        gains[metric] = g
        print(
            f"  {metric.upper():>3} = 1e-5: uncoded needs {g['ebn0_uncoded_db']:7.4f} dB, "
            f"RS needs {g['ebn0_rs_db']:7.4f} dB, gain = {g['gain_db']:6.4f} dB"
        )
    print("  BER metric compares Eq. (10) (approximate post-decoding bit error")
    print("  rate) against Eq. (1); FER metric compares Eq. (8) against Eq. (11)")
    print("  with n = 8936 bits. The two differ because a 8936-bit frame needs a")
    print("  much lower bit error rate to reach 1e-5 at the frame level.")
    print()

    print("Check 4: against the asymptotic-coding-gain bound (a formula, not a")
    print("         remembered figure)")
    print("-" * 78)
    g_inf = 10.0 * math.log10(RS_RATE * (RS_E + 1))
    print(f"G_inf = 10 log10(R (E+1)) = 10 log10({RS_RATE:.6f} x {RS_E + 1}) = "
          f"{g_inf:.4f} dB")
    print("  [Lin & Costello 2004 Ch. 1/12 coding-gain discussion; Sklar 2001 Ch. 8]")
    for metric, g in gains.items():
        ok = 0.0 < g["gain_db"] < g_inf
        all_pass &= ok
        print(
            f"  {metric.upper():>3}: measured gain {g['gain_db']:6.4f} dB, "
            f"0 < gain < {g_inf:.4f} dB -> {'PASS' if ok else 'FAIL'}"
        )
    print("  Real gain well below the asymptote is the expected behaviour of a")
    print("  high-rate hard-decision block code at a moderate error rate.")
    print()

    print("Check 5: the reference range the specification asked for")
    print("-" * 78)
    print("STATUS: NOT VERIFIED in this environment.")
    print("  The CCSDS Green Book 130.1-G carries the Reed-Solomon performance")
    print("  curves. This build container has no access to that document, so no")
    print("  numeric range and no page is quoted from it. A figure recalled from")
    print("  memory and attached to a page number would be an invented citation.")
    print("  The measured numbers stand on their own and are reported above; the")
    print("  comparison against a published range remains open and is recorded as")
    print("  a deviation in VALIDATION.md and README Limitations.")
    print()

    print("Convolutional leg: free distance, computed from the trellis")
    print("-" * 78)
    code = ConvCode()
    d_free = free_distance(code)
    dfree_ok = d_free == 10
    all_pass &= dfree_ok
    print(f"CCSDS (171, 133) rate-1/2 K={code.k}: computed d_free = {d_free}")
    print(f"literature value for this code: 10 -> {'PASS' if dfree_ok else 'FAIL'}")
    print(f"with the G2 inversion switched off: "
          f"{free_distance(ConvCode(invert_g2=False))} (must be unchanged)")
    print()

    payload = {
        "seed": SEED,
        "interleave": INTERLEAVE,
        "code_rate": RS_RATE,
        "ebn0_uncoded_ber_1e-5_db": unc_ber,
        "gain_ber_1e-5_db": gains["ber"]["gain_db"],
        "gain_fer_1e-5_db": gains["fer"]["gain_db"],
        "ebn0_rs_ber_1e-5_db": gains["ber"]["ebn0_rs_db"],
        "ebn0_rs_fer_1e-5_db": gains["fer"]["ebn0_rs_db"],
        "asymptotic_gain_db": g_inf,
        "conv_free_distance_computed": int(d_free),
        "reference_range_status": "NOT_VERIFIED_NO_DOCUMENT_ACCESS",
        "all_pass": bool(all_pass),
    }
    with open("rs_coding_gain.json", "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print("wrote rs_coding_gain.json")
    print(f"ALL CHECKS PASS: {all_pass}")
    print(f"wall time: {time.time() - t0:.1f} s")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
