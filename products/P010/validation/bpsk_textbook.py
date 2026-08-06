"""Cross-check BPSK analytic BER against textbook values.

References checked:
1. Proakis & Salehi, "Digital Communications", 5th ed. 2008, Eq. (4.3-13):
   Pb = Q(sqrt(2 Eb/N0)). The classic benchmark read off Fig. 4.3-1 and used
   throughout the literature: Pb = 1e-5 at Eb/N0 ~ 9.6 dB.
   Exact inversion: Q(x) = 1e-5 at x = 4.264891, so Eb/N0 = x^2/2 = 9.0946
   linear = 9.588 dB. At exactly 9.6 dB the exact Pb is 9.80e-6.
2. Sklar, "Digital Communications: Fundamentals and Applications", 2nd ed.
   2001, Sec. 4.7.1 worked example: BPSK at Eb/N0 = 9.6 dB => Pb ~ 1e-5.
3. Q-function spot values against scipy.stats.norm.sf (independent path
   through scipy) and the A&S normal table value Q(1) = 0.1587.

Writes bpsk_textbook_output.txt. Runtime < 1 s.
"""

import sys
from pathlib import Path

from scipy.optimize import brentq
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from berbench import analytic_ber, qfunc  # noqa: E402

OUT = Path(__file__).resolve().parent / "bpsk_textbook_output.txt"


def main() -> None:
    lines = ["BPSK textbook cross-check (berbench 0.1.0)", "=" * 60]

    # 1) The 9.6 dB @ 1e-5 benchmark (Proakis Fig. 4.3-1 / Sklar Sec. 4.7.1)
    ber_96 = analytic_ber("bpsk", 9.6).ber[0]
    lines.append(f"Pb at Eb/N0 = 9.6 dB           : {ber_96:.6e} (textbook: ~1e-5)")
    # exact Eb/N0 that yields exactly 1e-5
    snr_at_1e5 = brentq(lambda s: analytic_ber("bpsk", s).ber[0] - 1e-5, 5.0, 12.0)
    lines.append(f"Eb/N0 for Pb = 1e-5 (inverted) : {snr_at_1e5:.4f} dB "
                 "(textbook quotes ~9.6 dB)")
    ok1 = abs(ber_96 / 1e-5 - 1.0) < 0.05 and abs(snr_at_1e5 - 9.6) < 0.05
    lines.append(f"PASS: {ok1} (Pb within 5% of 1e-5 at 9.6 dB; inverted SNR within "
                 "0.05 dB of 9.6)")

    # 2) Q(sqrt(20)) at 10 dB — exact value via independent scipy path
    ber_10 = analytic_ber("bpsk", 10.0).ber[0]
    ref_10 = float(norm.sf(np.sqrt(20.0)))
    lines.append(f"Pb at Eb/N0 = 10 dB            : {ber_10:.9e}")
    lines.append(f"scipy norm.sf(sqrt(20))        : {ref_10:.9e}  "
                 f"rel diff {abs(ber_10 - ref_10) / ref_10:.2e}")
    ok2 = abs(ber_10 - ref_10) / ref_10 < 1e-12

    # 3) Q(1) vs Abramowitz & Stegun Table 26.1: 1 - 0.8413447461 = 0.1586552539
    q1 = qfunc(1.0)
    lines.append(f"Q(1)                           : {q1:.10f} (A&S: 0.1586552539)")
    ok3 = abs(q1 - 0.1586552539) < 1e-9

    # 4) OOK = BPSK shifted by exactly 3.0103 dB (10 log10 2)
    shift = 10 * np.log10(2.0)
    d = analytic_ber("ook", 8.0).ber[0] - analytic_ber("bpsk", 8.0 - shift).ber[0]
    lines.append(f"OOK(8 dB) - BPSK(8-3.0103 dB)  : {d:.3e} (identity, expect ~0)")
    ok4 = abs(d) < 1e-15

    lines.append("")
    lines.append(f"ALL CHECKS PASS: {ok1 and ok2 and ok3 and ok4}")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
