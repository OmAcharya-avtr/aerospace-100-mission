"""M-PPM: exact orthogonal-signalling BER vs union bound.

Demonstrates the documented relationship (Proakis & Salehi 2008, Sec. 4.4):
- the union bound (M-1) Q(sqrt(Es/N0)) is an UPPER bound on the exact SER;
- the bound tightens as SNR grows (ratio -> 1);
- for M = 2 the exact expression reduces to the closed-form binary
  orthogonal result Pb = Q(sqrt(Eb/N0)) (identity check, machine precision);
- the exact expression is cross-checked against an independent
  scipy.integrate.quad evaluation of Proakis Eq. (4.4-17).

Writes ppm_bound_vs_exact_output.txt. Runtime ~2 s.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from berbench import analytic_ber, qfunc  # noqa: E402

OUT = Path(__file__).resolve().parent / "ppm_bound_vs_exact_output.txt"


def ser_exact_quad(gamma_s: float, m: int) -> float:
    """Independent evaluation of Proakis Eq. (4.4-17) with adaptive quadrature.

    Integrates the error-form integrand phi(y-a) * (1 - Phi(y)^(M-1)) with a
    pure relative tolerance. The naive form 1 - Integral[phi Phi^(M-1)]
    suffers catastrophic cancellation below SER ~ 1e-12 and cannot serve as a
    reference in the deep tail; this form (evaluated by scipy's adaptive
    QUADPACK, a completely different quadrature than the package's
    Gauss-Hermite) stays accurate there.
    """
    a = np.sqrt(2.0 * gamma_s)

    def integrand(y: float) -> float:
        return norm.pdf(y - a) * (-np.expm1((m - 1) * norm.logcdf(y)))

    val, _ = quad(integrand, a - 40.0, a + 12.0, epsabs=0.0, epsrel=1e-11, limit=800)
    return val


def main() -> None:
    lines = ["M-PPM exact vs union bound (berbench 0.1.0)", "=" * 78]

    # identity: M=2 exact == Q(sqrt(gamma)) closed form
    snr = np.array([0.0, 4.0, 8.0, 12.0])
    ex2 = analytic_ber("ppm", snr, M=2).ber
    closed = qfunc(np.sqrt(10 ** (snr / 10)))
    lines.append(f"M=2 identity max |exact - Q(sqrt(g))| : {np.abs(ex2 - closed).max():.3e}")
    ok_id = np.abs(ex2 - closed).max() < 1e-12

    lines.append("")
    lines.append(f"{'M':>3} {'Eb/N0 dB':>9} {'BER exact':>12} {'BER union':>12} "
                 f"{'ratio U/E':>10} {'quad SER xcheck rel':>20}")
    ok_bound = ok_quad = True
    tighten = []
    for m in (4, 16, 64):
        k = int(np.log2(m))
        ratios = []
        for s in (0.0, 4.0, 8.0, 12.0):
            ex = analytic_ber("ppm", s, M=m).ber[0]
            ub = analytic_ber("ppm", s, M=m, ppm_method="union").ber[0]
            ok_bound &= ub >= ex - 1e-15
            gamma_s = k * 10 ** (s / 10)
            ser_q = ser_exact_quad(gamma_s, m)
            ber_q = ser_q * m / (2 * (m - 1))
            rel = abs(ex - ber_q) / max(ber_q, 1e-300)
            ok_quad &= rel < 1e-6
            ratios.append(ub / ex)
            lines.append(f"{m:>3} {s:>9.1f} {ex:>12.4e} {ub:>12.4e} "
                         f"{ub / ex:>10.4f} {rel:>20.2e}")
        tighten.append(ratios[-1] < ratios[0] and abs(ratios[-1] - 1) < 0.05)
    lines.append("")
    lines.append(f"M=2 reduces to closed form           : {ok_id}")
    lines.append(f"union bound >= exact everywhere      : {ok_bound}")
    lines.append(f"bound ratio -> 1 at high SNR (all M) : {all(tighten)}")
    lines.append(f"exact matches independent quad <1e-6 : {ok_quad}")
    lines.append("")
    lines.append(f"ALL CHECKS PASS: {ok_id and ok_bound and all(tighten) and ok_quad}")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
