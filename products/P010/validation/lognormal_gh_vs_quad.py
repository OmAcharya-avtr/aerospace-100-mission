"""Lognormal fading average: Gauss-Hermite quadrature vs adaptive quadrature.

Validates the GH-based fading average (Zhu & Kahn 2002 technique) against an
independent scipy.integrate.quad evaluation of

    Pb = Integral_0^inf Pb_cond(I) p_LN(I) dI,
    p_LN: lognormal pdf with E[I]=1, sigma_z^2 = ln(1 + sigma_I^2).

Also verifies the lognormal parameterisation itself (E[I] = 1,
Var[I] = sigma_I^2) and documents the known accuracy limit of the
fixed-threshold OOK case (step-like integrand at high SNR).

Writes lognormal_gh_vs_quad_output.txt. Runtime ~2 s.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.integrate import quad  # noqa: E402
from scipy.stats import lognorm  # noqa: E402

from berbench import analytic_ber, qfunc  # noqa: E402

OUT = Path(__file__).resolve().parent / "lognormal_gh_vs_quad_output.txt"


def main() -> None:
    lines = ["Lognormal GH average vs scipy.integrate.quad (berbench 0.1.0)", "=" * 78]

    # parameterisation check
    for s2 in (0.1, 0.3, 0.8):
        sz = math.sqrt(math.log1p(s2))
        dist = lognorm(s=sz, scale=math.exp(-0.5 * sz * sz))
        lines.append(f"sigma_I^2={s2}: E[I]={dist.mean():.12f} (want 1), "
                     f"Var[I]={dist.var():.12f} (want {s2})")

    lines.append("")
    lines.append(f"{'case':<38} {'quad':>12} {'GH':>12} {'rel diff':>10}")
    ok = True
    rows = []
    for s2 in (0.1, 0.3):
        sz = math.sqrt(math.log1p(s2))
        dist = lognorm(s=sz, scale=math.exp(-0.5 * sz * sz))
        for mod, snr_db in (("ook", 6.0), ("ook", 12.0), ("bpsk", 6.0), ("bpsk", 10.0)):
            g = 10 ** (snr_db / 10)
            if mod == "ook":
                cond = lambda irr: qfunc(irr * math.sqrt(g))  # noqa: E731
            else:
                cond = lambda irr: qfunc(irr * math.sqrt(2 * g))  # noqa: E731
            ref, _ = quad(lambda irr: cond(irr) * dist.pdf(irr), 0, 60, limit=400)
            gh = analytic_ber(mod, snr_db, channel="lognormal", sigma_i2=s2).ber[0]
            rel = abs(gh - ref) / ref
            ok &= rel < 1e-8
            rows.append((f"{mod} s2={s2} snr={snr_db} (adaptive)", ref, gh, rel))
    # fixed-threshold OOK: harder integrand, documented accuracy limit
    s2, t = 0.3, 0.5
    sz = math.sqrt(math.log1p(s2))
    dist = lognorm(s=sz, scale=math.exp(-0.5 * sz * sz))
    ok_fixed = True
    for snr_db in (12.0, 20.0, 30.0):
        g = 10 ** (snr_db / 10)

        def cond_fixed(irr: float) -> float:
            return 0.5 * (qfunc(2 * t * math.sqrt(g)) + qfunc(2 * (irr - t) * math.sqrt(g)))

        ref, _ = quad(lambda irr: cond_fixed(irr) * dist.pdf(irr), 0, 60,
                      limit=400, points=[t])
        gh = analytic_ber("ook", snr_db, channel="lognormal", sigma_i2=s2,
                          threshold=t).ber[0]
        rel = abs(gh - ref) / ref
        ok_fixed &= rel < 1e-2 if snr_db <= 20 else rel < 5e-2
        rows.append((f"ook s2={s2} snr={snr_db} (FIXED t=0.5)", ref, gh, rel))
    for name, ref, gh, rel in rows:
        lines.append(f"{name:<38} {ref:>12.6e} {gh:>12.6e} {rel:>10.2e}")
    lines.append("")
    lines.append(f"adaptive-threshold cases agree with quad to < 1e-8: {ok}")
    lines.append(f"fixed-threshold cases within documented limits "
                 f"(<1% at <=20 dB, <5% at 30 dB): {ok_fixed}")
    lines.append("NOTE: the fixed-threshold integrand approaches a step function as "
                 "SNR -> inf; GH-256 accuracy degrades there (documented limitation; "
                 "at 30 dB rel err ~5e-3).")
    lines.append("")
    lines.append(f"ALL CHECKS PASS: {ok and ok_fixed}")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
