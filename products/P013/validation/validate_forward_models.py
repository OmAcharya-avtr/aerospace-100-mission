"""Validation 1: the forward models against their analytic and published references.

Run:  python validation/validate_forward_models.py > validation/validate_forward_models_output.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import integrate, special

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turbscope import (  # noqa: E402
    PathGeometry,
    dimm_coefficient,
    dimm_variance,
    fried_from_average,
    scintillation_weight,
    weight_normalisation,
)
from turbscope.scintillation import (  # noqa: E402
    RYTOV_COEFFICIENT,
    aperture_parameter_sq,
    rytov_variance_from_average,
    scintillation_index,
)

ARCSEC = 180.0 * 3600.0 / np.pi


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def section_1_rytov_coefficient() -> None:
    rule("1. Rytov coefficient derived from the double integral (not copied)")
    print(
        "sigma_I^2 = 8 pi^2 k^2 L int_0^1 int_0^inf kappa Phi_n [1-cos(L kappa^2 xi(1-xi)/k)]"
        " dkappa dxi\n"
        "Phi_n = 0.033 Cn2 kappa^(-11/3)  (Kolmogorov; Andrews & Phillips 2005 Eq. 8.10)\n"
    )
    exact_inner = -special.gamma(-5.0 / 6.0) * np.cos(5.0 * np.pi / 12.0)
    print(f"  analytic inner integral  int u^(-11/6)(1-cos u) du = {exact_inner:.12f}")
    print(f"  C_R = 8 pi^2 (0.033)(0.5) x that = {RYTOV_COEFFICIENT:.12f}")
    print(f"  literature value quoted as 2.25            relative difference "
          f"{abs(RYTOV_COEFFICIENT / 2.25 - 1):.4e}")

    beta_norm = float(special.beta(11.0 / 6.0, 11.0 / 6.0))
    sph = RYTOV_COEFFICIENT * beta_norm
    pln = RYTOV_COEFFICIENT * 6.0 / 11.0
    print(f"\n  uniform-path spherical constant C_R*B(11/6,11/6) = {sph:.10f}")
    print(f"    literature 0.5  (A&P Eq. 8.13)           relative difference "
          f"{abs(sph / 0.5 - 1):.4e}")
    print(f"  uniform-path plane constant     C_R*(6/11)   = {pln:.10f}")
    print(f"    literature 1.23 (A&P Eq. 8.9)            relative difference "
          f"{abs(pln / 1.23 - 1):.4e}")

    # brute-force numerical double integral as an independent check
    def inner(a: float) -> float:
        v1, _ = integrate.quad(lambda u: u ** (-11 / 6) * (1 - np.cos(u)), 0, 60, limit=500)
        v2, _ = integrate.quad(
            lambda u: u ** (-11 / 6), 60, np.inf, weight="cos", wvar=1.0, limit=500
        )
        tail = 60 ** (-5 / 6) / (5 / 6)
        return 0.5 * a ** (5 / 6) * (v1 + tail - v2)

    k, ell, cn2 = 2 * np.pi / 1.55e-6, 1000.0, 1e-15
    num, _ = integrate.quad(
        lambda xi: 8 * np.pi**2 * k**2 * ell * 0.033 * cn2 * inner(ell * xi * (1 - xi) / k),
        0.0,
        1.0,
        limit=300,
    )
    closed = sph * cn2 * k ** (7 / 6) * ell ** (11 / 6)
    print(f"\n  brute-force double integral (L=1km, 1550nm, Cn2=1e-15): {num:.10e}")
    print(f"  closed form C_R*B*Cn2*k^(7/6)*L^(11/6):                 {closed:.10e}")
    print(f"  relative difference:                                    {abs(num / closed - 1):.3e}")


def section_2_quadrature_convergence() -> None:
    rule("2. Simpson quadrature convergence of the scintillation kernel")
    exact = weight_normalisation("scintillation", "spherical")
    print(f"exact int_0^1 u^(5/6)(1-u)^(5/6) du = B(11/6,11/6) = {exact:.12f}")
    print(f"{'N points':>10} {'Simpson value':>18} {'relative error':>16}")
    for n in (51, 101, 201, 401, 801, 1601, 3201):
        u = np.linspace(0.0, 1.0, n)
        got = float(integrate.simpson(scintillation_weight(u, "spherical"), x=u))
        print(f"{n:>10} {got:>18.12f} {abs(got / exact - 1):>16.3e}")
    print("\nThe dataset uses N = 201 (relative error 9.2e-06), three orders of magnitude")
    print("below the smallest measurement noise in the suite.")


def section_3_dimm_known_answers() -> None:
    rule("3. DIMM forward model against hand arithmetic (Sarazin & Roddier 1990)")
    d_sub, d_base, lam, r0 = 0.06, 0.20, 500e-9, 0.10
    k_l = dimm_coefficient(d_sub, d_base, "longitudinal")
    k_t = dimm_coefficient(d_sub, d_base, "transverse")
    print("K_l = 0.179 D^(-1/3) - 0.0968 d^(-1/3)")
    print("    = 0.179*2.5543653 - 0.0968*1.7099759 = 0.4572314 - 0.1655257")
    print(f"    hand 0.2917057   code {k_l:.7f}   diff {abs(k_l - 0.2917057):.2e}")
    print("K_t = 0.179 D^(-1/3) - 0.145 d^(-1/3)")
    print("    = 0.4572314 - 0.2479465")
    print(f"    hand 0.2092849   code {k_t:.7f}   diff {abs(k_t - 0.2092849):.2e}")
    var_l = dimm_variance(r0, lam, d_sub, d_base, "longitudinal")
    print("\nsigma_l^2 = 2 lambda^2 r0^(-5/3) K_l")
    print("          = 2*(5e-7)^2 * 10^(5/3) * 0.2917057")
    print("          = 5e-13 * 46.415888 * 0.2917057")
    print(f"    hand 6.769888e-12 rad^2   code {var_l:.6e} rad^2")
    print(f"    rms  {np.sqrt(var_l) * ARCSEC:.6f} arcsec (0.536681 by hand)")
    print("\nFried r0 for a uniform Cn2 = 1e-15, L = 1000 m, plane wave, 500 nm:")
    print("  r0 = [0.423 k^2 L Cn2]^(-3/5) = 66.797483^(-0.6)")
    r0_code = fried_from_average(1e-15, PathGeometry(1000.0, 500e-9, "plane"))
    print(f"    hand 0.0803792 m   code {r0_code:.7f} m   diff {abs(r0_code - 0.0803792):.2e}")


def section_4_large_aperture_scaling() -> None:
    rule("4. Large-aperture weak-regime scaling exponents")
    print("Wang, Ochs & Clifford (1978), JOSA 68(3) 334-338, showed that a large-aperture")
    print("scintillometer obeys Cn2 ~ sigma^2 D^(7/3) L^(-3), i.e. sigma^2 ~ Cn2 D^(-7/3) L^3.")
    print("Their instrument is double-ended (transmitter AND receiver apertures); the model")
    print("here is single-ended (point source, finite receiver aperture), so the CONSTANT")
    print("differs and is not compared -- only the aperture and path-length EXPONENTS.\n")
    lam, cn2, ell = 1.55e-6, 1e-16, 1000.0  # weak, so sigma_I^2 ~ beta_0^2 exactly
    print(f"{'D range (m)':>16} {'d^2 range':>20} {'d log sigma/d log D':>22}")
    for lo, hi in ((0.30, 1.20), (1.0, 4.0), (3.0, 12.0), (10.0, 40.0)):
        diameters = np.geomspace(lo, hi, 9)
        p = PathGeometry(ell, lam)
        beta = rytov_variance_from_average(cn2, p)
        vals = [float(scintillation_index(beta, aperture_parameter_sq(d, p))) for d in diameters]
        slope = float(np.polyfit(np.log(diameters), np.log(vals), 1)[0])
        d2lo = aperture_parameter_sq(lo, p)
        d2hi = aperture_parameter_sq(hi, p)
        print(f"{lo:6.2f} - {hi:6.2f} {d2lo:9.3g} - {d2hi:8.3g} {slope:>22.4f}")
    print(f"{'large-scale term':>16} {'-':>20} {-7 / 3:>22.4f}")
    print(f"{'small-scale term':>16} {'-':>20} {-2.0:>22.4f}")

    lengths = np.geomspace(600.0, 2000.0, 9)
    vals = []
    for ln in lengths:
        p = PathGeometry(ln, lam)
        beta = rytov_variance_from_average(cn2, p)
        vals.append(float(scintillation_index(beta, aperture_parameter_sq(3.0, p))))
    slope_l = float(np.polyfit(np.log(lengths), np.log(vals), 1)[0])
    print(f"\n  d(log sigma_I^2)/d(log L) at D = 3 m: {slope_l:+.4f}   (theory +3, "
          f"deviation {abs(slope_l - 3.0):.4f})")
    print("\nRESULT: PARTIAL - the pure D^(-7/3) law is NOT reproduced.")
    print("Andrews & Phillips Eq. 9.60 is the sum of a large-scale term that falls as")
    print("d^(-7/3) and a small-scale term that falls as d^(-2); the composite is")
    print("therefore not a pure power law.  The measured slope sits between the two")
    print("(-2.20 at instrument-sized apertures, drifting toward -2 as the small-scale")
    print("term takes over above d^2 ~ 6e4).  Wang et al.'s D^(7/3) law corresponds to")
    print("the large-scale term alone.  This is reported as a discrepancy, not a pass:")
    print("callers must not assume a clean D^(7/3) aperture law from this package.")


def main() -> int:
    print("TurbScope validation 1 - forward models")
    print("numpy", np.__version__)
    section_1_rytov_coefficient()
    section_2_quadrature_convergence()
    section_3_dimm_known_answers()
    section_4_large_aperture_scaling()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
