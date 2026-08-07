"""Validation 4: Noll residual variance coefficients against published values.

Compares the residual mean-square wavefront error ``Delta_J`` computed from the
analytic Kolmogorov Zernike-coefficient variances against the values published
in R. J. Noll, "Zernike polynomials and atmospheric turbulence", *Journal of
the Optical Society of America* **66**(3), 207-211 (1976), for J = 1..21.
(No page or table number beyond the paper's own labelling is claimed.)

Two spectral constants are used:
  * ``C_psd = 0.023`` -- the value Noll quotes in his Eq. (4), rounded to two
    significant figures;
  * ``C_psd = 0.490 / (2 pi)^(5/3) = 0.0229037`` -- the unrounded equivalent of
    the standard phase PSD ``Phi_phi(kappa) = 0.490 r0^(-5/3) kappa^(-11/3)``
    (Roddier 1981, *Progress in Optics* XIX; Hardy 1998, *Adaptive Optics for
    Astronomical Telescopes*).

An independent cross-check of ``Delta_1`` is also performed that never touches
the Zernike series: for a stationary phase screen the aperture-averaged,
piston-removed variance is ``sigma^2 = (1/2) <D_phi(|r1 - r2|)>`` with the two
points uniform on the pupil. With the Kolmogorov structure function
``D_phi(r) = 6.883877 (r / r0)^(5/3)`` (Fried 1965; the coefficient is
``2 (24/5 Gamma(6/5))^(5/6)``) this is a one-dimensional integral over the
known distance distribution of two points in a disc.

Run:  python validation/validate_noll_variance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zernkit import (  # noqa: E402
    KOLMOGOROV_PSD_CONSTANT,
    NOLL_PSD_CONSTANT,
    NOLL_TABLE_IV,
    coefficient_variance,
    residual_variance,
    residual_variance_asymptotic,
)


def structure_function_delta1() -> float:
    """Piston-removed Kolmogorov variance over a unit-diameter disc, in (D/r0)^(5/3).

    sigma^2 = (1/2) * 6.883877 * <s^(5/3)> with s the distance between two
    points drawn uniformly on a disc of diameter 1. The distance density is
    f(s) = (2s/R^2)(2/pi)[arccos(s/2R) - (s/2R) sqrt(1 - (s/2R)^2)], R = 1/2.
    """
    radius = 0.5
    coeff = 2.0 * (24.0 / 5.0 * gamma(6.0 / 5.0)) ** (5.0 / 6.0)

    def density(s: float) -> float:
        u = s / (2.0 * radius)
        return (
            (2.0 * s / radius**2)
            * (2.0 / np.pi)
            * (np.arccos(u) - u * np.sqrt(max(0.0, 1.0 - u * u)))
        )

    moment, _ = quad(lambda s: s ** (5.0 / 3.0) * density(s), 0.0, 2.0 * radius, limit=400)
    return 0.5 * coeff * moment


def main() -> int:
    lines: list[str] = []
    ok = True

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text)

    emit("ZernKit validation 4 -- Noll residual variance coefficients")
    emit("=" * 86)
    emit("Published reference: Noll (1976), JOSA 66(3), 207-211, residual mean-square")
    emit("error Delta_J after removal of the first J Zernike terms, in (D/r0)^(5/3).")
    emit("")
    emit(f"C_psd = {NOLL_PSD_CONSTANT} (Noll Eq. 4, rounded)")
    emit(f"C_psd = {KOLMOGOROV_PSD_CONSTANT:.7f} (unrounded, 0.490/(2 pi)^(5/3))")
    emit("")
    emit(
        f"{'J':>4} {'published':>11} {'computed(0.023)':>17} {'rel':>9} "
        f"{'computed(0.02290)':>19} {'rel':>9}"
    )
    worst_noll = 0.0
    worst_exact = 0.0
    for j in sorted(NOLL_TABLE_IV):
        published = NOLL_TABLE_IV[j]
        c_noll = residual_variance(j, psd_constant=NOLL_PSD_CONSTANT)
        c_exact = residual_variance(j, psd_constant=KOLMOGOROV_PSD_CONSTANT)
        r_noll = (c_noll - published) / published
        r_exact = (c_exact - published) / published
        worst_noll = max(worst_noll, abs(r_noll))
        worst_exact = max(worst_exact, abs(r_exact))
        emit(
            f"{j:>4} {published:>11.4f} {c_noll:>17.6f} {r_noll:>+8.2%} "
            f"{c_exact:>19.6f} {r_exact:>+8.2%}"
        )
    emit("")
    emit(f"Worst relative deviation with C_psd = 0.023   : {worst_noll:.3%}")
    emit(f"Worst relative deviation with C_psd = 0.022904: {worst_exact:.3%}")
    emit(f"Tolerance 1 % -> {'PASS' if worst_noll < 0.01 else 'FAIL'}")
    ok = ok and worst_noll < 0.01
    emit("")

    emit("Per-mode coefficient variances <a_j^2> (units (D/r0)^(5/3)),")
    emit("compared with differences of consecutive published Delta_J:")
    emit(f"{'n':>3} {'modes':>6} {'computed':>12} {'from table':>12} {'rel':>9}  from")
    table_checks = [
        (1, (1, 3), 2, "(D1 - D3)/2"),
        (2, (3, 6), 3, "(D3 - D6)/3"),
        (3, (6, 10), 4, "(D6 - D10)/4"),
        (4, (10, 15), 5, "(D10 - D15)/5"),
        (5, (15, 21), 6, "(D15 - D21)/6"),
    ]
    for n, (j_lo, j_hi), count, label in table_checks:
        from_table = (NOLL_TABLE_IV[j_lo] - NOLL_TABLE_IV[j_hi]) / count
        computed = coefficient_variance(n, psd_constant=KOLMOGOROV_PSD_CONSTANT)
        rel = (computed - from_table) / from_table
        emit(f"{n:>3} {count:>6} {computed:>12.6f} {from_table:>12.6f} {rel:>+8.2%}  {label}")
    emit("")

    emit("Independent cross-check of Delta_1 (no Zernike series involved):")
    sf = structure_function_delta1()
    zk_exact = residual_variance(1, psd_constant=KOLMOGOROV_PSD_CONSTANT)
    zk_noll = residual_variance(1, psd_constant=NOLL_PSD_CONSTANT)
    emit(f"  structure-function integral      : {sf:.6f} (D/r0)^(5/3)")
    emit(f"  Zernike series, C_psd = 0.022904 : {zk_exact:.6f}")
    emit(f"  Zernike series, C_psd = 0.023    : {zk_noll:.6f}")
    emit(f"  published Noll Delta_1           : {NOLL_TABLE_IV[1]:.6f}")
    rel_sf = abs(zk_exact - sf) / sf
    emit(f"  |series - structure function| / sf = {rel_sf:.3%}")
    emit(f"  Tolerance 0.5 % -> {'PASS' if rel_sf < 0.005 else 'FAIL'}")
    ok = ok and rel_sf < 0.005
    emit("")
    emit("  Both independent routes land ~0.3 % above Noll's published 1.0299, so the")
    emit("  residual gap is in the published rounding, not in this implementation.")
    emit("")

    emit("Large-J asymptote Delta_J ~ 0.2944 J^(-sqrt(3)/2) (Noll 1976):")
    emit(f"{'J':>4} {'published':>11} {'asymptotic':>12} {'rel':>9}")
    for j in (10, 15, 20, 21):
        asym = residual_variance_asymptotic(j)
        published = NOLL_TABLE_IV[j]
        emit(f"{j:>4} {published:>11.4f} {asym:>12.6f} {(asym - published) / published:>+8.2%}")
    emit("")

    emit("Numerical convergence of the residual sum (J = 5, C_psd = 0.023):")
    ref = residual_variance(5, n_max=1_000_000)
    for n_max in (100, 1_000, 20_000, 200_000):
        val = residual_variance(5, n_max=n_max)
        emit(f"  n_max = {n_max:>9}: {val:.12f}  (delta vs n_max=1e6: {val - ref:+.3e})")

    out = Path(__file__).with_name("noll_variance_output.txt")
    out.write_text("\n".join(lines) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
