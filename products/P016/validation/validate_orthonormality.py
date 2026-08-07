"""Validation 1: orthonormality of the Zernike basis on the unit disc.

Checks the analytic relation (Noll 1976, JOSA 66(3), Eq. 3)

    (1/pi) int_0^{2pi} int_0^1 Z_i Z_j rho drho dtheta = delta_ij

by numerical quadrature, and reports the worst absolute deviation of the
computed Gram matrix from the identity.

Two quadratures are used deliberately:

A. Gauss-Legendre in rho x uniform trapezoid in theta. The integrand is a
   polynomial in rho and a trigonometric polynomial in theta, so both rules
   are exact up to round-off for sufficient nodes. This tests the *library*.
B. A uniform Cartesian pupil grid with a circular mask -- what a real
   wavefront array looks like. This tests nothing about the library and
   everything about sampling: the deviation here is the discretisation error
   an ordinary user will actually experience, and it is reported so nobody
   mistakes A for B.

Run:  python validation/validate_orthonormality.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zernkit import mode_name, noll_to_nm, unit_disc_grid, zernike_noll  # noqa: E402

N_MODES = 36  # Noll j = 1..36, radial orders 0..7


def gram_gauss_legendre(n_modes: int, n_rho: int, n_theta: int) -> np.ndarray:
    """Gram matrix from Gauss-Legendre(rho) x uniform(theta) quadrature."""
    nodes, weights = np.polynomial.legendre.leggauss(n_rho)
    rho = 0.5 * (nodes + 1.0)
    w_rho = 0.5 * weights
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    rr, tt = np.meshgrid(rho, theta, indexing="ij")
    weight = ((w_rho[:, None] * rr) * (2.0 * np.pi / n_theta) / np.pi).ravel()
    modes = np.array([zernike_noll(j, rr, tt).ravel() for j in range(1, n_modes + 1)])
    return (modes * weight) @ modes.T


def gram_cartesian(n_modes: int, n_pix: int) -> np.ndarray:
    """Gram matrix from a masked uniform Cartesian grid (naive pixel sum)."""
    x, y, mask = unit_disc_grid(n_pix)
    xm, ym = x[mask], y[mask]
    rho = np.hypot(xm, ym)
    theta = np.arctan2(ym, xm)
    modes = np.array([zernike_noll(j, rho, theta) for j in range(1, n_modes + 1)])
    return (modes @ modes.T) / xm.size


def main() -> int:
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text)

    emit("ZernKit validation 1 -- orthonormality on the unit disc")
    emit("=" * 72)
    emit(f"Modes tested: Noll j = 1..{N_MODES} (radial orders 0..7)")
    emit("Reference: analytic Kronecker delta, Noll 1976 JOSA 66(3) Eq. 3")
    emit("")

    emit("A. Gauss-Legendre(rho) x uniform(theta) quadrature")
    emit(f"{'n_rho':>8} {'n_theta':>8} {'max|G-I| diag':>16} {'max|G| off-diag':>18}")
    best = None
    for n_rho, n_theta in ((20, 64), (40, 128), (80, 256), (120, 512)):
        gram = gram_gauss_legendre(N_MODES, n_rho, n_theta)
        diag = np.max(np.abs(np.diag(gram) - 1.0))
        off = np.max(np.abs(gram - np.diag(np.diag(gram))))
        emit(f"{n_rho:>8} {n_theta:>8} {diag:>16.3e} {off:>18.3e}")
        best = (diag, off)
    assert best is not None
    worst_a = max(best)
    emit("")
    emit(f"Worst absolute deviation from the identity (finest rule): {worst_a:.3e}")
    emit(f"Tolerance 1e-12 -> {'PASS' if worst_a < 1e-12 else 'FAIL'}")
    emit("")

    emit("Per-mode diagonal at the finest rule (should be exactly 1):")
    gram = gram_gauss_legendre(N_MODES, 120, 512)
    for j in (1, 2, 4, 6, 8, 11, 15, 22, 36):
        n, m = noll_to_nm(j)
        emit(
            f"  j={j:>3}  (n={n}, m={m:+d})  {mode_name(n, m):<30} "
            f"<Z,Z> = {gram[j - 1, j - 1]:.15f}"
        )
    emit("")

    emit("B. Uniform Cartesian pupil grid with circular mask (discretisation error)")
    emit(f"{'n_pix':>8} {'points in pupil':>16} {'max|G-I| diag':>16} {'max|G| off-diag':>18}")
    for n_pix in (32, 64, 128, 256, 512):
        gram_c = gram_cartesian(N_MODES, n_pix)
        diag = np.max(np.abs(np.diag(gram_c) - 1.0))
        off = np.max(np.abs(gram_c - np.diag(np.diag(gram_c))))
        _, _, mask = unit_disc_grid(n_pix)
        emit(f"{n_pix:>8} {int(mask.sum()):>16} {diag:>16.3e} {off:>18.3e}")
    emit("")
    emit("Note: B does not converge to machine precision. A masked square grid is")
    emit("a first-order-accurate quadrature of a disc (the boundary is jagged), so")
    emit("orthogonality is only approximate on sampled data. This is why")
    emit("zernkit.fitting uses least squares rather than projection integrals.")

    out = Path(__file__).with_name("orthonormality_output.txt")
    out.write_text("\n".join(lines) + "\n")
    return 0 if worst_a < 1e-12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
