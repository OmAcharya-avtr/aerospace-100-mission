"""Validation 1 -- noise-free exactness of the three reconstructors.

Checks, in order:

1. Modal (Zernike) least squares recovers an injected coefficient vector from
   noise-free slopes to numerical tolerance.
2. Southwell zonal least squares recovers a quadratic phase exactly (its
   trapezoidal relation is algebraically exact for quadratics).
3. Fried zonal least squares recovers the same phase exactly **modulo its
   analytically identified null space** (piston + waffle), and the size of the
   unobservable part is reported rather than hidden.
4. The singular spectra of both geometry operators, showing the rank deficiency
   that motivates the regularisation.
5. The size of the model error that remains when the slopes are true
   subaperture **area averages** rather than centre point samples -- the
   approximation the whole product rests on.

Run:  python validation/validate_noise_free.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wavelab.geometry import SubapertureGeometry, build_geometry_matrices  # noqa: E402
from wavelab.reconstruct import ModalReconstructor, ZonalReconstructor  # noqa: E402
from wavelab.zernike import zernike_gradient_noll, zernike_noll  # noqa: E402

TOL_EXACT = 1e-12  # rad; a few hundred times the double-precision epsilon
MODES = tuple(range(2, 22))
QUADRATIC = {2: 0.7, 3: -0.4, 4: 1.3, 5: 0.6, 6: -0.9}

results: list[tuple[str, str, float, float, bool]] = []


def record(name: str, quantity: str, value: float, tol: float, ok: bool) -> None:
    results.append((name, quantity, value, tol, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {quantity} = {value:.4e}  (tolerance {tol:.1e})")


def quadratic_slopes(geom: SubapertureGeometry) -> np.ndarray:
    cx, cy = geom.subaperture_centres()
    ux = np.zeros_like(cx)
    uy = np.zeros_like(cy)
    for j, c in QUADRATIC.items():
        gx, gy = zernike_gradient_noll(j, cx, cy)
        ux += c * gx
        uy += c * gy
    return np.concatenate([ux, uy]) * geom.scaled_slope_factor


def quadratic_phase(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.zeros_like(x)
    for j, c in QUADRATIC.items():
        p += c * zernike_noll(j, x, y)
    return p - p.mean()


def area_average_slopes(geom: SubapertureGeometry, quad: int = 16) -> np.ndarray:
    """Slopes as the true area average of the gradient over each subaperture."""
    cx, cy = geom.subaperture_centres()
    half = 1.0 / geom.n_sub  # half a subaperture in normalised coordinates
    off = ((np.arange(quad) + 0.5) / quad - 0.5) * 2.0 * half
    ox, oy = np.meshgrid(off, off, indexing="xy")
    ux = np.zeros_like(cx)
    uy = np.zeros_like(cy)
    for j, c in QUADRATIC.items():
        gx, gy = zernike_gradient_noll(
            j, (cx[:, None, None] + ox).ravel(), (cy[:, None, None] + oy).ravel()
        )
        ux += c * gx.reshape(cx.size, quad, quad).mean(axis=(1, 2))
        uy += c * gy.reshape(cy.size, quad, quad).mean(axis=(1, 2))
    return np.concatenate([ux, uy]) * geom.scaled_slope_factor


def main() -> int:
    geom = SubapertureGeometry(n_sub=8, diameter=1.0, fill_threshold=0.5)
    print("WaveLab validation 1 -- noise-free exactness")
    print(f"geometry: n_sub = {geom.n_sub}, D = {geom.diameter} m, pitch = {geom.pitch} m")
    print(f"illuminated subapertures = {geom.n_valid_sub}, slopes = {geom.n_slopes}")
    print(f"Southwell phase points = {geom.southwell_points()[0].size}")
    print(f"Fried phase points     = {geom.fried_points()[0].size}")
    print(f"modes estimated (Noll)  = {MODES[0]}..{MODES[-1]} ({len(MODES)} modes)")
    print()

    print("[1] Modal least squares: pure Zernike in, same Zernike out")
    rec = ModalReconstructor(geom, MODES, "tsvd", 1e-8)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        a = rng.normal(size=len(MODES))
        worst = max(worst, float(np.max(np.abs(rec.reconstruct(rec.forward(a)) - a))))
    record("modal", "worst |a_hat - a| over 200 random vectors [rad]", worst, TOL_EXACT,
           worst < TOL_EXACT)
    print()

    print("[2] Southwell zonal: quadratic phase (Noll j = 2,3,4,5,6)")
    u = quadratic_slopes(geom)
    zs = ZonalReconstructor(geom, "southwell", "tsvd", 1e-6)
    px, py = zs.phase_points()
    truth = quadratic_phase(px, py)
    err_s = float(np.max(np.abs(zs.reconstruct(u) - truth)))
    print(f"  true phase RMS over the points = {np.std(truth):.4f} rad")
    record("southwell", "max |p_hat - p| [rad]", err_s, TOL_EXACT, err_s < TOL_EXACT)
    print()

    print("[3] Fried zonal: same phase, reported modulo its null space")
    zf = ZonalReconstructor(geom, "fried", "tsvd", 1e-6)
    fx, fy = zf.phase_points()
    truth_f = quadratic_phase(fx, fy)
    a_mat = build_geometry_matrices(geom, "fried").a
    _, sv, vt = np.linalg.svd(a_mat)
    n_null = int(np.count_nonzero(sv <= 1e-9 * sv[0]))
    null = vt[len(sv) - n_null :]
    resid = zf.reconstruct(u) - truth_f
    observable = resid - null.T @ (null @ resid)
    print(f"  null-space dimension = {n_null} (piston + waffle)")
    print(f"  unobservable component of the residual = {np.abs(null @ resid)} rad")
    err_f = float(np.max(np.abs(observable)))
    record("fried", "max |observable part of p_hat - p| [rad]", err_f, TOL_EXACT,
           err_f < TOL_EXACT)
    print()

    print("[4] Singular spectra of the geometry operators A")
    for name in ("southwell", "fried"):
        s = np.linalg.svd(build_geometry_matrices(geom, name).a, compute_uv=False)
        n_small = int(np.count_nonzero(s <= 1e-9 * s[0]))
        print(f"  {name:10s} shape {build_geometry_matrices(geom, name).a.shape}"
              f" sigma_max = {s[0]:.4f}  sigma_min(nonzero) = {s[s > 1e-9 * s[0]][-1]:.4e}"
              f"  condition = {s[0] / s[s > 1e-9 * s[0]][-1]:.2f}"
              f"  #(sigma <= 1e-9 sigma_max) = {n_small}")
    print()

    print("[5] Model error from point-sampled versus area-averaged subaperture slopes")
    u_area = area_average_slopes(geom)
    rel = float(np.max(np.abs(u_area - u)) / np.max(np.abs(u)))
    print(f"  max |u_area - u_point| / max |u_point| = {rel:.4e}")
    est_area = rec.reconstruct(u_area)
    a_true = np.zeros(len(MODES))
    for j, c in QUADRATIC.items():
        a_true[MODES.index(j)] = c
    coef_err = float(np.sqrt(np.sum((est_area - a_true) ** 2)))
    print(f"  resulting modal residual wavefront RMS = {coef_err:.4e} rad")
    print(f"  as a fraction of the input wavefront RMS "
          f"({np.sqrt(np.sum(a_true**2)):.4f} rad) = {coef_err / np.sqrt(np.sum(a_true**2)):.4e}")
    # This is a documented model error, not a pass/fail check: it is reported.
    print()

    n_fail = sum(1 for r in results if not r[4])
    print(f"SUMMARY: {len(results) - n_fail} passed, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
