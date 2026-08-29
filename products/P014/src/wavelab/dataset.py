"""Deterministic synthetic slope datasets from Kolmogorov phase screens.

**Everything here is synthetic.** No laboratory, on-sky or flight data is used
anywhere in this product; see `DATASET_CARD.md`.

Construction
------------
A screen is generated on an FFT grid whose spacing is exactly **half the
subaperture pitch**, so that both the Fried phase points (cell corners) and the
Southwell phase points (cell centres) fall on grid nodes exactly. Nothing is
interpolated: the phase and both gradient components are read off grid nodes,
and the gradients are the exact spectral derivatives of the band-limited screen
(:mod:`wavelab.turbulence`).

Per sample the generator produces

* ``u`` -- the noise-free scaled slopes ``pitch * grad(phi)`` [rad], point
  sampled at the illuminated subaperture centres, x components then y;
* ``coeffs`` -- the least-squares Zernike coefficients [rad] of the same screen
  over the pupil, in the Noll-orthonormal basis (:mod:`wavelab.zernike`);
* ``phase_southwell`` / ``phase_fried`` -- the true phase [rad] at each
  geometry's own reconstruction points, piston removed, for zonal error
  scoring.

Two model errors are deliberately left in, because they are real:

* A physical subaperture measures the **area average** of the gradient; this
  generator point samples it at the subaperture centre. The difference is
  ``O(h^2)`` in the gradient's curvature and is quantified in
  `validation/VALIDATION.md`.
* The screen contains spatial frequencies far above the ``n_modes`` Zernike
  modes being estimated, so the label is a projection, not a complete
  description, and both the classical and the learned estimator suffer the
  resulting aliasing. This is what a real sensor sees.

Turbulence strength enters only as the ratio ``D / r0``. Because a Kolmogorov
phase screen scales as ``r0**(-5/6)`` (the PSD scales as ``r0**(-5/3)``,
Roddier 1981), screens are generated once at a reference ``r0`` and rescaled
exactly per sample; no regeneration is needed to change strength.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .geometry import SubapertureGeometry
from .turbulence import KolmogorovScreens
from .zernike import zernike_basis

__all__ = ["SlopeDataset", "generate_dataset", "make_measurements", "DEFAULT_NOLL_INDICES"]

#: Noll indices estimated by default: tip/tilt through the fourth radial order.
DEFAULT_NOLL_INDICES: tuple[int, ...] = tuple(range(2, 22))


@dataclass(frozen=True)
class SlopeDataset:
    """A batch of noise-free synthetic slope samples with Zernike labels.

    Attributes
    ----------
    u : ndarray, shape (n, n_slopes)
        Noise-free scaled slopes [rad].
    coeffs : ndarray, shape (n, n_modes)
        Least-squares Zernike coefficients of the true screen [rad].
    phase_southwell : ndarray, shape (n, n_southwell)
        True phase at the Southwell points [rad], piston removed.
    phase_fried : ndarray, shape (n, n_fried)
        True phase at the Fried points [rad], piston removed.
    d_over_r0 : ndarray, shape (n,)
        Turbulence strength of each sample [-].
    noll_indices : tuple[int, ...]
        Noll indices of the coefficient columns.
    seed : int
        Seed that produced this dataset.
    """

    u: NDArray[np.float64]
    coeffs: NDArray[np.float64]
    phase_southwell: NDArray[np.float64]
    phase_fried: NDArray[np.float64]
    d_over_r0: NDArray[np.float64]
    noll_indices: tuple[int, ...]
    seed: int

    def __len__(self) -> int:
        return int(self.u.shape[0])


def _grid_indices(geom: SubapertureGeometry, n_grid: int) -> dict[str, NDArray[np.int_]]:
    """Exact FFT-grid indices of the sampling points (no interpolation)."""
    n = int(geom.n_sub)
    c = n_grid // 2
    if c + n > n_grid or c - n < 0:
        raise ValueError(
            f"n_grid={n_grid} is too small for n_sub={n}; need n_grid >= {2 * n + 2}"
        )
    # Grid spacing is pitch/2, so offsets are in half-pitch units.
    centre_off = 2 * np.arange(n) + 1 - n  # subaperture centres
    corner_off = 2 * np.arange(n + 1) - n  # cell corners
    return {
        "centre": c + centre_off,
        "corner": c + corner_off,
    }


def generate_dataset(
    geom: SubapertureGeometry,
    n_samples: int,
    seed: int,
    noll_indices: tuple[int, ...] = DEFAULT_NOLL_INDICES,
    d_over_r0_range: tuple[float, float] = (10.0, 10.0),
    n_grid: int = 64,
    n_subharmonics: int = 3,
    chunk: int = 256,
) -> SlopeDataset:
    """Generate a deterministic synthetic slope dataset.

    Parameters
    ----------
    geom : SubapertureGeometry
        Lenslet layout.
    n_samples : int
        Number of independent screens [-], >= 1.
    seed : int
        Seed for ``numpy.random.default_rng``. Fixing it makes the dataset
        bit-for-bit reproducible.
    noll_indices : tuple[int, ...]
        Noll indices to label (all >= 2; piston is unobservable).
    d_over_r0_range : (float, float)
        Inclusive log-uniform range of ``D / r0`` [-]. Equal endpoints fix it.
    n_grid : int
        FFT grid side [-]. Must satisfy ``n_grid >= 2 * n_sub + 2``; the grid
        spacing is ``pitch / 2`` so the screen period is ``n_grid / 2``
        subapertures.
    n_subharmonics : int
        Lane et al. (1992) subharmonic levels [-].
    chunk : int
        Screens generated per FFT batch [-]; affects memory only, not results.

    Returns
    -------
    SlopeDataset
    """
    if not isinstance(geom, SubapertureGeometry):
        raise TypeError(f"geom must be a SubapertureGeometry, got {type(geom).__name__}")
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise TypeError(f"n_samples must be an integer, got {n_samples!r}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    js = tuple(int(j) for j in noll_indices)
    if len(js) == 0 or any(j < 2 for j in js):
        raise ValueError("noll_indices must be non-empty with every index >= 2")
    lo, hi = (float(d_over_r0_range[0]), float(d_over_r0_range[1]))
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0.0 or hi < lo:
        raise ValueError(f"d_over_r0_range must satisfy 0 < lo <= hi, got {d_over_r0_range!r}")
    if isinstance(chunk, bool) or not isinstance(chunk, (int, np.integer)) or chunk < 1:
        raise ValueError(f"chunk must be an integer >= 1, got {chunk!r}")

    idx = _grid_indices(geom, int(n_grid))
    n = int(geom.n_sub)
    pitch = geom.pitch
    diameter = float(geom.diameter)
    r0_ref = diameter / 10.0

    screens = KolmogorovScreens(
        n_grid=int(n_grid),
        dx=pitch / 2.0,
        r0=r0_ref,
        n_subharmonics=int(n_subharmonics),
        seed=int(seed),
    )
    rng_scale = np.random.default_rng(int(seed) + 1_000_003)

    # --- sampling index sets -------------------------------------------------
    mask = geom.mask
    cy, cx = np.nonzero(mask)  # row-major over [iy, ix], matches slope ordering
    sub_iy = idx["centre"][cy]
    sub_ix = idx["centre"][cx]

    corner_active = geom.corner_mask()
    fy, fx = np.nonzero(corner_active)
    fr_iy = idx["corner"][fy]
    fr_ix = idx["corner"][fx]

    # Pupil sampling for the Zernike projection: every grid node of the
    # half-pitch grid that lies inside the pupil.
    centre_index = int(n_grid) // 2
    off = np.arange(-n, n + 1)
    pup_index = centre_index + off
    gy, gx = np.meshgrid(pup_index, pup_index, indexing="ij")
    xn, yn = np.meshgrid(off / n, off / n, indexing="xy")
    inside = xn**2 + yn**2 <= 1.0
    pup_iy, pup_ix = gy[inside], gx[inside]
    basis = zernike_basis(list(js), xn[inside], yn[inside])
    proj = np.linalg.pinv(basis)

    sw_pts = geom.southwell_points()[0].size
    fr_pts = fr_iy.size

    u_all = np.empty((int(n_samples), geom.n_slopes))
    c_all = np.empty((int(n_samples), len(js)))
    sw_all = np.empty((int(n_samples), sw_pts))
    fr_all = np.empty((int(n_samples), fr_pts))

    done = 0
    while done < int(n_samples):
        k = min(int(chunk), int(n_samples) - done)
        phase, gxs, gys = screens.generate(k)
        sl = slice(done, done + k)
        u_all[sl, : geom.n_valid_sub] = gxs[:, sub_iy, sub_ix] * pitch
        u_all[sl, geom.n_valid_sub :] = gys[:, sub_iy, sub_ix] * pitch
        pup = phase[:, pup_iy, pup_ix]
        pup = pup - pup.mean(axis=1, keepdims=True)
        c_all[sl] = pup @ proj.T
        sw = phase[:, sub_iy, sub_ix]
        sw_all[sl] = sw - sw.mean(axis=1, keepdims=True)
        fr = phase[:, fr_iy, fr_ix]
        fr_all[sl] = fr - fr.mean(axis=1, keepdims=True)
        done += k

    if hi > lo:
        d_r0 = np.exp(rng_scale.uniform(np.log(lo), np.log(hi), size=int(n_samples)))
    else:
        d_r0 = np.full(int(n_samples), lo)
    # phase amplitude scales as r0**(-5/6) = (D/r0)**(5/6) / D**(5/6)
    scale = (d_r0 / 10.0) ** (5.0 / 6.0)
    s = scale[:, None]
    return SlopeDataset(
        u=u_all * s,
        coeffs=c_all * s,
        phase_southwell=sw_all * s,
        phase_fried=fr_all * s,
        d_over_r0=d_r0,
        noll_indices=js,
        seed=int(seed),
    )


def make_measurements(
    u_clean: NDArray[np.float64],
    n_photons: NDArray[np.float64] | float,
    dropout_rate: NDArray[np.float64] | float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """Apply photon noise and subaperture dropout to clean slopes.

    Parameters
    ----------
    u_clean : ndarray, shape (n, 2 * n_sub)
        Noise-free scaled slopes [rad].
    n_photons : float or ndarray, shape (n,)
        Photons per subaperture [-], > 0.
    dropout_rate : float or ndarray, shape (n,)
        Per-sample probability of losing a subaperture [-], in ``[0, 1)``.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    u_meas : ndarray, shape (n, 2 * n_sub)
        Noisy slopes with dropped subapertures set to exactly 0.0 [rad].
    available : ndarray of bool, shape (n, n_sub)
        Subaperture availability.
    n_photons : ndarray, shape (n,)
        The per-sample photon counts actually used.
    """
    from .noise import photon_slope_noise

    arr = np.asarray(u_clean, dtype=float)
    if arr.ndim != 2 or arr.shape[1] % 2 != 0:
        raise ValueError(f"u_clean must be 2-D with an even width, got shape {arr.shape}")
    n, two_ns = arr.shape
    n_sub = two_ns // 2
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")

    nph = np.broadcast_to(np.asarray(n_photons, dtype=float), (n,)).copy()
    rate = np.broadcast_to(np.asarray(dropout_rate, dtype=float), (n,)).copy()
    if np.any(~np.isfinite(rate)) or np.any(rate < 0.0) or np.any(rate >= 1.0):
        raise ValueError("dropout_rate must be finite and in [0, 1)")

    sigma = photon_slope_noise(nph)[:, None]
    u_meas = arr + sigma * rng.standard_normal(arr.shape)
    available = rng.random((n, n_sub)) >= rate[:, None]
    u_meas = u_meas * np.concatenate([available, available], axis=1)
    return u_meas, available, nph
